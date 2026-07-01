# FINDINGS — real H100 measurements (gemma_2b)

All numbers from `feat/jax-multihost-v1` on a runpod **4× H100 80GB** node, using the corrected profiler
(device time = median BLOCKED step; MFU = 6·N·D / device_time). Random-init weights (structure is what
throughput depends on).

## Single-GPU step

| config | device ms/step | MFU | dominant |
|---|---|---|---|
| gemma_2b, batch 4, full AdamW | 211.9 | **32.1%** | matmul-FFN 58%; attention fused (see below) |
| gemma_2b, batch 4, fwd+bwd (no opt) | 195.6 | **34.7%** | — |

## Attention cost — measured by ablation (the trace can't isolate it)

The step-breakdown showed attention ≈ 0%, which is a **trace artifact**: on GPU, XLA fuses gemma's masked
`jax.nn.softmax` into the adjacent einsum fusion, so its time is attributed to matmul-FFN. Measured directly
(`scripts/profile_attention.py`, gemma_2b shapes: B=4, seq=866, 8q/1kv GQA, Hd=256, ×18 layers):

| attention impl | ms/layer | ms/step (×18) | % of 211.9 ms step |
|---|---:|---:|---:|
| naive (gemma.py einsum+softmax+einsum) | 0.549 | 9.89 | **4.7%** |
| flash (`dot_product_attention`, xla) | 0.277 | 4.98 | 2.4% |

⇒ **Attention is ~4.7% of the step, not 0%.** Fused attention would save only ~2.3% of the step — below the
~15–20% gate, so it is **not worth it** (attention is small because the FFN `mlp_dim=16384` GEMMs dominate at
this modest sequence length). The V2 conclusion — **compute/GEMM-bound, no worthwhile single-GPU kernel
headroom** — stands, now on a real measurement rather than a bogus 0%.

## FSDP strong scaling (fixed global batch 4, fwd+bwd, `sharding.fsdp_sharding` across N GPUs)

| GPUs | device ms/step | speedup | aggregate TFLOP/s | per-GPU MFU |
|-----:|---------------:|--------:|------------------:|------------:|
| 1 | 195.6 | 1.00× | 344 | 34.7% |
| 2 | 169.3 | 1.16× | 398 | 20.1% |
| 4 | 135.5 | 1.44× | 497 | 12.5% |

**Reading:** FSDP works (device time drops with more GPUs) but scales **poorly at this batch**. At global batch 4,
4-GPU FSDP = **1 sample/GPU**, so the per-layer **all-gather of sharded params** dominates (comms-bound) —
per-GPU MFU collapses 35% → 12.5%. FSDP shards *memory* well but needs enough per-GPU compute to hide the
all-gather.

## FSDP weak scaling (global batch = 4 × N, i.e. 4 samples/GPU constant)

| GPUs | global batch | device ms/step | aggregate TFLOP/s | per-GPU MFU |
|-----:|-------------:|---------------:|------------------:|------------:|
| 1 | 4 | 195.5 | 344 | 34.8% |
| 2 | 8 | 249.9 | 539 | 27.2% |
| 4 | 16 | 274.6 | **980** | 24.8% |

**Reading:** with per-GPU compute held constant, **aggregate throughput scales 2.85× on 4 GPUs** (344→980
TFLOP/s) at **24.8% MFU** — far better than strong scaling (1.44× / 12.5%). The residual loss: device time grows
195→275 ms (should stay flat) ⇒ ~71% of ideal weak scaling. That growth is the **FSDP all-gather not fully
overlapping compute** — the classic comms/compute-overlap opportunity (XLA latency-hiding scheduler, collective
placement/tuning).

### Strong vs weak (4 GPUs)
| | speedup / aggregate | 4-GPU per-GPU MFU |
|---|---|---|
| strong (fixed batch 4) | 1.44× (497 TFLOP/s) | 12.5% |
| weak (batch 16 = 4/GPU) | 2.85× (980 TFLOP/s) | 24.8% |

**Conclusion:** the model step is single-GPU compute-bound (no kernel headroom); throughput scaling comes from
FSDP, which needs **enough per-GPU batch** (weak scaling) to be efficient, and whose remaining ~30% overhead is
**all-gather comms** → the next real optimization is **comms/compute overlap**, not anything in the model step.

## Multi-host device model (see MULTIHOST_VALIDATION.md)

4 processes × 1 H100 via `jax.distributed`: `device_count`=4 global, mesh spans all 4. Core multi-host validated
on real CUDA. (Cross-*process* NCCL collective hung on this node — env issue, flagged; single-process multi-GPU
FSDP above is the reliable single-node path and produced the scaling curve.)

## Instrument note

Earlier "91% data-wait" and "0.2% MFU" were XLA-metric artifacts (trace-op-duration sum under-counts device
time ~12×; `cost_analysis` FLOPs under-reports ~100×). Fixed: blocked step time + 6·N·D. See DISPATCH_PROFILE.md
and STEP_BREAKDOWN.md.

---
*(Prior version of this file was the CPU-sim scaling scaffold from V1/M5; superseded by these real-H100 numbers.)*
