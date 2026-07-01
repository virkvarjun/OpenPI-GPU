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

## FlashAttention — implemented, and it's a real win (bigger than the ablation predicted)

Wired `jax.nn.dot_product_attention` into gemma behind `Pi0Config.use_flash_attention` (parity: bit-identical
loss vs naive, maxdiff **0.0**; `pi0_flash_test.py`). End-to-end gemma_2b fwd+bwd, batch 4, H100:

| attention | device ms/step | MFU |
|---|---:|---:|
| naive | 195.85 | 34.7% |
| **flash** | **177.61** | **38.3%** |

**9.3% faster (18 ms/step)** at pi0's current sequence — *larger* than the forward-only ablation (~2.3%) because
the **backward pass benefits most**: naive stores and backprops through the full T×S attention matrix, while
flash recomputes it in the fused kernel. Plus the memory win (no T×S matrix), which enables bigger batches /
longer context.

### Where it matters more (attention grows ~seq²; `profile_attention.py` ablation)
| seq | naive attn % of step | flash-xla | naive attn-matrix mem |
|----:|---------------------:|----------:|----------------------:|
| 866 (pi0 today) | 4.7% | 2.4% | 0.14 GB/layer |
| 2048 | 22% | 8.3% | 0.81 GB/layer |
| 4096 | 86% | 30% | 3.22 GB/layer |

⇒ At long context (more cameras/frames), flash goes from "nice" to **essential** (memory + up to ~56% step time).

### Honest caveats
- **cuDNN flash is unavailable for gemma_2b**: `head dim must be ≤ 128, got 256` (confirmed — head_dim=128 runs
  cuDNN fine at 1.5% of step). So `flash-xla` is the usable fused path here; a Pallas/splash kernel (or head_dim
  ≤128) would be needed for the cuDNN flash kernel.
- **PagedAttention does NOT apply**: it's an inference-serving KV-cache memory manager (autoregressive decode).
  The training step has no KV cache. It would only matter for pi0-FAST *serving*.

**Verdict:** enable `use_flash_attention=True` — free 9.3% + memory now, and the key unlock for longer context.
The compute is still GEMM-dominated (matmul-FFN ~58%), so scale-out (FSDP) remains the main throughput lever.

## Comms/compute overlap — DONE (tuned XLA flags, modest real gain)

The weak-scaling residual (device time grows with GPUs) is FSDP all-gather/reduce-scatter not fully hidden behind
compute. Tuned XLA flags (`scripts/fsdp_xla_flags.sh`: latency-hiding scheduler + pipelined async collectives +
large combine thresholds) vs default, weak scaling, **flash attention on** (so 1-GPU anchor is 178 ms, not 195):

| GPUs | default ms | tuned ms | gain | tuned aggregate | tuned MFU |
|-----:|-----------:|---------:|-----:|----------------:|----------:|
| 1 | 178.2 | 178.2 | — (no collectives) | 378 TFLOP/s | 38.1% |
| 2 | 232.3 | 227.0 | **2.3%** | 593 TFLOP/s | 29.9% |
| 4 | 255.9 | 246.0 | **3.9%** | 1094 TFLOP/s | 27.6% |

**The gain is real but modest, and grows with GPU count** (more comms to hide → 2.3% @ 2-GPU, 3.9% @ 4-GPU).
Honest reading: **recent jaxlib already overlaps most collectives by default**, so the tuned flags recover only
the last few percent — the FSDP comms are already largely hidden. The remaining weak-scaling gap (178→246 ms even
tuned) is now mostly the *fundamental* un-hideable reduce-scatter/all-gather on the critical path, not a tuning
miss. So: ~4% free at 4-GPU, worth setting, but there's **no large overlap headroom left** — the comms tax is
close to its practical floor for this model/config. See `figures/comms_overlap.png`.

## Figures

`python scripts/plot_results.py` regenerates all of these into `figures/` from the measured numbers:
`fsdp_scaling_time.png`, `fsdp_scaling_throughput.png`, `fsdp_scaling_mfu.png` (strong vs weak),
`attention_vs_seq.png` (attention grows ~seq²), `flash_end_to_end.png` (9.3% win), `comms_overlap.png`.

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
