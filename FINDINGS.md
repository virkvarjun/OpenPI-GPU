# FINDINGS — real H100 measurements (gemma_2b)

All numbers from `feat/jax-multihost-v1` on a runpod **4× H100 80GB** node, using the corrected profiler
(device time = median BLOCKED step; MFU = 6·N·D / device_time). Random-init weights (structure is what
throughput depends on).

## Single-GPU step

| config | device ms/step | MFU | dominant |
|---|---|---|---|
| gemma_2b, batch 4, full AdamW | 211.9 | **32.1%** | matmul-FFN 58%, attention ~0% |
| gemma_2b, batch 4, fwd+bwd (no opt) | 195.6 | **34.7%** | — |

⇒ The step is **compute-bound, GEMM-saturated, attention ≈ 0%** → no single-GPU kernel headroom (V2 conclusion).

## FSDP strong scaling (fixed global batch 4, fwd+bwd, `sharding.fsdp_sharding` across N GPUs)

| GPUs | device ms/step | speedup | aggregate TFLOP/s | per-GPU MFU |
|-----:|---------------:|--------:|------------------:|------------:|
| 1 | 195.6 | 1.00× | 344 | 34.7% |
| 2 | 169.3 | 1.16× | 398 | 20.1% |
| 4 | 135.5 | 1.44× | 497 | 12.5% |

**Reading:** FSDP works (device time drops with more GPUs) but scales **poorly at this batch**. At global batch 4,
4-GPU FSDP = **1 sample/GPU**, so the per-layer **all-gather of sharded params** dominates (comms-bound) —
per-GPU MFU collapses 35% → 12.5%. This is the expected FSDP regime: it shards *memory* well but needs enough
per-GPU compute to hide the all-gather. **Lever: weak scaling** — grow the global batch with GPU count so each
GPU keeps ~batch-4 of compute; aggregate throughput should then scale much closer to linear at ~35% MFU.
(Weak-scaling sweep is the natural next measurement.)

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
