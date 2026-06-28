# TRAIN_LOOP_BREAKDOWN — real loop with data pipeline (clean-wall data-wait)

> ✅ **Overlapped train.py-style loop**, `gemma_2b` bf16 fwd+bwd, batch=1, FakeData pipeline (host numpy gen +
> collate + H2D, num_workers=0 equivalent), on an NVIDIA GeForce RTX 5090, via runpod. Wall is timed **without**
> the profiler (Phase A) so it isn't inflated by trace overhead; op composition is from a separate short traced
> window (Phase B). `data-wait = wall − device_busy` is genuine host/input cost not overlapped behind compute.

**Window: 12 steps | clean wall 130.58 ms/step | device-busy 10.89 ms/step | data-wait 119.69 ms/step (91.7%)**

| category | device ms/step | % wall |
|----------|---------------:|-------:|
| matmul-FFN | 7.084 | 5.4% |
| other | 3.394 | 2.6% |
| vision | 0.257 | 0.2% |
| embedding | 0.159 | 0.1% |
| attention | 0.000 | 0.0% |
| collectives | 0.000 | 0.0% |
| optimizer | 0.000 | 0.0% |
| **data-wait** | 119.69 | **91.7%** |

## The measurement journey (why this number is trustworthy now)

- Single-step microbench: 95.5% "data-wait" — but it had a *fixed on-device input*, so that was pure
  `jax.profiler` overhead, not data-wait. Flagged, not used.
- First loop attempt: 95.9% "data-wait" at 264 ms/step wall — still inflated, because wall was timed *inside*
  the profiler trace. Fixed by measuring clean wall (Phase A) separately from the traced op window (Phase B).
- **This run:** clean wall is 130.6 ms/step (the ~134 ms difference *was* trace overhead). data-wait is now the
  real, un-traced host/input cost: **119.7 ms/step vs 10.9 ms compute.**

## What's solid

1. **GPU compute ≈ 10.9 ms/step and is GEMM-bound** (matmul-FFN ~65% of device-busy); **attention softmax ≈ 0%.**
   ⇒ **No fused-attention headroom**, and the GEMMs are XLA-saturated (don't touch). Compute is *not* the
   bottleneck.
2. **The GPU is ~92% idle waiting on the host.** Even cleanly measured, the per-step host/input path
   (≈120 ms) dwarfs compute (≈11 ms). The bottleneck is the **input/host pipeline**, not the model.

## Caveat on the absolute number

This uses the FakeData + `num_workers=0` path generated inline: **synchronous** host generation with **no
prefetch**, and `make_array_from_process_local_data` called **per leaf per step** (multi-host-oriented, not
optimized for single-host). So ~120 ms over-states what a tuned, prefetching loader would show. The
*qualitative* conclusion is robust (host/input-bound, compute cheap, no attention headroom); the *absolute*
data-wait should be re-measured against the real torch loader with `num_workers>0` prefetch (and a real dataset
to fold in image-decode/disk).

## Routing decision (V2)

- ❌ fused attention — attention ≈ 0% (ruled out by measurement).
- ❌ matmul/GEMM — XLA-saturated.
- ❌ collectives — single-host here; revisit multi-host.
- ✅ **Input/host pipeline is the measured bottleneck.** Levers: background **prefetch + compute/load overlap**,
  **batched H2D** (one transfer per batch instead of per-leaf `make_array_from_process_local_data`), and larger
  batches to amortize per-step host overhead. Next: re-measure with the real prefetching loader to choose the
  single highest-impact change, then BEFORE→change→AFTER with the parity gate.
