# DISPATCH_PROFILE — the "data-wait" was a profiler artifact; the step is COMPUTE-BOUND

gemma_2b bf16 fwd+bwd, batch=1, fixed on-device input (no data pipeline), RTX 5090.

```
dispatch-only (async jstep() call, no block) =   5.29 ms/step
blocked step (block_until_ready)             = 131.20 ms/step
cProfile of the dispatch loop: dominated by block_until_ready; nnx pytree
flatten/unflatten = ~0.001 s per 1000 calls (negligible). param-state leaves = 50.
```

## What this proves

- **Python dispatch is cheap (~5 ms).** Not the bottleneck. The nnx merge / pytree flatten per call is negligible.
- **The real device step is ~131 ms** (block_until_ready). And the train-loop clean wall was ~130 ms/step ⇒
  **clean_wall ≈ blocked device step ⇒ the loop is COMPUTE-BOUND. The GPU is fed; there is ~no data-wait.**
- **The earlier "91% data-wait" was an instrument artifact.** The attribution device-busy (sum of
  `jax.profiler` Chrome-trace op `dur` fields) was **10.9 ms** — it under-counted true device time (~131 ms) by
  ~12×. `data-wait = wall − undercounted_busy` then manufactured a fake 119 ms. Summed Chrome-trace op
  durations ≠ total device-busy (async/fusion/gaps); the blocked step time is the reliable device measure.

## Sanity check (independent)

gemma_2b fwd+bwd at batch 1 ≈ 6·2.5e9·(~800 tokens) ≈ 1.2e13 FLOP. RTX 5090 ≈ ~100 TFLOP/s bf16 ⇒ ~120 ms at
modest MFU. The measured 131 ms matches — it is genuinely compute-bound, not input-starved.

## Corrected V2 conclusion (the honest endpoint)

1. **The gemma_2b training step is compute-bound (~131 ms/step device on one RTX 5090), GEMM-dominated,
   attention ≈ 0%.** No fused-attention / compute-kernel headroom; XLA saturates the GEMMs.
2. **There is NO data-wait / input-pipeline bottleneck** — that was a profiler-undercounting artifact. The GPU
   is kept fed (clean wall ≈ device step).
3. ⇒ **No profile-identified optimization headroom in the single-GPU step itself.** The lever for this
   compute-bound, GEMM-saturated workload is **throughput via more devices** — i.e. exactly Sharder V1's
   multi-host FSDP/data-parallel path, not a kernel or input-pipeline change.

## Instrument limitation to fix (follow-up)

`attribution.py` should derive device-busy from **device-side step time** (blocked) or XLA cost-analysis, not
from summing Chrome-trace op durations, which under-counts. The op *composition* (relative %) is still
directionally useful; the absolute device-busy / data-wait split from it is not.
