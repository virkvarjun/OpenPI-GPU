# TRAIN_LOOP_BREAKDOWN — real prefetching loader (clean-wall data-wait)

> ✅ **Overlapped train.py-style loop with the REAL openpi `TorchDataLoader`** (num_workers=4 background
> prefetch + collate + `make_array_from_process_local_data` H2D), `gemma_2b` bf16 fwd+bwd, **batch=1**, FakeData
> (numpy gen, no disk), on an NVIDIA GeForce RTX 5090, via runpod. Clean wall timed WITHOUT the profiler
> (Phase A); op composition from a separate traced window (Phase B).

**Window: 12 steps | clean wall 130.38 ms/step | device-busy 10.90 ms/step | data-wait 119.48 ms/step (91.6%)**

| category | device ms/step | % wall |
|----------|---------------:|-------:|
| matmul-FFN | 7.072 | 5.4% |
| other | 3.416 | 2.6% |
| vision | 0.257 | 0.2% |
| embedding | 0.156 | 0.1% |
| attention | 0.000 | 0.0% |
| collectives | 0.000 | 0.0% |
| optimizer | 0.000 | 0.0% |
| **data-wait** | 119.48 | **91.6%** |

## The decisive finding: prefetch does NOT help

This num_workers=4 prefetching run (data-wait 119.5 ms) is **essentially identical** to the num_workers=0
inline run (119.7 ms). Background workers hide *data production* — yet data-wait is unchanged. **So the host
bottleneck is not generating/loading data**, it's the **per-step main-thread overhead that prefetch can't
hide**:
- **`make_array_from_process_local_data` called per-leaf, per-step** to assemble each batch as a (global) device
  array — a multi-host primitive with real Python overhead, run on the main thread (~10 leaves/step).
- **jitted-step dispatch** of the gemma_2b executable (large pytree of params/args).

## Big caveat: this is batch=1

At batch=1 the device does only ~11 ms of work while the *fixed* per-step host overhead (~119 ms) dominates —
so 91.6% over-states host-boundness for realistic training. Larger batches amortize the per-step host cost over
more samples **and** grow compute, so the compute/host ratio improves sharply. The batch-size sweep
(`TRAIN_LOOP_SCALING.md`) characterizes this.

## Robust conclusions (across 4 consistent GPU runs)

1. **GPU compute ≈ 11 ms/step, GEMM-bound, attention ≈ 0%** → no fused-attention / compute-kernel headroom; XLA
   saturates the GEMMs.
2. **The loop is host-bound at small batch by per-step assembly + dispatch, NOT by data production** (prefetch
   doesn't move the needle).

## Routing decision (V2)

- ❌ fused attention / GEMM kernels — ruled out by measurement (attention 0%, GEMM saturated).
- ✅ **Reduce per-step host overhead**: (a) **batched H2D** — assemble the batch with one device transfer
  instead of per-leaf `make_array_from_process_local_data`; (b) **realistic batch sizes** to amortize fixed
  dispatch/assembly cost. The batch sweep confirms which dominates before we implement.
