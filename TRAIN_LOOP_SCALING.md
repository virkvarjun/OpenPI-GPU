# TRAIN_LOOP_SCALING — batch-size sweep (gemma_2b, RTX 5090, real prefetching loader)

Real openpi `TorchDataLoader` (num_workers=4 prefetch) + gemma_2b bf16 fwd+bwd, clean-wall (profiler-free)
timing. FakeData (numpy gen, no disk).

| batch | clean wall ms/step | device-busy ms/step | data-wait ms/step | data-wait % | device/sample | data-wait/sample |
|------:|-------------------:|--------------------:|------------------:|:-----------:|:-------------:|:----------------:|
| 1 | 130.4 | 10.9 | 119.5 | 91.6% | 10.9 | 119.5 |
| 2 | 219.0 | 16.8 | 202.2 | 92.3% | 8.4 | 101.1 |
| 4 | 383.6 | 37.1 | 346.4 | 90.3% | 9.3 | 86.6 |
| 8 | 724.9 | 65.5 | 659.3 | 91.0% | 8.2 | 82.4 |

## Reading

- **data-wait stays ~91% at every batch size and its absolute value scales ~linearly with batch** (119→659 ms).
  ⇒ the host cost is **data-volume work**, NOT a fixed per-step dispatch cost (which would stay constant and
  shrink as a %). Bigger batches barely move the host-boundness.
- **Prefetch (num_workers=4) does not hide it** — identical to num_workers=0. So it's main-thread work the
  background workers can't overlap: the **per-leaf `make_array_from_process_local_data` H2D assembly** (and/or
  the FakeData host generation), which scale with batch bytes.
- device-busy scales with batch (compute grows), per-sample compute amortizes slightly (10.9→8.2 ms).

## Conclusions (V2, measured across many GPU runs)

1. **Model compute is GEMM-bound, attention ≈ 0%** → no fused-attention / compute-kernel headroom (ruled out by
   measurement — the instrument's core job).
2. **The training loop is host-bound (~91%) at all batch sizes**, and the host cost is **data-volume-bound**
   (H2D assembly / generation), not dispatch — and not hidden by prefetch.

## Caveat that gates the FINAL pick

The host cost here mixes two things the sweep can't separate: (a) `make_array_from_process_local_data` per-leaf
H2D assembly (a real framework inefficiency — pathologically slow for the byte volume), and (b) FakeData
`np.random` generation (an artifact — a real dataset decodes JPEGs, a different profile). The next targeted
measurement (time `next_batch()` vs the bare `make_array` vs a real dataset) separates them and picks the single
highest-impact change — most likely **batched single-transfer H2D** replacing the per-leaf assembly.
