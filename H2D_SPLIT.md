# H2D_SPLIT — targeted measurement REFUTES the make_array hypothesis

gemma_2b-shaped batch on the data-axis sharding, RTX 5090. Median over 20 iters.

| batch | bytes | gen (np.random) | make_array per-leaf | device_put batched | make_array/device_put |
|------:|------:|----------------:|--------------------:|-------------------:|:---------------------:|
| 1 | 1.8 MB | 4.24 ms | **0.41 ms** | 0.32 ms | 1.3× |
| 4 | 7.3 MB | 18.2 ms | **0.44 ms** | 0.51 ms | 0.9× |
| 8 | 14.5 MB | 35.5 ms | **0.86 ms** | 0.81 ms | 1.1× |
| 16 | 29.0 MB | 73.6 ms | **2.26 ms** | 2.37 ms | 1.0× |

## Verdict: do NOT implement batched H2D

- `make_array_from_process_local_data` is **0.4–2.3 ms** — sub-millisecond to ~2 ms. **Not** the bottleneck.
- A batched `device_put` is the **same** speed (≈1×). The proposed optimization would move nothing.
- `np.random` generation (4–74 ms) is a FakeData artifact, not a production cost — and even it is far smaller
  than the loop's measured data-wait (119–659 ms/step).

## So what IS the loop's 119–659 ms/step host cost?

By elimination it is **not** data assembly (make_array, 0.4–2 ms) and **not** generation (hidden by prefetch;
identical data-wait at num_workers=0 vs 4). In the async-dispatch loop the per-step wall is the **host time to
launch the gemma_2b step** — i.e. **jitted-step dispatch / Python-side launch overhead** for this large model —
which dwarfs the 11 ms of device compute at small batch. Pinning that precisely needs a python-side dispatch
profile (cProfile / jax dispatch timing), not an H2D change.

## Net for V2

1. **Compute: GEMM-bound, attention ≈ 0% → no model-kernel headroom.** (final)
2. **make_array H2D is fast → batched-H2D optimization is unwarranted (refuted by measurement).**
3. The dominant cost is host-side step-launch overhead, confounded by the small-batch FakeData setup. The
   honest, measurement-respecting move is to NOT optimize a refuted target; the next real step is a python-side
   dispatch profile (or a real-dataset run) before committing to any change.
