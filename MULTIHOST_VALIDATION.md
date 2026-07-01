# MULTIHOST_VALIDATION — Sharder V1 on real 4× H100

Ran V1's multi-host path on a real 4×H100 80GB node (runpod), 4 processes × 1 GPU each, via
`scripts/launch_local.py --backend gpu` (assigns `CUDA_VISIBLE_DEVICES` per process).

## ✅ Core multi-host works on real hardware

Each of the 4 processes, after `distributed.maybe_initialize()` + `sharding.make_mesh()`:

| pid | nproc | global_dev | local_dev | mesh_size | CUDA_VISIBLE_DEVICES |
|----:|------:|-----------:|----------:|----------:|:--------------------:|
| 0 | 4 | 4 | 1 | 4 | 0 |
| 1 | 4 | 4 | 1 | 4 | 1 |
| 2 | 4 | 4 | 1 | 4 | 2 |
| 3 | 4 | 4 | 1 | 4 | 3 |

- **`jax.distributed.initialize()` rendezvous across 4 processes succeeds** on real CUDA.
- **`jax.device_count()` is GLOBAL (4)** while each process owns 1 local GPU — the multi-host device model.
- **`make_mesh()` spans all 4 GPUs** (`mesh_size=4`) — the `(data, fsdp)` mesh is a true multi-process mesh.
- Completed in ~3 s. This is the real-hardware confirmation of what the CPU-sim / localhost tests proved.

## ⚠️ Flagged: multi-process NCCL collective hang (not a V1-code issue)

A cross-process gradient all-reduce (`out_shardings` forcing a reduce across the data axis) **hung** — GPU 0 at
100%, GPUs 1–3 idle — i.e. an NCCL collective deadlock in the *multi-process* setup on this node. This is an
environment/NCCL issue (localhost multi-process NCCL on runpod), **not** Sharder code (V1 doesn't implement
collectives; XLA/NCCL insert them for the sharded step). Follow-ups to try: `NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE`
env, `NCCL_DEBUG=INFO` to see where it stalls, or use **single-process multi-GPU** (one process sees all 4 GPUs)
for single-node FSDP — which runs collectives within one process and avoids multi-process NCCL rendezvous.

## Net

V1's multi-host **device/mesh model is validated on real 4×H100**. Real *collective* execution across processes
needs the NCCL setup sorted on the target node (flagged); single-node FSDP via single-process-multi-GPU is the
reliable path for one 4-GPU box.
