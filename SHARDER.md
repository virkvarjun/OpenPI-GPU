# Sharder

**Sharder** adds a **fault-tolerant, multi-host JAX training path** to Physical Intelligence's `openpi`.

This repository is **not** a GitHub fork of upstream openpi. It is its own project
(`github.com/virkvarjun/OpenPI-JAX-`) that **vendors** a copy of upstream `openpi` as its working tree, so
Sharder's changes edit the real `src/openpi/...` internals directly. The from-scratch JAX recreation that this
repo started as is preserved on the **`scaffold`** branch.

- **Plan & codebase map:** [PLAN.md](PLAN.md)
- **Upstream baseline:** vendored from `Physical-Intelligence/openpi` @ main (see the "Vendor upstream …" commit).
  Omitted from the vendor: upstream `.git`, `.gitmodules`, and the `third_party/{aloha,libero}` sim-eval
  submodules (not needed for training).

## Why

Upstream openpi's own README states *"the current training script does not yet support multi-node training."*
The mesh is already 2D (`batch`×`fsdp`) but there is **no `jax.distributed` init, no gradient accumulation, and
the checkpoint path drops the data-iterator position.** Sharder closes exactly those gaps.

## Build order (approved)

1. **Core 1** — multi-host init (`jax.distributed`, GPU/CPU-first) + gradient accumulation (`lax.scan` + `remat`).
2. **Core 4** — profiling / MFU harness (built first, alongside Core 1, so every later change is measured).
3. **Core 2** — host-sharded data (deterministic per-`process_index` sharding, no duplication, weighted mixing).
4. **Core 3** — fault tolerance (all-state Orbax checkpoints incl. iterator position, topology-independent
   restore, elastic restart).

## Validation

No TPU pod / multi-node cluster is used. Everything is proven on the **cheap-validation ladder**: CPU device
simulation (`XLA_FLAGS=--xla_force_host_platform_device_count=8`) and localhost multi-process via
`jax.distributed.initialize()`. Anything only verifiable on real accelerators is explicitly flagged.

> Note: upstream pins `jax[cuda12]==0.5.3`, which has no macOS/CUDA wheels. Local development uses a **CPU-only**
> JAX build of the same version for the cheap ladder; CUDA/TPU paths are exercised only on Linux accelerators.
