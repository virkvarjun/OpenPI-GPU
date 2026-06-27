# Sharder — Multi-Host, Fault-Tolerant JAX Training for `openpi`

**Step 0 deliverable.** This maps the upstream `openpi` codebase, reports exactly what the
`mli0603/openpi-comet` fork already shipped, distills the MaxText reference patterns we'll port, and proposes a
milestone plan with the exact files each change touches. **Nothing here is built yet — awaiting your review.**

All findings below were read from source in shallow clones of the three repos, not from memory:
- upstream `Physical-Intelligence/openpi` (the contribution target)
- `mli0603/openpi-comet` (prior art — the headroom we aim past)
- `google/maxtext` (the reference patterns we port)

---

## 0. The single most important correction to the brief

> Core 1 says: "generalize the existing **1D fsdp sharding** into a **2D (data, fsdp) mesh**."

**The mesh is already 2D.** `training/sharding.py:17-23` builds
`jax.make_mesh((device_count // fsdp_devices, fsdp_devices), ("batch", "fsdp"))`, and `scripts/train.py:209`
already shards the batch over **both** axes (`DATA_AXIS = ("batch", "fsdp")`) while params shard along `fsdp`
only (`fsdp_sharding`, `sharding.py:48-102`). So `fsdp_devices=1 ⇒ pure data-parallel`,
`fsdp_devices=N ⇒ FSDP` — the additive axis you want **exists today**.

What is genuinely missing — and is the real Core 1 — is:
1. **`jax.distributed.initialize()`** — it appears **nowhere** in upstream (`scripts/train.py` has no call).
   The whole script assumes `jax.device_count()` = *local* devices on one host.
2. **Gradient accumulation** — there is **no** microbatching / accumulation path anywhere (net-new).
3. Making the existing 2D mesh **span hosts** (it does so automatically once `jax.distributed` is up — the
   comet fork proves this needs *no* new mesh code).

I'll keep your `data`/`fsdp` axis naming and the "additive, single-host stays identical" guarantee, but re-aim
Core 1's labor at **multi-host init + grad accumulation**, not at re-deriving a 2D mesh that already exists.

---

## 1. Upstream `openpi` codebase map (verified, with file:line)

### Stack & versions (`openpi/pyproject.toml`)
- **Model framework: `flax.nnx`** — *not* Linen. Confirmed `models/model.py:10 from flax import nnx`;
  `models/__init__` exports nnx modules; `shared/nnx_utils.py` exists.
- Pins: `flax==0.10.2`, `jax[cuda12]==0.5.3`, `orbax-checkpoint==0.11.13`, `optax` (via flax),
  `ml_collections==1.0.0`, `tyro>=0.9.5`, Python `>=3.11`. RLDS extra: `tensorflow-cpu==2.15.0`,
  `tensorflow-datasets==4.9.9`, `dlimp` (only ships cp311 wheels → **must use Python 3.11**).
- `[tool.uv] override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]` — tensorstore underpins Orbax.

### Train entrypoint — `scripts/train.py` (281 lines)
- `main()` (`:194`): `batch_size % jax.device_count()` check (`:198`) → **assumes single-host device count**;
  `mesh = sharding.make_mesh(config.fsdp_devices)` (`:208`); `data_sharding = NS(mesh, P(DATA_AXIS))` (`:209`);
  `replicated = NS(mesh, P())` (`:210`).
- `init_train_state()` (`:84-133`): `jax.eval_shape(init)` → `fsdp_sharding(shape, mesh)` for `state_sharding`;
  weights loaded on host then `jax.jit(init, out_shardings=state_sharding, donate_argnums=(1,))`.
- `train_step()` (`:136-191`): `nnx.merge(model_def, params)` → `nnx.value_and_grad(loss_fn, DiffState(0, trainable_filter))`
  → `optax` update → optional EMA (`:169-175`). **Single grad over the full batch — no accumulation.**
- Loop (`:259-273`): `with sharding.set_mesh(mesh): ptrain_step(...)`; checkpoint every `save_interval`.
- **No `jax.distributed`, no `jax.profiler`, no microbatching.**

### Sharding — `training/sharding.py` (102 lines)
- `make_mesh(num_fsdp_devices)` → 2D `(batch, fsdp)` (`:17-23`).
- `fsdp_sharding(pytree, mesh, min_size_mbytes=4)` (`:48-102`): replicates scalars/vectors/small (<4 MiB)
  arrays; shards larger tensors along the **largest axis divisible by the fsdp dim** (`:83-93`); replicates if
  no divisible axis. `mesh.shape[FSDP_AXIS]==1 ⇒ replicate everything` (`:72`) — the clean single-host path.
- `set_mesh` context manager + `activation_sharding_constraint` (`:26-45`) for in-model activation sharding.

### TrainState — `training/utils.py:14-24` (`@flax.struct.dataclass`, is a pytree)
`step`, `params: nnx.State`, `model_def: nnx.GraphDef`, `tx: optax.GradientTransformation` (pytree_node=False),
`opt_state`, `ema_decay: float|None` (pytree_node=False), `ema_params: nnx.State|None`. **EMA already exists.**

### Config registry — `training/config.py` (989 lines)
- `TrainConfig` dataclass (`:466-557`). Relevant fields: `model`, `lr_schedule` (CosineDecay default),
  `optimizer` (AdamW), `ema_decay=0.99`, `data: DataConfigFactory`, `seed=42`, `batch_size=32`,
  `num_workers=2`, `num_train_steps=30_000`, `save_interval=1_000`, `keep_period=5_000`, `overwrite`, `resume`,
  **`fsdp_devices: int = 1`** (`:535`).
- Registry: `_CONFIGS` list → `_CONFIGS_DICT = {c.name: c}` (`:975`) → `cli()` uses
  `tyro.extras.overridable_config_cli(...)` (`:978-979`). **New configs = append to `_CONFIGS`; new fields =
  add to the dataclass.** This is where our `data_axis`/`grad_accum_steps`/multi-host knobs land.

### Optimizer — `training/optimizer.py`
`create_optimizer(optimizer, lr_schedule, weight_decay_mask=None)` (`:105-109`); `CosineDecaySchedule`
(warmup 1k, peak 2.5e-5, decay 30k) and `RsqrtDecaySchedule`; `AdamW` (b1 .9, b2 .95, clip 1.0).

### Data loaders — `training/data_loader.py` (540) + `training/droid_rlds_dataset.py` (248)
- `create_data_loader(config, *, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False, framework="jax")`
  (`:223-231`). Numpy→global-array handoff **already uses** `jax.make_array_from_process_local_data(sharding, x)`
  (torch path `:466`, RLDS path `:527`). `local_batch_size = batch_size // jax.process_count()` (`:322`).
- **Hard block:** `if jax.process_count() > 1: raise NotImplementedError("Data loading with multiple processes
  is not supported.")` (`:412` torch, `:502` RLDS). **No `jax.process_index()` anywhere** ⇒ every process would
  currently load *identical* data. This guard is exactly what Core 2 removes.
- LeRobot path: torch `DataLoader`, `multiprocessing.get_context("spawn")` when `num_workers>0`, worker sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` (`:478-483`); DDP `DistributedSampler` only on the pytorch framework
  branch (`:308-322`).
- DROID/RLDS path: pure **tf.data** (no torch workers at all — that's the "num_workers=0" reality). Parallelism
  via `num_parallel_reads/calls = AUTOTUNE`; `tf.config.set_visible_devices([], "GPU")` (`:58-59`) so TF can't
  grab accelerator memory; `__iter__` is `yield from dataset.as_numpy_iterator()` (`:243`). **No host sharding**
  (no `process_index`), global shuffle buffer (250k). *Re-read the exact in-file rationale before touching.*
- **`DataLoader` protocol exposes `data_config()` only — no iterator/epoch/shuffle position is checkpointable.**

### Checkpointing — `training/checkpoints.py` (159 lines)
- `ocp.CheckpointManager` with items `{assets: CallbackHandler, train_state: PyTreeCheckpointHandler,
  params: PyTreeCheckpointHandler}`, `max_to_keep=1`, `keep_period`, **async** (`AsyncOptions(timeout_secs=7200)`)
  (`:40-53`).
- `CallbackHandler.save` guards `if jax.process_index() == 0` (`:125`) — so it's *partly* multi-host-aware.
- **`restore_state(...)` does `del data_loader` (`:95`) — the data-iterator position is NOT saved or restored.**
  Save only writes norm-stats into `assets`. This is the concrete fault-tolerance gap for Core 3.
- Orbax `PyTreeCheckpointHandler` is GSPMD-aware: given target shardings it reshards on restore — topology-
  independent restore is *achievable* but not yet exercised.

---

## 2. What `openpi-comet` already did (prior art) — and the headroom

**Verdict: TRUE multi-host**, not single-node FSDP. Evidence: new `scripts/train_dist.py:207-218` calls
`jax.distributed.initialize(coordinator_address=MASTER_ADDR:MASTER_PORT, process_id=WORLD_RANK,
num_processes=WORLD_SIZE, local_device_ids=...)`, launched across nodes via Lepton env vars
(`README.md:175-193`, config name `…_gpu40` = 5×8 GPUs).

They shipped:
- **`train_dist.py`**: distributed init, per-process RNG (`fold_in(rng, process_index)`), cross-host string
  broadcast (`multihost_utils.broadcast_one_to_all`), process-0-only wandb. Mesh code **unchanged** — they ride
  the existing `(batch, fsdp)` mesh, which spans hosts automatically.
- **`checkpoints_dist.py`**: multi-host-safe Orbax by **disabling** OCDBT, the ArrayMetadata store, and **async**
  (forced synchronous + `wait_until_finished()`) — explicitly "to avoid multi-host synchronization issues."
- **Host-sharded data**: replaced the `NotImplementedError` with coarse `chunks[process_index::process_count]`
  striping (`data_loader.py:353-377`), `seed + process_index`.
- **Multi-dataset**: `data: Sequence[DataConfigFactory]` + `sample_weights: list[float]` knob — but the actual
  weighted sampling is offloaded to an **external `behavior` package** (not in the fork).

**Headroom they left (our novelty), all confirmed absent:**
- ❌ **Gradient accumulation / microbatching** (grep: none).
- ❌ **Fault tolerance / elastic / preemption** (grep: none — a node death kills the run; recovery is only
  manual `--resume`).
- ❌ **Async *and* reshardable multi-host checkpoints** — they *disabled* async to dodge bugs; no save-on-N /
  restore-on-M logic; no data-iterator checkpointing.
- ❌ **Smart host-aware data sharding** — static `chunks[i::N]` index-striping, not a globally-shuffled,
  weighted, rebalancing sampler; cross-host mixture consistency not enforced.
- ❌ **Profiling/MFU harness.**

**So Sharder's defensible contribution = grad-accum (scan+remat) + elastic fault tolerance + async reshardable
multi-host Orbax (incl. data-iterator state) + a profiling/MFU harness + a correct host-sharded sampler** — the
exact four Cores you specified, aimed precisely at this gap.

---

## 3. MaxText patterns we port (reference, with file:line)

MaxText pins newer libs (`jax==0.7.1`, `orbax>=0.11.25`, `grain`) — **API drift vs openpi's `jax==0.5.3`,
`orbax==0.11.13` is the #1 risk; every API below must be re-verified against the pinned versions before use.**

1. **`jax.distributed.initialize`** (`utils/max_utils.py:234-406`): guard with `is_initialized()` + single-proc
   flag; **TPU = no args** (auto-discovery); **GPU/CPU = explicit** coordinator/process_id/num_processes from
   env. (Matches comet's GPU pattern — we generalize to cover both.)
2. **Mesh** (`utils/maxtext_utils.py:1808-1885`): ICI (intra-slice) vs DCN (inter-slice) split via
   `create_hybrid_device_mesh(ici, dcn, devices)`; logical-axis-rules indirection so model code names *logical*
   axes. *For us: keep `(data, fsdp)` for v1; note hybrid ICI/DCN as a later option for real pods.*
3. **Per-host data → global array** (`input_pipeline/multihost_dataloading.py:50-167`): load local shard → split
   across `mesh.local_devices` → `device_put` → `make_array_from_single_device_arrays`. *openpi already uses the
   simpler `make_array_from_process_local_data`; we keep that unless we need the explicit split.*
4. **Checkpointing** (`common/checkpointing.py`): abstract-state restore with per-array
   `ArrayRestoreArgs(sharding=...)` ⇒ **free resharding across topologies** (our save-on-4/restore-on-8 test);
   single-replica-broadcast restore for scale; `EmergencyCheckpointManager` (local+persistent dirs) for fast
   preemption recovery; a `GrainCheckpointHandler` registered as a Composite `"iter"` item to checkpoint the
   **data iterator per process**; `ElasticIterator` keeps a single global scalar that survives resize. ⚠️
   `SingleReplicaArrayHandler`, `EmergencyCheckpointManager`, `save_decision_policy_lib` may not exist / differ
   in `orbax==0.11.13` — verify first; fall back to stock `CheckpointManager` + `ArrayRestoreArgs` if so.
5. **Grad accumulation** (`utils/gradient_accumulation.py:26-178`): reshape batch → `(num_microbatches, micro,
   …)`, `lax.scan` over microbatches accumulating summed grads/loss/weights in the carry; normalize at the end;
   keep sharding annotations **outside** the scan; remat lives in the model layers. *This is our scan+remat
   reduce-scatter-incrementally pattern — the carry holds the running grad so all-layer grads never co-exist.*

---

## 4. Cheap-validation ladder (how every milestone is proven without a pod)

| Rung | Mechanism | Proves |
|---|---|---|
| 0 | single real CPU device | parity baseline |
| 1 | `XLA_FLAGS="--xla_force_host_platform_device_count=8"` | mesh shapes, sharding, grad-accum numerics |
| 2 | `jax.distributed.initialize()` across N localhost processes (one coordinator) | true multi-process code path: per-host data, multi-host Orbax, cross-host collectives |
| 3 | kill -9 a process mid-run, relaunch | elastic restart + checkpoint resume |
| ⛔ | real TPU pod / multi-node GPU | **flagged, not run here**: ICI/DCN perf, MFU at scale, NCCL/ICI bandwidth |

Anything only verifiable on rung ⛔ will be clearly marked in code and PRs.

---

## 5. Milestone plan (your ordering: Core 1 → 4 → 2 → 3), files each touches

Each milestone = small reviewable commits with the JAX mechanics explained in the message; single-host path
stays byte-identical (guarded by `process_count()==1` / `grad_accum_steps==1` / `data_axis==1`).

### Core 1 — Multi-host training + gradient accumulation
- **New** `src/openpi/training/distributed.py`: `maybe_initialize(config)` (guarded `jax.distributed.initialize`,
  TPU no-arg + GPU/CPU env-driven, like MaxText §3.1 / comet).
- **`scripts/train.py`**: call init at top; `batch_size % jax.device_count()` already uses the *global* count
  post-init (correct); per-process RNG `fold_in(rng, process_index)`; wandb/print guarded to process 0.
- **`training/sharding.py`**: no change to `make_mesh` (already 2D); maybe expose explicit `P("fsdp")` /
  `P("data")` helpers for clarity.
- **`training/utils.py` (`train_step`)** + **new** `training/grad_accum.py`: microbatch reshape + `lax.scan` +
  `jax.remat` accumulation; `grad_accum_steps` knob in `TrainConfig` (`config.py`).
- **Tests** (`xla_force_host_platform_device_count=8`): numerical parity single-device vs 8-device same
  seed/data over K steps (bf16 tol); grad-accum(steps=4, micro=B/4) ≈ full-batch(B) to bf16 tol.

### Core 4 — Profiling/MFU harness (built **alongside** Core 1, used by all later work)
- **New** `src/openpi/training/profiling.py`: `jax.profiler` trace capture (device-side, not wall-clock),
  median per-step **device** time, FLOPs/step (from model config) → **MFU** + roofline position.
- **`scripts/train.py`**: optional `--profile` window (capture steps N..N+k), emit trace + a one-line MFU report.
- **Tests**: smoke-run the profiler on CPU sim; assert a trace file + finite MFU number are produced (absolute
  MFU is only meaningful on rung ⛔ — flagged).

### Core 2 — Host-sharded data
- **`training/data_loader.py`**: remove the `process_count()>1` `NotImplementedError`; deterministic shard by
  `(process_index, process_count)`; ensure **no cross-host duplication** and correct global batch reassembly via
  the existing `make_array_from_process_local_data`; prefetch overlap.
- **`training/droid_rlds_dataset.py`**: add `tf.data` host sharding (`process_index`/`process_count`) *after*
  confirming the num_workers=0 rationale in-file; keep tf GPU hidden.
- **Multi-dataset mixing**: a `sample_weights` knob (like comet) but with a **globally-consistent, host-aware**
  sampler (each host draws its disjoint slice of the same global mixture/shuffle) — the comet headroom.
- **Tests**: determinism — two runs of one config produce identical loss curves **and** identical data order;
  no-duplication — union of per-host shards = full dataset, intersection = ∅, over an epoch on rung 2.

### Core 3 — Fault tolerance
- **`training/checkpoints.py`** (or new `checkpoints_dist.py`): checkpoint **ALL** state — params, opt_state,
  EMA, step, **and data-iterator position** (port MaxText's iterator-as-Composite-item §3.4); topology-
  independent restore via abstract-state + `ArrayRestoreArgs(sharding=...)`; keep async if `orbax==0.11.13`
  allows safe multi-host async, else synchronous fallback (comet's lesson).
- **New** `scripts/train_elastic.py` (or a loop flag): elastic-restart wrapper — on process death, re-init
  rendezvous and resume from last checkpoint.
- **Tests**: reshard round-trip — save on 4 sim devices, restore on 8, training continues & loss is continuous;
  fault injection — `kill` a process mid-run on rung 2/3, assert clean resume from last checkpoint with
  identical post-resume trajectory.

---

## 6. Versions to pin (verified against repo source; runtime-verify in the first impl step)

`python==3.11`, `jax==0.5.3`, `flax==0.10.2` (**nnx**), `orbax-checkpoint==0.11.13`, `optax` (flax-pinned),
`ml_collections==1.0.0`, `tyro>=0.9.5`, `tensorstore==0.1.74`, `ml-dtypes==0.4.1`; RLDS extra:
`tensorflow-cpu==2.15.0`, `tensorflow-datasets==4.9.9`. **Caveat:** the Orbax emergency-checkpoint / single-
replica / save-decision-policy APIs seen in MaxText are from `orbax>=0.11.25`; on `0.11.13` I will verify each
and fall back to stock `CheckpointManager` + `ArrayRestoreArgs` resharding where they're absent.

---

## 7. Open decisions for you (blocking before I write Core code)

1. **Repo target.** Sharder modifies real `openpi` internals (`training/sharding.py`, `data_loader.py`,
   `checkpoints.py`, `scripts/train.py`). Should this repo (`OpenPI-JAX-`) become a **fork of upstream openpi**
   where I add Sharder under `src/openpi/training/` (clean upstream PRs) — my current from-scratch JAX scaffold
   moved to a branch/subdir? Or keep them side by side? *Recommendation: fork upstream openpi as the working
   tree; preserve the scaffold on a branch.* This is a "large refactor" so I'm not doing it without your call.
2. **First accelerator target for the code path:** TPU-style (no-arg init) or GPU/CPU (env-driven init) as the
   primary, with the other as the guarded alternative? *Recommendation: GPU/CPU env-driven first (matches the
   localhost-multiprocess cheap ladder and comet), TPU path added but only ⛔-verifiable.*
3. **Core 1 scope check:** OK to re-aim Core 1 at **multi-host init + grad-accum** (since the 2D mesh already
   exists), keeping your `data`/`fsdp` naming and single-host-identical guarantee?

**Stopping here for your review per Step 0. No Core code written.**
