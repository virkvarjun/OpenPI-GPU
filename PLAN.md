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

## 7. Open decisions — RESOLVED (2026-06-26)

1. **Repo target → VENDOR, not a GitHub fork.** This repo stays its own project and vendors upstream openpi as
   the working tree; Sharder edits `src/openpi/...` directly. The from-scratch scaffold is preserved on the
   `scaffold` branch. (Done — see the "Vendor upstream …" commit and [SHARDER.md](SHARDER.md).)
2. **First accelerator target → GPU/CPU env-driven init first**, with the TPU no-arg path added as the guarded
   alternative (⛔-verifiable only).
3. **Core 1 scope → re-aimed at multi-host init + grad-accum**, keeping the `data`/`fsdp` naming and the
   single-host-identical guarantee. (The 2D mesh already exists; we do not rebuild it.)

**Next:** stand up a CPU-only dev env (jax 0.5.3 CPU) for the cheap ladder, confirm the vendored tree imports +
upstream tests pass under CPU simulation, then begin Core 1 + Core 4 in small, explained commits.

---

# Core 2 — Host-sharded data pipeline

## Step 2.0 — understand before building (this section). **No Core-2 code written; stop after this.**

> ⚠️ **Repo-state caveat (read first).** At HEAD (`cc77bab`) the working tree is *vendored upstream openpi*;
> **Core 1 and Core 4 are not committed here** (`src/openpi/training/{distributed,profiling,grad_accum}.py` are
> absent; `scripts/train.py` has no `jax.distributed`/`process_index`). The design below therefore attaches to
> the data axis that *already exists* in the vendored tree — `sharding.DATA_AXIS = ("batch","fsdp")` and the
> `data_sharding = NamedSharding(mesh, P(DATA_AXIS))` built in `scripts/train.py:209`. Two of the Step-2.1 tests
> (multi-process via `jax.distributed.initialize()`; the Core-4 overlap profile) need Core 1 / Core 4 actually
> in-tree. **Before approving 2.1, confirm where Core 1/4 live (or have me build them first).**

### 2.0.1 — Spec of openpi's CURRENT data path (verified, file:line)

There are **two** loaders, chosen in `create_data_loader` (`data_loader.py:223-268`) by whether
`data_config.rlds_data_dir` is set:

**A) Default LeRobot loader (map-style / torch).** `create_torch_data_loader` (`:271-337`).
- Dataset is **map-style** (`Dataset` protocol, `__getitem__`+`__len__`, `:22-29`): `LeRobotDataset` (random
  access) wrapped in `TransformedDataset` (repack → data transforms → `Normalize` → model transforms,
  `:172-191`). `repo_id == "fake"` → `FakeDataset` of random spec-shaped samples (`:99-127`) — **our CPU-ladder
  dataset, no downloads/weights needed**.
- Batched & iterated by `TorchDataLoader` (`:381-468`): a `torch.utils.data.DataLoader` with
  `batch_size=local_batch_size`, `shuffle`, seeded `torch.Generator` (`:432-433`), `drop_last=True`,
  `collate_fn=_collate_fn` (np.stack, `:471-475`), `num_workers` via `spawn` + `persistent_workers` (`:428-446`).
- **Host→device handoff:** each yielded numpy batch becomes a *global* `jax.Array` via
  `jax.make_array_from_process_local_data(self._sharding, x)` (`:466`). Default sharding when none passed is a 1D
  `Mesh(jax.devices(), ("B",))`, `P("B")` (`:420-425`).
- **Per-host batch already computed:** `local_batch_size = batch_size // jax.process_count()` (`:322`).
- **The hard block:** `if jax.process_count() > 1: raise NotImplementedError("Data loading with multiple
  processes is not supported.")` (`:412`). And **no `jax.process_index()` anywhere** → if the guard were simply
  removed, every host would draw the *same* shuffled indices and `make_array_from_process_local_data` would
  assemble a global batch of **duplicated** data. This is the core gap.
- PyTorch-DDP branch (`:309-318`) *does* use a `DistributedSampler` (disjoint per rank) — but only on
  `framework=="pytorch"`, not the JAX path.

**B) RLDS/tf.data DROID loader (streaming / iterable).** `create_rlds_data_loader` (`:340-378`) →
`DroidRldsDataset` (`droid_rlds_dataset.py:36-248`).
- Pure **tf.data** graph built in `__init__`: `dl.DLataset.from_rlds(... num_parallel_reads=AUTOTUNE)` (`:68-70`)
  → success-filter → `repeat()` → `traj_map(restructure/chunk_actions, AUTOTUNE)` → `flatten(AUTOTUNE)` →
  idle-filter → `frame_map(decode_images, AUTOTUNE)` (`:171-222`). **Multi-dataset mixing already exists here**:
  per-source `RLDSDataset.weight` (`:28-33`, must sum to 1.0 `:62`) fed to
  `dl.DLataset.sample_from_datasets(all_datasets, weights=weights)` (`:232`), then a global
  `shuffle(250_000)` → `batch(batch_size)` → `with_ram_budget(1)` (`:233-236`).
- Iterated via `dataset.as_numpy_iterator()` (`:243`); `RLDSDataLoader` (`data_loader.py:486-527`) is a shallow
  wrapper that just applies the **same** `make_array_from_process_local_data` handoff (`:527`) and the **same**
  `process_count()>1` block (`:502`). No `process_index` / `shard()` anywhere.

**C) Why the RLDS loader is pinned to `num_workers=0` (the real reason).**
`num_workers` is a `TrainConfig` field (`config.py:509`, default 2) but it is **only consumed by the torch
path** (`data_loader.py:264,332,439`); `create_rlds_data_loader` never threads it anywhere. Configs that use
RLDS set `num_workers=0` with the comment *"RLDS DataLoader requires num_workers=0, handles multi-processing
internally"* (`config.py:859,893`; `misc/polaris_config.py:68…`). The substance:
- tf.data **owns its parallelism in-process** via C++ threadpools — `AUTOTUNE` parallel file reads (`:69`) and
  parallel `traj_map`/`flatten`/`frame_map` (`:171-222`), plus internal prefetch and `with_ram_budget` (`:236`).
- Wrapping that in a torch `DataLoader` with `num_workers>0` uses **`spawn`** (`data_loader.py:430`), and each
  worker would **rebuild the entire tf.data graph** — re-reading files, re-allocating the 250k-frame shuffle
  buffer, rebuilding the `StaticHashTable` idle-filter, and re-running the process-global
  `tf.config.set_visible_devices([], "GPU")` (`droid_rlds_dataset.py:59`). That multiplies RAM/I/O and risks TF
  init conflicts, for **zero** benefit since tf.data is already parallel.
- **Decision (understanding-based, not reflex): KEEP `num_workers=0` for the RLDS path.** Host scaling for RLDS
  comes from **`dataset.shard(process_count, process_index)` inside the tf.data graph**, not from torch workers.
  (We will, separately, still allow `num_workers>0` on the *LeRobot* path — that path benefits from worker
  parallelism for Python-side `__getitem__`/transform CPU work.)

**D) Iterator state / resumability — NONE today.** The `DataLoader` protocol exposes only `data_config()`
(`data_loader.py:42-50`); both loaders loop `while True` with a local `num_items` counter (`:452-468`,
`:515-527`). `checkpoints.restore_state` does `del data_loader` (`checkpoints.py:95`). So a resumed run **does
not** continue on the right examples — the gap Core 2 must close and Core 3 will persist.

### 2.0.2 — Diff vs `openpi-comet`, and our headroom

| Concern | `openpi-comet` (prior art) | Sharder headroom |
|---|---|---|
| Host sharding (LeRobot) | Replaced the `NotImplementedError` with static **`chunks[process_index::process_count]`** index striping (`data_loader.py:353-377`), `seed+process_index` | A **deterministic global-shuffle → disjoint per-host slice** via a custom index sampler (not raw mod-striping), so shuffle quality is preserved and the slice is provably a partition |
| RLDS host sharding | Not added (rode the existing mesh; RLDS path largely untouched) | `dataset.shard(host_count, host_id)` attached **early per-source** in the tf.data graph |
| Multi-dataset mixing | `data: Sequence[DataConfigFactory]` + `sample_weights`, but the actual weighted sampling is **offloaded to an external `behavior` package** (not in-repo, not host-consistent) | A **first-class, in-repo** mixing knob; reuse/extend the RLDS `weight` mechanism that already exists upstream; deterministic and identical mixture across hosts |
| Resume / iterator state | None | **Checkpointable iterator position** (`get_state`/`set_state`) → exact resume, wired to Core 3 |
| Upstream-mergeability | New `train_dist.py`, external dep, disabled async ckpt | **Additive, single-host-identical, no new heavy deps**, behind `process_count()`/weight knobs — designed to be a clean upstream PR |

### 2.0.3 — Proposed design

**Primary path = the LeRobot map-style loader.** Map-style + a custom sampler gives *explicit control over
example order and position*, which makes all four properties (no-dup, determinism, mixing, resume) clean and
exactly checkpointable. (Streaming tf.data resume needs heavy tf iterator checkpoints — handled as the secondary
path, coarser.)

1. **Deterministic host-aware sampler** (new). A seeded sampler produces one global permutation per epoch
   (`epoch_seed = base_seed ⊕ epoch`) and yields **only this host's disjoint slice**
   `perm[process_index :: process_count]` (a true partition; asserted in tests). Replaces both the
   `process_count()>1` guard and the default JAX shuffle. Single-host (`process_count()==1`) → identical order to
   today.
2. **Checkpointable position.** Sampler tracks `(epoch, index_within_epoch)` and its base seed → `get_state()` /
   `set_state()`. Restoring reconstructs the exact remaining stream (no repeats/skips).
3. **Multi-dataset mixing knob.** Generalize `TrainConfig.data` to accept a weighted list (mirroring comet's
   ergonomics but in-repo): a `ConcatDataset` of the per-config datasets + a **weighted index sampler** that
   draws source-dataset ids by configured proportion from a seeded stream, then a within-source index — fully
   deterministic and identical across hosts (each host then takes its disjoint slice). Reuse the existing
   `RLDSDataset.weight` semantics for the RLDS path; add an analogous top-level `sample_weights` for LeRobot.
4. **Mesh consistency.** Keep yielding global arrays via `make_array_from_process_local_data` along the existing
   `DATA_AXIS` (Core 1's `data` axis) — unchanged handoff, correct per-host local batch (`batch_size //
   process_count`).
5. **Prefetch/overlap.** Background host→device prefetch (depth N) so batch *k+1* transfers during step *k*’s
   compute; validated as *hidden* via the Core-4 profiler (HW-gated for true throughput).
6. **RLDS secondary path.** Add `dataset.shard(process_count, process_index)` early per-source in
   `DroidRldsDataset`; keep `num_workers=0`; iterator-state via tf.data checkpoint (or documented coarse resume
   first). Mixing already supported by `weight`.

**Iterator-state hook → Core 3.** Extend the `DataLoader` protocol with `get_state() -> PyTree` /
`set_state(state)`. Core 3 saves it as a **per-process Orbax Composite `"data_iter"` item** (the MaxText
`GrainCheckpointHandler` pattern), restored topology-independently alongside the train state.

### 2.0.4 — Exact files touched (Step 2.1)

- `src/openpi/training/data_loader.py` — replace the two `process_count()>1` guards with the host-aware sampler;
  add `get_state`/`set_state` to the `DataLoader` protocol + `DataLoaderImpl`/`TorchDataLoader`/`RLDSDataLoader`;
  wire prefetch; weighted multi-dataset sampling for the torch path.
- **New** `src/openpi/training/data_sharding.py` — the deterministic host-aware index sampler, the weighted
  multi-dataset index mixer, and the iterator-state (`get/set`) helpers (kept separate so `data_loader.py` stays
  a thin wiring layer and the logic is unit-testable in isolation).
- `src/openpi/training/droid_rlds_dataset.py` — add `process_index`/`process_count` params and early
  `.shard(...)`; keep `num_workers=0`.
- `src/openpi/training/config.py` — add the in-repo mixing knob (`sample_weights` / list-valued `data`) to the
  `TrainConfig`/`DataConfigFactory` registry; document the kept `num_workers` semantics.
- **New** `src/openpi/training/data_sharding_test.py` (+ extend upstream `data_loader_test.py`) — the five tests
  below, on the CPU-sim ladder with `FakeDataset`.

### 2.0.5 — Test plan (cheap-validation ladder)

| Test | Mechanism | Asserts |
|---|---|---|
| No-duplication | 8 sim hosts (`--xla_force_host_platform_device_count=8` and/or N localhost processes), `FakeDataset` | ⋃ per-host index slices = full shard set, pairwise ∩ = ∅ over an epoch |
| Determinism | two runs, same config+seed | identical example order **and** identical loss inputs |
| Mixing fidelity | K datasets, configured weights, N batches | realized proportions ≈ weights within tolerance |
| Resume correctness | `get_state` mid-epoch → `set_state` → continue | stream resumes with **no repeated or skipped** examples |
| Overlap sanity | Core-4 profiler | data-wait hidden behind compute on sim devices *(true throughput is HW-gated — flagged)* |

**STOP — awaiting your review of this section before writing any Core-2 code.**

---

# Sharder V1 — true multi-HOST training (branch `feat/jax-multihost-v1`)

> **This section is the authoritative, in-scope plan for V1 and supersedes the broader scoping above for this
> branch.** V1 is deliberately narrow: bring openpi's *existing* JAX/FSDP path up across multiple hosts. We do
> **not** touch the model, the mesh, FSDP, Orbax, or DROID mixing — those already exist and work.

## V1 scope (the only in-scope work)
- **G1** — `jax.distributed.initialize()` in `scripts/train.py:main()`, a **no-op when single-process**.
- **G2** — remove the hard `raise NotImplementedError("Data loading with multiple processes is not supported.")`
  in both `TorchDataLoader` and `RLDSDataLoader`.
- **G3** — per-process **disjoint, deterministic** dataset sharding (machine *k* of *N* reads a non-overlapping
  slice), for **both** the torch (LeRobot) and RLDS (DROID) loaders. Keep the existing
  `make_array_from_process_local_data` assembly. **No new mixing-weight knob** (DROID already has one).
- **G6** — a device-side `jax.profiler` MFU/roofline harness to measure step time + device utilization.
- **Correctness:** multi-process learning must **match** single-process (bf16 tol over K steps).

**Explicitly OUT of V1** (deferred / not in scope): G4 exact-resume / iterator-state checkpointing (leave the
`del data_loader` in `restore_state` alone — flagged for V1.1), G5 fault tolerance / elastic restart, gradient
accumulation (see audit below — not present, not needed), any model / mesh / FSDP / mixing / serving changes.

## Step-0 audit findings (verified this session)
1. **No redundant/conflicting code to revert.** `diff -rq src/openpi <pristine upstream clone>` is **empty** —
   our `src/openpi` is byte-identical to upstream openpi. The earlier "build a 2D mesh" premise only ever lived
   in PLAN.md *prose*; **zero** source changes were made to `make_mesh`/`fsdp_sharding`/`train_step`/loaders.
   `training/` contains only upstream files (no `distributed.py`/`profiling.py`/`grad_accum.py`). **Recommended
   action: nothing to revert in code; this V1 section corrects the plan's scope.**
2. **Q(a) — gradient accumulation? NO.** `scripts/train.py:train_step` (`:136-191`) computes a single
   `nnx.value_and_grad(loss_fn, argnums=DiffState(0, trainable_filter))` over the full batch, then `tx.update` +
   `optax.apply_updates`. No `scan`/`remat`/microbatch (grep confirms). **⇒ no scan+remat work in V1.**
3. **Q(b) — RLDS process-aware structure? ABSENT.** `droid_rlds_dataset.py` has no `process_index`/
   `process_count`/`.shard`; the only relevant op is `sample_from_datasets(all_datasets, weights)` (`:232`).
   **⇒ G3 adds per-source `.shard(...)` before `sample_from_datasets`, built from scratch.**

## API verification (against installed jax 0.5.3, not memory)
`jax.distributed.initialize(coordinator_address, num_processes, process_id, local_device_ids,
cluster_detection_method, initialization_timeout=300, coordinator_bind_address)` ✓;
`jax.distributed.is_initialized()` ✓; `jax.{process_index,process_count,device_count,local_device_count}` ✓;
`jax.make_array_from_process_local_data(sharding, local_data, global_shape=None)` ✓;
`compiled.cost_analysis()['flops']` ✓ (device FLOPs for MFU); `--xla_force_host_platform_device_count=N` ✓.
> Note: on **CPU** `initialize()` does **not** auto-detect — coordinator/num_processes/process_id must be passed
> explicitly (our localhost launcher provides them). `jax[cuda12]==0.5.3` has no macOS wheels, so local dev uses
> **CPU jax 0.5.3** (installed & verified); GPU/TPU init is ⛔ HW-gated.

## Design — exact files & functions

### G1 — cluster bring-up  → **new** `src/openpi/training/distributed.py`, edit `scripts/train.py`
- `maybe_initialize() -> None`: if `jax.distributed.is_initialized()` → return; if the rendezvous env vars are
  present (coordinator address, num_processes, process_id — set by GPU/CPU launchers) →
  `jax.distributed.initialize(...)` with them; **else no-op** (single-process). No env ⇒ single-host path is
  byte-identical to today.
- `scripts/train.py:main()`: call `distributed.maybe_initialize()` as the **first** line, before any device/mesh
  use. `make_mesh(config.fsdp_devices)` then automatically spans all processes' devices (it already uses
  `jax.device_count()`, which becomes global after init) — **no mesh change**.

### G2 — remove the hard guard  → edit `src/openpi/training/data_loader.py`
- Delete the `raise NotImplementedError("Data loading with multiple processes is not supported.")` at
  `TorchDataLoader.__init__` (`~:412`) and `RLDSDataLoader.__init__` (`~:502`). Removal is paired with G3 in the
  same milestone so multi-process is never enabled without disjoint sharding (no silent duplication).

### G3 — per-process disjoint, deterministic sharding
**Torch / LeRobot (map-style) → edit `data_loader.py`, optional small helper.**
- Add a **jax-process-aware sampler** keyed on `jax.process_index()` / `jax.process_count()` that shards
  **within each global batch** (the key to exact parity): given a deterministic global index order `order`
  (seeded permutation if `shuffle`, else sequential), for global step `s` the global batch is
  `order[s·B : (s+1)·B]`, and process *k* yields its contiguous slice `order[s·B + k·local : s·B + (k+1)·local]`
  (`local = B // process_count`, already computed at `:322`). `make_array_from_process_local_data` then
  reassembles the **identical** global batch in process order → **bit-parity** with single-process.
  - *Why not vanilla `DistributedSampler`*: its `indices[rank::N]` interleaves across the whole dataset, not
    within a batch, so the per-step global batch composition would differ from single-process and break parity.
    We reuse its seeding idea but shard within-batch.
- Single-host (`process_count()==1`) → the slice is the whole batch → identical order to today (additive).

**RLDS / DROID (streaming) → edit `droid_rlds_dataset.py`.**
- In `prepare_single_dataset`, immediately after `dl.DLataset.from_rlds(... shuffle=shuffle ...)` and **before**
  `.repeat()` / per-frame maps and **before** `sample_from_datasets`, add
  `dataset = dataset.shard(num_shards=jax.process_count(), index=jax.process_index())`. Each host reads a
  disjoint file/element slice of **each** source → no cross-host duplication; per-source weights are preserved
  per host (mixing unchanged). Requires the source **file-shuffle seed to be identical across hosts** so
  `.shard` partitions the same ordering deterministically. *Verify `dl.DLataset.shard` passthrough to
  `tf.data.Dataset.shard` at implementation time (flag).*
- Keep `num_workers=0` for RLDS (tf.data owns parallelism; see Core-2 §2.0.1.C).

### G6 — MFU/roofline harness  → **new** `src/openpi/training/profiling.py`, edit `scripts/train.py`
- Device-side timing: median over a step window with `jax.block_until_ready`; optional `jax.profiler.trace`
  (Perfetto) for the window. FLOPs/step from the compiled step's `cost_analysis()['flops']` (device-accurate);
  bytes from cost_analysis for the roofline. `MFU = flops_per_step / (median_step_s · peak_device_flops)` with a
  configurable `peak_device_flops`.
- `scripts/train.py`: behind a `--profile` flag, capture steps `[N, N+k]`, emit a one-line
  `step_ms / TFLOP·s⁻¹ / MFU% / {compute|memory}-bound` report + trace. **Absolute MFU is only meaningful on
  real accelerators** — on CPU sim it validates the plumbing + relative step time (flagged).

## Milestones (small commits; single-host green at every commit) — **V1 COMPLETE on cheap ladder**
- **M1 (G6)** ✅ `badd806`+`50f5aa2` — `profiling.py` (measure_step_time/step_cost/mfu_report/profile_step) +
  env-gated `train.py` `--profile` wiring; 7 tests green; baseline line emitted on CPU sim.
- **M2 (G1)** ✅ `bc78a59` — `distributed.py::maybe_initialize()` (no-op single-process) + `train.py` first-line
  wiring + `scripts/launch_local.py`; integration test spawns 2 procs, asserts global `device_count` + mesh span.
- **M3 (G2+G3)** ✅ `583c7bf`+`912adac` — within-batch sharding math (`data_sharding.py`, 10 tests) + removed both
  `NotImplementedError` guards + torch `_WithinBatchShardSampler` + RLDS per-source `.shard` (flagged HW-gated).
- **M4 (correctness)** ✅ `da91793` — multi-process parity via the real `make_array_from_process_local_data`:
  per-process shard == reference slice, and global loss+grad == single-process reference (fp32 tol). Full V1
  suite 21 green.
- **M5 (scaling scaffold)** ✅ `612eeb3` — `scripts/scaling_study.py` sweeps 1/2/4/8 devices through the profiler
  → `FINDINGS.md`; CPU shared-core artifact flagged; real throughput/interconnect HW-gated.

**HW/env-gated (flagged, not run here):** the literal pi0 model multi-host step (heavy env + `jax[cuda12]` has no
macOS wheels); absolute MFU; real interconnect scaling; RLDS `.shard` on a real DROID slice. Parity is proven
with the real assembly primitive + a stand-in loss (the *data path* is what V1 changes).

## Tests (V1 subset; written with each milestone)
parity (bf16 tol, K steps, torch/FakeData) · no-duplication (⋃ disjoint, covers shard set) · determinism
(identical example order across two runs) · single-host-untouched (additive guarantee) · MFU/scaling at 1/2/4/8
(flagged HW-gated). *(G4 resume & G5 fault-injection tests are V1.1 — not written now.)*

## Files touched (summary)
**new:** `training/distributed.py` (G1), `training/profiling.py` (G6), a multihost data test (G3), `FINDINGS.md`
(M5). **edit:** `scripts/train.py` (G1+G6 wiring), `training/data_loader.py` (G2 guards + G3 torch sampler),
`training/droid_rlds_dataset.py` (G3 `.shard`). **untouched:** `sharding.py`, model, `checkpoints.py`,
`optimizer.py`, mixing.

## Env / HW-gated flags
Local dev = CPU jax 0.5.3 (verified). Full parity test needs flax-nnx model deps (+`data_loader.py` imports
torch/lerobot at module top) — heavier CPU env stood up at M1/M4; pure-sampler tests (no-dup/determinism) need
neither. **HW-gated (scaffold + flag only):** absolute MFU, real interconnect scaling curve, GPU/TPU
`initialize()` auto-detect, RLDS bit-parity (250k shuffle buffer makes exact single-vs-multi parity infeasible —
RLDS asserts no-dup + determinism, not bit-parity; exact parity is proven on the map-style path).

**STOP — Step 0 complete. Awaiting your approval of this V1 plan before writing M1 code.**
