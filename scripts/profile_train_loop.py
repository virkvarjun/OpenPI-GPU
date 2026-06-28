"""V2: attribute the REAL train loop with the REAL prefetching data loader — trustworthy data-wait.

Runs the actual openpi `TorchDataLoader` (torch DataLoader with `num_workers` background prefetch, openpi's
collate, and the `make_array_from_process_local_data` H2D assembly) feeding the real gemma_2b step, in a
train.py-style overlapped loop (dispatch async, pull next batch during compute, block once). Wall is timed
WITHOUT the profiler (Phase A — `jax.profiler.trace` has large per-op overhead that fakes "data-wait"); a short
separate traced window (Phase B) gives the device op-category composition. `data-wait = clean_wall - device_busy`
is then the genuine input-pipeline cost a *prefetching* loader leaves un-hidden behind compute.

Dataset is a numpy generator over the model's input shapes (FakeData equivalent, no disk) so prefetch workers do
no jax/CUDA work; swap in a LeRobot/RLDS dataset to fold in image-decode/disk. gemma_2b runs forward+backward in
bf16 (fits 32GB; AdamW state would not).
"""

from __future__ import annotations

import argparse
import tempfile
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
from openpi.models import pi0_config
from openpi.training import attribution


def _bf16(params):
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )


def _host_sample(spec):
    """Numpy array of the per-sample shape (drop the batch dim; collate re-adds it). Workers: pure numpy, no jax."""
    shape = tuple(spec.shape[1:])
    dt = np.dtype(spec.dtype)
    if dt == np.float32:
        return np.random.standard_normal(shape).astype(np.float32)
    if dt == np.int32:
        return np.random.randint(0, 100, shape).astype(np.int32)
    if dt == bool:
        return np.ones(shape, dtype=bool)
    return np.zeros(shape, dtype=dt)


def _gen(spec_tree):
    # inputs_spec has optional None fields (token_ar_mask/token_loss_mask are pi0-fast only) — omit them;
    # Observation.from_dict treats them as absent.
    if spec_tree is None:
        return None
    if isinstance(spec_tree, dict):
        return {k: g for k, v in spec_tree.items() if (g := _gen(v)) is not None}
    return _host_sample(spec_tree)


class _FakeNumpyDataset:
    """Map-style dataset that generates host numpy samples — runs in torch prefetch workers without touching jax."""

    def __init__(self, model_cfg, num_samples: int = 8192):
        obs_spec, act_spec = model_cfg.inputs_spec(batch_size=1)
        self._specs = {**obs_spec.to_dict(), "actions": act_spec}
        self._n = num_samples

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return _gen(self._specs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="gemma_2b")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4, help="torch DataLoader prefetch workers")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=12)
    args = p.parse_args()

    small = args.variant == "dummy"
    model_cfg = pi0_config.Pi0Config(
        paligemma_variant=args.variant,
        action_expert_variant="dummy" if small else "gemma_300m",
        action_horizon=4 if small else 50,
        max_token_len=8 if small else 48,
    )

    device = jax.devices()[0]
    mesh = sharding.make_mesh(1)
    data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    with at.disable_typechecking():
        dataset = _FakeNumpyDataset(model_cfg)
        loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=args.batch_size,
            sharding=data_sh,
            shuffle=False,
            num_workers=args.num_workers,
            seed=0,
        )
        data_iter = iter(loader)

        def to_obs_act(batch):
            return _model.Observation.from_dict(batch), batch["actions"]

        rng = jax.random.key(0)
        model = model_cfg.create(rng)
        model_def = nnx.graphdef(model)
        params = nnx.state(model)
        if not small:
            params = _bf16(params)

        def loss_fn(model, rng, obs, act):
            return jnp.mean(model.compute_loss(rng, obs, act, train=True))

        def step(params, rng, obs, act):
            m = nnx.merge(model_def, params)
            m.train()
            return nnx.value_and_grad(loss_fn)(m, rng, obs, act)

        jstep = jax.jit(step)

        obs, act = to_obs_act(next(data_iter))
        compiled = jstep.lower(params, rng, obs, act).compile()

        # Warmup (compile + spin up prefetch workers).
        for _ in range(args.warmup):
            obs, act = to_obs_act(next(data_iter))
            jax.block_until_ready(jstep(params, rng, obs, act))

        # PHASE A — clean wall (NO profiler): overlapped loop, pull next batch during compute, block once.
        t0 = time.perf_counter()
        losses = []
        obs, act = to_obs_act(next(data_iter))
        for _ in range(args.iters):
            loss, grads = jstep(params, rng, obs, act)
            losses.append(loss)
            del grads  # don't accumulate per-step gradients (~5GB bf16 each)
            obs, act = to_obs_act(next(data_iter))  # prefetched dequeue + H2D while the GPU computes
        jax.block_until_ready(losses)
        clean_wall_us = (time.perf_counter() - t0) * 1e6

        # PHASE B — short traced window (fixed batch) for device op-category composition only.
        trace_dir = tempfile.mkdtemp()
        with jax.profiler.trace(trace_dir):
            for _ in range(min(8, args.iters)):
                loss, grads = jstep(params, rng, obs, act)
                jax.block_until_ready(loss)
                del grads
        trace_steps = min(8, args.iters)

    scope_of = attribution._hlo_scope_map(compiled)  # noqa: SLF001
    per_inst = attribution._parse_trace_durations(trace_dir)  # noqa: SLF001
    cat = dict.fromkeys(attribution.CATEGORIES, 0.0)
    busy_total = 0.0
    for inst, dur in per_inst.items():
        scope = scope_of.get(inst) or scope_of.get(inst.split(".clone")[0]) or ""
        cat[attribution.classify(inst, scope)] += dur
        busy_total += dur
    device_busy_per_step = busy_total / trace_steps
    wall_per_step = clean_wall_us / args.iters
    data_wait_per_step = max(0.0, wall_per_step - device_busy_per_step)
    bd = attribution.Breakdown(
        category_us={k: v / trace_steps * args.iters for k, v in cat.items()},
        device_busy_us=device_busy_per_step * args.iters,
        wall_us=clean_wall_us,
        n_steps=args.iters,
    )
    pct = bd.percentages()

    lines = [
        "# TRAIN_LOOP_BREAKDOWN — real prefetching loader (clean-wall data-wait)",
        "",
        f"> ✅ **Overlapped train.py-style loop with the REAL openpi TorchDataLoader** (num_workers="
        f"{args.num_workers} background prefetch + collate + H2D), `{args.variant}` bf16 fwd+bwd, "
        f"batch={args.batch_size}, FakeData (numpy gen, no disk), on {device.device_kind} ({device.platform}). "
        "Clean wall timed WITHOUT the profiler (Phase A); op composition from a separate traced window (Phase B). "
        "`data-wait = wall - device_busy` is input-pipeline cost a *prefetching* loader leaves un-hidden.",
        "",
        f"Window: {bd.n_steps} steps | clean wall {wall_per_step / 1e3:.3f} ms/step | "
        f"device-busy {device_busy_per_step / 1e3:.3f} ms/step | data-wait {data_wait_per_step / 1e3:.3f} ms/step "
        f"({data_wait_per_step / wall_per_step * 100:.1f}%)",
        "",
        "| category | device ms/step | % wall |",
        "|----------|---------------:|-------:|",
    ]
    for c in (*attribution.CATEGORIES, "data-wait"):
        ms = (bd.category_us.get(c, bd.data_wait_us if c == "data-wait" else 0.0)) / 1e3 / bd.n_steps
        lines.append(f"| {c} | {ms:.4f} | {pct[c] * 100:.1f}% |")
    lines += [
        "",
        f"**Dominant: `{bd.dominant()}`** ({pct[bd.dominant()] * 100:.1f}% of clean wall).",
        "",
        "data-wait here = what the prefetching loader (collate + H2D, workers overlapped) leaves un-hidden behind",
        "compute. Low ⇒ compute-bound (GEMM); high ⇒ the input path is the bottleneck even with prefetch.",
        "",
    ]
    with open("TRAIN_LOOP_BREAKDOWN.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("[loop] wrote TRAIN_LOOP_BREAKDOWN.md")


if __name__ == "__main__":
    main()
