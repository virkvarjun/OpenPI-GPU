"""V2: attribute the REAL train loop (data pipeline feeding the GPU) — trustworthy data-wait.

The single-step microbench (`profile_step_breakdown.py`) can't measure data-wait: it re-runs a fixed on-device
input, so `wall - device_busy` there is just host dispatch overhead. This script runs the actual train.py-style
loop — build a batch on the host, dispatch the jitted step **asynchronously**, build the next batch while the GPU
computes (overlap), block once at the end — and attributes it. Now `data-wait = wall - device_busy` is the
genuine input-pipeline cost NOT hidden behind compute: if the host pipeline keeps the GPU fed, data-wait → ~0;
if not, the input pipeline is the measured bottleneck.

This mirrors the `FakeDataConfig` + `num_workers=0` path: per step we generate host (numpy) arrays of the model's
input shapes, collate, and move them to the device via the same `make_array_from_process_local_data` assembly the
real loader uses. We generate inline rather than importing `training.data_loader` to avoid its torch/lerobot
import (whose CUDA libs conflict with jax on this box); with num_workers=0 the two paths are equivalent (no torch
worker pool). Swap in a real LeRobot/RLDS loader to fold in image-decode/disk cost. gemma_2b runs forward+backward
in bf16 (fits 32GB; full AdamW state would not).
"""

from __future__ import annotations

import argparse
import tempfile
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
import openpi.models.model as _model
from openpi.models import pi0_config
from openpi.training import attribution


def _bf16(params):
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        params,
    )


def _host_array(spec):
    """A host (numpy) array matching a ShapeDtypeStruct — the kind a real data loader produces before H2D."""
    if spec.dtype == jnp.float32:
        return np.random.standard_normal(spec.shape).astype(np.float32)
    if spec.dtype == jnp.int32:
        return np.random.randint(0, 100, spec.shape).astype(np.int32)
    if spec.dtype == bool:
        return np.ones(spec.shape, dtype=bool)
    return np.zeros(spec.shape, dtype=spec.dtype)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="gemma_2b")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2, help="data loader prefetch workers")
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
    obs_spec, act_spec = model_cfg.inputs_spec(batch_size=args.batch_size)
    obs_dict_spec = obs_spec.to_dict()

    def next_batch():
        # Host-side numpy generation + collate, then H2D via the loader's assembly primitive (the data pipeline).
        obs_np = jax.tree.map(_host_array, obs_dict_spec)
        act_np = jax.tree.map(_host_array, act_spec)
        obs_dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sh, x), obs_np)
        act_dev = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sh, x), act_np)
        return _model.Observation.from_dict(obs_dev), act_dev

    with at.disable_typechecking():
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

        obs, act = next_batch()
        compiled = jstep.lower(params, rng, obs, act).compile()

        # Warmup (compile).
        for _ in range(args.warmup):
            obs, act = next_batch()
            jax.block_until_ready(jstep(params, rng, obs, act))

        # PHASE A — CLEAN wall (NO profiler): the overlapped loop, timed honestly. jax.profiler.trace adds large
        # per-op recording overhead, so timing inside it inflates wall and fakes "data-wait". We time clean here.
        # Keep only the loss SCALAR to block on — the full grad pytree (~5GB bf16) must not accumulate.
        def run_window():
            losses = []
            obs, act = next_batch()
            for _ in range(args.iters):
                loss, grads = jstep(params, rng, obs, act)
                losses.append(loss)
                del grads
                obs, act = next_batch()  # host gen + H2D while the GPU runs the dispatched step
            jax.block_until_ready(losses)

        t0 = time.perf_counter()
        run_window()
        clean_wall_us = (time.perf_counter() - t0) * 1e6

        # PHASE B — TRACE a few fixed-input steps purely for the device op-category composition.
        trace_dir = tempfile.mkdtemp()
        obs, act = next_batch()
        with jax.profiler.trace(trace_dir):
            for _ in range(min(8, args.iters)):
                loss, grads = jstep(params, rng, obs, act)
                jax.block_until_ready(loss)
                del grads
        trace_steps = min(8, args.iters)

    # Device-op composition (Phase B) + clean per-step timing (Phase A).
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
    # Express category times per-step and as % of CLEAN wall.
    bd = attribution.Breakdown(
        category_us={k: v / trace_steps * args.iters for k, v in cat.items()},
        device_busy_us=device_busy_per_step * args.iters,
        wall_us=clean_wall_us,
        n_steps=args.iters,
    )
    pct = bd.percentages()

    lines = [
        "# TRAIN_LOOP_BREAKDOWN — real loop with data pipeline (clean-wall data-wait)",
        "",
        f"> ✅ **Overlapped train.py-style loop**, `{args.variant}` bf16 fwd+bwd, batch={args.batch_size}, "
        f"FakeData pipeline (host numpy gen + collate + H2D, num_workers=0 equivalent), on {device.device_kind} "
        f"({device.platform}). Wall is timed WITHOUT the profiler (Phase A) so it's not inflated by trace overhead; "
        "op composition is from a separate short traced window (Phase B). `data-wait = wall - device_busy` is the "
        "genuine host/input cost NOT overlapped behind compute (≈0 ⇒ the GPU is kept fed).",
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
        "Interpretation: data-wait here = FakeData host gen + collate + H2D + dispatch not overlapped by compute.",
        "With a real LeRobot/RLDS config it also includes image decode + disk. Low data-wait ⇒ compute-bound (GEMM,",
        "per the single-step breakdown); high data-wait ⇒ the host/input path is the measured bottleneck.",
        "",
    ]
    with open("TRAIN_LOOP_BREAKDOWN.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("[loop] wrote TRAIN_LOOP_BREAKDOWN.md")


if __name__ == "__main__":
    main()
