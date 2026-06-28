"""V2: confirm the loop's host bottleneck is jitted-step DISPATCH (Python launch overhead), and show where.

Times, for a fixed on-device gemma_2b batch (so no data pipeline is involved):
  - dispatch-only: the `jstep(...)` Python call that returns an async future WITHOUT blocking — pure host launch
    overhead (pytree flatten of the param state, sharding checks, donation, enqueue).
  - blocked step : `block_until_ready(jstep(...))` — launch + actual device compute.
Then cProfile's the dispatch loop and prints the top cumulative-time host functions, so we see WHAT in dispatch
costs the ~115 ms/step the loop attributed to "data-wait".
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from openpi.models import pi0_config


def _host(spec):
    dt = np.dtype(spec.dtype)
    if dt == np.float32:
        return np.random.standard_normal(spec.shape).astype(np.float32)
    if dt == np.int32:
        return np.random.randint(0, 100, spec.shape).astype(np.int32)
    if dt == bool:
        return np.ones(spec.shape, dtype=bool)
    return np.zeros(spec.shape, dtype=dt)


def _gen(t):
    if t is None:
        return None
    if isinstance(t, dict):
        return {k: g for k, v in t.items() if (g := _gen(v)) is not None}
    return _host(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    import openpi.models.model as _model

    with at.disable_typechecking():
        cfg = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", action_horizon=50, max_token_len=48
        )
        obs_spec, act_spec = cfg.inputs_spec(batch_size=args.batch_size)
        mesh = sharding.make_mesh(1)
        data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        obs = _model.Observation.from_dict(
            jax.tree.map(lambda x: jax.device_put(x, data_sh), _gen(obs_spec.to_dict()))
        )
        act = jax.device_put(_gen(act_spec), data_sh)

        rng = jax.random.key(0)
        model = cfg.create(rng)
        model_def = nnx.graphdef(model)
        params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        )
        n_param_leaves = len(jax.tree.leaves(params))
        del model  # free the original fp32 param copy (~10GB) — only model_def + bf16 params are needed
        jax.block_until_ready(params)

        def step(params, rng, obs, act):
            m = nnx.merge(model_def, params)
            m.train()
            return nnx.value_and_grad(lambda mm: jnp.mean(mm.compute_loss(rng, obs, act, train=True)))(m)

        jstep = jax.jit(step)
        jax.block_until_ready(jstep(params, rng, obs, act))  # compile + warmup

        N = 20
        # (1) dispatch-only: time the python call that returns an async future; don't block.
        d = []
        for _ in range(N):
            t = time.perf_counter()
            out = jstep(params, rng, obs, act)
            d.append((time.perf_counter() - t) * 1e3)
            jax.block_until_ready(out)  # drain between iters so the queue doesn't back up
        dispatch_ms = statistics.median(d)

        # (2) blocked step.
        b = []
        for _ in range(N):
            t = time.perf_counter()
            jax.block_until_ready(jstep(params, rng, obs, act))
            b.append((time.perf_counter() - t) * 1e3)
        blocked_ms = statistics.median(b)

        # (3) cProfile the dispatch loop to see WHERE the host time goes.
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(N):
            out = jstep(params, rng, obs, act)
        jax.block_until_ready(out)
        pr.disable()

    print(f"\ngemma_2b batch={args.batch_size} | param-state leaves={n_param_leaves}")
    print(f"dispatch-only (async call, no block) = {dispatch_ms:.2f} ms/step")
    print(f"blocked step (launch + compute)      = {blocked_ms:.2f} ms/step")
    print("\n=== top host functions by cumulative time (dispatch loop) ===")
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
    # keep it compact: drop the header noise
    for line in s.getvalue().splitlines():
        if "/" in line or "function calls" in line or line.strip().startswith("ncalls"):
            print(line)


if __name__ == "__main__":
    main()
