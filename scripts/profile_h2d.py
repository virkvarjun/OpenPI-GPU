"""V2 targeted split-measurement: isolate the host-side per-step cost into its components.

The loop breakdown showed ~91% "data-wait" that scales with batch and isn't hidden by prefetch. This pins down
which host component dominates by timing each in isolation, for a gemma_2b-shaped batch on the data-axis sharding:

  - gen        : np.random host generation of the batch (FakeData artifact; a real dataset decodes JPEGs instead)
  - make_array : per-leaf jax.make_array_from_process_local_data(sharding, leaf)  -- what the loader does today
  - device_put : a single jax.device_put(batch_pytree, sharding)                  -- the proposed batched H2D

If make_array >> device_put, the per-leaf assembly is the real, optimizable bottleneck (the FakeData gen is the
artifact). This is the BEFORE/AFTER for the batched-H2D change.
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
import openpi.training.sharding as sharding
from openpi.models import pi0_config


def _host(spec):
    dt = np.dtype(spec.dtype)
    shape = tuple(spec.shape)  # full [B, ...] from inputs_spec(batch_size=B)
    if dt == np.float32:
        return np.random.standard_normal(shape).astype(np.float32)
    if dt == np.int32:
        return np.random.randint(0, 100, shape).astype(np.int32)
    if dt == bool:
        return np.ones(shape, dtype=bool)
    return np.zeros(shape, dtype=dt)


def _gen_tree(spec_tree):
    if spec_tree is None:
        return None
    if isinstance(spec_tree, dict):
        return {k: g for k, v in spec_tree.items() if (g := _gen_tree(v)) is not None}
    return _host(spec_tree)


def _median_ms(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t) * 1e3)
    return statistics.median(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    args = p.parse_args()

    with at.disable_typechecking():
        cfg = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", action_horizon=50, max_token_len=48
        )
        obs_spec, act_spec = cfg.inputs_spec(batch_size=args.batch_size)
        specs = {**obs_spec.to_dict(), "actions": act_spec}

        mesh = sharding.make_mesh(1)
        data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

        gen_ms = _median_ms(lambda: _gen_tree(specs))

        batch = _gen_tree(specs)  # fixed host batch, reused so we time only the H2D
        make_array_ms = _median_ms(
            lambda: jax.block_until_ready(
                jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sh, x), batch)
            )
        )
        device_put_ms = _median_ms(lambda: jax.block_until_ready(jax.device_put(batch, data_sh)))

    nbytes = sum(np.asarray(x).nbytes for x in jax.tree.leaves(batch))
    print(
        f"batch={args.batch_size} bytes={nbytes/1e6:.1f}MB | "
        f"gen={gen_ms:.2f}ms | make_array(per-leaf)={make_array_ms:.2f}ms | device_put(batched)={device_put_ms:.2f}ms | "
        f"speedup={make_array_ms/device_put_ms:.1f}x"
    )


if __name__ == "__main__":
    main()
