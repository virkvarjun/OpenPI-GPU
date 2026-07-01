"""V1/V2: measure real FSDP strong-scaling of the gemma_2b step across the visible GPUs (single process).

Single-process multi-GPU (set CUDA_VISIBLE_DEVICES to pick 1/2/4 GPUs) shards the gemma_2b params along the mesh
`fsdp` axis via the REAL `sharding.fsdp_sharding`, shards the batch along the `data` axis, and runs the fwd+bwd
step with in/out shardings — exercising V1's mesh + FSDP path on real hardware, with collectives (all-gather of
sharded params) running within one process (reliable NCCL, unlike multi-process). Reports the real blocked
device time + aggregate/per-GPU MFU (6·N·D), so 1→2→4 GPU strong scaling is measurable with a fixed global batch.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--batch-size", type=int, default=4, help="GLOBAL batch (sharded over the data axis)")
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--peak-flops", type=float, default=990e12)
    args = p.parse_args()

    n_gpus = jax.device_count()
    with at.disable_typechecking():
        cfg = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", action_horizon=50, max_token_len=48
        )
        # Full FSDP: shard params over all visible GPUs. make_mesh(num_fsdp_devices=n) -> (1, n).
        mesh = sharding.make_mesh(num_fsdp_devices=n_gpus)
        # Shard the LEADING (batch) dim over the data axis — a single-element spec works for any-rank leaf
        # (images [B,H,W,C], masks [B]); a 2-elem spec would fail on the rank-1 image_masks. Matches train.py.
        data_sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

        rng = jax.random.key(0)
        model = cfg.create(rng)
        model_def = nnx.graphdef(model)
        params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            nnx.state(model),
        )
        n_params = sum(int(x.size) for x in jax.tree.leaves(params) if hasattr(x, "size"))
        del model

        # Real FSDP param sharding (V1's sharding.py), then place params accordingly.
        state_sharding = sharding.fsdp_sharding(params, mesh)
        params = jax.device_put(params, state_sharding)

        obs_spec, act_spec = cfg.inputs_spec(batch_size=args.batch_size)
        obs = jax.device_put(_gen(obs_spec.to_dict()), data_sh)
        act = jax.device_put(_gen(act_spec), data_sh)
        from openpi.models import model as _model

        obs = _model.Observation.from_dict(obs)

        def step(params, rng, obs, act):
            m = nnx.merge(model_def, params)
            m.train()
            return nnx.value_and_grad(lambda mm: jnp.mean(mm.compute_loss(rng, obs, act, train=True)))(m)

        jstep = jax.jit(step, in_shardings=(state_sharding, None, data_sh, data_sh), out_shardings=state_sharding)
        with sharding.set_mesh(mesh):
            jax.block_until_ready(jstep(params, rng, obs, act))  # compile + warmup
            for _ in range(2):
                jax.block_until_ready(jstep(params, rng, obs, act))
            samples = []
            for _ in range(args.iters):
                t = time.perf_counter()
                jax.block_until_ready(jstep(params, rng, obs, act))
                samples.append((time.perf_counter() - t) * 1e3)
        device_ms = statistics.median(samples)

    img_patches = (224 // 14) ** 2
    tokens = args.batch_size * (3 * img_patches + 48 + 50)
    flops = 6.0 * n_params * tokens
    achieved = flops / (device_ms / 1e3)
    mfu_agg = achieved / (n_gpus * args.peak_flops) * 100
    print(
        f"FSDP n_gpus={n_gpus} | global_batch={args.batch_size} | {n_params/1e9:.2f}B params | "
        f"device {device_ms:.1f} ms/step | {achieved/1e12:.0f} TFLOP/s aggregate | {mfu_agg:.1f}% per-GPU MFU"
    )


if __name__ == "__main__":
    main()
