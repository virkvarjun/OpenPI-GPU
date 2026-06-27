"""V2/M1: produce STEP_BREAKDOWN.md by attributing the REAL openpi pi0 training step.

Runs the actual `openpi.models.pi0` model (gemma attention with the naive masked softmax, the FFN, the action
expert, flow-matching loss) through a faithful nnx `train_step` (value_and_grad + optax AdamW, mirroring
scripts/train.py), and attributes measured device time into attention | matmul-FFN | collectives | optimizer |
other | data-wait via `openpi.training.attribution`.

To run on the cheap CPU ladder we use the `dummy` gemma variant (width=64, depth=4). The CODE PATH is the real
model; the category *proportions* are NOT production numbers — small dims + CPU change the mix (e.g. data-wait
and "other" are inflated by tiny-op dispatch). Collect production proportions by running this attribution on
`scripts/train.py`'s ptrain_step on a real accelerator slice (HW-gated). The instrument and the decision gate
are what's delivered here.
"""

from __future__ import annotations

import argparse
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
from openpi.models import pi0_config
from openpi.training import attribution


def _fake_inputs(cfg, rng, batch_size=1):
    obs_spec, act_spec = cfg.inputs_spec(batch_size=batch_size)

    def rand(spec):
        if spec.dtype == jnp.float32:
            return jax.random.uniform(rng, spec.shape, minval=-1.0, maxval=1.0)
        if spec.dtype == jnp.int32:
            return jax.random.randint(rng, spec.shape, 0, 100)
        if spec.dtype == bool:
            return jnp.ones(spec.shape, bool)
        return jnp.zeros(spec.shape, spec.dtype)

    return jax.tree.map(rand, obs_spec), jax.tree.map(rand, act_spec)


def build_step(variant: str = "dummy", batch_size: int = 1, optimizer: str = "adamw"):
    # On a real GPU/TPU slice, pass --variant gemma_2b for the production breakdown. On CPU keep `dummy`.
    small = variant == "dummy"
    cfg = pi0_config.Pi0Config(
        paligemma_variant=variant,
        action_expert_variant="dummy" if small else "gemma_300m",
        action_horizon=4 if small else 50,
        max_token_len=8 if small else 48,
    )
    rng = jax.random.key(0)
    model = cfg.create(rng)
    model_def = nnx.graphdef(model)
    params = nnx.state(model)
    # gemma_2b params/grads in fp32 alone exceed a 32GB GPU (~2.5B x 4B x2). Cast to bf16 for real variants --
    # this is how mixed-precision training runs anyway, and attribution measures op structure, not numerics.
    if not small:
        params = jax.tree.map(
            lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            params,
        )
    obs, act = _fake_inputs(cfg, rng, batch_size=batch_size)

    def loss_fn(model, rng, obs, act):
        return jnp.mean(model.compute_loss(rng, obs, act, train=True))

    if optimizer == "none":
        # Forward + backward only. This is the bottleneck region (attention/FFN/vision/data-wait) and avoids the
        # optimizer state, which is what makes full AdamW on gemma_2b exceed a 32GB GPU.
        def step(params, rng, obs, act):
            model = nnx.merge(model_def, params)
            model.train()
            return nnx.value_and_grad(loss_fn)(model, rng, obs, act)

        jstep = jax.jit(step)
        args = (params, rng, obs, act)
    else:
        # AdamW doubles param memory (fp32 m+v); SGD-momentum is ~1x and fits gemma_2b on 32GB.
        tx = optax.sgd(1e-3, momentum=0.9) if optimizer == "sgd" else optax.adamw(1e-4)
        opt_state = tx.init(params)

        def step(params, opt_state, rng, obs, act):
            model = nnx.merge(model_def, params)
            model.train()
            loss, grads = nnx.value_and_grad(loss_fn)(model, rng, obs, act)
            updates, opt_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        jstep = jax.jit(step)
        args = (params, opt_state, rng, obs, act)

    compiled = jstep.lower(*args).compile()
    return (lambda: jstep(*args)), compiled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default="dummy", help="paligemma variant: dummy (CPU) | gemma_2b (real GPU/TPU)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "none"],
                        help="none = forward+backward only (fits gemma_2b on 32GB); sgd = light state")
    args = parser.parse_args()

    device = jax.devices()[0]
    is_cpu = device.platform == "cpu"
    label = f"`{args.variant}` variant, optimizer={args.optimizer}, batch={args.batch_size}, on {device.device_kind} ({device.platform})"
    if is_cpu or args.variant == "dummy":
        flag = (f"> ⚠️ **Not production: {label}.** The code path is the real model, but tiny dims and/or CPU "
                "inflate data-wait/other (per-op dispatch) and shrink the GEMM/attention share. Run "
                "`--variant gemma_2b` on a real accelerator for production proportions.")
    else:
        flag = (f"> ✅ **Production-scale run: {label}.** Random-init weights (structure is what attribution "
                "measures). `optimizer=none` profiles forward+backward — the bottleneck region; the AdamW update "
                "is small and elementwise.")

    # openpi guards model fns with jaxtyping runtime checks; disable them for AOT lowering/profiling (matches
    # how training/checkpoints.py wraps jitted regions).
    with at.disable_typechecking():
        run_step, compiled = build_step(variant=args.variant, batch_size=args.batch_size, optimizer=args.optimizer)
        trace_dir = tempfile.mkdtemp()
        bd = attribution.attribute_step(run_step, compiled, trace_dir=trace_dir, warmup=args.warmup, iters=args.iters)
    pct = bd.percentages()

    lines = [
        "# STEP_BREAKDOWN — openpi pi0 training step (attribution)",
        "",
        "Real `openpi.models.pi0` step (gemma naive-softmax attention + FFN + action expert + flow-matching",
        "loss), attributed by HLO named-scope + op type.",
        "",
        flag,
        "",
        f"Window: {bd.n_steps} steps | wall {bd.wall_us / 1e3 / bd.n_steps:.3f} ms/step | "
        f"device-busy {bd.device_busy_us / 1e3 / bd.n_steps:.3f} ms/step",
        "",
        "| category | device ms/step | % wall |",
        "|----------|---------------:|-------:|",
    ]
    order = [*attribution.CATEGORIES, "data-wait"]
    for c in order:
        ms = (bd.category_us.get(c, bd.data_wait_us if c == "data-wait" else 0.0)) / 1e3 / bd.n_steps
        lines.append(f"| {c} | {ms:.4f} | {pct[c] * 100:.1f}% |")

    dominant = bd.dominant()
    lines += [
        "",
        f"**Dominant (this run): `{dominant}`** ({pct[dominant] * 100:.1f}% of wall).",
        "",
        "## Decision gate (apply to PRODUCTION proportions, not these CPU/dummy ones)",
        "- **data-wait dominates** → input pipeline: image decode, prefetch depth, H2D overlap.",
        "- **collectives dominate** (multi-host) → comms/compute overlap, collective placement, mesh/fsdp tuning.",
        "- **attention > ~15-20%** → fused attention (jax.nn.dot_product_attention flash; then Pallas/splash). "
        "Do NOT touch saturated matmul/GEMM.",
        "- **memory-bound / OOM** → tune the remat policy off `nothing_saveable` to recover recompute.",
        "",
        "On real hardware the GEMM/attention share rises sharply (large dims) and data-wait/other shrink; the",
        "instrument is unchanged. This artifact + the harness are the V2/M1 deliverable; the optimization target",
        "is chosen from the production breakdown.",
        "",
    ]
    with open("STEP_BREAKDOWN.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\n[breakdown] wrote STEP_BREAKDOWN.md")


if __name__ == "__main__":
    main()
