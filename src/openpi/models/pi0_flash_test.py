"""Parity test for the flash-attention path: flash must match the naive gemma attention on the real pi0 model.

`use_flash_attention` swaps gemma's naive einsum+softmax+einsum for `jax.nn.dot_product_attention`. This checks
that, with identical init + inputs, the model's `compute_loss` is (bf16-)equal — so enabling flash is a safe,
behavior-preserving optimization (the risky bits are the block-causal mask, the head_dim**-0.5 pre-scaling, and
GQA, all of which this exercises end-to-end).
"""

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
from openpi.models import pi0_config


def _model_and_inputs(use_flash: bool):
    cfg = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        max_token_len=8,
        use_flash_attention=use_flash,
    )
    rng = jax.random.key(0)
    model = cfg.create(rng)
    obs_spec, act_spec = cfg.inputs_spec(batch_size=2)

    def rand(s):
        if s.dtype == jnp.float32:
            return jax.random.uniform(rng, s.shape, minval=-1, maxval=1)
        if s.dtype == jnp.int32:
            return jax.random.randint(rng, s.shape, 0, 100)
        if s.dtype == bool:
            return jnp.ones(s.shape, bool)
        return jnp.zeros(s.shape, s.dtype)

    return model, rng, jax.tree.map(rand, obs_spec), jax.tree.map(rand, act_spec)


def test_flash_matches_naive_training():
    with at.disable_typechecking():
        m0, rng, obs, act = _model_and_inputs(False)
        m1, *_ = _model_and_inputs(True)
        naive = np.asarray(m0.compute_loss(rng, obs, act, train=True))
        flash = np.asarray(m1.compute_loss(rng, obs, act, train=True))
    assert np.max(np.abs(naive - flash)) < 2e-2, f"training: flash != naive: {np.max(np.abs(naive - flash))}"


def test_flash_matches_naive_inference():
    # Inference uses the KV-cache (different mask shape) — verify flash matches there too.
    with at.disable_typechecking():
        m0, rng, obs, _ = _model_and_inputs(False)
        m1, *_ = _model_and_inputs(True)
        naive = np.asarray(m0.sample_actions(rng, obs, num_steps=4))
        flash = np.asarray(m1.sample_actions(rng, obs, num_steps=4))
    assert np.max(np.abs(naive - flash)) < 5e-2, f"inference: flash != naive: {np.max(np.abs(naive - flash))}"
