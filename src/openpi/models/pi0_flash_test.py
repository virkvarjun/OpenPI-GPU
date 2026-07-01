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


def _loss(use_flash: bool) -> np.ndarray:
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

    obs = jax.tree.map(rand, obs_spec)
    act = jax.tree.map(rand, act_spec)
    return np.asarray(model.compute_loss(rng, obs, act, train=True))


def test_flash_attention_matches_naive():
    with at.disable_typechecking():
        naive = _loss(use_flash=False)
        flash = _loss(use_flash=True)
    assert np.max(np.abs(naive - flash)) < 2e-2, f"flash != naive: maxdiff {np.max(np.abs(naive - flash))}"
