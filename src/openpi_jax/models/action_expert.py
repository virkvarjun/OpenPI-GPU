"""Action-expert projections for π0's flow-matching head.

The action expert is a small Gemma (see ``gemma.GEMMA_300M``). It receives the robot state and a chunk of
*noisy* actions, embeds them into the shared transformer width, and after the joint attention pass projects the
final action tokens back to a per-timestep velocity used by the flow-matching ODE.

This module only holds the input/output projections + the flow-matching time embedding; the transformer trunk
is shared and lives in ``pi0.py``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi_jax.shared.typing import Array


def sinusoidal_time_embedding(t: Array, dim: int, max_period: float = 10_000.0) -> Array:
    """Embed a continuous flow-matching time ``t in [0, 1]`` into a ``dim``-vector. t: [B] -> [B, dim]."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = t[:, None].astype(jnp.float32) * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class ActionEncoder(nn.Module):
    """Projects (state, noisy_actions, time) into action-expert token embeddings."""

    width: int
    action_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, state: Array, noisy_actions: Array, time: Array) -> Array:
        # state: [B, S]; noisy_actions: [B, horizon, action_dim]; time: [B]
        b, horizon, _ = noisy_actions.shape
        state_tok = nn.Dense(self.width, dtype=self.dtype, name="state_in")(state)[:, None, :]
        act_tok = nn.Dense(self.width, dtype=self.dtype, name="action_in")(noisy_actions)

        t_emb = sinusoidal_time_embedding(time, self.width)
        t_emb = nn.Dense(self.width, dtype=self.dtype, name="time_in")(t_emb)[:, None, :]
        act_tok = act_tok + t_emb  # broadcast time across the horizon

        # [B, 1 + horizon, width]: a single state token followed by one token per action step.
        return jnp.concatenate([state_tok, act_tok], axis=1)


class ActionDecoder(nn.Module):
    """Projects action-expert output tokens back to a per-step velocity field."""

    action_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, tokens: Array) -> Array:
        # tokens: [B, 1 + horizon, width] -> drop the leading state token -> velocity [B, horizon, action_dim]
        act_tokens = tokens[:, 1:, :]
        return nn.Dense(self.action_dim, dtype=jnp.float32, name="action_out")(act_tokens)
