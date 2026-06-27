"""π0: a flow-matching vision-language-action model.

Training: given an observation and the ground-truth action chunk ``a``, sample a flow-matching time ``t`` and
noise ``e``, form the interpolant ``x_t = t * e + (1 - t) * a``, and regress the model's predicted velocity onto
the target ``u = e - a`` (rectified-flow parameterization). Inference: integrate the learned velocity field from
noise ``x_1 ~ N(0, I)`` back to ``x_0`` (the action) with a few Euler steps.

The transformer trunk (shared attention over [image tokens | text tokens | state+action tokens] with the
VLM/action-expert weight split) is intentionally stubbed here — see ``_backbone`` — so the data, training, and
serving scaffolding can be exercised end-to-end before the heavy modeling lands. See docs/ROADMAP.md.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi_jax.models.action_expert import ActionDecoder, ActionEncoder
from openpi_jax.models.gemma import GEMMA_2B, GEMMA_300M, GemmaConfig
from openpi_jax.models.observation import Observation
from openpi_jax.models.siglip import SiglipConfig, SiglipVisionModel
from openpi_jax.shared.typing import Array, PRNGKey


@dataclasses.dataclass(frozen=True)
class Pi0Config:
    action_dim: int = 7
    action_horizon: int = 50
    max_token_len: int = 48
    num_inference_steps: int = 10
    vlm: GemmaConfig = GEMMA_2B
    action_expert: GemmaConfig = GEMMA_300M
    vision: SiglipConfig = SiglipConfig()
    dtype: str = "bfloat16"

    @property
    def width(self) -> int:
        # Shared attention width — both experts project into the larger VLM width.
        return self.vlm.width


class Pi0(nn.Module):
    config: Pi0Config

    def setup(self):
        cfg = self.config
        self.vision = SiglipVisionModel(cfg.vision)
        self.action_encoder = ActionEncoder(cfg.width, cfg.action_dim, jnp.dtype(cfg.dtype))
        self.action_decoder = ActionDecoder(cfg.action_dim, jnp.dtype(cfg.dtype))
        # Project SigLIP tokens into the LLM width.
        self.image_proj = nn.Dense(cfg.width, name="image_proj")
        self.prompt_embed = nn.Embed(cfg.vlm.vocab_size, cfg.width, name="prompt_embed")

    # --- shared trunk ----------------------------------------------------

    def _embed_prefix(self, obs: Observation) -> Array:
        """Embed images + language into the prefix token stream [B, prefix_len, width]."""
        img_tokens = []
        for image in obs.images.values():
            tok = self.image_proj(self.vision(image))  # [B, num_patches, width]
            img_tokens.append(tok)
        img_tokens = jnp.concatenate(img_tokens, axis=1) if img_tokens else None

        txt_tokens = self.prompt_embed(obs.tokenized_prompt)  # [B, L, width]
        if img_tokens is None:
            return txt_tokens
        return jnp.concatenate([img_tokens, txt_tokens], axis=1)

    def _backbone(self, prefix: Array, suffix: Array) -> Array:
        """Joint attention over [prefix | suffix]; returns the suffix (action) tokens.

        TODO(modeling): implement the dual-expert Gemma trunk:
          - RoPE positions over the concatenated sequence
          - block-causal mask (prefix attends within itself; suffix attends to prefix + causally to suffix)
          - VLM weights on prefix tokens, action-expert weights on suffix tokens, shared QKV attention
        For now this is an identity pass-through so shapes flow and the scaffolding runs.
        """
        return suffix

    # --- flow matching ---------------------------------------------------

    def compute_loss(self, rng: PRNGKey, obs: Observation, actions: Array) -> Array:
        """Rectified-flow regression loss. actions: [B, horizon, action_dim]."""
        b = actions.shape[0]
        t_rng, n_rng = jax.random.split(rng)

        time = jax.random.uniform(t_rng, (b,))  # [B] in [0, 1)
        noise = jax.random.normal(n_rng, actions.shape)
        x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
        target_v = noise - actions  # rectified-flow target velocity

        pred_v = self._predict_velocity(obs, x_t, time)
        return jnp.mean(jnp.square(pred_v - target_v))

    def _predict_velocity(self, obs: Observation, noisy_actions: Array, time: Array) -> Array:
        prefix = self._embed_prefix(obs)
        suffix = self.action_encoder(obs.state, noisy_actions, time)
        out = self._backbone(prefix, suffix)
        return self.action_decoder(out)

    def sample_actions(self, rng: PRNGKey, obs: Observation) -> Array:
        """Integrate the velocity field from noise to an action chunk via Euler steps."""
        cfg = self.config
        b = obs.state.shape[0]
        x = jax.random.normal(rng, (b, cfg.action_horizon, cfg.action_dim))
        dt = 1.0 / cfg.num_inference_steps
        # Integrate from t=1 (noise) down to t=0 (clean actions).
        for i in range(cfg.num_inference_steps):
            t = jnp.full((b,), 1.0 - i * dt)
            v = self._predict_velocity(obs, x, t)
            x = x - dt * v
        return x

    def __call__(self, rng: PRNGKey, obs: Observation, actions: Array) -> Array:
        # Default forward = training loss, so `model.init`/`apply` cover all params.
        return self.compute_loss(rng, obs, actions)
