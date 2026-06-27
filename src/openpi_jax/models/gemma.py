"""Gemma decoder backbone with dual-stream (VLM + action expert) attention.

π0's key architectural trick is a *mixture of experts over token type*: vision+language tokens are processed by
the large Gemma weights ("VLM expert"), while state+action tokens are processed by a smaller Gemma
("action expert"). Both streams share a single self-attention over the concatenated sequence, so actions can
attend to the visual/language context. This module implements the building blocks; the two-expert wiring lives
in ``pi0.py``.
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi_jax.shared.typing import Array


@dataclasses.dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int = 257_152
    norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    dtype: str = "bfloat16"


# Reference presets (approximate, for the publicly described π0 configuration).
GEMMA_2B = GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
GEMMA_300M = GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)


def rms_norm(x: Array, scale: Array, eps: float) -> Array:
    dtype = x.dtype
    x = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + eps)
    return (x * (1.0 + scale)).astype(dtype)


def apply_rope(x: Array, positions: Array, theta: float) -> Array:
    """Rotary position embedding. x: [B, T, H, head_dim]; positions: [B, T]."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    freqs = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[..., None].astype(jnp.float32) * freqs  # [B, T, half]
    angles = angles[:, :, None, :]  # broadcast over heads
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    x1, x2 = jnp.split(x, 2, axis=-1)
    out = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    return out.astype(x.dtype)


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array) -> Array:
        scale = self.param("scale", nn.initializers.zeros, (x.shape[-1],))
        return rms_norm(x, scale, self.eps)


class FeedForward(nn.Module):
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array) -> Array:
        d = x.shape[-1]
        gate = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype, name="gating")(x)
        up = nn.Dense(self.mlp_dim, use_bias=False, dtype=self.dtype, name="up")(x)
        x = nn.gelu(gate, approximate=True) * up
        return nn.Dense(d, use_bias=False, dtype=self.dtype, name="down")(x)
