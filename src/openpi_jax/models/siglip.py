"""SigLIP vision encoder (ViT) — produces a sequence of image tokens for the VLM.

This is a standard Vision Transformer: patchify -> + position embeddings -> N transformer blocks. π0 uses the
SigLIP-So400m variant from PaliGemma. Weights can be ported from the PaliGemma checkpoint (see docs/CHECKPOINTS).
"""

from __future__ import annotations

import dataclasses

import flax.linen as nn
import jax.numpy as jnp

from openpi_jax.shared.typing import Array


@dataclasses.dataclass(frozen=True)
class SiglipConfig:
    image_size: int = 224
    patch_size: int = 14
    width: int = 1152
    depth: int = 27
    num_heads: int = 16
    mlp_dim: int = 4304
    dtype: str = "bfloat16"

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array) -> Array:
        d = x.shape[-1]
        x = nn.Dense(self.mlp_dim, dtype=self.dtype)(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(d, dtype=self.dtype)(x)
        return x


class EncoderBlock(nn.Module):
    num_heads: int
    mlp_dim: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: Array) -> Array:
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dtype=self.dtype)(y, y)
        x = x + y
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(self.mlp_dim, dtype=self.dtype)(y)
        return x + y


class SiglipVisionModel(nn.Module):
    config: SiglipConfig

    @nn.compact
    def __call__(self, images: Array) -> Array:
        """images: [B, H, W, 3] in [-1, 1] -> tokens: [B, num_patches, width]."""
        cfg = self.config
        dtype = jnp.dtype(cfg.dtype)
        x = nn.Conv(
            cfg.width,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            dtype=dtype,
            name="embedding",
        )(images)
        b = x.shape[0]
        x = x.reshape(b, -1, cfg.width)  # [B, num_patches, width]

        pos = self.param("pos_embed", nn.initializers.normal(0.02), (1, cfg.num_patches, cfg.width))
        x = x + pos.astype(dtype)

        for i in range(cfg.depth):
            x = EncoderBlock(cfg.num_heads, cfg.mlp_dim, dtype=dtype, name=f"block_{i}")(x)
        x = nn.LayerNorm(dtype=dtype, name="post_ln")(x)
        return x
