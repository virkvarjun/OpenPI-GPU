"""The model-facing observation/action containers.

These mirror the structured inputs OpenPI feeds π0: one or more camera images (with masks indicating which
cameras are present), a tokenized language prompt, and the robot proprioceptive state.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from openpi_jax.shared.typing import Array


@dataclasses.dataclass
class Observation:
    """A single (batched) observation fed to the model.

    Shapes use ``B`` for batch, ``H/W`` image size, ``C`` cameras, ``L`` token length, ``S`` state dim.
    """

    # Dict of camera-name -> image array, each [B, H, W, 3] in [-1, 1].
    images: dict[str, Array]
    # Dict of camera-name -> bool [B], whether that camera is present this step.
    image_masks: dict[str, Array]
    # Tokenized language prompt, [B, L] int32.
    tokenized_prompt: Array
    # Attention mask for the prompt, [B, L] bool.
    tokenized_prompt_mask: Array
    # Proprioceptive state, [B, S] float32.
    state: Array

    @classmethod
    def from_dict(cls, batch: dict) -> Observation:
        """Build an Observation from a raw data-pipeline dict, with light validation."""
        required = ("images", "image_masks", "tokenized_prompt", "tokenized_prompt_mask", "state")
        missing = [k for k in required if k not in batch]
        if missing:
            raise KeyError(f"observation batch missing keys: {missing}")
        return cls(
            images=batch["images"],
            image_masks=batch["image_masks"],
            tokenized_prompt=batch["tokenized_prompt"].astype(jnp.int32),
            tokenized_prompt_mask=batch["tokenized_prompt_mask"].astype(bool),
            state=batch["state"].astype(jnp.float32),
        )


@dataclasses.dataclass
class Actions:
    """A chunk of actions, [B, horizon, action_dim] float32."""

    value: Array
