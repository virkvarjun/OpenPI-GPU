"""A ``Policy`` bundles a trained model with its transforms for closed-loop inference.

The serving server (``serve.py``) calls ``Policy.infer(obs_dict) -> action_dict`` once per control step:
  raw obs -> input transforms (tokenize, normalize) -> model.sample_actions -> output transforms (unnormalize).
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np

from openpi_jax.models.observation import Observation
from openpi_jax.models.pi0 import Pi0
from openpi_jax.transforms.base import Compose


@dataclasses.dataclass
class Policy:
    model: Pi0
    params: dict
    input_transform: Compose | None = None
    output_transform: Compose | None = None
    seed: int = 0

    def __post_init__(self):
        self._rng = jax.random.key(self.seed)

        def _sample(params, rng, batch):
            obs = Observation.from_dict(batch)
            return self.model.apply({"params": params}, rng, obs, method=self.model.sample_actions)

        self._sample = jax.jit(_sample)

    def infer(self, obs: dict) -> dict:
        """obs: a raw single-step observation dict -> {"actions": [horizon, action_dim]}."""
        if self.input_transform is not None:
            obs = self.input_transform(obs)

        # Add a batch dim of 1.
        batch = jax.tree_util.tree_map(lambda x: np.asarray(x)[None], obs)
        self._rng, step_rng = jax.random.split(self._rng)
        actions = self._sample(self.params, step_rng, batch)
        actions = np.asarray(actions)[0]  # drop batch dim

        out = {"actions": actions}
        if self.output_transform is not None:
            out = self.output_transform(out)
        return out
