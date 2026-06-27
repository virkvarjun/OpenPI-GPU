"""End-to-end smoke run with the fake data loader — no external data or weights needed.

    python examples/quickstart.py

Builds a (tiny) π0, runs one loss step, and samples an action chunk. Useful as a first sanity check after
installing the package.
"""

import jax
import numpy as np

from openpi_jax.data.dataset import make_data_loader
from openpi_jax.models.observation import Observation
from openpi_jax.models.pi0 import Pi0
from openpi_jax.training.config import get_config


def main():
    cfg = get_config("debug")
    loader = make_data_loader(cfg.data, action_dim=cfg.model.action_dim, horizon=cfg.model.action_horizon)
    batch = next(iter(loader))
    obs = Observation.from_dict(batch)

    model = Pi0(cfg.model)
    rng = jax.random.key(0)
    variables = model.init(rng, rng, obs, batch["actions"])

    loss = model.apply(variables, rng, obs, batch["actions"])
    actions = model.apply(variables, rng, obs, method=model.sample_actions)

    print(f"loss: {float(loss):.4f}")
    print(f"sampled action chunk: {np.asarray(actions).shape}")


if __name__ == "__main__":
    main()
