"""Smoke tests: exercise the scaffolding end-to-end with the fake data loader on CPU.

These don't assert model quality — they assert the shapes, configs, and the train/inference plumbing all wire
together without errors. Heavy modeling lands later (see docs/ROADMAP.md); update these as it does.
"""

import jax
import numpy as np
import pytest

from openpi_jax.data.dataset import make_data_loader
from openpi_jax.models.observation import Observation
from openpi_jax.models.pi0 import Pi0
from openpi_jax.shared.normalize import Normalizer, NormStats
from openpi_jax.training.config import get_config


def test_configs_resolve():
    for name in ("debug", "pi0_aloha_sim", "pi0_droid"):
        cfg = get_config(name)
        assert cfg.name == name
        assert cfg.model.action_dim > 0


def test_fake_loader_shapes():
    cfg = get_config("debug")
    loader = make_data_loader(cfg.data, action_dim=cfg.model.action_dim, horizon=cfg.model.action_horizon)
    batch = next(iter(loader))
    b = cfg.data.batch_size
    assert batch["actions"].shape == (b, cfg.model.action_horizon, cfg.model.action_dim)
    assert batch["state"].shape[0] == b
    obs = Observation.from_dict(batch)
    assert obs.tokenized_prompt.dtype == np.int32 or str(obs.tokenized_prompt.dtype) == "int32"


def test_normalizer_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 7)).astype(np.float32)
    stats = {"actions": NormStats(mean=x.mean(0), std=x.std(0))}
    norm = Normalizer(stats, mode="mean_std")
    out = norm.unnormalize(norm.normalize({"actions": x}))["actions"]
    np.testing.assert_allclose(out, x, rtol=1e-4, atol=1e-4)


def test_pi0_loss_and_sample_runs():
    cfg = get_config("debug")
    loader = make_data_loader(cfg.data, action_dim=cfg.model.action_dim, horizon=cfg.model.action_horizon)
    batch = next(iter(loader))
    obs = Observation.from_dict(batch)
    actions = batch["actions"]

    model = Pi0(cfg.model)
    rng = jax.random.key(0)
    variables = model.init(rng, rng, obs, actions)

    loss = model.apply(variables, rng, obs, actions)
    assert np.isfinite(float(loss))

    sampled = model.apply(variables, rng, obs, method=model.sample_actions)
    assert sampled.shape == (cfg.data.batch_size, cfg.model.action_horizon, cfg.model.action_dim)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
