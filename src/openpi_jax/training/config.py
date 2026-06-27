"""Training configuration registry.

Every experiment is a named ``TrainConfig``. The CLI (``openpi-train --config <name>``) looks the name up in
``CONFIGS``. This mirrors OpenPI's config-as-code approach: configs are plain dataclasses, fully introspectable
and overridable from the command line via ``tyro``.
"""

from __future__ import annotations

import dataclasses

from openpi_jax.models.pi0 import Pi0Config


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    peak_lr: float = 2.5e-5
    warmup_steps: int = 1_000
    decay_steps: int = 30_000
    weight_decay: float = 1e-4
    b1: float = 0.9
    b2: float = 0.95
    grad_clip_norm: float = 1.0


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot dataset repo id, e.g. "lerobot/aloha_sim_transfer_cube_human".
    repo_id: str = "lerobot/aloha_sim_transfer_cube_human"
    batch_size: int = 32
    num_workers: int = 8
    # Camera keys present in the dataset, mapped to model image names.
    camera_keys: tuple[str, ...] = ("base_0_rgb",)
    # Normalization mode: "mean_std" or "quantile".
    norm_mode: str = "quantile"


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: str
    model: Pi0Config = dataclasses.field(default_factory=Pi0Config)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)

    num_train_steps: int = 30_000
    log_every: int = 100
    save_every: int = 5_000
    seed: int = 0
    checkpoint_dir: str = "checkpoints"

    # Optional path to weights to warm-start from (e.g. a ported PaliGemma checkpoint).
    pretrained_path: str | None = None


# --- registry ------------------------------------------------------------

CONFIGS: dict[str, TrainConfig] = {
    "pi0_aloha_sim": TrainConfig(
        name="pi0_aloha_sim",
        model=Pi0Config(action_dim=14, action_horizon=50),
        data=DataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            camera_keys=("base_0_rgb",),
            batch_size=32,
        ),
    ),
    "pi0_droid": TrainConfig(
        name="pi0_droid",
        model=Pi0Config(action_dim=8, action_horizon=16),
        data=DataConfig(
            repo_id="lerobot/droid",
            camera_keys=("base_0_rgb", "left_wrist_0_rgb"),
            batch_size=64,
        ),
    ),
    "debug": TrainConfig(
        name="debug",
        model=Pi0Config(action_dim=7, action_horizon=8, num_inference_steps=4),
        data=DataConfig(batch_size=2, num_workers=0),
        num_train_steps=10,
        log_every=1,
        save_every=10,
    ),
}


def get_config(name: str) -> TrainConfig:
    if name not in CONFIGS:
        raise KeyError(f"unknown config '{name}'. available: {sorted(CONFIGS)}")
    return CONFIGS[name]
