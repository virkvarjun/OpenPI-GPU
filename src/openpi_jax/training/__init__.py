"""Training: config registry, optimizer, checkpointing, sharding, and the train loop."""

from openpi_jax.training.config import CONFIGS, TrainConfig, get_config

__all__ = ["CONFIGS", "TrainConfig", "get_config"]
