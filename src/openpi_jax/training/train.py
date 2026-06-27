"""Training entrypoint for π0.

Run via the console script: ``openpi-train --config pi0_aloha_sim`` (see pyproject ``[project.scripts]``), or
``python -m openpi_jax.training.train --config debug``.

The loop is deliberately small and readable: build model+optimizer, jit a single train step, iterate the data
loader. Heavy pieces (real data, the model trunk, sharding across devices) are stubbed and flagged with TODOs.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import optax

from openpi_jax.data.dataset import make_data_loader
from openpi_jax.models.observation import Observation
from openpi_jax.models.pi0 import Pi0
from openpi_jax.training import checkpoints
from openpi_jax.training.config import TrainConfig, get_config
from openpi_jax.training.optimizer import make_optimizer


@dataclasses.dataclass
class TrainState:
    params: optax.Params
    opt_state: optax.OptState
    step: int


def init_train_state(
    cfg: TrainConfig, rng: jax.Array, sample_batch: dict
) -> tuple[Pi0, optax.GradientTransformation, TrainState]:
    model = Pi0(cfg.model)
    obs = Observation.from_dict(sample_batch)
    actions = sample_batch["actions"]

    init_rng, loss_rng = jax.random.split(rng)
    variables = model.init(init_rng, loss_rng, obs, actions)
    params = variables["params"]

    tx = make_optimizer(cfg.optimizer)
    opt_state = tx.init(params)
    return model, tx, TrainState(params=params, opt_state=opt_state, step=0)


@functools.partial(jax.jit, static_argnums=(0, 1))
def train_step(model: Pi0, tx: optax.GradientTransformation, state: TrainState, rng: jax.Array, batch: dict):
    obs = Observation.from_dict(batch)
    actions = batch["actions"]

    def loss_fn(params):
        return model.apply({"params": params}, rng, obs, actions)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, opt_state = tx.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    new_state = TrainState(params=params, opt_state=opt_state, step=state.step + 1)
    return new_state, {"loss": loss}


def train(cfg: TrainConfig) -> None:
    print(f"[openpi-jax] training '{cfg.name}' on {jax.devices()}")
    rng = jax.random.key(cfg.seed)

    loader = make_data_loader(cfg.data, action_dim=cfg.model.action_dim, horizon=cfg.model.action_horizon)
    sample_batch = next(iter(loader))

    init_rng, rng = jax.random.split(rng)
    model, tx, state = init_train_state(cfg, init_rng, sample_batch)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[openpi-jax] initialized {n_params/1e6:.1f}M params")

    for step, batch in zip(range(cfg.num_train_steps), loader, strict=False):
        step_rng, rng = jax.random.split(rng)
        state, metrics = train_step(model, tx, state, step_rng, batch)

        if step % cfg.log_every == 0:
            loss = float(jnp.asarray(metrics["loss"]))
            print(f"[openpi-jax] step {step:>6} | loss {loss:.4f}")
        if step > 0 and step % cfg.save_every == 0:
            checkpoints.save(f"{cfg.checkpoint_dir}/{cfg.name}", step, dataclasses.asdict(state))

    print("[openpi-jax] done.")


def main() -> None:
    import tyro

    @dataclasses.dataclass
    class Args:
        config: str = "debug"

    args = tyro.cli(Args)
    train(get_config(args.config))


if __name__ == "__main__":
    main()
