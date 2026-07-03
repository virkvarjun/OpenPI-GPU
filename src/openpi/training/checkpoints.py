from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import jax.experimental.multihost_utils
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()

    # The destructive setup (wipe on overwrite, the exists guard, mkdir) must run on exactly one process. On a
    # single machine every process shares this filesystem, so if all of them raced here one would create the dir
    # and the others would spuriously raise FileExistsError (then the survivors would hang on the next
    # collective). Do it on process 0 only, then barrier so peers proceed once the dir is ready. Single-process
    # runs (process_count()==1) take the identical path as before.
    multiproc = jax.process_count() > 1
    if not multiproc or jax.process_index() == 0:
        if checkpoint_dir.exists():
            if overwrite:
                checkpoint_dir.rmtree()
                logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
            elif not resume:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                    "to indicate how to handle it."
                )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if multiproc:
        jax.experimental.multihost_utils.sync_global_devices("initialize_checkpoint_dir")

    # Decide resuming uniformly on every process from the (now-ready) directory: resume only if asked AND a real
    # checkpoint exists. This replaces the old per-branch `resuming` flag and keeps all processes in agreement.
    resuming = False

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Resume only if requested AND a real checkpoint exists. (If --resume was passed but the run never reached
    # its first checkpoint, there is nothing to restore, so start fresh instead of failing.)
    if resume and tuple(mngr.all_steps()) not in [(), (0,)]:
        resuming = True
    elif resume:
        logging.info("Checkpoint directory has no checkpoints yet; starting fresh instead of resuming.")

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    merged = _merge_params(restored["train_state"], restored["params"])

    # G4: seek the data loader to the restored step so training resumes on exactly the right examples. The shard
    # sampler order is a pure function of (seed, epoch), so the step counter alone determines (epoch, offset) —
    # no separate iterator blob is checkpointed. No-op for loaders without the sampler (e.g. RLDS -> coarse resume).
    data_loader.set_state({}, resume_step=int(merged.step))
    return merged


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
