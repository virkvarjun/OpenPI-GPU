"""G1: multi-host cluster bring-up for the JAX training path.

`maybe_initialize()` calls `jax.distributed.initialize()` exactly once, and is a **no-op when single-process**,
so the existing single-host path is byte-for-byte unchanged (no env set ⇒ nothing happens ⇒ `process_count()==1`).

Two entry routes, both opt-in:
  - **Explicit rendezvous** (our localhost cheap-ladder launcher, and GPU/CPU clusters): all three of
    `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID` are set → we pass them through. On CPU,
    `initialize()` cannot auto-detect, so this explicit route is required there.
  - **Cluster auto-detection** (TPU pods / SLURM-GPU): opt in with `JAX_CLUSTER_AUTODETECT=1` and call
    `initialize()` with no args, letting JAX discover the topology. ⛔ Only verifiable on real hardware.

After init, `jax.device_count()` becomes the GLOBAL device count across all processes, so the existing
`make_mesh(config.fsdp_devices)` mesh spans every host with no change to `sharding.py`.
"""

from __future__ import annotations

import logging
import os

import jax

# Env var names our launcher sets (kept JAX-prefixed for familiarity; these are *our* contract, not JAX's).
COORDINATOR_ADDRESS = "JAX_COORDINATOR_ADDRESS"
NUM_PROCESSES = "JAX_NUM_PROCESSES"
PROCESS_ID = "JAX_PROCESS_ID"
AUTODETECT = "JAX_CLUSTER_AUTODETECT"


def maybe_initialize(env: dict[str, str] | None = None) -> bool:
    """Initialize the JAX distributed runtime if a multi-process launch is requested.

    Returns True if the distributed runtime is active (now or already), False for the single-process no-op.
    Safe to call unconditionally at the top of `main()`.
    """
    env = os.environ if env is None else env

    if jax.distributed.is_initialized():
        return True

    addr, nprocs, pid = env.get(COORDINATOR_ADDRESS), env.get(NUM_PROCESSES), env.get(PROCESS_ID)
    if addr and nprocs and pid:
        jax.distributed.initialize(
            coordinator_address=addr,
            num_processes=int(nprocs),
            process_id=int(pid),
        )
        logging.info(
            "[distributed] initialized via explicit rendezvous: process %s/%s at %s | "
            "global_devices=%d local_devices=%d",
            pid,
            nprocs,
            addr,
            jax.device_count(),
            jax.local_device_count(),
        )
        return True

    if env.get(AUTODETECT, "").lower() in ("1", "true", "yes"):
        jax.distributed.initialize()  # TPU/GPU cluster auto-discovery; HW-gated.
        logging.info(
            "[distributed] initialized via cluster auto-detection | global_devices=%d local_devices=%d",
            jax.device_count(),
            jax.local_device_count(),
        )
        return True

    # Single-process: no-op. process_count()==1, device_count() is the local device count, mesh is single-host.
    return False
