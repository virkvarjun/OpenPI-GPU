"""Launch N local processes that form a `jax.distributed` cluster on localhost (cheap-ladder multi-host).

This is the rung-2 of the validation ladder from PLAN.md: real multi-process JAX without a real cluster. It sets
the rendezvous env vars (`JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`, `JAX_PROCESS_ID`) that
`openpi.training.distributed.maybe_initialize()` reads, plus `--xla_force_host_platform_device_count` so each
process owns some CPU devices, then runs the given command in each and waits.

Examples:
  # 4 processes, 1 CPU device each (global device_count == 4):
  python scripts/launch_local.py --nproc 4 -- python -c "import jax; print(jax.process_index(), jax.device_count())"

  # Run training across 2 local processes:
  python scripts/launch_local.py --nproc 2 -- python scripts/train.py <config> --exp_name=x

Everything after `--` is the per-process command, run verbatim in each process.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nproc", type=int, required=True, help="number of local processes")
    parser.add_argument("--devices-per-proc", type=int, default=1, help="CPU devices simulated per process")
    parser.add_argument("--coordinator-port", type=int, default=None, help="defaults to a free port")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <command> run in each process")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given; pass it after `--`")

    port = args.coordinator_port or _free_port()
    procs: list[subprocess.Popen] = []
    for i in range(args.nproc):
        env = os.environ.copy()
        env["JAX_COORDINATOR_ADDRESS"] = f"localhost:{port}"
        env["JAX_NUM_PROCESSES"] = str(args.nproc)
        env["JAX_PROCESS_ID"] = str(i)
        # Each process simulates its own CPU devices; the cluster's global device_count is the sum.
        env["XLA_FLAGS"] = (
            f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={args.devices_per_proc}".strip()
        )
        procs.append(subprocess.Popen(command, env=env))

    exit_code = 0
    for p in procs:
        exit_code |= p.wait()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
