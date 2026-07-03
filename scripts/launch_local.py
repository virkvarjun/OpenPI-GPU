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
import signal
import socket
import subprocess
import sys
import time


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nproc", type=int, required=True, help="number of local processes")
    parser.add_argument("--devices-per-proc", type=int, default=1, help="devices per process (CPU-sim count, or GPUs per process)")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                        help="cpu = simulated CPU devices (cheap ladder); gpu = real GPUs via CUDA_VISIBLE_DEVICES")
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
        if args.backend == "gpu":
            # Real multi-host: one process per GPU (or a contiguous GPU slice). Global device_count == sum.
            lo = i * args.devices_per_proc
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in range(lo, lo + args.devices_per_proc))
        else:
            # Cheap ladder: each process simulates its own CPU devices; global device_count is the sum.
            env["XLA_FLAGS"] = (
                f"{env.get('XLA_FLAGS', '')} --xla_force_host_platform_device_count={args.devices_per_proc}".strip()
            )
        procs.append(subprocess.Popen(command, env=env))

    # Tear down the whole group if any process exits non-zero. Local processes share a coordinator and run
    # collectives, so a survivor would otherwise block forever on the next barrier waiting for a dead peer.
    # A launcher must fail the group on any worker failure (like torchrun/mpirun) — this is also what lets the
    # elastic supervisor (G5) observe a clean non-zero exit and relaunch.
    while True:
        finished = [p for p in procs if p.poll() is not None]
        failed = [p for p in finished if p.returncode != 0]
        if failed:
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGKILL)
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            return failed[0].returncode or 1
        if len(finished) == len(procs):
            return 0
        time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(main())
