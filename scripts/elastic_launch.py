"""G5: elastic-restart supervisor for fault-tolerant multi-host training.

Runs a training command and, if it exits non-zero (a process died / node preempted), RE-LAUNCHES it with the
resume flag so it continues from the last checkpoint — up to ``--max-retries``, with backoff. This composes the
pieces already built: G1 (`jax.distributed` bring-up), Orbax checkpointing, and G4 (exact deterministic resume —
the restarted run continues on precisely the right examples via the step counter).

Usage:
  python scripts/elastic_launch.py --max-retries 5 -- \
      python scripts/launch_local.py --nproc 4 --backend gpu -- python scripts/train.py <config> --resume

The wrapped command should be idempotent under restart: checkpoint on an interval and support the resume flag
(train.py does both). The first attempt runs as-is; every retry appends ``--resume-flag`` (default ``--resume``)
if it is not already present, so a fresh run starts clean and restarts resume.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-retries", type=int, default=5, help="restarts after the initial attempt")
    parser.add_argument("--backoff-seconds", type=float, default=5.0, help="base backoff between restarts")
    parser.add_argument("--resume-flag", default="--resume", help="flag appended on restart to resume")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- <training command>")
    args = parser.parse_args()

    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("no command given; pass it after `--`")

    for attempt in range(args.max_retries + 1):
        run_cmd = list(command)
        if attempt > 0 and args.resume_flag not in run_cmd:
            run_cmd.append(args.resume_flag)  # restarts resume from the last checkpoint
        print(f"[elastic] attempt {attempt}/{args.max_retries}: {' '.join(run_cmd)}", flush=True)

        rc = subprocess.run(run_cmd).returncode
        if rc == 0:
            print(f"[elastic] succeeded on attempt {attempt}", flush=True)
            return 0

        if attempt == args.max_retries:
            print(f"[elastic] exhausted {args.max_retries} retries; last rc={rc}", flush=True)
            return rc

        delay = args.backoff_seconds * (2**attempt)
        print(f"[elastic] attempt {attempt} died (rc={rc}); restarting from last checkpoint in {delay:.0f}s", flush=True)
        time.sleep(delay)

    return 1


if __name__ == "__main__":
    sys.exit(main())
