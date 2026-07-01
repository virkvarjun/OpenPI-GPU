"""G5 fault-injection test: a crashed run is restarted by the elastic supervisor and resumes exactly.

Uses a stand-in "trainer" (no model/GPU) that mimics real training: process a step, checkpoint the step counter,
repeat; the FIRST run crashes mid-way. `elastic_launch.py` must detect the non-zero exit, restart with
`--resume`, and the restarted run continues from the checkpoint — with every step processed exactly once (no lost
or duplicated work). This validates the elastic-restart + exact-resume mechanism on the cheap ladder.
"""

import pathlib
import subprocess
import sys
import textwrap

_REPO = pathlib.Path(__file__).resolve().parents[1]

_TRAINER = textwrap.dedent(
    """
    import os, sys, pathlib
    ckpt = pathlib.Path(os.environ["CKPT"]); log = pathlib.Path(os.environ["LOG"])
    N, KILL_AT = 6, 3
    resume = "--resume" in sys.argv
    start = int(ckpt.read_text()) if (resume and ckpt.exists()) else 0
    for step in range(start, N):
        with log.open("a") as f:           # "process" the step (record it)
            f.write(f"{step}\\n")
        ckpt.write_text(str(step + 1))     # checkpoint AFTER the step (save_interval=1 -> exact resume)
        if not resume and step == KILL_AT: # inject a crash on the first run only
            os._exit(1)
    """
)


def test_elastic_restart_resumes_exactly(tmp_path):
    trainer = tmp_path / "trainer.py"
    trainer.write_text(_TRAINER)
    ckpt = tmp_path / "ckpt.txt"
    log = tmp_path / "processed.log"

    import os

    env = {**os.environ, "CKPT": str(ckpt), "LOG": str(log)}
    res = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "elastic_launch.py"),
            "--max-retries", "3",
            "--backoff-seconds", "0",
            "--",
            sys.executable, str(trainer),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, f"elastic did not recover:\n{res.stdout}\n{res.stderr}"

    processed = [int(x) for x in log.read_text().split()]
    # Exact resume: every step 0..5 processed exactly once (crash at 3 -> restart -> continue 4,5).
    assert processed == list(range(6)), f"lost/duplicated work: {processed}"
    assert len(processed) == len(set(processed))  # no duplicates
    assert "restarting from last checkpoint" in res.stdout  # the supervisor actually restarted
