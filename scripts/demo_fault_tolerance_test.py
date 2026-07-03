"""Unit tests for the fault-tolerance demo's pure logic.

These exercise the parts that must be correct regardless of the (flaky, environment-dependent) subprocess
training: the resume-exactness proof, the command the demo builds per mode, worker-PID parsing, and checkpoint
discovery. They need no GPU, no training, and no multi-process, so they run in CI.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import demo_fault_tolerance as d  # noqa: E402


def _state(mode="sharder", nproc=1, exp="demoX", ckpt="/tmp/x"):
    return d.State(mode=mode, nproc=nproc, exp_name=exp, ckpt_dir=pathlib.Path(ckpt))


def test_resume_proof_is_exact_across_steps():
    # The whole point: the resumed run's next global batch equals the never-crashed run's, byte-for-byte.
    for step in (0, 1, 24, 120, 350):
        epoch, off, ref, res, identical, hr, hs = d.resume_proof(step, seed=42, batch=2, nproc=1)
        assert identical, f"step {step}: {ref} != {res}"
        assert hr == hs
        assert len(ref) == 2


def test_resume_proof_matches_for_multiprocess_reassembly():
    # Reassembling the per-process slices (nproc=2) must reconstruct the same global batch.
    _, _, ref, res, identical, _, _ = d.resume_proof(50, seed=42, batch=2, nproc=2)
    assert identical and ref == res


def test_sharder_command_wraps_with_elastic_supervisor():
    cmd = d.full_command(_state(mode="sharder", nproc=1), steps=100, save_interval=10)
    assert "scripts/elastic_launch.py" in cmd
    assert "scripts/train.py" in cmd
    assert "launch_local.py" not in " ".join(cmd)  # nproc=1 runs train directly


def test_naive_command_has_no_supervisor():
    cmd = d.full_command(_state(mode="naive", nproc=1), steps=100, save_interval=10)
    assert "scripts/elastic_launch.py" not in " ".join(cmd)
    assert "scripts/train.py" in cmd


def test_multiprocess_inserts_launch_local():
    cmd = d.full_command(_state(mode="sharder", nproc=2), steps=100, save_interval=10)
    joined = " ".join(cmd)
    assert "scripts/launch_local.py" in joined and "--nproc 2" in joined


def test_parse_workers_isolates_real_train_processes():
    ps = "\n".join(
        [
            "  PID COMMAND",
            "40412 python scripts/elastic_launch.py --max-retries 5 -- python scripts/train.py debug --exp_name demoX",
            "40415 python scripts/launch_local.py --nproc 2 -- python scripts/train.py debug --exp_name demoX",
            "40418 python scripts/train.py debug --exp_name demoX --no-overwrite",
            "40419 python scripts/train.py debug --exp_name demoX --no-overwrite",
            "40500 python scripts/train.py debug --exp_name OTHER",
            "40600 grep scripts/train.py",
        ]
    )
    assert d._parse_workers(ps, "demoX") == [40418, 40419]  # only the two real workers for this exp


def test_checkpoint_step_reads_latest(tmp_path):
    base = tmp_path
    (base / d.CONFIG_NAME / "demoX" / "2").mkdir(parents=True)
    (base / d.CONFIG_NAME / "demoX" / "4").mkdir(parents=True)
    (base / d.CONFIG_NAME / "demoX" / "4.orbax-checkpoint-tmp").mkdir(parents=True)  # partial, ignored
    assert d.checkpoint_step(base, "demoX") == 4
    assert d.checkpoint_step(base, "missing") is None


def test_sparkline_renders_within_bounds():
    s = d.sparkline([3.0, 2.0, 1.0, 2.5, 0.5])
    assert s and all(ch in d.SPARK for ch in s)
