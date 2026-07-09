"""Unit tests for the pure parts of the cinematic demo orchestrator (no processes spawned)."""

import argparse
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import demo_sharder as demo


def _args(**over):
    base = {
        "nproc": 3, "backend": "cpu", "steps": 240, "save_interval": 25, "batch_size": 6,
        "fsdp_devices": 1, "script": None, "kill_proc": None, "profile": False, "no_coda": False,
        "serve": 0, "out": None, "ckpt_dir": "/tmp/sharder_cine_test", "live": True, "replay": None,
    }
    base.update(over)
    return argparse.Namespace(**base)


def _orch(tmp_path, **over):
    log = demo.EventLog(tmp_path / "events.jsonl")
    return demo.Orchestrator(_args(**over), log, "testrun"), log


# ---- line parsing against REAL output formats (train.py / launch_local.py / elastic_launch.py) ----

def test_parses_spawn_step_and_proc_up(tmp_path):
    orch, log = _orch(tmp_path)
    orch._parse("[launcher] spawned proc 0 pid 111")
    orch._parse("[proc 0] 16:47:55.340 [I] Running on: somehost (process 0/3)  (7243:train.py:208)")
    orch._parse("[proc 0] Step 12: grad_norm=6.4051, loss=2.5568, param_norm=477.8596")
    types = [e["type"] for e in log.events]
    assert types == ["line", "proc_up", "line", "step"]
    up = next(e for e in log.events if e["type"] == "proc_up")
    assert up["proc"] == 0 and up["pid"] == 111 and up["process_count"] == 3
    step = next(e for e in log.events if e["type"] == "step")
    assert step["step"] == 12 and step["loss"] == 2.5568 and step["grad_norm"] == 6.4051


def test_handshake_fires_once_when_all_procs_up(tmp_path):
    orch, log = _orch(tmp_path)
    for i in range(3):
        orch._parse(f"[proc {i}] Running on: h (process {i}/3)")
    assert [e["type"] for e in log.events if e["type"] == "handshake"] == ["handshake"]


def test_hlo_collectives_emitted_once(tmp_path):
    orch, log = _orch(tmp_path)
    line = '[demo] hlo-collectives: {"all-reduce": 161, "all-gather": 54}'
    orch._parse(f"[proc 0] {line}")
    orch._parse(f"[proc 1] {line}")
    evs = [e for e in log.events if e["type"] == "collective_profile"]
    assert len(evs) == 1
    assert evs[0]["ops"] == {"all-reduce": 161, "all-gather": 54}


def test_supervisor_and_exit_lines(tmp_path):
    orch, log = _orch(tmp_path)
    orch._parse("[launcher] spawned proc 1 pid 222")
    orch._parse("[launcher] proc 1 pid 222 exited rc=-9")
    orch._parse("[elastic] attempt 0 died (rc=247); restarting from last checkpoint in 1s")
    orch._parse("[elastic] attempt 1/2: python ...")
    types = [e["type"] for e in log.events]
    assert types == ["proc_exit", "supervisor", "supervisor"]
    assert log.events[0]["exit_code"] == -9
    assert log.events[2]["attempt"] == 1


def test_resume_detected_on_step_rewind(tmp_path):
    orch, log = _orch(tmp_path)
    orch.st.ckpt_step = 25
    orch.st.killed_at = {"proc": 1, "step": 40, "ckpt_step": 25}
    orch._parse("[proc 0] Step 40: loss=1.5000")
    orch._parse("[proc 0] Step 26: loss=1.6000")  # counter went backwards -> a resume happened
    resumes = [e for e in log.events if e["type"] == "resume"]
    assert len(resumes) == 1 and resumes[0]["ckpt_step"] == 25
    # the proof anchors on the step the run actually resumed at (its first post-restore batch)
    proof = next(e for e in log.events if e["type"] == "proof")
    assert proof["identical"] is True and proof["step"] == 26


def test_resume_detected_when_kill_lands_on_checkpoint_step(tmp_path):
    # kill at step N with ckpt@N: the resumed counter comes back at the SAME number, so the strict
    # rewind check alone would miss it — the relaunch ([elastic] attempt > 0) arms an <= comparison.
    orch, log = _orch(tmp_path)
    orch.st.killed_at = {"proc": 0, "step": 4, "ckpt_step": 4}
    orch._parse("[proc 0] Step 4: loss=1.5000")
    orch._parse("[elastic] attempt 1/2: python ...")
    orch._parse("[proc 0] Step 4: loss=1.5000")
    resumes = [e for e in log.events if e["type"] == "resume"]
    assert len(resumes) == 1 and resumes[0]["from_step"] == 4
    proof = next(e for e in log.events if e["type"] == "proof")
    assert proof["step"] == 4 and proof["identical"] is True
    # a later ordinary step must NOT re-trigger
    orch._parse("[proc 0] Step 5: loss=1.4000")
    assert len([e for e in log.events if e["type"] == "resume"]) == 1


# ---- the PROOF and the CODA (real openpi.training.data_sharding) ----

@pytest.mark.parametrize("step", [0, 25, 170, 200])
@pytest.mark.parametrize("nproc", [2, 3])
def test_resume_proof_exact(step, nproc):
    p = demo.resume_proof(step, batch=2 * nproc, nproc=nproc)
    assert p["identical"] is True
    assert p["sha_ref"] == p["sha_res"]
    assert len(p["reference"]) == 2 * nproc


def test_coda_naive_striping_differs_within_batch_matches():
    c = demo.striping_coda(25, batch=6, nproc=3)
    assert c["within_batch_identical"] is True
    assert c["naive_identical"] is False  # the whole point of within-batch sharding
    # the naive model (dataset k::N + per-process shuffle) must differ in MEMBERSHIP, not just order —
    # order-only difference would be gradient-equivalent (batch mean is permutation-invariant)
    assert c["naive_same_examples"] is False
    assert sorted(c["reference"]) != sorted(c["naive_stripe"])


def test_event_log_is_valid_jsonl(tmp_path):
    log = demo.EventLog(tmp_path / "e.jsonl")
    log.emit("meta", run_id="x")
    log.emit("step", proc=0, step=1, loss=2.0, wall_ms=None)
    lines = (tmp_path / "e.jsonl").read_text().strip().splitlines()
    assert [json.loads(ln)["type"] for ln in lines] == ["meta", "step"]
