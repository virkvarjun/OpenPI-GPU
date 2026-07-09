"""Sharder cinematic demo — orchestrates a REAL fault-tolerant training run and emits a JSONL event stream.

This drives the exact chain a production run uses — nothing is reimplemented and no number is invented:

    elastic_launch.py (G5 supervisor)
      └─ launch_local.py --tag-output (G1 rendezvous, per-process line attribution)
           └─ N × train.py debug (real train step, FakeData, Orbax checkpoints, G4 exact resume)

Every rendered element in demo/index.html comes from an event in the JSONL log this script appends to:
real per-process log lines, real step/loss values, the collective ops present in the compiled step's HLO
(SHARDER_DEMO_HLO=1), real checkpoint directories on disk, the supervisor's own [elastic] lines, and the
deterministic-resume proof computed by openpi.training.data_sharding.

HONESTY: this is MULTI-PROCESS ON ONE MACHINE. It exercises the identical G1/G3/G4/G5 code path a multi-node
run would use. It is NOT a validated multi-node / NCCL-fabric run and never claims to be.

Usage:
  # live run, browser UI at http://localhost:7777 (kill a pane by clicking it, or press k / 0-9 then k):
  python scripts/demo_sharder.py --live --nproc 3 --serve 7777

  # deterministic take for recording (kills proc 1 when step 40 is reached):
  python scripts/demo_sharder.py --live --nproc 3 --script kill@step=40 --serve 7777

  # replay a recorded run with full cinematic timing control:
  python scripts/demo_sharder.py --replay demo/runs/<run>.jsonl --serve 7777
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import http.server
import json
import os
import pathlib
import re
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# From the real code path this demo drives (same constants as demo_fault_tolerance.py):
#   config "debug": FakeData, dummy pi0, seed 42, shuffle True     (openpi.training.config)
#   FakeDataset(num_samples=1024)                                  (openpi.training.data_loader:145)
DATASET_LEN = 1024
CONFIG_NAME = "debug"
SEED = 42  # config "debug" seed; asserted against the real config at startup when importable.

# ---- real-output line patterns (see scripts/train.py, launch_local.py, elastic_launch.py) ----
TAG_RE = re.compile(r"^\[proc (\d+)\] (.*)$")
SPAWN_RE = re.compile(r"^\[launcher\] spawned proc (\d+) pid (\d+)")
EXIT_RE = re.compile(r"^\[launcher\] proc (\d+) pid (\d+) exited rc=(-?\d+)")
STEP_RE = re.compile(r"Step (\d+): (.*)")
KV_RE = re.compile(r"(\w+)=([0-9.eE+-]+)")
PROC_UP_RE = re.compile(r"Running on: (\S+) \(process (\d+)/(\d+)\)")
HLO_RE = re.compile(r"\[demo\] hlo-collectives: (\{.*\})")
PROFILE_RE = re.compile(r"\[profile\] (.*)")
ELASTIC_RE = re.compile(r"\[elastic\] (.*)")
ATTEMPT_RE = re.compile(r"\[elastic\] attempt (\d+)/(\d+)")


# ---------------------------------------------------------------------------
# event log: single source of truth for LIVE rendering and REPLAY
# ---------------------------------------------------------------------------
class EventLog:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.events: list[dict] = []
        self.cond = threading.Condition()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)
        self._t0 = time.time()

    def emit(self, type_: str, **fields):
        ev = {"t": round(time.time() - self._t0, 3), "type": type_, **fields}
        with self.cond:
            self.events.append(ev)
            self._fh.write(json.dumps(ev) + "\n")
            self.cond.notify_all()
        return ev


# ---------------------------------------------------------------------------
# the deterministic-resume PROOF and the naive-striping CODA (real data_sharding)
# ---------------------------------------------------------------------------
def resume_proof(step: int, batch: int, nproc: int) -> dict:
    """The global batch a resumed run reads at `step` vs the one a never-crashed run reads. Real code:
    openpi.training.data_sharding — order is a pure function of (seed, epoch); the step counter fixes position."""
    import numpy as np

    from openpi.training import data_sharding as ds

    epoch, offset = ds.resume_position(step, DATASET_LEN, batch)
    reference = ds.global_batch_indices(DATASET_LEN, batch, offset, shuffle=True, seed=SEED, epoch=epoch)
    local = batch // nproc
    resumed = np.concatenate([
        ds.process_index_stream(
            DATASET_LEN, batch, k, nproc, shuffle=True, seed=SEED, epoch=epoch, start_batch=offset
        )[:local]
        for k in range(nproc)
    ])
    sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]  # noqa: E731
    return {
        "step": step,
        "epoch": epoch,
        "offset": offset,
        "reference": reference.tolist(),
        "resumed": resumed.tolist(),
        "sha_ref": sha(reference),
        "sha_res": sha(resumed.astype(reference.dtype)),
        "identical": bool(np.array_equal(reference, resumed.astype(reference.dtype))),
    }


def striping_coda(step: int, batch: int, nproc: int) -> dict:
    """Naive order[k::N] striping (how openpi-comet shards) vs our within-batch shard, both against the
    single-process reference batch at `step`. Same real global_order for all three."""
    import numpy as np

    from openpi.training import data_sharding as ds

    epoch, offset = ds.resume_position(step, DATASET_LEN, batch)
    order = ds.global_order(DATASET_LEN, shuffle=True, seed=SEED, epoch=epoch)
    reference = ds.global_batch_indices(DATASET_LEN, batch, offset, shuffle=True, seed=SEED, epoch=epoch)
    local = batch // nproc
    naive = np.concatenate([order[k::nproc][offset * local : (offset + 1) * local] for k in range(nproc)])
    within = np.concatenate([
        ds.process_index_stream(DATASET_LEN, batch, k, nproc, shuffle=True, seed=SEED, epoch=epoch,
                                start_batch=offset)[:local]
        for k in range(nproc)
    ])
    return {
        "step": step,
        "reference": reference.tolist(),
        "naive_stripe": naive.tolist(),
        "within_batch": within.tolist(),
        "naive_identical": bool(np.array_equal(reference, naive)),
        "within_batch_identical": bool(np.array_equal(reference, within.astype(reference.dtype))),
    }


# ---------------------------------------------------------------------------
# orchestrator: run the real chain, parse tagged output into events
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RunState:
    nproc: int
    pids: dict[int, int] = dataclasses.field(default_factory=dict)  # proc -> live train.py pid
    last_step: dict[int, int] = dataclasses.field(default_factory=dict)  # proc -> last step seen
    last_step_t: dict[int, float] = dataclasses.field(default_factory=dict)  # proc -> arrival time
    up: set = dataclasses.field(default_factory=set)
    attempt: int = 0
    ckpt_step: int | None = None
    kill_pending: str | None = None  # "step=N" trigger
    killed_at: dict | None = None  # {"step":, "ckpt_step":, "proc":}
    resumes: int = 0
    hlo_emitted: bool = False
    handshake_done: bool = False
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)


class Orchestrator:
    def __init__(self, args, log: EventLog, run_id: str):
        self.args = args
        self.log = log
        self.st = RunState(nproc=args.nproc)
        self.proc: subprocess.Popen | None = None
        self.exp_name = run_id
        self.ckpt_dir = pathlib.Path(args.ckpt_dir)
        if args.script and args.script.startswith("kill@step="):
            self.st.kill_pending = args.script.split("kill@step=")[1]

    # ---- real command chain ----
    def train_cmd(self) -> list[str]:
        return [
            sys.executable, "scripts/train.py", CONFIG_NAME,
            "--exp_name", self.exp_name, "--no-overwrite",
            "--num_train_steps", str(self.args.steps),
            "--save_interval", str(self.args.save_interval),
            "--log_interval", "1",
            "--batch_size", str(self.args.batch_size),
            "--fsdp_devices", str(self.args.fsdp_devices),
            "--checkpoint_base_dir", str(self.ckpt_dir),
        ]

    def full_cmd(self) -> list[str]:
        launch = [
            sys.executable, "scripts/launch_local.py",
            "--nproc", str(self.args.nproc), "--backend", self.args.backend, "--tag-output", "--",
        ] + self.train_cmd()
        return [
            sys.executable, "scripts/elastic_launch.py",
            "--max-retries", "2", "--backoff-seconds", "1", "--",
        ] + launch

    def child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        pp = f"{REPO}/src:{REPO}/packages/openpi-client/src"
        env["PYTHONPATH"] = pp + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["WANDB_MODE"] = "disabled"
        env["PYTHONUNBUFFERED"] = "1"
        env["SHARDER_DEMO_HLO"] = "1"
        if self.args.profile:
            env["SHARDER_PROFILE"] = "1"
        if self.args.backend == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
            # Multi-process on CPU: stop each process from grabbing every core (thrashes the collectives).
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[k] = "1"
        return env

    # ---- lifecycle ----
    def start(self):
        cmd = self.full_cmd()
        self.log.emit(
            "meta",
            run_id=self.exp_name,
            mode="live",
            nproc=self.args.nproc,
            backend=self.args.backend,
            config=CONFIG_NAME,
            batch_size=self.args.batch_size,
            fsdp_devices=self.args.fsdp_devices,
            seed=SEED,
            dataset_len=DATASET_LEN,
            steps=self.args.steps,
            save_interval=self.args.save_interval,
            kill_script=self.args.script,
            git_sha=_git_sha(),
            cmd=" ".join(cmd),
            honesty="multi-process on one machine; identical G1/G3/G4/G5 code path a multi-node run uses; "
                    "NOT a validated multi-node/NCCL run",
        )
        self.proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=self.child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,  # own process group so stop() can tear down the whole real chain
        )
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._watch_ckpts, daemon=True).start()

    def _read(self):
        st = self.st
        for raw in self.proc.stdout:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            self._parse(line)
        rc = self.proc.wait()
        ok = rc == 0
        with st.lock:
            final = max(st.last_step.values(), default=-1)
        self.log.emit("end", ok=ok, rc=rc, final_step=final, resumes=st.resumes)

    def _parse(self, line: str):  # noqa: PLR0912
        st = self.st
        m = SPAWN_RE.match(line)
        if m:
            with st.lock:
                st.pids[int(m.group(1))] = int(m.group(2))
            return
        m = EXIT_RE.match(line)
        if m:
            proc, pid, rc = int(m.group(1)), int(m.group(2)), int(m.group(3))
            with st.lock:
                if st.pids.get(proc) == pid:
                    del st.pids[proc]
                st.up.discard(proc)
            self.log.emit("proc_exit", proc=proc, pid=pid, exit_code=rc)
            return
        m = TAG_RE.match(line)
        if m:
            proc, text = int(m.group(1)), m.group(2)
            self._parse_proc_line(proc, text)
            return
        m = ATTEMPT_RE.match(line)
        if m:
            attempt = int(m.group(1))
            with st.lock:
                st.attempt = attempt
                if attempt > 0:  # relaunch: pane pids/handshake reset, panes will come back up
                    st.up.clear()
                    st.handshake_done = False
            self.log.emit("supervisor", attempt=attempt, max_retries=int(m.group(2)), text=line)
            return
        if ELASTIC_RE.match(line):
            self.log.emit("supervisor", attempt=st.attempt, text=line)
            return
        self.log.emit("line", proc=None, text=line)

    def _parse_proc_line(self, proc: int, text: str):
        st = self.st
        self.log.emit("line", proc=proc, text=text)

        m = PROC_UP_RE.search(text)
        if m:
            with st.lock:
                st.up.add(proc)
                pid = st.pids.get(proc)
                all_up = len(st.up) == st.nproc and not st.handshake_done
                if all_up:
                    st.handshake_done = True
            self.log.emit("proc_up", proc=proc, pid=pid, host=m.group(1),
                          process_index=int(m.group(2)), process_count=int(m.group(3)))
            if all_up:
                self.log.emit("handshake", nproc=st.nproc, attempt=st.attempt)
            return

        m = HLO_RE.search(text)
        if m and not st.hlo_emitted:
            st.hlo_emitted = True
            try:
                ops = json.loads(m.group(1))
            except json.JSONDecodeError:
                ops = {}
            self.log.emit("collective_profile", source="hlo", ops=ops)
            return

        m = PROFILE_RE.search(text)
        if m:
            fields = {k: float(v) for k, v in KV_RE.findall(m.group(1))}
            self.log.emit("profile", proc=proc, text=m.group(1), **{k: v for k, v in fields.items() if k != "proc"})
            return

        m = STEP_RE.search(text)
        if m:
            step = int(m.group(1))
            vals = {k: float(v) for k, v in KV_RE.findall(m.group(2))}
            now = time.time()
            with st.lock:
                prev_step, prev_t = st.last_step.get(proc), st.last_step_t.get(proc)
                wall_ms = round((now - prev_t) * 1000, 1) if prev_t is not None else None
                rewound = prev_step is not None and step < prev_step - 1
                st.last_step[proc] = step
                st.last_step_t[proc] = now
                ck = st.ckpt_step
            if rewound:
                with st.lock:
                    st.resumes += 1
                    resumes = st.resumes
                self.log.emit("resume", proc=proc, from_step=step, prev_step=prev_step,
                              ckpt_step=ck, resumes=resumes, detected_by="step_counter_rewind")
                if self.st.killed_at is not None and resumes == 1:
                    self._emit_proofs()
            self.log.emit("step", proc=proc, step=step, wall_ms=wall_ms,
                          loss=vals.get("loss"), **{k: v for k, v in vals.items() if k != "loss"})
            self._maybe_scripted_kill(step)

    # ---- checkpoints: real Orbax directories on disk ----
    def _ckpt_path(self) -> pathlib.Path:
        return self.ckpt_dir / CONFIG_NAME / self.exp_name

    def _watch_ckpts(self):
        seen: set[int] = set()
        while self.proc is None or self.proc.poll() is None:
            d = self._ckpt_path()
            if d.exists():
                for p in sorted(d.iterdir()):
                    if p.name.isdigit() and int(p.name) not in seen:
                        step = int(p.name)
                        seen.add(step)
                        with self.st.lock:
                            self.st.ckpt_step = max(step, self.st.ckpt_step or 0)
                        self.log.emit("ckpt", step=step, dir=str(p))
            time.sleep(0.3)

    # ---- the kill ----
    def _maybe_scripted_kill(self, step: int):
        st = self.st
        with st.lock:
            trigger = st.kill_pending
        if trigger is None or st.killed_at is not None:
            return
        if step >= int(trigger):
            self.kill(self.args.kill_proc)

    def kill(self, proc_idx: int | None) -> bool:
        st = self.st
        with st.lock:
            if not st.pids:
                return False
            if proc_idx is None or proc_idx not in st.pids:
                proc_idx = sorted(st.pids)[len(st.pids) // 2]  # default: the middle pane
            pid = st.pids[proc_idx]
            step = st.last_step.get(proc_idx, -1)
            ck = st.ckpt_step
            st.kill_pending = None
            st.killed_at = {"proc": proc_idx, "step": step, "ckpt_step": ck}
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return False
        self.log.emit("kill", proc=proc_idx, pid=pid, step_at_kill=step, ckpt_step=ck, signal="SIGKILL")
        return True

    # ---- proof + coda, computed from the real sharding math at the resume point ----
    def _emit_proofs(self):
        ck = self.st.killed_at.get("ckpt_step") or 0
        try:
            self.log.emit("proof", **resume_proof(ck, self.args.batch_size, self.args.nproc))
            if not self.args.no_coda:
                self.log.emit("coda", **striping_coda(ck, self.args.batch_size, self.args.nproc))
        except Exception as e:  # noqa: BLE001 — a proof failure must be visible, never silent
            self.log.emit("line", proc=None, text=f"[demo] PROOF FAILED: {e!r}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# HTTP: serves demo/index.html, streams events (SSE), accepts kill clicks
# ---------------------------------------------------------------------------
def make_handler(log: EventLog | None, orch: Orchestrator | None, replay_path: pathlib.Path | None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                page = (REPO / "demo" / "index.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
            elif path == "/log":
                src = replay_path if replay_path else (log.path if log else None)
                body = src.read_bytes() if src and src.exists() else b""
                self._send(200, body, "application/x-ndjson")
            elif path == "/events" and log is not None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                i = 0
                try:
                    while True:
                        with log.cond:
                            log.cond.wait(timeout=10.0)
                            chunk = log.events[i:]
                            i = len(log.events)
                        for ev in chunk:
                            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/kill" and orch is not None:
                q = urllib.parse.parse_qs(parsed.query)
                proc = int(q["proc"][0]) if q.get("proc") else None
                ok = orch.kill(proc)
                self._send(200, json.dumps({"ok": ok}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="spin up the real training chain")
    mode.add_argument("--replay", default=None, metavar="LOG.jsonl", help="serve a recorded run for replay")
    ap.add_argument("--nproc", type=int, default=3)
    ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--save-interval", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=None, help="global batch (default 2*nproc)")
    ap.add_argument("--fsdp-devices", type=int, default=1)
    ap.add_argument("--script", default=None, help='deterministic trigger, e.g. "kill@step=40"')
    ap.add_argument("--kill-proc", type=int, default=None, help="which process the kill targets (default: middle)")
    ap.add_argument("--profile", action="store_true", help="also enable SHARDER_PROFILE in the workers")
    ap.add_argument("--no-coda", action="store_true", help="skip the naive-striping coda event")
    ap.add_argument("--serve", type=int, default=7777, help="HTTP port for the renderer (0 = no server)")
    ap.add_argument("--out", default=None, help="JSONL path (default demo/runs/<run_id>.jsonl)")
    ap.add_argument("--ckpt-dir", default="/tmp/sharder_cine")
    args = ap.parse_args()

    if args.batch_size is None:
        args.batch_size = 2 * args.nproc
    if args.batch_size % args.nproc != 0:
        ap.error(f"--batch-size {args.batch_size} must be divisible by --nproc {args.nproc}")

    # Assert our constants against the real config when the deps are importable (keeps the proof honest).
    try:
        from openpi.training import config as _config

        cfg = _config.get_config(CONFIG_NAME)
        assert cfg.seed == SEED, f"config seed {cfg.seed} != {SEED}"
    except ImportError:
        pass

    if args.replay:
        replay_path = pathlib.Path(args.replay)
        if not replay_path.exists():
            ap.error(f"no such log: {replay_path}")
        server = ThreadingHTTPServer(("", args.serve), make_handler(None, None, replay_path))
        print(f"[demo] replaying {replay_path}  →  http://localhost:{args.serve}/?mode=replay")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    run_id = f"cine_{int(time.time())}"
    out = pathlib.Path(args.out) if args.out else REPO / "demo" / "runs" / f"{run_id}.jsonl"
    log = EventLog(out)
    orch = Orchestrator(args, log, run_id)

    server = None
    if args.serve:
        server = ThreadingHTTPServer(("", args.serve), make_handler(log, orch, None))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"[demo] live  →  http://localhost:{args.serve}/   (event log: {out})")

    orch.start()
    try:
        while orch.proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        orch.stop()
    finally:
        if server:
            server.shutdown()
    print(f"[demo] run finished. event log: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
