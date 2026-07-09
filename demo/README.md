# Sharder — cinematic terminal demo

A terminal is what distributed training actually looks like. This demo applies cinematic language — dark
canvas, monospace, glowing traces, camera moves, a HUD — to **real terminal panes streaming real logs** from a
real fault-tolerant training run. The only thing added is the connective tissue a terminal normally hides: the
collectives between processes, and the supervisor's decision to restart.

## Honesty (read this first)

- **These are PROCESSES, not nodes.** Every run is multi-process on one machine. It exercises the same
  distributed bring-up, within-batch data sharding, exact deterministic resume, and restart supervision a
  multi-node run would use. It is **not** a validated multi-node / NCCL-fabric run, and neither this README,
  the renderer, nor the video ever claims it is. The footnote saying so stays on screen for the whole video.
- **Every rendered number is produced by the real code.** Steps, losses, ms, checkpoint steps, batch indices,
  hashes, pids, exit codes — all parsed from the run's own output or computed by `openpi.training.data_sharding`
  and recorded in a JSONL event log. Pause the video on any frame; the table below says where each number
  comes from.
- **The polished video is a replay of a real run.** LIVE mode appends every event to a JSONL log; REPLAY renders
  that log with timing compression only (the on-screen chrome says so). Values are verbatim. The recorded master
  take is committed at `demo/runs/master_3xh100.jsonl` — a real run on 3× H100 (1 process per GPU,
  `--fsdp_devices 3`): scripted SIGKILL of process 1 at step 110, supervisor relaunch, all three processes
  resumed from the step-100 checkpoint 27 s later, proof identical.
- **What the H100 numbers are and aren't.** The HUD's live numbers are from the demo run itself (a deliberately
  tiny `debug`-config model — that's what makes kill/resume takes cheap and repeatable). The end card's
  performance numbers (flash attention +9.3 %, weak scaling 2.85× / 980 TFLOP/s, 32.1 % MFU) are gemma_2b
  measurements from [FINDINGS.md](../FINDINGS.md), and the card labels them "measured separately — not this run".

## Run it

```bash
# LIVE (any machine, CPU): 3 real processes, browser UI at http://localhost:7777
python scripts/demo_sharder.py --live --nproc 3 --serve 7777
# kill a process by clicking its pane, or press a digit then `k` (e.g. `1 k`)

# LIVE on GPUs (1 process per GPU) with a deterministic scripted kill — how the master take was recorded:
python scripts/demo_sharder.py --live --nproc 3 --backend gpu --fsdp-devices 3 --batch-size 6 \
    --steps 200 --save-interval 25 --script kill@step=110 --serve 7777

# REPLAY the committed master take:
python scripts/demo_sharder.py --replay demo/runs/master_3xh100.jsonl --serve 7777
# then open  http://localhost:7777/?mode=replay
#   &capture=1   -> the 20s recording cut: chrome hidden, beat-budgeted timeline
#                   (title/handshake 2.5s · steady 4s · kill 2s · supervisor 2s · resume 2s · proof ~4.5s · end card ~2s)
#   &speed=2     -> playback rate ([ and ] adjust live; space pauses)
#   &skipto=86   -> jump near a beat (86≈kill, 114≈proof) when reviewing
```

To record the video: open the replay with `&capture=1` at 1080p+ fullscreen and screen-capture the browser —
the cut runs 20 seconds, ending on the proof verdict and the end card (which holds from ~18.5s, so stop the
recording at 20s). Time is remapped onto the beat budget (values verbatim, and the wall clock always shows the
real run time); the naive-striping coda plays only in interactive replay, not in the capture cut.

## What each rendered element is fed by

| On-screen element | Produced by (repo file) |
|---|---|
| Pane step lines (`step 00110 │ loss 2.4702 │ 155 ms │ ckpt@100`) | `scripts/train.py` (`pbar.write(f"Step {step}: …")`), attributed per process by `scripts/launch_local.py --tag-output` |
| Handshake (`process k/3`, pids) | `scripts/train.py` "Running on:" line + `src/openpi/training/distributed.py` G1 init; pids from the launcher's spawn lines |
| Trace pulse types & counts (legend `all-reduce×45 …`) | The compiled train step's own HLO, counted by the `SHARDER_DEMO_HLO=1` probe in `scripts/train.py` (vocabulary from `src/openpi/training/attribution.py`). The renderer draws only the ops the executable actually contains |
| Per-pane / HUD ms | Wall time between that process's consecutive real step lines (labeled "wall"); with `--profile`, the device median from `src/openpi/training/profiling.py` (labeled "device") |
| MFU slot | Only populated if the real profiler reports one (`SHARDER_PROFILE=1` + `SHARDER_PEAK_FLOPS`); renders "—" otherwise — never invented |
| CKPT chip / `ckpt@N` | Real Orbax checkpoint directories on disk (`src/openpi/training/checkpoints.py`), watched by the orchestrator |
| Supervisor pane | Verbatim `[elastic]` lines from `scripts/elastic_launch.py` |
| Stall + group death | `scripts/launch_local.py` group-teardown (a survivor would block on the dead peer's collective; the launcher fails the group — that's what the supervisor observes) |
| Kill / exit codes | Real `SIGKILL` on the worker pid; exit codes as observed (`rc=-9` at the launcher, `rc=247` at the supervisor) |
| Resume rewind | The step counter coming back at/below the death step in the worker's own output |
| THE PROOF (indices, SHAs) | `src/openpi/training/data_sharding.py`: `resume_position`, `global_batch_indices`, `process_index_stream` — computed at the observed resume step, recorded in the log |
| Coda rows | Same module. "Naive" = dataset `k::N` stripe + per-process shuffle (the comet-style approach): batch **membership** diverges. Within-batch reconstructs the reference bit-identically |
| End-card FINDINGS block | `FINDINGS.md` (4× H100 gemma_2b measurements; labeled "measured separately — not this run") |
| Loss curve | The run's real loss values, keyed by step — a resume overlays the same x, which is why the curve continues unbroken |

## Event log (JSONL, one event per line)

`meta` (run config, git sha, honesty note) · `proc_up`/`handshake` · `line` (raw per-process output) ·
`step` (step, loss, wall_ms) · `collective_profile` (HLO op counts) · `ckpt` · `kill` · `proc_exit` ·
`supervisor` · `resume` · `proof` · `coda` · `profile` · `end`. The renderer consumes only this stream, in both
modes — one source of truth. `scripts/demo_sharder_test.py` unit-tests the parsers and the proof/coda math.
