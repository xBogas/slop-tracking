# Acoustic tracking of a motor boat (post-processed)

Estimate the position and velocity over time of a moving motor boat from
recordings on three acoustic receivers: first with a towed pinger
(phase 1), then from the boat's own radiated noise (phase 2). All
processing is offline.

## Setup

Receivers:

- **Two buoy hydrophones**, single channel, at ~10 m depth (correction: 10 m above the seabed, ~3 m depth), on a line
  parallel to the coast. Positions are known in a predefined coordinate
  system, but the buoys may drift and real-time GPS on them is not
  guaranteed. The two buoys are synchronized with each other.
- **One 4-channel shore receiver**, on the coastline roughly midway
  between the buoys, at a known exact GPS position.
- **Sync signal**: a dedicated signal is emitted at the start so the time
  offsets between all receivers can be measured.

Target:

- One motor boat. It starts with a straight-line run, then moves freely.
- No other boats are expected in the area.
- No GPS ground truth for the boat will be provided.

### Phase 1: with pinger

- The pinger hangs from a buoy towed ~5 m behind the boat, at ~10 m depth.
- Band: ~30–40 kHz.
- We choose the waveform and the ping interval. Absolute emission times
  are unknown (the start time is not controlled).
- Run plan: a straight-line run first, then a zig-zag path. We do not
  control the path, and no onboard data (GPS, IMU, steering, throttle)
  will be available: the pinger recordings are the only data.

### Phase 2: without pinger

- Same boat and receivers, no pinger. Only the boat's radiated noise is
  available.

### What the data turned out to be (group 4, "ForFun")

- `data/Rx902_Group4.wav`, `data/Rx01_Group4.wav`: buoy hydrophones,
  mono int16, 96 kHz, 1020 s from 15:59:00 local (the first ForFun run).
- `data/grp4_15-56-00wav`: the "shore receiver" is a **USBL**, 4 channels,
  float32, 96 kHz, 1341 s. Its clock is offset by 102.82 s from the buoys,
  not the 180 s its file name implies.
- `data/LocReceivers.xlsx`: receiver lat/lon and run times (ForFun: 15:59,
  16:06, 16:14, minute resolution). Runs 1–2 have the pinger; run 3 is radiated noise only.
  The config uses run times from the data instead (the boat passes each start fix 85–100 s
  after the logged time; `processing/notes/J_run_times.md`). Date: 2026-10-20 (inferred from
  the sheet's creation time).
- Pinger used: LFM sweep 30→40 kHz, 100 ms, 1 s interval.
- USBL array: a 50.5 cm near-straight line (gaps 19.0 / 12.5 / 19.1 cm),
  with a wall 1.0–1.25 m behind it.

Still unknown: buoy mooring type, true buoy positions during the runs,
sound speed, and the scoring metric and output format.

## Goals

1. Design the pinger waveform (within 30–40 kHz) and the ping interval.
   **Done:** LFM 30→40 kHz, 100 ms, 1 s.
2. Estimate the boat's position and velocity over time from the pinger
   recordings. **Done for runs 1–2:** see `processing/README.md`.
3. Estimate the boat's position and velocity over time without the
   pinger, from its radiated noise. **Done for 796–1020 s** (the boat is
   audible only from ~805 s; the buoy files end at 1020 s), with a 58 s gap
   without measurements: see `processing/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `motion_model/` | Boat motion models: a kinematic synthetic truth track, the towed-pinger model and its inversion (pinger → boat), and the coordinated-turn (CT) model used by the tracker. See `motion_model/README.md`. |
| `estimation/` | Batch factor-graph smoother: TOA with unknown emission times (phase 1), TDOA (phase 2), USBL bearings, buoy drift, clock offset and drift, sound speed, water current; `demo.py` is the measurement-level simulator. See `estimation/README.md`. |
| `processing/` | Real-data pipeline for phase 1: detection, association, USBL calibration/bearings, solve, truth-free validation. See `processing/README.md`. |
| `multipath/` | Multipath-ranging study (simulation). On the real data it doesn't give usable ranges (`processing/notes/D_multipath.md`). |
| `tests/` | `uv run pytest` (~35 s). |
| `data/` | Recordings and receiver positions (not modified). |
| `out/` | Generated outputs; tracks in `out/phase1/track.csv` and `out/phase2/track.csv`. |
| `tasks/` | `todo.md` (progress), `lessons.md`. |

## Running

Dependencies (numpy, scipy, matplotlib) are managed with
[uv](https://docs.astral.sh/uv/) via `pyproject.toml` / `uv.lock`.

```
uv sync                                 # create .venv and install
uv run python motion_model/demo.py      # synthetic truth track + plots
uv run python estimation/demo.py        # simulated phase 1 + 2 solve
uv run python estimation/current.py     # water-current estimation (several minutes)
uv run pytest                           # tests
```

Real-data pipelines: see `processing/README.md`
(`processing/run_phase1.py` and `processing/run_phase2.py` produce the tracks).
