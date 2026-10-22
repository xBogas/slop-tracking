# CLAUDE.md

Context for working in this workspace. The full problem statement is in
`README.md`; read it first.

## Problem in one paragraph

Post-processed acoustic localization of one motor boat using three
receivers: two synchronized single-channel buoy hydrophones (10 m above the seabed, ~3 m depth,
positions nominally known but may drift, no guaranteed GPS) and one
4-channel shore array at a known GPS position midway between them. A
start-of-run sync signal gives the inter-receiver clock offsets. Phase 1:
a pinger (30–40 kHz, waveform and interval are ours to design, emission
times unknown) towed ~5 m behind the boat at ~10 m depth; the run is a
straight line followed by a zig-zag. Phase 2: no
pinger, only the boat's radiated noise. Goals: (1) design the pinger
waveform and interval, (2) track position + velocity with the pinger,
(3) track position + velocity from radiated noise. No ground truth.

## The experiment is fixed

We only post-process. We do not control the boat's path, and get no
onboard logs and no ground truth. Do not propose changes to the run plan
(extra circles, standard zig-zags, onboard logging). Work with what the
acoustic data can show, and validate without truth (e.g. held-out ping
prediction, residual statistics).

## Real setup (from the data; overrides the generic description)

- Receivers (ENU m, origin = the USBL GPS antenna): Rx902 (-338.5, -365.0),
  Rx01 (-109.8, -702.8), USBL (-3.4, -3.6). The array is 5 m from its GPS
  (operator); the direction (seaward, 223° true) is assumed. Positions in
  `processing/forfun.toml` (`gps_offset`, `gps_offset_bearing`).
- The "4-channel shore receiver" is a USBL: a near-straight line, gaps
  19.0 / 12.5 / 19.1 cm from the delays. The operator says 15.5 cm uniform,
  but that fits the delays at 10.7 µs rms against 0.44 µs (any channel order,
  with or without channel biases), so `usbl.py` keeps its self-calibration.
  Axis ch0→ch3 at 128° magnetic (operator) ≈ 133° true. The data give 130–132°,
  so this is used as the orientation prior (±3°). There is a wall 1.0–1.25 m behind
  it. No elevation information. Its clock is offset by 102.82 s from the
  buoys (not the 180 s the file name implies) and drifts by about -90 ppm.
- Pinger: LFM 30→40 kHz, 100 ms, period 1.000290 s, 0.13 ms emission
  jitter. Pings run 52.4–795.6 s buoy time (runs 1–2); run 3 has none.
- Not observable from the data, so set by priors: the sound speed
  (1521 ± 4 m/s assumed) and a common shift of the buoys and the track.
  Rx01 is pulled ~28 m north of its GPS position (a position error or a
  timing delay; undecided).
- Run times (`[[run]]` start/end) come from the data, not the organisers' minute log (15:59 / 16:06 /
  16:14 = 0 / 420 / 900 s): each run starts where the boat passes its start fix (102 / 514 / ~985 s;
  run 2 ends at 797 s, pinger off). The pinger runs continuously from 52.4 s (before run 1) to 796.6 s
  (no restart; ping 360 at 412 s is a missed detection in the turn 387–444 s). `notes/J_run_times.md`.
- Date 2026-10-20 (inferred). Tide: `[site] tide` = +0.02 m above MSL (Sibenik prediction; surge unknown,
  `tide_sigma` 0.2 m placeholder); `bathy.Bathymetry.depth` = charted depth + tide.
- Boat start fixes per run: `[[run]] lat/lon` in `forfun.toml` (`Run.en`), no time
  stamp. The track passes 9-12 m from them; `run_phase1.py --fixes` uses them as
  priors (`notes/H_start_fixes.md`). They pull everything 5-15 m north; Rx01's
  offset from Rx902 is unchanged.
- Charted depth (EMODnet, `data/bathymetry/`, `processing/bathy.py`): 13 m at
  the buoys, 6–9 m under the track. So the pinger is not at 10 m, and the
  hydrophones are moored 10 m above the seabed (operator): 3.28 m (Rx902) and 2.65 m (Rx01) deep, derived in
  `config.py` from `height_above_seabed`, the DTM and the tide. The 24–27 m in `notes/D_multipath.md` is wrong.
- Multipath ranging doesn't work on this data (diffuse surface echo,
  depth/range scale ambiguity). Don't re-add the `mp_range` factor without
  new information (pinger depth logger, charted water depth).
- Multipath as measurements (`multipath_paths.py`, `notes/I_multipath_paths.md`): with the hydrophones at
  ~3 m, S is unresolved (< 1/B) and B is separable; only 19 of 740 chirps give a usable delay (mostly B on
  Rx01), and those bottom bounces want a pinger deeper than the configured 3 m (edge of the 1–5 m grid).
  The list is `out/multipath/paths_phase1.csv`.
- Isovelocity is the weak link for multipath delays: a 0.05–0.2 m/s per m gradient moves them by 5–160 µs.
  Hydrophone depth now comes from the operator (10 m above the seabed); `notes/K_hydrophone_depth.md` predates it.

- Phase 2 (noise): the boat is audible only ~805–1020 s on the buoys
  (to ~1165 s on the USBL). Carrier-phase GCC between receivers fails;
  use the envelope (`noise_tdoa.py`). The envelope TDOA has a ±2–3 ms
  slowly varying error (multipath centroid), measured on the pinger band.
  USBL bearings must use c = 1520 m/s (the array calibration value; only
  pos/c is determined). The USBL channels share zero-lag electrical
  crosstalk. No measurements 926–984 s.

## Unknowns (do not assume silently)

Buoy mooring type, true buoy positions, sound-speed profile, USBL depth
(placeholder 2 m), boat noise-source depth (placeholder 1 m), scoring metric and output format. When code needs one
of these, make it a parameter with a clearly marked placeholder default,
and say so.

## Code layout

- `processing/`: real-data phase 1 pipeline (`detect.py` → `associate.py`
  → `usbl.py` → `run_phase1.py` + `validate.py`) and phase 2 pipeline
  (`noise_tdoa.py`, `noise_bearing.py` → `run_phase2.py`, which reads
  `out/phase1/state.npz` via `posterior.py`); config in `forfun.toml`;
  per-step findings in `processing/notes/`. See `processing/README.md`.
- `processing/export.py` → `out/export/`: the phase 1 measurements per receiver as CSV (synced first
  arrivals, USBL bearing and range, multipath per buoy), every reading with a sigma; columns in
  `out/export/README.md`.
- `processing/pipeline.py` rebuilds stale stages (scripts and notebooks) after a
  config, input or code change; `--status` says why. When adding a step or a
  notebook, add it to `pipeline.stages()`. Never copy config values or earlier
  results into code; key any cache with `stamp.key` / `stamp.load` / `stamp.save`.
- `motion_model/boat_model.py`: `towed_body_track` and its inverse
  `boat_from_towed_body`, `ct_predict` / `ct_process_noise` / `ct_track`
  (vectorized coordinated-turn model on `[x, y, chi, U, omega]`). No dynamic
  model (no data to fit one); `demo.py` writes a kinematic synthetic truth.
- `estimation/fg_tracker.py`: batch factor-graph smoother (own LM, sparse
  finite-difference Jacobian, Huber then Cauchy on measurements, Gaussian
  priors). Globals `[b_S, alpha, skew, c, cx, cy, d_S]`. Optional meas:
  `toa_sig`, `bearing_sig`, `mp_range*`, `rr*`. `estimation/demo.py`
  simulates measurements; `current.py` estimates water current.
- `multipath/`: simulation study of multipath ranging.

## Conventions

- World frame x east, y north; heading/course measured from +x, CCW
  positive. Body frame u forward, v to port, r CCW.
- Depth is negative z.
- In phase 1 the estimated emitter is the **pinger**, not the boat; the
  boat-track conversion needs the tow offset model.
- Plain numpy/scipy/matplotlib; no GTSAM or other heavy dependencies.
- Environment is managed with uv (`pyproject.toml`, `uv.lock`, `.venv`).
  Run scripts with `uv run python ...`; add dependencies with `uv add`,
  never `pip install`. The system `python3` has no numpy.
- `processing/recordings.py` (not `io.py`: that name shadows the stdlib
  once `processing/` is on `sys.path`).
- Tests: `uv run pytest`. `tests/conftest.py` puts the module dirs on
  `sys.path`; two modules are named `demo.py`, so tests load
  `estimation/demo.py` by path.
- Git: `.gitignore` drops `data/*` (keeps the folder), `out/` except `out/export/` (measurement CSVs), images and `.venv`. Notebooks pass
  through the `nbclean` filter (`.gitattributes`; outputs and metadata stripped on commit, local copies
  keep them). After a fresh clone, set it up with `git config filter.nbclean.clean "uv run --quiet jupyter
  nbconvert --clear-output --ClearMetadataPreprocessor.enabled=True --to notebook --stdout --stdin
  --log-level=ERROR"` and `git config filter.nbclean.smudge cat`.

## Key findings so far

- Accuracy is limited by receiver geometry and the buoy-position/sound-speed
  priors, not by the measurements (held-out TDOAs predicted to ~60 µs).
- Buoy position, sound speed and USBL clock offset trade off against each
  other; the USBL bearings halve the position uncertainty.
- Water current is not observable on this path; solve with
  `motion="ground"` (velocity over ground).
