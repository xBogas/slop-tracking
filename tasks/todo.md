# Phase 1 processing (ForFun, runs 1–2)

Plan: `~/.claude/plans/vivid-humming-island.md`

## 1. Shared ground (coordinator)
- [x] `processing/config.py` + `processing/forfun.toml` (receivers, pinger, runs, priors, ENU)
- [x] `processing/recordings.py` (buoy int16 mono, USBL float32 4-ch extensionless)
- [x] Import-path fixes (`motion-model` → `motion_model`) in estimation/ and multipath/
- [x] `uv add --dev pytest`, `tests/conftest.py`

## 2. Parallel
- [x] A: `processing/detect.py`, `processing/associate.py` → TOA table, USBL coarse offset
- [x] B: `fg_tracker.py` (USBL drift, multipath-range factor, range-rate factor, start prior, grid seed), `estimation/demo.py`, `tow_point_from_body`
- [x] C: `processing/usbl.py` (spacing + wall validation, bearings)

## 3. After A
- [x] D: `processing/multipath_range.py` on real pings (negative result: not usable)

## 4. Integration (coordinator)
- [x] `processing/run_phase1.py`, `processing/validate.py` (TOAs + USBL bearings)
- [x] Run on runs 1–2, review results
- [x] Docs: README, CLAUDE.md, processing/README.md (estimation/, motion_model/ READMEs by agent B)

## Review

- Pipeline runs end to end on runs 1–2 (`processing/run_phase1.py`, ~10 s after the cached detection).
- Track: run 1 straight NW leg at 2.3 m/s, run 2 zig-zag back E at 1.2 m/s; graph 1-sigma 6.4 / 9.0 m median.
- Validation: held-out TDOA 60–66 µs (1.4 sigma), bearing residual 0.8° rms, no residual trends.
- Systematic ~6–12 m from unobservable priors: sound speed (fixed to 1521 ± 4 assumed) and buoy positions (Rx01 pulled 28 m N).
- User assumptions corrected by data: USBL gaps 19.0/12.5/19.1 cm (not 15 uniform); wall 1.0–1.25 m (not 0.3–0.8); USBL offset 102.82 s (not 180).
- Multipath ranging: negative on real data (diffuse surface echo, depth–range scale ambiguity).
- Next: phase 2 (run 3, radiated noise); settle Rx01 position vs timing delay; a CTD/sound-speed value if one exists.


# Phase 2: radiated noise (run 3)

Plan: ~/.claude/plans/vivid-humming-island.md. Solve window 796–1020 s
buoy time (boat audible from ~805 s; buoy files end at 1020 s).

- [x] Agent E: `processing/noise_tdoa.py` (envelope xcorr, click blanking, trace extraction, closure gating) + pinger-band method check + tests
- [x] Agent F: `processing/noise_bearing.py` (USBL inter-channel GCC on noise, calibrated array) + tests
- [x] `fg_tracker.py`: per-epoch `tdoa_sig`; noise-mode init with start prior; regression test
- [x] `forfun.toml` `[noise]` section (src_depth placeholder 1 m)
- [x] `processing/run_phase2.py`: graph, track.csv, bearings_only.csv, plots
- [x] `validate.py` phase 2 report (residuals, held-out, closure, method check)
- [x] Docs: notes E/F, processing/README, README, CLAUDE.md

## Review (phase 2)

- Track 796–1020 s, 1 s epochs: 826–926 s at 3.7 m/s (1-sigma 3.9 m),
  984–1020 s accelerating west to ~9 m/s (5.2 m); 926–984 s has no
  measurements (motion model only, 1-sigma up to 75 m, flagged `weak`).
  Starts 6.4 m from the phase 1 end.
- Validation: residual rms 0.2–0.8 sigma; held-out 10 s blocks to
  0.2–0.8 ms and 0.4 deg; pinger-band method check bias < 0.6 ms.
- 41 tests pass.
- Open: multipath (bottom-echo delay per buoy, water depth as a global) as a
  range factor to cover the weak geometry — proposed to the user, not started.

# Config → outputs → notebooks stay in sync (2026-10-21)

Review findings:
- No dependency tracking. Every stage is run by hand, and nothing notices when
  `forfun.toml` changes (now: `mp_range_phase1.npz` 13:59 is older than
  `toa_phase1.npz` 21:30, and both notebooks are older than `out/phase1`).
- Caches keyed on file existence only: `multipath/arrivals_*`, `profile_*_zr10`,
  `usbl/pings.npz`; `family_grid_*` keyed on ping indices only (not track, buoys, c, depths).
- Config values copied as constants: `noise_tdoa.py` POS / B_S / D_S / C_SOUND=1513.5 /
  pinger depth 10; `noise_bearing.read_globals` fallback numbers; `C_SOUND=1520` in
  detect / associate / multipath_range; `multipath_range.main(zr=10)`; notebook
  `ZR_EX=10`, z_r=10 in the phase 2 cell, "nominal 10 m" label, `mp.C_SOUND`.

Plan:
- [x] `processing/stamp.py`: input key = hash(used config sections, input files, script source).
- [x] Self-checking caches: arrivals, profiles, usbl pings, family grid store their key; recompute on mismatch.
- [x] `processing/pipeline.py`: stage table (scripts + notebooks, inputs, outputs); reruns stale
      stages in order, executes notebooks with nbconvert; `--status` / `--dry-run`.
- [x] Replace copied constants with config / upstream outputs.
- [x] Tests for stamp + staleness; run the pipeline end to end; README section.

Review:
- Full pipeline run from scratch: 13 stages, ~11 min, all ok; `--status` clean afterwards.
- Comment-only toml edit: nothing stale. `[pinger].depth` edit: every stage but detect and the chirp notebook.
- Cycles removed: usbl.py now calibrates on out/phase1_toa_only/ (run_phase1 --no-bearing);
  run_phase1 reads mp_range only with --mp (the ranges are empty).
- Coherent multipath profiles (c 1520) no longer match (prior 1529): fit skipped until `multipath_range.py --full`.
- 50 tests pass.

# Multipath paths → measurements notebook (2026-10-22)

Goal (notes/D2_multipath.md): predict each path from the bathymetry, extract paths with
geometry windows on the Doppler-matched pulse, pick the direct path as the earliest
qualifying arrival, associate peaks to paths, and end with how many chirps give usable
multipath measurements plus the measurement list. No particle filter.

- [x] `bathy.eigenrays`: also return the arrival angle at the receiver
- [x] `processing/multipath_paths.py`: predictions per ping (nominal depths + window grid over
      pinger depth, hydrophone depth, tide; placeholders), Doppler per path, OMP components,
      false-alarm level from the pre-arrival zone, direct-path pick, association, persistence,
      sigma (CRLB, empirical), common-depth consistency; writes out/multipath/paths_phase1.{npz,csv}
- [x] `notebooks/multipath_paths.ipynb`: prediction, image vs ray trace (slope, cross-track),
      Doppler/coupling, windows + extraction on example pings, association echogram, counts, list
- [x] pipeline stages + tests + README/notes/CLAUDE.md

## Review (multipath measurements)

- Split: `multipath_predict.py` (ray-traced predictions, cached ~8 min) and `multipath_paths.py` (5 s).
  `run_phase1.py` now also saves `toa_pred` for the direct-path window. Stages `multipath_paths` and
  `nb_multipath_paths` added; 54 tests pass; the full pipeline runs clean (0 stale).
- Result: 45/740 chirps with a usable multipath delay (SB 26, BS 28; no S or B). SB and BS want different
  depths (joint misfit 45 µs against 5–10 µs scatter), so only SB is a candidate measurement.

# Run times from the data + tide (2026-10-22)

- [x] Validate run times (notes/J_run_times.md): pinger continuous 52.4–796.6 s; the log's minute times are
      85–100 s before the boat passes each start fix; differences agree within 1 min
- [x] `[[run]]` start/end from the data (102 / 514 / 985 s; run 2 ends at 797 s); detect + chirp notebook search
      from the file start; `--fixes` searches from 60 s before the run start
- [x] `[site] date/tide/tide_sigma` (2026-10-20, +0.02 m above MSL from the Sibenik prediction, 0.2 m surge
      placeholder); `Bathymetry.depth` adds the tide; multipath windows span tide ± tide_sigma
- [x] Full pipeline rerun (0 stale), 55 tests pass

## Review

- Detections, pings (740) and held-out TDOAs unchanged; fixes land at 102 s and 514 s.
- Multipath: 46/740 chirps usable (was 45); joint SB/BS misfit 31 µs on Rx902 (was 45), still 2–6 × scatter.
- Open: hydrophone-depth fit with a sound-speed gradient (notes/K); BELLHOP not installed (rule: numpy/scipy).

# Export per receiver (2026-10-22)

- [x] `processing/export.py` → out/export/: first_arrival_<rx>.csv (USBL synced by the phase 1 clock model),
      usbl_bearing_range.csv, multipath_<buoy>.csv, README.md (columns); pipeline stage `export`
- [x] run_phase1 saves emission-time covariances (total, with the globals; conditional on globals + buoys)
      → every reading has a total and a ping-to-ping sigma; `FactorGraph.conditional_std` + test
- [x] multipath_paths: sigma on every row (`sigma_basis`), direct-path time for absolute path arrivals

## Review

- 732 / 691 / 553 chirps; TOA sigma 55 / 63 / 47 µs (USBL total 3.8 ms from the clock offset); range sigma
  7.7 / 7.9 / 4.7 m total, 0.2–0.25 m ping-to-ping; bearings 552 (0.86° random, 1.33° total; residual 0.80° rms).
- The USBL "range" is c × (toa − phase 1 emission time): it equals the track range (MAD 1 cm), not an
  independent measurement.

# Hydrophones 10 m above the seabed (2026-10-22)

- [x] `height_above_seabed` in `[[receiver]]`; `config.py` derives depth = DTM depth + tide - height
      (Rx902 3.28 m, Rx01 2.65 m); per-buoy depth in the phase 1/2 graphs; test
- [x] Full rerun (0 stale, 57 tests): tracking unchanged; multipath 19/740 usable (B on Rx01), bottom bounces
      want a pinger deeper than 3 m (grid edge)
- [x] Notebook findings, notes I/G/K, CLAUDE.md, README corrected
- Open: widen the pinger-depth grid (now 1–5 m) to see where the bottom bounces put the pinger
