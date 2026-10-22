# Phase 1 processing: real pinger recordings → track

Turns the group 4 ("ForFun") recordings in `data/` into pinger and boat
position/velocity tracks for runs 1 and 2. Configuration is in `forfun.toml`;
detailed per-step notes are in `notes/`.

## Run

```
uv run python processing/pipeline.py           # rebuild whatever is out of date (scripts + notebooks)
uv run python processing/pipeline.py --status  # what is stale, and why
```

`pipeline.py` reruns a stage when its config sections, input files or code
(the script plus every repo module it imports) changed since its last run,
or an output is missing; running it makes the stages that read its outputs
stale in turn. Order and timings (16 cores):

```
detect.py                    ~3.5 min  ping detections, all receivers (reruns only on [[receiver]] file/clock,
                                       pinger waveform or run time changes)
associate.py                 ~7 s      ping index, USBL clock offset -> out/toa_phase1.npz
run_phase1.py --no-bearing   ~10 s     TOA-only solve -> out/phase1_toa_only/ (usbl.py's orientation reference)
usbl.py                      ~15 s     USBL geometry, wall, bearings -> out/usbl/
run_phase1.py                ~10 s     solve + validation -> out/phase1/   (--fixes: out/phase1_fixes/)
multipath_range.py           ~1 min    ridge diagnostics (negative result, see below) -> out/multipath/
multipath_paths.py           ~8 min    predicted paths + multipath measurement list -> out/multipath/paths_phase1.*
                                       (the ray-traced predictions are cached; ~5 s when only association changes)
export.py                    ~2 s      per-receiver CSVs with sigmas -> out/export/ (column guide: out/export/README.md)
noise_tdoa.py, noise_bearing.py, run_phase2.py   phase 2, see below
notebooks/*.ipynb            executed in place
```

`out/phase1/state.npz` also holds `toa_pred` (the TOA the solution predicts per ping and receiver, rows as
`track.csv`); `multipath_paths.py` centres its direct-path window on it.

`run_phase1.py` flags: `--no-bearing`, `--mp` (add the multipath ranges; empty), `--fixes`, `--no-validate`, `--verbose`.

Nothing downstream copies config values or earlier results into the code:
phase 2 reads the phase 1 posterior from `out/phase1/state.npz`
(`posterior.py`), and the caches inside steps (`out/multipath/arrivals_*`,
`profile_*`, `family_grid_*`, `paths_predict_*`, `paths_components_*`, `out/usbl/pings.npz`) store a key of their
inputs (`stamp.py`) and are recomputed when it no longer matches. The
coherent multipath fit profiles (~25 min) are only recomputed with
`multipath_range.py --full`; until then the fit is skipped, not run on stale profiles.

## Steps

| File | What it does |
|---|---|
| `config.py`, `forfun.toml` | Receivers, pinger, runs, priors; WGS84 → local ENU (origin = USBL). |
| `recordings.py` | Memory-mapped access to the WAVs (buoys int16 mono, USBL float32 4-ch). |
| `detect.py` | Matched filter with the measured per-receiver pulse shape, first-arrival rule, multipath-aware TOA fit, Doppler time scale. |
| `associate.py` | Ping interval, ping index per receiver, USBL clock offset (integer second resolved from the emission-jitter fingerprint). |
| `usbl.py` | Inter-channel delays, array self-calibration, wall echo, bearings in the array frame. |
| `multipath_range.py` | Image-source ranging from multipath. Doesn't work on this data (see below); output kept empty. |
| `bathy.py` | EMODnet bathymetry in ENU; eigenrays (delay, launch and arrival angle) over a sloping bottom (`notebooks/multipath_reflections.ipynb`). |
| `multipath_predict.py` | Per ping and buoy: each path's delay, angles and Doppler from the bathymetry; search windows over the unknown depths and tide; image-method and cross-slope checks; Doppler-coupling budget. |
| `export.py` | Per-receiver CSVs on the buoy clock: first arrivals (USBL synced by the phase 1 clock model) with sigma and one-way range; USBL bearing + range; multipath delays per buoy. Random and common sigmas kept apart. |
| `multipath_paths.py` | Multipath as measurements (`notes/I_multipath_paths.md`, `notebooks/multipath_paths.ipynb`): OMP paths, false-alarm level, earliest-arrival direct path, association, persistence → `out/multipath/paths_phase1.csv`. |
| `run_phase1.py` | Builds and solves the factor graph (`estimation/fg_tracker.py`), pinger → boat conversion, CSV + plots. |
| `pipeline.py`, `stamp.py` | Stage table and staleness; input keys and self-checking npz caches. |
| `posterior.py` | Phase 1 posterior (clock, c, buoys) for the phase 2 steps. |
| `validate.py` | Truth-free checks: normalized residuals, held-out-ping TDOA prediction, residual trends. |

Output `out/phase1/track.csv`, one row per ping: `ping, run, t_emit_buoy_s`
(seconds since the buoy files' start, 15:59:00 local), `pinger_e/n`,
`pinger_lat/lon`, `boat_e/n`, `boat_lat/lon` (pinger + 5 m tow + 3 m stern
offset), `v_e, v_n, speed`, `sig_e, sig_n` (graph 1-sigma), `n_receivers`.
`globals.txt` holds the clock, sound speed and buoy estimates.

## What the data showed

- **Pinger:** LFM 30→40 kHz, 100.0 ms, period 1.000290 s, 0.13 ms emission jitter common to all receivers. Pings from 52.4 s to 795.6 s buoy time, continuous across the run 1/2 boundary, so both runs are solved as one graph.
- **Detections:** Rx902 98 %, Rx01 92–94 %. USBL 53 % in run 1 (a gap at buoy time 309–394 s) and 95 % in run 2.
- **USBL clock:** usbl_time = buoy_time + 102.822 s. The file name suggests 180 s, which is wrong. The whole second comes from the jitter fingerprint (z ≈ 15); the graph fits the fine part (±3.7 ms) and a −87 ± 11 ppm drift.
- **USBL array:** a near-straight line, channels 0-1-2-3, gaps 19.0 / 12.5 / 19.1 cm (50.5 cm long), **not uniform 15 cm**. Ch1 sits 1.65 cm off the line, which resolves left/right. No elevation information. Bearings are good to 0.85° per ping; the graph estimates the orientation at −40.4°.
- **Wall behind the USBL:** a plane reflector 1.0–1.25 m away (per channel), **not 30–80 cm**. The echo is −12 dB and doesn't bias delays or TOAs.
- **Multipath ranging:** negative. The surface echo is diffuse at 30–40 kHz, and depths and range trade off exactly (a scale ambiguity). The byproduct "water depth ≈ 24–27 m" is wrong: EMODnet charts 13 m at the buoys and 6–9 m under the track (`notes/G_multipath_bathymetry.md`).

## Results (runs 1–2)

| | Run 1 (straight) | Run 2 (zig-zag) |
|---|---|---|
| Pings | 368 | 376 |
| Pinger speed, median | 2.3 m/s | 1.2 m/s |
| Graph 1-sigma, median | 6.4 m | 9.0 m |

Validation (no truth):
- TOA residuals are below their sigmas (rms 0.3); 0.1–0.4 % are outliers.
- Held-out TDOAs are predicted to 60–66 µs median (1.4σ).
- Bearing residuals are 0.8° rms.
- Residuals show no trend over time.

## Limits

- **Sound speed is not observable.** Three receivers plus a free emission time per ping leave no redundancy within a ping. It is held by a physical prior (1521 ± 4 m/s, assumed; replace with a CTD value if one exists). Loosening it moves the track by ~6 m median.
- **Rx01 is pulled ~28 m north of its GPS position** in every variant. A position error and a constant Rx01 timing delay are indistinguishable. Forcing the buoys to their GPS positions moves the track ~9 m.
- **Systematic error:** the model choices above add ~6–12 m on top of the graph's 1-sigma. The weakest direction is a common translation of the buoys and the track, anchored only by the USBL TOA and bearing.
- **Pinger → boat conversion:** assumes a taut 5 m line plus 3 m to the boat centre (tractrix model).

# Phase 2 processing: radiated noise → boat track (run 3)

```
uv run python processing/noise_tdoa.py          # ~30 s: envelope TDOAs -> out/phase2/tdoa_noise.npz
uv run python processing/noise_tdoa.py --pinger # ~2 min: method check on the pinger band, runs 1-2
uv run python processing/noise_bearing.py       # ~30 s: USBL bearings from noise -> out/phase2/bearings_noise.npz
uv run python processing/run_phase2.py          # solve + validation -> out/phase2/
```

`run_phase2.py` needs `out/phase1/state.npz` (written by `run_phase1.py`):
the phase 1 posterior is used as tight priors (sound speed, USBL clock,
array orientation, buoy positions) and the last phase 1 boat position as the
start prior. Settings are in `[noise]` in `forfun.toml`.

| File | What it does |
|---|---|
| `noise_tdoa.py` | Inter-receiver TDOAs from the noise **envelope** (6 sub-bands in 2–20 kHz, clicks blanked, 4 s windows), line stacking + Viterbi trace extraction, delay-rate warping, USBL pairs by a joint search along the closure line. See `notes/E_noise_tdoa.md`. |
| `noise_bearing.py` | USBL bearings from the noise (2–28 kHz SCOT GCC, clicks blanked, crosstalk removed, steered-response seed), phase 1 array calibration. See `notes/F_noise_bearing.md`. |
| `run_phase2.py` | Factor graph in `mode="noise"` on a 1 s grid, 796–1020 s; `track.csv`, `bearings_only.csv`, plots. |
| `validate.py` | `report_phase2`: normalized residuals, held-out 10 s blocks, pinger-band method check. |

Output `out/phase2/track.csv`, one row per second: `t_buoy_s, boat_e/n,
boat_lat/lon, v_e, v_n, speed, sig_e, sig_n, n_tdoa, has_bearing, weak`
(`weak` = 1-sigma > 30 m or fewer than two measurements). `bearings_only.csv`:
USBL bearings after the buoy files end (1021–1164 s), no range.

## What the data showed (phase 2)

- The boat is audible only from ~805 s (after the last ping) to the end of the buoy files at 1020 s, and on the USBL to ~1165 s. It is not visible in the noise during runs 1–2.
- Carrier-phase cross-correlation between receivers fails (waveguide decorrelation over 400–700 m; snapping-shrimp clicks). The envelope cross-correlation works: Rx902–Rx01 in 153 windows (0.22 ms random), the USBL pairs in 79 (0.85 ms), only where the buoy pair is detected.
- Method check (same code on the pinger in runs 1–2 vs the pinger track): bias < 0.6 ms on all pairs, ~1 ms random, plus a slowly varying ±2–3 ms (the envelope peak follows the multipath energy centroid). `tdoa_sig` includes a 2 ms systematic term for this.
- USBL noise bearings: 263 good 1 s windows, ~0.7°; the pinger-band check reproduces the phase 1 bearings to 0.08°. All four channels share an electrical crosstalk at zero lag (removed). Bearings use c = 1520 m/s, the speed the array was calibrated with (only pos/c is determined).
- No measurements at all 926–984 s (buoy pair loses the trace; the boat is near endfire of the USBL).

## Results (phase 2)

| Segment [s] | Speed, median | 1-sigma, median | Weak epochs |
|---|---|---|---|
| 796–826 (start, few measurements) | 1.1 m/s | 7.5 m | 22/30 |
| 826–926 | 3.7 m/s | 3.9 m | 7/100 |
| 926–984 (no data: motion model only) | — | 47 m | 58/58 |
| 984–1020 (accelerating west) | 9.2 m/s | 5.2 m | 3/37 |

The track starts 6.4 m from the phase 1 end position. Residuals are below
their sigmas (rms 0.2–0.8); held-out 10 s blocks are predicted to
0.2–0.8 ms and 0.4°. The 926–984 s loop in the plot is the smoother's
interpolation, not a measurement.

Limits: the 2 ms systematic TDOA term is a conservative placeholder taken from
the pinger band; the noise-source depth (1 m) is a placeholder; the graph
treats the overlapping 4 s windows as independent, so its 1-sigma is
optimistic where the systematic error dominates.
