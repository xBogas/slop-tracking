# A: detection and association (phase 1, runs 1-2)

Code: `processing/detect.py`, `processing/associate.py`; tests:
`tests/test_detect.py`, `tests/test_associate.py` (9 tests, all pass).
Outputs: `out/detections_<rx>.npz`, `out/toa_phase1.npz`, logs
`out/detect_log.txt`, `out/associate_log.txt`.

```
uv run python processing/detect.py        # ~3.5 min wall (16 cores), all three receivers
uv run python processing/associate.py     # ~7 s
```

## Key numbers

| Item | Value |
|---|---|
| Pulse | LFM 30 -> 40 kHz, **100.0 ms** (matched-filter gain vs T peaks between 99.8 and 100.0 ms; 99.6 and 100.4 ms are 10-20 % lower), sharp edges (< 1 ms, at the smoothing limit), no taper in the transmit. The received amplitude across the sweep varies per receiver (see below). |
| Ping interval | Tp = **1.000290 s** (long-baseline emission slope; about ±2e-5 from the geometry). The nominal 1.000 s is off by 290 ppm. |
| Pinger on/off | first ping arrives at **52.41 s** buoy time (Rx902; Rx01 52.45 s); nothing coherent before. Last ping about **795.6 s** (the SNR falls 52 -> 24 dB over the last 6 s on Rx902, then stops). It **kept running across the run boundary** at 420 s (Rx902 sees a steady 1.001 s arrival interval and no phase jump). 740 pings, k = 0..743. |
| Emission jitter | **0.127 ms rms**, common to all receivers (the TOA second differences of Rx902 / Rx01 / USBL correlate at 0.79-0.83). So the emission times are **not** e0 + k Tp: give them a free per-ping variable with a ~0.13 ms chain sigma. |
| TOA sigma (empirical, receiver-specific) | Rx902 **50 µs**, Rx01 **62 µs**, USBL **48 µs** (from the pairwise TDOA second differences, where the common jitter cancels). Heavy tails: 1-2 % of TDOA d2 values are beyond 4 sigma (first-arrival flips). The CRLB is 0.4-1.4 µs, so the TOAs are limited by multipath and source motion, not by SNR. |
| USBL clock offset | usbl_time = buoy_time + **102.8137 s**. The integer is resolved by the jitter fingerprint (**z = 15.0** vs Rx902 and **14.2** vs Rx01). The fine value is geometry-limited: **±10 ms** (cost +10 %), per run 102.8254 / 102.8038 s, with a drift term 102.8116 s at usbl t = 500 s and -52 ppm. |
| Buoy-buoy index lag | 0 (the fingerprint z = 16.5; every other lag has no real hyperbolic fix at all). |

## Measured pulse and receiver responses

- T and edges: a grid search of the matched-filter gain over T, f0 and f1
  on about 20 pings per buoy peaks sharply at exactly 30-40 kHz and 100 ms.
  The dechirped envelope rises and falls within about 1 ms (the limit of
  the smoothing used), so the transmit is rectangular.
- The received pulse shape is not flat. `pulse_shape` averages the
  dechirped first arrival over up to 400 pings per channel; it is stored
  as `shapes` in `detections_<rx>.npz`, n_u x n_ch complex. Amplitude at
  u = 0.05 / 0.25 / 0.5 / 0.75 / 0.95 of the pulse (30 -> 40 kHz):
  - Rx902: 1.03 / 1.49 / 0.77 / 1.02 / 0.51
  - Rx01: 1.51 / 1.71 / 0.89 / 0.47 / 0.24 (strong high-frequency roll-off, about -16 dB at 40 kHz)
  - USBL ch0-3: about 0.7 / 1.0 / 0.8 / 1.2 / 1.2 (rises with frequency)

  Each receiver also has a quadratic phase of 0.6-3.6 rad, which looks like
  0.15-0.9 m/s of Doppler. With the ideal LFM model, single-path fits left
  20-60 % of the window energy unexplained and the fitter placed spurious
  near-coincident paths; with the measured shape, that is gone.

## Detection method (detect.py)

1. Coarse pass: an 8 s chunked FFT matched filter that is also the
   band-pass, with channels summed incoherently. The noise floor is the
   median per 1 s block and the threshold is **20 dB** MF SNR; at 15 dB the
   USBL gave several false anchors per second. Anchors are clustered over
   ±0.3 s.
2. `index_pings` (in associate.py) assigns the per-receiver ping index and
   drops anchors inconsistent with their neighbours: 3 / 2 / 11 anchors on
   Rx902 / Rx01 / USBL.
3. Doppler and TOA. **Finding: the per-pulse Doppler (the peak of matched
   filter power vs time scale) does not work on these recordings.**
   - The LFM ridge is flat: about -1 dB at 1.5 m/s mismatch.
   - Overlapping multipath tilts it, so the estimates scattered by 2-5 m/s
     from ping to ping, bimodal on Rx01. Because the delay-Doppler coupling
     is 0.40 s per unit eta (0.26 ms per m/s), the "refined" TOAs were then
     noisier than the eta = 1 peak.

   So the reference time scale for the TOA comes from each receiver's own
   arrival sequence: eta_k = Tp / (dt/dk), from a robust local quadratic
   over ±6 pings, iterated TOA -> eta -> TOA (converges to < 0.002 m/s).
   With the nominal Tp, the 290 ppm Tp error biases eta equally on all
   receivers, which moves all TOAs of a ping by the same 0.12 ms; that is
   absorbed by the emission time.
4. First arrival: the earliest local maximum within 10 dB of the strongest
   in the 12 ms before it. Then a greedy variable-projection multipath fit
   (up to 4 paths) over a window from -0.6 to +1.6 ms, with the
   cross-ambiguity of the measured pulse shape. TOA = the earliest fitted
   path within 10 dB of the strongest.
   - How often the first arrival is not the strongest: Rx902 32 %, Rx01 52 %,
     USBL 54 %.
   - Typical first-to-strongest gaps: 0.21-0.26 ms on the buoys (surface
     reflection) and 0.12 ms on the USBL (a near-surface sign-flipped echo
     about 50-90 µs behind).
   - On the USBL the strongest later path sits at **1.1-1.6 ms in 30 % of
     pings**, which would be a wall about 0.85-1.2 m away, a bit more than
     the 30-80 cm guess (agent C to confirm). The fit window ends at 1.6 ms,
     so a later echo would be cut.
5. Per-ping Doppler `rr_dop`: the profile likelihood over eta of the same
   multipath fit (a joint least squares over eta and the delays gets stuck
   on the ridge). In synthetic tests it is accurate to < 0.04 m/s. On the
   real data its scatter around the arrival-track range rate is 0.67 m/s
   (Rx902), 1.0 m/s (Rx01) and 0.33 m/s (USBL), and Rx01 is offset by
   -0.75 m/s. Use it only as a weak, independent velocity cue.
6. USBL: the candidate comes from the 4-channel sum and the fits are per
   channel. Channels further than 0.25 ms from the median are dropped
   (0.2-1.8 % per channel). Per-channel offsets from the mean span about
   ±150 µs (ch0, ch3) and ±60 µs (ch1, ch2), consistent with a 4-element
   line of about 45 cm (3 x 15 cm). The array TOA is the channel mean with
   missing channels filled by their smoothed offset (`array_toa`). The
   inter-channel second differences are only 15-20 µs per channel.

## Association method (associate.py)

- **Lags: jitter fingerprint.** Correlate the clipped second differences
  of the TOA sequences between receivers for candidate lags of ±3. Results
  at lags -3..+3:

  | Pair | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
  |---|---|---|---|---|---|---|---|
  | Rx01 vs Rx902 | .13 | .00 | -.46 | **.79** | -.45 | .04 | .07 |
  | USBL vs Rx902 | .13 | .03 | -.51 | **.83** | -.54 | .07 | .11 |
  | USBL vs Rx01 | .14 | -.01 | -.48 | **.82** | -.54 | .04 | .14 |

  The ±1 values are the expected white-jitter sidelobes. This needs no
  geometry. Validated on simulation: a 100 % correct association with 5 %
  misses, false alarms, a 0.35 s error on the coarse offset and apparent
  cross-receiver delays up to 0.69 s.
- **Geometry cost** (as asked: per-ping fixes plus clock smoothness, for
  all 49 lag hypotheses, each with a 1-D search of b over ±1 s). It
  rejects every wrong buoy lag, which has no real fix. It does **not**
  separate the USBL lags:
  - costs 0.881 m (chosen) vs 0.886 m (one ping less) vs 0.920 m (one ping
    more); `usbl_offset_cost_gap` = 0.005 m;
  - the reason: the pinger stays outside the receiver triangle, where the
    hyperbolic GDOP is large, and ±15 m of buoy-position error gives
    0.5-1 ms of smooth emission-time residuals, more than the one-ping lag
    signature (rr Tp / c, about 1 ms at the 1.5 m/s seen here);
  - a fit with free buoy positions, c, USBL depth and drift was degenerate
    without priors and inconclusive with them, so it was left to the graph.

  The chosen lag agrees with the geometry minimum, but only by that thin
  margin. The **fingerprint is the decisive evidence**, with a margin of
  z = 15.
- **"First ping anchors the clocks"** is consistent with this: Rx902 and
  Rx01 both see ping k = 0 at 52.41 / 52.45 s. The USBL's first detection
  is ping k = 3, because it misses the first 3 pings at the 20 dB threshold.
- Tp: the slope of the per-ping emission times between the first and last
  30 fixes. Per-run quadratic clock fits give 1.00039 and 0.99998 s with
  0.5-0.8 ms residuals, spoilt by geometry; do not use those.

## Detection rate and SNR

Rates are relative to pings k_first..k_last of each run. SNR is the
matched-filter output SNR of the first path (in-band SNR = MF - 30 dB).

| Run (pings) | Receiver | Rate | MF SNR p10 / p50 / p90 [dB] |
|---|---|---|---|
| run1, k 0-367 (52-420 s) | Rx902 | 98.6 % | 30.5 / 45.1 / 53.8 |
| | Rx01 | 91.6 % | 24.1 / 36.5 / 44.9 |
| | USBL | 53.3 % | 18.2 / 31.8 / 44.5 |
| | all three | 51.1 % | |
| run2, k 368-743 (420-796 s) | Rx902 | 98.1 % | 35.8 / 43.5 / 48.1 |
| | Rx01 | 94.1 % | 25.4 / 33.3 / 39.6 |
| | USBL | 94.9 % | 23.5 / 31.4 / 40.8 |
| | all three | 89.9 % | |

Anomalies:
- **USBL gaps:** 85 pings at 309-394 s, plus 7-14 ping gaps between 276
  and 404 s. Most of the run-1 USBL loss is there; the pinger is
  presumably far away or masked.
- Rx01 has a 12-ping gap at 411 s.
- No double detections survive (±0.3 s clustering plus `index_pings`).
- One isolated USBL detection 38 s before the pinger starts (buoy time
  14.3 s) was dropped; it could be the sync signal and is worth a look.
- The range rates stay within ±1.9 m/s on all receivers over the whole
  recording, so the pinger moves slowly (about 1.5 m/s along the fixes).

## Output formats

`out/detections_<rx>.npz`
- `det`: structured array, one row per detection, with fields:
  - `t_rx`: TOA of the pulse start, receiver clock [s];
  - `t_buoy`: `t_rx` minus the *coarse* offset (102.7 s for the USBL; use
    the TOA table instead);
  - `snr_db`, `snr_peak_db`;
  - `eta`: the reference scale used;
  - `rr_dop`: per-ping Doppler [m/s], referred to the nominal Tp;
  - `rr_track`: range rate from the arrival track with the nominal Tp;
  - `amp`, `n_paths`;
  - `dt_strong`: delay from the first to the strongest arrival [s];
  - `echo_db`, `echo_dt`: the strongest later fitted path;
  - `sigma_toa`, `sigma_rr`: CRLB-style;
  - `t_coarse`: the eta = 1 anchor;
  - `k`: per-receiver ping index.
- `t_ch` (n, n_ch): per-channel TOAs (receiver clock).
- `shapes` (200, n_ch) complex: the measured pulse shapes.
- `offset`, `fs`, `t0`, `t1`, `n_anchors`, `n_rejected`, `runtime_s`.

`out/toa_phase1.npz` (K = 740)
- `receivers` = [Rx902, Rx01, USBL]; `k` (K,): ping index, 0 = first ping
  (52.41 s); `run` (K,): 0 = run1, 1 = run2 (from the earliest TOA).
- `toa` (K, 3): TOA of the pulse start in **buoy time**, with the USBL
  already shifted by `usbl_offset`. NaN where missing; `toa_mask`.
- `snr` (K, 3) [dB]; `toa_sigma` (3,) [s]: empirical receiver-specific
  sigmas, excluding the common jitter; `toa_sigma_crlb` (K, 3);
  `emission_jitter` [s]: the common part.
- `range_rate` (K, 3) [m/s], opening > 0, plus `rr_mask`: the per-ping
  Doppler, corrected to the fitted Tp; noisy (0.3-1 m/s), but independent
  of the TOAs. `rr_track` (K, 3), plus `rr_track_mask`: the smooth range
  rate from the TOA sequence (corrected to the fitted Tp). It is derived
  from the TOAs, so do not use it as an extra measurement next to them.
- `Tp` [s]; `clock_runs` (rows: run, k_mean, e_mean, Tp, quad, resid_rms,
  n; run = -1 for both runs).
- `usbl_offset` [s] (usbl_time = buoy_time + offset);
  `usbl_offset_margin` (fingerprint z vs Rx902) and
  `usbl_offset_margin_rx01`; `usbl_offset_cost`, `_cost_next`, `_cost_gap`
  [m] (geometry); `usbl_offset_halfwidth` [s]; `usbl_offset_drift` = (b at
  usbl t = 500 s, drift ppm, cost); `usbl_offset_runs` (rows: run, b,
  cost, n).
- `usbl_lag`, `rx01_lag`, `lag_offset_*`, `rx01_margin`,
  `geometry_choice_agrees`, `hypotheses` (rows: cost, d01, du, b),
  `jitter_corr_*`, `lag_candidates`.
- `usbl_toa_ch` (K, 4): per-channel USBL TOAs in buoy time.
- `det_index` (K, 3): row -> detection index per receiver (-1 = none).
- `pos_fix` (K, 2) ENU [m] and `emission_fix` (K,) [s]: per-ping hyperbolic
  fixes, nominal geometry, c = 1520, pinger at 10 m, USBL at the placeholder
  2 m; a starting point only. `emission_resid`.
- `rx_xyz` (3, 3), `sound_speed`, `pinger_depth`.

## Runtimes (16 cores)

| Step | Rx902 | Rx01 | USBL |
|---|---|---|---|
| Coarse pass | 9 s | 9 s | 30 s |
| Full detection (3 refine passes, 2 shape estimates) | 50 s | 48 s | 108 s |

Association takes 7 s. The tests take about 11 s.

## Tests

- `test_detect`: single-ping mode on a synthetic LFM with Doppler (-8 to
  +12 m/s), a sign-flipped echo inside 1/B (70 µs, -6 dB) and a wall echo
  (0.7 ms) at 50 dB MF SNR: TOA error < 10 µs and range-rate error
  < 0.2 m/s (observed < 0.04 m/s). With the eta-matched replica at 40 dB
  MF SNR, the TOA error is < 10 µs. Also tested: the coupling coefficient,
  no false alarms in noise, and `index_pings` with misses, a gap and false
  alarms.
- `test_associate`: a simulated straight leg plus zig-zag, 5 % misses, 4
  false alarms per receiver, 0.1 ms emission jitter, 20 µs TOA noise, the
  true offset 0.35 s from the coarse guess. Checked: rows 100 % consistent,
  b within 1 ms, fingerprint z > 3, Tp within 1e-5.

## Open problems

- **Fine USBL offset.** ±10 ms from geometry alone (15 m), and the per-run
  values differ by 21 ms. Fit it in the graph, with buoy positions, c and
  possibly a USBL drift (here -52 ppm, poorly determined).
- **Emission times.** The pinger has 0.13 ms of white jitter, and its
  clock may wander slowly; the per-run quadratic clocks do not fit to the
  jitter level, though geometry may be to blame. The graph needs free
  per-ping emission times.
- **Per-receiver constant delays** (hardware group delay, the pulse-shape
  reference) are not calibrated. Buoy-to-buoy constant biases go straight
  into the TDOA; the USBL one is absorbed into b.
- **First-arrival flips** (direct vs surface reflection, 0.2-0.3 ms apart)
  cause about 1-2 % TOA outliers, so use a robust loss in the graph. Some
  real direct paths may still be missed when they are > 10 dB below the
  surface path.
- **Weak per-ping Doppler.** Accurate in simulation, but 0.3-1 m/s noise on
  the real data, with a -0.75 m/s offset on Rx01.
- The USBL depth (2 m) is a placeholder. It sets the surface-echo delay
  (about 50-90 µs) and enters the fixes.
