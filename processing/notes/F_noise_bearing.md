# F: USBL bearings from radiated noise (phase 2, run 3)

Code `processing/noise_bearing.py` (method in the module docstring), tests
`tests/test_noise_bearing.py` (4 synthetic tests, ~7 s). Run:
`uv run python processing/noise_bearing.py` (~30 s on 16 cores, incl. the
method check; `--no-check`, `--compare`, `--win`, `--band`, `--weight`,
`--click-k`, `--no-crosstalk`). Outputs go to `out/phase2/`.

## Result (defaults: 1 s windows on a 1 s grid, 2-28 kHz, SCOT, clicks blanked at 8x)
- 263 of 443 windows (796-1238 s buoy time) are good, in three spans:
  815-919 s (101), 983-1046 s (63), 1062-1164 s (97).
- No usable bearings in 920-982 s, 1047-1061 s or after 1165 s. 920-982 s:
  the boat is near endfire (az_array ~30-35 deg). That is along the wall
  line (the wall normal is at 119 deg), the response is smeared and the wall
  cross terms fall inside the physical lag range. 1047-1061 s: broadband
  transients. After 1165 s: the boat is weak; a faint track is only visible
  with 4 s windows.
- Coverage: az_array 34-140 deg. ENU bearing (alpha = -40.4 deg) runs
  -167 deg -> -75 deg -> -160 deg -> -180 deg, so the boat stays SW to W of
  the USBL, toward the sea.
- Precision:
  - formal sigma from the delay fit: 0.26 deg (median);
  - scatter about a local quadratic in time (+-7.5 s): **0.66 deg**;
  - `sigma` = sqrt(formal^2 + 0.66^2): median 0.71 deg, p90 1.0 deg.
  - Plane-wave rms is 1.6 us for good windows, against 12.7 us for the
    mirror solution, so the side is resolved per window as in phase 1.
- Consecutive windows are anti-correlated (lag-1 -0.33). Phase 1 showed the
  same (-0.5). Here the windows are continuous noise, not 1 s pings, so this
  is not a ping-sampling artefact. It fits a ~2 s array (or source) motion.
  2 s windows (overlapping, 1 s step) bring the scatter down to ~0.42 deg,
  and 4 s windows to ~0.27 deg (measured with c = 1513.5; the change of c
  does not matter here).

## Method check (pinger band, runs 1-2)
- The same windowed code was run on 30-40 kHz (no notch, no crosstalk
  removal; blanking on) on the 1 s grid 52.5-795.5 s. Each window was
  compared with the phase 1 per-ping `az_array` of the ping that lies wholly
  inside it.
- n = 468 (all good phase 1 pings), median |diff| **0.08 deg**,
  bias +0.02 deg, robust sigma **0.11 deg**; 1.9 % differ by more than 2 deg.
- This first exposed a **sound-speed inconsistency**. With c = 1513.5 and
  the phase 1 `pos`, the difference was bias -0.24 deg, ranging from
  +0.31 deg at az 40 deg to -0.37 deg at 150 deg, exactly the cos(az)
  scaling with c. `pos` was self-calibrated with c = 1520, and only pos / c
  is determined. The bearing therefore must use the calibration c (stored as
  `sound_speed` in both npz files). The graph's c (1513.5) does not belong
  here: the true c cancels.
- When no ping is in the window (USBL gap 309-394 s), the pinger band locks
  onto fixed lines at ~80 and ~95 deg. That is background (crosstalk plus
  something stationary), not the pinger.

## Choices, with evidence (`--compare`; good windows / scatter)
- Band:
  - 1-6 kHz: 75 / 1.00 deg;
  - 2-10 kHz: 175 / 0.99 deg;
  - 2-20 kHz: 255 / 0.84 deg;
  - 5-20 kHz: 266 / 0.78 deg;
  - **2-28 kHz: 266 / 0.66 deg**;
  - 5-28 kHz: 270 / 0.65 deg.
  - The boat is coherent across the array well above 20 kHz (coherence up
    to 30+ kHz at 1000 s), and the wider band sharpens the peak. 2-28 kHz
    keeps the low end for the weaker periods. The band stops below the
    pinger notch (29.5-40.5 kHz) and the anti-alias roll-off (~40 kHz).
  - Bearings in 2-20 kHz differ from the default by 0.25 deg median.
- Weighting: SCOT and PHAT are within noise (2-28 kHz: 0.66 against
  0.69 deg, 266 against 268 windows). SCOT is kept because its peak is the
  band-mean coherency, an interpretable quality number (`coh`).
- Click blanking matters most:
  - no blanking: 161 good windows;
  - k = 12: 237;
  - **k = 8: 263**;
  - k = 6: 271 (scatter 0.70);
  - k = 5: 278 (scatter 0.74).
  - Without blanking the median GCC coherency doubles (0.057 against
    0.028): the clicks are coherent across the array and take over weak
    windows. Blanking zeroes 0.3 % of the samples (p90 0.7 %).
- **Common-mode crosstalk** (new finding):
  - All 4 channels share a weak broadband component at exactly zero lag,
    also on the off-line ch1 (an acoustic broadside arrival would give
    tau01 = -11 us). It shows in every band up to 47 kHz, i.e. beyond the
    anti-alias roll-off, so it is electrical.
  - It is stationary over the whole file: Re S01 in 12-28 kHz is
    (1.1-1.4)e-15 at all quiet times, coherency ~0.07.
  - It is removed by subtracting the median over all windows of Re S_ij(f),
    smoothed over 1 kHz.
  - Without removal there are *more* "good" windows (286), but the extra
    41 are all at az = -90.7 deg. Zero delays fit best on the back side
    because of the ch1 offset. They form a neighbour-consistent fake track
    in 920-973 s and 1184-1213 s. Removal plus the front-side mask
    (sin az > 0) excludes them.
- Wall echo:
  - The direct x echo cross terms lie at tau +- (0.2-1.6 ms). Delays are
    taken at the local GCC maximum reached uphill from the steered-response
    prediction of the plane wave.
  - In the synthetic test with a -12 dB wall echo, the delays stay within
    5 us of the truth and the bearings within 1 deg.
  - On the data, the wall is a likely cause of the endfire gap (920-982 s),
    where the echo lag drops to ~0.2 ms.
- Estimator: the per-pair delays -> `usbl.bearing_ls` (as in phase 1) and
  the steered-response maximum (`az_srp`) agree to 0.05 deg median. The
  scatter is 0.63 against 0.65 deg, so `az_array` is the bearing_ls one.

## Output `out/phase2/bearings_noise.npz`
- Per window (K = 443):
  - `t`: buoy time of the window centre;
  - `t_usbl` = t + 102.821832 + d_S (t - 52.14);
  - `az_array` [rad]: same frame and convention as phase 1;
  - `az_srp`;
  - `sigma` [rad]: total; also `sigma_formal`;
  - `mask`;
  - quality: `coh` (SCOT peak, coherency), `snr` (GCC peak / off-peak rms),
    `contrast` (steered response), `rms` (plane-wave fit) [s], `az_mirror`,
    `rms_mirror`, `nb_resid` (vs neighbours), `az_resid` (vs local
    quadratic), `blanked`, `band_rms` (K, 4);
  - `tau` (K, 6) [s]: pair delays t_j - t_i in `pairs` order (the usbl.py
    PAIRS); `tau_peak`;
  - `az_enu`: alpha_graph + handedness * az_array.
- Globals:
  - copied from the phase 1 file: `handedness` (-1), `alpha` (-42.4 deg),
    `pos`, `sound_speed` (1520, the calibration c);
  - `alpha_graph` (-40.4 deg, from out/phase1/globals.txt);
  - `sigma_emp`, `band`, `win`, `weight`, `usbl_offset`, `usbl_drift`,
    `t_ref`;
  - method check: `check_t`, `check_az`, `check_az_p1`, `check_diff`.
- Graph use, the same as phase 1: meas bearing = handedness * az_array,
  modelled as atan2(p - usbl) - alpha. So theta_enu = alpha +
  handedness * az_array = alpha - az_array.

## Caveats
- `sigma` does not include the geometry systematic (phase 1 `sigma_sys`,
  0.3 deg near broadside, 1.2-1.8 deg near 40 / 150 deg). It is correlated
  over time and mostly absorbed by alpha.
- Windows are 1 s apart and not overlapping. Their errors are
  anti-correlated, so treating them as independent is slightly
  conservative.
- A boat noise source is not a point at the pinger position: the bearing is
  to the boat (propeller/engine), not to the towed body.
