# C: USBL array, wall, bearings (phase 1)

Code `processing/usbl.py` (method in the module docstring), tests
`tests/test_usbl.py` (3 synthetic tests, pass). Run:
`uv run python processing/usbl.py` (~15 s). It uses A's `out/toa_phase1.npz`
ping list when it is there (`--pings own` runs the standalone finder instead).
Outputs go to `out/usbl/`.

## Channels
- No clipping: the largest sample in the whole file is 0.03 FS. No channel is dead.
- ch2 is noisier: its in-band noise PSD is -129 dB, against -139/-137/-136 dB
  for ch0/1/3 (its broadband RMS is 9 dB higher).
- Direct-path amplitude is the same on all channels within 1.3 dB (ch3 is
  -1.3 dB), so the gains match.
- MF SNR median (p10) in dB: ch0 38 (30), ch1 35 (26), ch2 27 (19), ch3 34 (26).
  The SNR changes with bearing: in run 2 ch2/ch3 drop to about 15-20 dB.
- Polarity: a sign flip would show as a half-cycle (14 us) misfit. None is
  seen: the plane-wave residual is 0.44 us rms.

## Array layout and spacing (array frame: x from ch0 to ch3, y toward the sources)
- Rank. The singular values of the delays relative to ch0 are 299 / 6.1 /
  0.6 us rms. The array is almost a straight line, with the channels in
  order 0-1-2-3 along it. The data are 2-D, so elevation cannot be observed.
- Positions (m): ch0 (0, 0), ch1 (0.189, 0.0165), ch2 (0.313, 0.000), ch3 (0.505, 0).
- Pair distances (bootstrap + sound speed ±10 m/s):

  | pair | distance |
  |---|---|
  | d01 | 0.190 ± 0.005 m |
  | d12 | 0.125 ± 0.002 m |
  | d23 | 0.191 ± 0.004 m |
  | d03 | 0.505 ± 0.011 m |

- Other checks on the scale:
  - The scale profile has its minimum at 0.50-0.51 m.
  - Hard endfire bound: max|tau03|·c = 0.451 m.
  - A ±2.6 us/channel ADC skew would move the length only to 0.495-0.516 m.
- **The "15 cm between each channel" does not hold.** The ratio of the
  positions along the line is 1 : 1.66 : 2.67 (0 / 18.9 / 31.3 / 50.5 cm).
  This ratio does not depend on c or on the scale. Uniform spacing would give
  1 : 2 : 3. The total length is 45-55 cm, not 45 cm with uniform steps.
  A's "~45 cm" is the endfire lower bound.
- ch1 sits 1.65 cm off the line. That small offset is what fixes the scale
  and resolves the per-ping left/right side.
- Assumption: no per-channel timing bias. Free biases trade with the scale
  under half-plane coverage, so they cannot be fitted.

## Wall
- **The echo is there, but the reflector is about 1.0-1.25 m away, not 30-80 cm.**
- A single plane-reflector model fits the echo trace well across all
  bearings and channels (see `wall_echo.png`):
  - normal 119.2 ± 0.5° in the array frame;
  - distance from each element: ch0 1.25, ch1 1.17, ch2 1.10, ch3 1.00 m (±0.01 m).
  - So the array line is at about 30° to the wall, with ch3 nearest.
- Echo amplitude is -12 dB re direct (median 0.25; background 0.07).
- Lag: 0.2 ms near azimuth 40° (grazing on the wall), rising to 1.64 ms
  near azimuth 120°.
- Effect on the measurements:
  - Changing the gate from ±0.05 to ±0.4 ms moves the bearings by < 0.01° (median).
  - The direct path is always earliest, so the first-arrival TOA is not biased.
  - 54 pings (run 1 start, azimuth ≈ 40°) have a lag < 0.3 ms. They are flagged
    in `wall_overlap`. Their scatter is slightly higher (0.95° against 0.7-0.8°).
- The first post-peak sidelobe is somewhat raised at 0.14-0.3 ms. That may
  be a weak near reflection (mount or surface); it is not resolved.
- The wall does not resolve the frame handedness. It lives in the same 2-D frame.

## Bearings
- 468 of 552 pings are good (`mask`). Rejected: plane-wave rms > 4 us, more
  than 5 sigma off the local track, on the mirror side, or isolated.
- Cycle slips (1/fc = 28.6 us) affect about 30 % of the raw pings and are
  repaired by `resolve_cycles`.
- Precision:
  - The formal sigma from the delay fit is 0.1°.
  - The truth-free scatter about a local quadratic in time is **0.85° rms**.
    It is flat in bearing and SNR, and the same with my own ping times,
    without the Doppler scale, or with a narrower gate. So it is in the data,
    not in the processing.
  - Consecutive pings are anti-correlated (lag-1 ≈ -0.5, a spectrum rising
    toward 0.5 Hz). That looks like ~2-3 s motion (waves; pinger or array)
    aliased by 1 s pings. It is not resolved.
  - `sigma` = sqrt(formal² + 0.85²), median 0.86°.
- Geometry systematic (`sigma_sys`, from the bootstrap; correlated over
  pings): 0.3° near broadside, 1.2-1.8° at 40° / 150°.
- Coverage in the array frame:
  - run 1: 39° to 148° (it sweeps through broadside);
  - run 2: 122° to 153° (within 27-58° of endfire).
  - Nothing beyond 155° or below 38°.
- Mirror. A pure line array cannot tell left from right. The ch1 offset
  makes the mirror solution fit at 9.6 us rms against 0.44 us, so the side
  is resolved per ping. The frame handedness (whether the frame is seen from
  above or below) cannot be seen in the delays.
- Orientation, from A's TOA-only track (`out/phase1/track.csv`):
  `az_enu = alpha + s * az_array` with alpha = -42.4° and **s = -1**.
  - Robust residual: 2.3° for s = -1 against 23° for s = +1, so the handedness is clear.
  - Residual against the track: run 1 median -2.4° / spread 1.2°; run 2 +1.0° / 1.7°.
    That is consistent with the geometry systematic plus track error.
  - So the array axis ch0 -> ch3 points to ENU -42.4°. For s = -1 the array's
    +y points to ENU azimuth alpha - 90° = -132° (SW): it faces the buoys and
    the sea. The wall normal maps to ENU -161°.

## Output `out/usbl/bearings_phase1.npz`
- Per ping (K = 552 USBL pings from A's table):
  - `k`: A's ping index;
  - `mask`: good ping;
  - `az_enu` [rad]: atan2(n, e) from the USBL, calibrated;
  - `az_array` [rad];
  - `sigma` [rad]: total; also `sigma_noise` (formal) and `sigma_sys` (geometry);
  - `quality` (0..1), `rms`, `az_mirror`, `rms_mirror`;
  - `t_usbl`: array-mean direct arrival, USBL clock;
  - `t_buoy` = t_usbl - 102.81372;
  - `t_ch` (K, 4): per-channel arrivals;
  - `tau` (K, 6): resolved pair delays t_j - t_i, with `pairs`; also `tau_raw`,
    `tau_coarse`, `cycle_margin`;
  - `snr` (K, 4), `scale`, `wall_lag` (K, 4), `wall_amp`, `wall_overlap`;
  - `el`: all NaN (not observable).
- Globals:
  - geometry: `pos` (4, 2), `pos_boot`, `dist`, `dist_sigma`, `sound_speed`,
    `prof_L`, `prof_cost`, `L_skew`;
  - wall: `wall_h0`, `wall_phi`, `wall_h`, `wall_h_sigma`;
  - orientation: `alpha`, `handedness`;
  - `sigma_emp`, `usbl_offset`, `ping_source`.

## Recommendations for the graph
- Bearing factor: wrap(az_enu_pred - az_enu) with sigma about 1.0° and a Huber loss.
- Better: use `az_array` with a global alpha (prior -42.4 ± 3°) and s = -1
  fixed. Then the orientation is estimated jointly and does not depend on
  A's first track.
- The geometry error (1-2° at the coverage edges, smooth in bearing) is
  mostly absorbed by alpha. If that is not enough, refit alpha per run.
- No elevation information.
- Do not use a pure-delay (TDOA) model with the nominal 15 cm spacing.

## Operator information (2026-10-21)
- Spacing 15.5 cm between channels. A fixed straight 15.5 cm array fits the
  pair delays at 10.7 us rms (sound speed free), 9.7 us with free
  per-channel timing biases, and 60-145 us with any other channel order.
  The self-calibrated geometry fits at 0.44 us. ch2 sits where 15.5 cm puts
  it (31.3 vs 31.0 cm); ch1 and ch3 are 3-4 cm further out. Kept the
  self-calibration; `element_spacing` in forfun.toml is the nominal value only.
- Axis ch1 -> ch4 (= ch0 -> ch3) at 128° magnetic, so ~133.4° true with a 5.4° E
  declination, ENU -43.4°. Independent of the data: usbl.py gives -40.1 to
  -40.8° and the graph -40.5°. The gap (~3°) is within a compass reading on a
  mount. `run_phase1.py` uses it as the alpha prior (±3°, placeholder).
- GPS antenna 5 m from the array centre. The direction is not given, so it is
  set seaward and normal to the array (223.4° true). The pinger
  track moves ~4.8 m with it (median); the residuals do not change.
