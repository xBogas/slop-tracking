# E: TDOAs from the boat's radiated noise (phase 2)

Code: `processing/noise_tdoa.py` (method in the module docstring). Tests:
`tests/test_noise_tdoa.py` (4 synthetic tests, ~3 s). Run:
`uv run python processing/noise_tdoa.py` (~30 s, 16 cores) and
`uv run python processing/noise_tdoa.py --pinger` (~2 min).
Outputs go to `out/phase2/`.

Convention: tdoa(i, j) = arrival at i minus arrival at j = (|p - r_i| - |p - r_j|)/c,
in buoy time, with the USBL clock (102.821832 s at 52.14 s, -87.4 ppm) removed.
Pairs: (Rx902, Rx01), (Rx902, USBL), (Rx01, USBL). The pinger check confirms
the sign and the clock model (see below).

## Result (buoy time 796-1020 s, 221 windows of 4 s at a 1 s step)

| pair | detected windows | spans [s] | random sigma (scatter) |
|---|---|---|---|
| Rx902-Rx01 | 153 | 807-926, 984-1018 | 0.22 ms |
| Rx902-USBL | 79 | 834-881, 890-902, 988-1017 (fragmented) | 0.85 ms |
| Rx01-USBL | 79 | same windows | 0.88 ms |

- The buoy pair runs from -195 ms (807 s) to -10 ms (926 s), then from 0 to -160 ms
  over 984-1018 s. There is no trace in 926-984 s.
- All USBL values come from the joint closure search, which needs a buoy-pair
  detection. No USBL-only trace passed the stricter threshold.
- Closure tdoa01 - (tdoa02 - tdoa12), with the USBL pairs re-peaked independently
  within +-5 ms of the joint lag: n = 97, median -0.34 ms, MAD 0.45 ms. The gate
  (> 4 MAD and > 1 ms) rejected the USBL pairs in 20 windows, mostly at 984-1018 s.
  After the gate: n = 77, median -0.33 ms, MAD 0.35 ms. Widening the re-peak
  window to +-10 ms leaves the core unchanged (MAD 0.43 ms), so the closure is a
  real check. Caveat: any error that belongs to one receiver (its path structure,
  the USBL's own noise) cancels in the closure. The closure therefore checks
  association and correlation noise, not timing. The USBL errors in tdoa02 and
  tdoa12 are correlated through the common USBL term.
- `tdoa_sig` = hypot(random, 2 ms systematic). The median is 2.0-2.2 ms, so the
  systematic term dominates (see the method check). `tdoa_sig_rand` holds the
  random part on its own.

## Method check on the pinger (`--pinger`, 52-796 s, same code, 30-40 kHz)

Measured minus predicted from `out/phase1/track.csv` (pinger depth 10 m, USBL
depth 2 m, c 1513.5 m/s):

| pair | det % | bias | MAD | random part | > 10 ms |
|---|---|---|---|---|---|
| Rx902-Rx01 | 100 | -0.52 ms | 2.96 ms | 0.99 ms | 4 |
| Rx902-USBL | 89.5 | -0.04 ms | 1.69 ms | 0.87 ms | 25 |
| Rx01-USBL | 89.5 | +0.61 ms | 2.20 ms | 0.99 ms | 22 |

- The sign convention and the USBL clock model are confirmed: the bias is below
  0.6 ms on every pair, including both USBL pairs.
- The error has a slowly varying part of +-2-3 ms that changes with geometry
  (about +2 ms in run 1 and -2 ms in run 2 on the buoy pair). On top of it sits
  ~1 ms of random error. The likely cause: the envelope correlation peaks at the
  energy centroid of the multipath arrivals, not at the direct path. The
  synthetic test shows the same thing: copies closer than the envelope
  resolution (a few ms) bias the peak. This is why SIG_SYS = 2 ms. Whether the
  noise band (2-20 kHz, a continuous source) has the same systematic error
  cannot be checked without truth. Treat 2 ms as a conservative placeholder;
  it is the parameter `Params.sig_sys`.
- The > 10 ms errors on the USBL pairs fall in 309-410 s. Phase 1 had no USBL
  pings there, so the reference track is weak at those times. The envelope trace
  there differs from the track by up to 15 ms.
- Sigma calibration: the truth-free random sigma matches the truth-based random
  part (z = 0.72-0.82), and the total sigma gives z = 0.7-1.3.

## Choices and the numbers behind them (noise, 796-1020 s)

Figure of merit: the median single-window z on a reference trace (taken from the
stacked correlogram), and the percentage of windows where the trace beats the
best off-trace peak. Values are listed as buoy / 902-USBL / 01-USBL.

| variant | on-trace z | trace wins |
|---|---|---|
| start: 3 bands, modulation 2-1000 Hz, beta 0.5 | 4.6 / 4.7 / 4.5 | 85 / 81 / 67 % |
| one 2-20 kHz band | 3.3 / 4.6 / 3.8 | 37 / 68 / 47 % |
| click blanking off | 4.1 / 2.5 / 3.2 | 17 / 3 / 5 % |
| whitening beta 0 / 1 | 4.4 / 3.9 (buoy) | 86 / 66 % (buoy) |
| window 8 s | 3.7 / 4.5 / 4.5 | 56 / 65 / 73 % |
| modulation band 2-200 Hz | 5.1 / 5.5 / 5.0 | 97 / 95 / 87 % |
| 6 log bands + 2-200 Hz (**used**) | 5.7 / 5.6 / 5.2 | 97 / 90 / 87 % |

Further results:
- 10 bands give little over 6. An envelope rate of 2 kHz equals 4 kHz; 4 kHz is
  kept for the sub-bin resolution. Using all 4 USBL channels instead of ch0 helps
  little (+0.1 z).
- Longer windows lose because the delay moves by up to 5 ms/s within a window.
  Two steps fix this:
  - Track-before-detect: a stack along straight lines over +-4 windows lifts
    the path z from 4-6 to 8-13 for detection.
  - Delay-rate warping: each window is re-correlated with envelope b resampled
    at the local delay rate. On the synthetic 3 ms/s drift it cuts the error
    from 2.0 ms to 0.04 ms (median) and raises z from 5.5 to 19.7. On the real
    buoy pair, the truth-free scatter drops from 1.7 ms to 0.22 ms. It does
    nothing on the pinger, whose delays are slow and whose error is systematic.
- The probe's "broadband transient at 985-990 s that lights up all lags" is the
  boat's power ramp (the envelope power rises ~10x over 20 s). Cutting the
  envelope below 2 Hz and z-scoring each window removes it (< 3 % of lags above
  z = 3). The power gate (GATE_K = 10) never triggers on this data.
- MIN_RUN = 5 removes an isolated 4-window detection at 939-942 s (z 6-7,
  -40 ms) on the buoy pair, which is probably clutter.

## Outputs

`out/phase2/tdoa_noise.npz`:
- Main keys: `t` (K,) window centres (buoy time), `names`, `pairs` (names),
  `pair_idx` [[0,1],[0,2],[1,2]], `tdoa`, `tdoa_sig`, `tdoa_mask` (K, 3).
  `tdoa` is NaN where the mask is False.
- Per-window diagnostics: `tdoa_sig_rand`, `sig_raw`, `snr` (stacked or joint z
  used for detection), `snr_window` (z of the warped single-window peak),
  `source` (1 = own trace, 2 = joint closure trace).
- Calibration and closure: `sig_scale`, `scatter_rms`, `closure_rejected`.
- Per-pair traces before the joint search: `tdoa_indep`, `snr_indep`,
  `mask_indep`. Joint search: `tdoa_joint02`, `snr_joint`.
- Correlograms: `gate`, `corr_XY` / `stack_XY` (K, L) z-scored single-window and
  stacked correlograms, `lags_XY` [s].
- Run settings: `params`, `clock`.

Other files:
- `tdoa_noise.png`: stacked correlograms with the detections.
- `tdoa_pinger_check.txt`, `.png`, `.npz`: the same keys plus `pred`.

Windows overlap (4 s windows at a 1 s step), so neighbouring values share data.
For a tracker, use every 4th window, or inflate sigma by about 2.

- Fragile: the picks depend on the lag-window widths. Updating `C_SOUND`
  1513.5 -> 1521.4 (plus B_S/D_S and a 5 m USBL move in `POS`) moved TDOA
  picks by up to 9 ms on all pairs, even Rx902-Rx01. The phase 2 held-out
  Rx902-Rx01 error went from 0.21 to 2.9 ms. Those constants were reverted;
  they are now stale against phase 1 (USBL is at (-3.4, -3.6), c = 1521.4).
