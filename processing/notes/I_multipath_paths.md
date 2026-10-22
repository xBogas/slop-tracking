# I: multipath as measurements (predicted paths, windows, association)

Notebook: `notebooks/multipath_paths.ipynb`. Code: `processing/multipath_predict.py` (paths from the
bathymetry, ~8 min on 16 cores, cached) and `processing/multipath_paths.py` (extraction, direct path,
association; output `out/multipath/paths_phase1.csv`). This tries the receiver design of `D2_multipath.md`
on the phase 1 pings of both buoys, without the tracker. Numbers are from the 2026-10-22 run.

## Method

- Eigenrays over the EMODnet profile at the track positions, at most two reflections (S, B, SB, BS). The
  delay window per family is the range over pinger depth 1–5 m, hydrophone depth ±2 m around the
  configured one (placeholders in `multipath_predict.py`), and the water level: the predicted tide
  (`[site] tide`, +0.02 m on 2026-10-20) ± `tide_sigma` (0.2 m, unmeasured surge).
- Doppler-matched compressed pulses (`multipath_range.extract`; η from the arrival sequence). Free OMP
  extraction: up to 15 components on the measured pulse model.
- False-alarm level: the 95th percentile of components found before the direct-path window, relative
  to the strongest. Direct path: the earliest component above it within ±1/B of the TOA the phase 1
  solution predicts (`state.npz: toa_pred`).
- A component is usable when it is unique (one family window holds it, and no other component is in
  that window), resolved (≥ 1/B after the direct path, ≥ 1/(2B) from its neighbours) and persistent
  (≥ 2 of the ±2 neighbouring pings at the predicted delay change ±30 µs).

## Results

- S and B arrive 0.02–0.14 ms after the direct path, below 1/B = 0.1 ms on most pings, because the
  hydrophones are 3 m above the seabed. They are never usable. SB and BS are clear of the direct path,
  but their windows are 0.25–0.4 ms wide (unknown depths).
- The flat-bottom image method misses the bottom families by 40–230 µs, so the sloping seabed matters.
  The cross-path slope (< 1°) changes B by ≤ 0.02 µs, so a 3D model is not needed.
- Coupling: 0.26 ms per m/s (100 ms chirp). Up to 1.3 ms bias without Doppler matching; ≤ 4 ns per path
  after the common time scale. Paths differ in Doppler by ≤ 0.4 Hz.
- A reflection is stronger than the direct path on 28 % (Rx902) and 42 % (Rx01) of pings.
- **46 of 740 chirps have ≥ 1 usable delay** (Rx902 36, Rx01 11). 9 chirps have two families on one
  buoy. Usable: SB 23 + 8, BS 22 + 3, B 0, S 0. Ping-to-ping scatter on Rx902: 5 µs (SB) and 15 µs (BS);
  CRLB 0.3 µs. (With the data-derived run times and the 2026-10-20 tide; before them: 45 chirps.)
- No common geometry. Each family alone fits one (zs, zr, water level) to 7–18 µs, but at different depths;
  together 31 µs on Rx902. The "BS" delays are the persistent late arrival near 0.78 ms (pings 18–53), and
  they fit BS only with the pinger at 1 m, the edge of the grid (see G). Only the SB delays are candidate
  measurements, and they constrain only the zs·zr product (the scale ambiguity of D).

## Update: hydrophones 10 m above the seabed (2026-10-22)

The operator corrected the hydrophone depth: 10 m above the seabed, i.e. 3.28 m (Rx902) and 2.65 m (Rx01) deep
with the tide. The numbers above are for 10 m depth. Rerun:

- S is now 0.01–0.04 ms behind the direct path (never separable); B 0.06–0.31 ms (above 1/B on 83–86 % of
  pings); SB 0.23–0.61 ms; BS 0.09–0.50 ms. B, SB and BS windows overlap; BS is never unique.
- **19 of 740 chirps usable** (Rx902 1, Rx01 18: B 10, SB 6, S 2; the 2 "S" sit in the window margin, far from the
  22 µs prediction). No chirp has two families.
- The 10 Rx01 bottom bounces measure 170–188 µs against 306 µs predicted at the configured depths; the best common
  geometry is a 5 m pinger and 4.6 m hydrophone (grid edge, 10 µs misfit vs 117 µs configured). SB alone fits the
  configured depths to 10 µs. The bottom bounces point to a pinger deeper than 3 m; the 1–5 m grid is too narrow.
- Rx902's persistent late arrival near 0.77 ms lies beyond every path with at most two reflections.
- Tracking is unchanged by the depth correction (held-out TDOA 67 / 59 µs; c 1521.6 m/s; Rx01 still ~28 m north).
