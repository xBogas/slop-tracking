# G: multipath against the EMODnet bathymetry

Notebook: `notebooks/multipath_reflections.ipynb` (~25 s; the first run adds ~3 min for the
family-delay grid, cached in `out/multipath/family_grid_<rx>.npz`). Code: `processing/bathy.py`
(EMODnet DTM in ENU, isovelocity eigenrays over a sloping bottom). Tests: `tests/test_bathy.py`.
DTM: `data/bathymetry/emodnet_mean_vodice.tif` (EMODnet WCS `emodnet__mean`, 1/16 arcmin;
the source here is the `croatia2004` survey); `bathy.download_emodnet()` fetches it again.
`data/bathymetry/E6_2024.tif` (EMODnet DTM 2024 tile E6) has the same 1/16 arcmin grid and
is identical over water; it only masks 36 more coastal cells as no-data.

## Corrections to D

- Charted depth: 13.3 m at Rx902, 12.9 m at Rx01, 5.8-9.3 m under the pinger track
  (shoal of 6-7 m, ~330-570 s). The "water depth 24-27 m" in D is wrong: it read the late
  ridge as a single bottom bounce.
- The pinger cannot be at 10 m. (Correction, 2026-10-22: the hydrophones are 10 m above the seabed, ~3 m deep, not at
  10 m depth; the ray-tracing results below were computed for 10 m depth. See note I.)
- D's coherent fit used D_GRID = 12-45 m, which excludes the charted depths. Its depth
  "running to the shallow edge" fits a shallower true depth. The fit has not been re-run.

## What the data shows

- An early cluster (0.05-0.6 ms; S, B, SB) about as strong as the direct path: median
  ≈ 0 dB on Rx902 and +3 dB on Rx01, relative to the direct path.
- One discrete late arrival per buoy (0.5-1.7 ms), flat in tau R (a real 1/R path). Single
  chirps show it as interference fades every 10-25 ms along the sweep (none over the shoal).
  It is present only while the water under the pinger is deeper than ~7.5 m
  (Rx902: 0 % of pings under 7.25 m, 50-75 % deeper); it vanishes over the shoal (confounded
  with position).
- Ray tracing over EMODnet, limited to paths with at most two reflections (`bathy.MAX_BOUNCES`):
  only BS reaches these delays, and its best fit is at the grid edge (zs 1 m, zr 12.5 m) with
  log-delay misfit 0.33 (Rx902) / 0.48 (Rx01), delays off by 40-60 %. So the late arrival is
  either a higher-order path or not a specular surface/bottom path. (Allowing up to 5
  reflections, every segment was fitted by some family, with no common zs, zr.) The sloping
  bottom does not reduce the within-segment scatter below the 1/R model's (2-12 %).
- Phase 2 noise: no moving multipath lags (predicted < 1.5 ms, inside the stationary
  autocorrelation).
