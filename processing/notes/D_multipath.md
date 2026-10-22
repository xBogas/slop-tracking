# D: multipath ranging on the real data (phase 1, buoys)

Code: `processing/multipath_range.py`. Tests: `tests/test_multipath_range.py`
(5 tests, about 3 s). Outputs are in `out/multipath/`: `mp_range_phase1.npz`,
`mp_log.txt`, and the caches `arrivals_<rx>.npz` and `profile_<rx>_zr10.npz`.

```
uv run python processing/multipath_range.py          # ~10 s with the caches
uv run python processing/multipath_range.py --full   # + coherent fit profiles, ~25 min on 16 cores
```

## Verdict

**Multipath does not give an independent per-ping range on these recordings.**
`mp_range` is NaN and `mp_range_mask` is False everywhere. Do not add the
`mp_range` factor to the graph. There are two reasons, and each one is
enough on its own:

1. **The surface echo is a diffuse cluster, not a specular copy.** The
   0.05-0.6 ms behind the direct path hold a jumble of 3-6 fitted paths
   whose phases do not show a consistent sign flip (rough sea at
   30-40 kHz). The coherent image-model fit therefore crowds image paths
   into that cluster:
   - Rx902 ranges come out +157 % against the TOA track (median);
   - Rx01 ranges come out +50 %;
   - only 2 % (Rx902) and 9 % (Rx01) of pings are within 10 %;
   - the water depth runs to the shallow edge of the grid (13 m), with a
     profile that decreases monotonically above 16 m.
2. **Even with clean paths the range scale is not observable.** In the far
   field every image delay is a quadratic form in the depths divided by
   c R:
   - surface: 2 zs zr / (c R)
   - bottom: 2 (D - zs)(D - zr) / (c R)
   - surface-bottom: 2 D (D + zs - zr) / (c R)

   So (zs, D, zr, R) and (l zs, l D, l zr, l² R) give the same delays.
   With zr fixed, the direct, surface and bottom paths still leave a 1-D
   family (`equivalent_geometry`). Example at R = 400 m, zs = 7 m,
   D = 24 m: zs = 6 or 8 m with R = 343 or 457 m (D = 22.4 or 25.5 m)
   reproduces the surface and bottom delays to < 1 µs. Only the
   double-bounce paths (shifted 70-140 µs) break the family, and they are
   not seen consistently. Even in synthetic data with all six specular
   paths, the fit lands on the wrong family member in 3 of 8 noise draws at
   700 m and 30 dB (it needs about 40 dB). On the real data the model
   mismatch alone is at about -6 dB.

## What the data does show

Echograms are the compressed pulse aligned on A's TOA (`echogram`). There
is one discrete late arrival per buoy, intermittent, tracked by Viterbi
without any geometry (`track_ridge`). Coverage of the detected pings:

| | run 1 | run 2 |
|---|---|---|
| Rx902 | 47 % (3 segments) | 38 % (2 segments) |
| Rx01 | 40 % (2 segments) | 11 % (2 segments) |

Its constant K = tau R (against the TOA graph track), per segment:

- **Rx902:** 0.305, 0.43, 0.33, 0.36 m s. One more segment, at 0.156 m s,
  is the tracker sitting on the edge of the surface clutter.
- **Rx01:** 0.39-0.41 m s, and 0.77-0.84 m s in two other segments. The
  factor-2 ratio fits bottom vs surface-bottom at D ≈ 24 m, so the same
  tracker follows different paths in different segments.

The surface-cluster centroid gives K ≈ 0.09 m s on Rx902 in run 1. That
means **zs ≈ 6.8 m** for zr = 10 m (zs = K c / (2 zr)). It depends on the
track ranges.

**Water depth (conditional on the track ranges, zs = 6.8 m, zr = 10 m):**

- under Rx902: median 24.4 m (p10-p90 20.5-26.6 m);
- under Rx01: median 26.6 m (p10-p90 25.4-33.4 m; the upper tail is the
  misidentified path).

K drifts by ±15-20 % between segments. That is bathymetry (and zs), not
noise, so D varies by 2-3 m along the track. A sloping bottom is one more
reason why the bottom path cannot fix the range scale.

## Graph questions (Rx01 position, sound speed)

The scale-free test asks whether tau R is constant within each ridge
segment, so it needs no zs, D, zr or c. Results for the TOA graph track
(`out/phase1/track.csv`):

| Buoy | Position | Robust scatter of tau R | At median range |
|---|---|---|---|
| Rx902 | GPS (-338.5, -365.0) | 4.59 % | 17 m at 381 m |
| Rx902 | graph (-331, -365) | 4.59 % | 17 m at 379 m |
| Rx01 | GPS (-109.8, -702.8) | 6.26 % | 45 m at 717 m |
| Rx01 | graph (-116, -669) | 6.25 % | 43 m at 683 m |

A ±80 m scan of the buoy position changes the cost by less than 0.2 % of
range, with no minimum: 4.76-4.99 % for Rx902 and 5.65-5.72 % for Rx01.
**The multipath data supports neither the GPS nor the shifted Rx01
position.** It has no power here, because the scatter comes from K
varying with depth, not from geometry.

Sound speed: every delay scales as 1/(c R) times the unknown depths, so
multipath cannot settle c either.

## Spectrum vs lag vs delay fit

I used the coherent lag-domain fit with variable projection (complex
amplitudes as nuisances, bounded |a_p / a_d|). It is the exact Gaussian
likelihood of the multipath model and needs no received spectrum W.

The demo's power-spectrum fit carries the same delay information and
removes only tau0, which is also a nuisance here. It is subject to the
same degeneracy and the same diffuse-surface mismatch. **I did not run it
on the real data**, because the failure comes from the data and not from
the domain of the fit.

The per-ping free-path extraction (`sparse_paths`, OMP) shows the
problem directly: about 10 paths per ping above 0.3 % of the energy,
mostly in the first 0.6 ms.

## Sensitivities

On the surface path, R = 2 zs zr / (c tau_s), so R is linear in zs and in
zr, and goes as 1/c:

- 0.5 m of zs is 7 % of range;
- 1 m of zr is 10 % of range.

On the bottom path, d ln R / dD ≈ 1/(D - zs) + 1/(D - zr) ≈ 12 % per
metre of water depth.

## Runtime

- Extraction: 5 s for both buoys (12 processes).
- Coherent fit profiles: 25 min (16 cores, 31 x 34 (zs, D) x 540 R
  x 19 tau0 per ping).
- Ridge tracking and statistics: < 1 s.

## Output (`mp_range_phase1.npz`, rows = `toa_phase1.npz`)

The fields the graph expects, all flagged unusable:

- `k`, `receivers`;
- `mp_range`, `mp_range_sig` (K, 3): NaN;
- `mp_range_mask` (K, 3): False;
- `zs` (K,): NaN;
- `zr`: 10, nominal;
- `D` (2,): implied depths 24.4 and 26.6 m, conditional as above.

Diagnostics:

- `mp_tau_ridge` (K, 3): ridge delay in seconds, NaN outside segments;
- `mp_ridge_contrast`, `mp_ridge_seg`;
- `K_track_gps`, `K_track_graph`;
- `D_implied_<rx>`;
- `full_fit_range`, `full_fit_zs`: the failed coherent fit.

The USBL column is NaN: it was not attempted (shallow water, wall echo).

## Recommendations

- Leave `mp_range` out of the graph (or keep the mask False). Use a robust
  loss for the 1-2 % direct/surface flips that A reported.
- Settling Rx01 / c / timing needs other information:
  - buoy GPS logs;
  - the USBL bearing (agent C);
  - a known-range calibration, such as the start position or a pass close
    to a buoy, with D measured on a chart.
- To make multipath ranging work in a future run:
  - put a pressure logger on the pinger, which fixes zs and removes the
    degeneracy;
  - log the hydrophone depth;
  - use a bathymetry chart for D.

  With zs and zr known, the bottom ridge alone gives R at about the 5 %
  K scatter seen here (17 m on Rx902), provided D is taken from a chart.
