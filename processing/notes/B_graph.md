# B: factor graph, simulator, tow inversion

Files: `estimation/fg_tracker.py`, `estimation/demo.py` (restored), `estimation/current.py`
(one-line fix for 7 globals), `motion_model/boat_model.py`, both READMEs,
`tests/test_fg.py`, `tests/test_tow.py`.

## API for integration (run_phase1.py)

- Receivers: index 0..nb-1 buoys (Rx902, Rx01), index nb = shore (USBL). `Geometry.shore` (3,),
  `buoys_nominal` (nb, 2), `buoy_depth`, `src_depth`.
- Globals `g = [b_S, alpha, skew, c, cx, cy, d_S]`, `fg_tracker.GLOBALS`, `fg.iG[i]`.
  `Config.g_mean` / `g_sig` must have 7 entries (ValueError otherwise). d_S prior 50 ppm.
  Shore clock at epoch k: `b_S + d_S * k * Tp`. With A's coarse USBL shift, b_S is the residual:
  set `g_sig[0]` to the coarse-offset uncertainty (default 0.2 ms).
- `meas` keys: `toa`, `toa_mask`, optional `toa_sig` (K, nr) [s]; `bearing`, `bearing_mask` (K,)
  optional; `mp_range`, `mp_range_mask`, `mp_range_sig` (K, nr) [m] optional (shore column can be
  masked); `rr`, `rr_mask`, `rr_sig` (K, nr) [m/s] optional, sign: d|p - r|/dt, positive = opening.
  NaNs are fine where masked. Missing sigma arrays fall back to `sig_toa`, `sig_mp` (5 m), `sig_rr` (0.1 m/s).
- `Config.start_prior = (x, y, sigma)` or None. `Config.region = (xmin, xmax, ymin, ymax)` or None
  (receivers' bbox ± 1.5 km), `grid_step` 25 m, `reseed_gap` 10, `u_max` 15 m/s.
- `Config.loss = "cauchy"` (default: Huber warm start, then Cauchy on measurement rows) or `"huber"`.
- After `initial_guess(fg)`: `fg.init_fixes` (K, 2) raw single-epoch fixes (NaN = none).
- Boat track: `boat_model.boat_from_towed_body(P, V, t, L=5, stern_offset=3) -> (tow_point, centre)`.

## Findings

- Huber on the priors (old code) let c run to 79 m/s on one seed; priors, chain and buoy drift are
  Gaussian now. Huber on one-sided outliers (late multipath) biased the buoy-line + track translation
  by 30-40 m; the Cauchy stage fixes it (pinger mean RMSE over 8 seeds 8.8 m, was ~22 m with outliers).
  Expect the same on real data: late multipath arrivals are one-sided.
- Weakest mode: common translation of the buoys and the track. Anchored only by the shore TOA
  (b_S prior) and the bearing. Buoy GPS or a tight b_S prior helps most.
- Simulation seed 1: pinger 11.5 m / 0.26 m/s (with mp + rr: 8.1 m / 0.14 m/s); noise 5.2 m / 0.39 m/s.
- d_S: recovered 40.6 ± 2.0 ppm (true 40) without buoy wander. With fast wander (3 m, 200 s) the
  pinger d_S scatters ±15 ppm vs a claimed 2 ppm: q_buoy 0.05 is too stiff for that wander.
  Tune q_buoy on real data (0.15-0.3 gave honest sigmas in simulation).
- Speed: vectorized CT (sinc form, smooth through omega = 0, which also fixes a zero FD derivative
  w.r.t. omega at omega = 0), COO sparsity build, own grouped FD Jacobian (~16 residual calls),
  symmetric-mode SuperLU with MMD ordering (7x faster than COLAMD). K = 300: pinger 3 s, noise 1 s;
  K = 600: pinger 8.5 s, noise 1.7 s (init + solve).
- Far start (track 1.1 km from the old seed, first 20 epochs and a 50 s stretch empty): the grid seed
  lands in the truth basin (17.7 m vs 18.0 m from a truth-started solve, outside the gaps).
  Inside long gaps only the motion model acts, and its path is not unique.
- Tow inversion: straight legs tow point < 1 mm, boat centre < 0.1 m; turns: centre up to 1.1 m.

## Open

- q_buoy and the motion noise are placeholders; tune on real data (chi2/dof, held-out pings).
- No analytic Jacobian (FD is fast enough at K <= 800).
- The range-rate factor assumes static receivers and the same sound speed as A's Doppler conversion.
- Phase 2 at long range (track ~1.3 km out) is information-limited (TDOA range is weak):
  ~270 m MAP error in simulation without multipath ranges.

- Huber warm-start tolerance 1e-3 -> 1e-4 (`FactorGraph.solve`). With the
  3° alpha prior, the Huber stage stopped on a plateau after 3 iterations
  (cost 22e3, 232 Rx902 outliers) instead of reaching ~3.3e3. Cauchy then
  crawled to a different minimum: the same track except pings 280-359 (end of
  run 1, no bearings, < 2 TOAs per ping), moved up to 44 m. In that minimum the
  lone USBL TOA at k = 342 was rejected at 483 sigma. That segment is weakly
  determined.
