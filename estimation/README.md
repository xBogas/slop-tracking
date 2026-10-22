# Factor-graph estimator for the boat track

Post-processed batch smoother: every unknown over the whole run is solved jointly as one sparse nonlinear least-squares problem (the MAP estimate of a factor graph). Plain numpy/scipy, no GTSAM needed; the solver is Levenberg-Marquardt with a sparse direct solve (symmetric-mode SuperLU, minimum-degree ordering), a grouped finite-difference Jacobian, a robust loss for outliers (Huber, then Cauchy on the measurements), and covariance from the Gauss-Newton information.

| File | What it is |
|---|---|
| `fg_tracker.py` | Variables, factors, solver, initialisation, marginal covariance |
| `demo.py` | Simulates phase 1 (pinger) and phase 2 (radiated noise) on the motion-model truth track and solves both |
| `current.py` | Water current two ways: as a graph variable, and as a grid sweep with profile-likelihood uncertainty |
| `demo_*.png`, `current_*.png`, `demo_*_estimate.csv` | Outputs |

Run from the repository root: `uv run python estimation/demo.py` (about 15 s), `uv run python estimation/current.py` (a few minutes, mostly the sweep). Tests: `uv run pytest tests/test_fg.py`.

## The graph

Variables (per epoch k = one ping in phase 1, one 1 s window in phase 2):

- `Z[k] = [x, y, chi, U, omega]` emitter state, from the motion-model thread's coordinated-turn model (`ct_predict`, `ct_process_noise` are imported, not copied).
- `T[k]` ping emission time (phase 1 only). It is unknown, but the pinger clock is regular, so a chain factor ties `T[k+1] - T[k] = Tp (1 + skew)`.
- `B[k, j]` position of buoy j. Random-walk drift factor, plus a mooring prior (15 m) on the first epoch.
- Globals `g = [b_S, alpha, skew, c, cx, cy, d_S]` (`fg_tracker.GLOBALS`; `Config.g_mean` / `g_sig` have 7 entries): shore clock offset `b_S` (prior from the sync signal; on real data the residual after the coarse USBL offset), shore-array orientation error `alpha`, pinger clock skew, sound speed `c` (prior 3 m/s, from a CTD cast), water current `(cx, cy)`, shore clock drift `d_S` (prior 50 ppm). The shore clock at epoch k is `b_S + d_S t_k` with `t_k = k Tp` (equal to `T[k] - T[0]` up to skew x drift).

Factors:

- Motion: `Z[k+1] = f_CT(Z[k]) (+ current * dt)`, whitened by the CT process noise.
- Phase 1, TOA: `arrival_i = T[k] + |p_k - r_i| / c + b_i(k)`. Using TOA with `T[k]` as a variable is equivalent to TDOA but keeps every receiver's information, and the ping chain turns TOA into near-ranges. Per-entry sigmas via `meas["toa_sig"]` (the USBL is weaker), else `Config.sig_toa`.
- Phase 2, TDOA: `(|p - r_i| - |p - r_j|)/c + b_i(k) - b_j(k)` for each receiver pair.
- Bearing from the 4-channel shore array: `atan2(p - shore) - alpha`.
- Optional multipath range (single-receiver horizontal range, e.g. from `multipath/`): `R_ik = |p_k - r_ik|_horizontal`.
- Optional range rate (Doppler): `rr_ik = (p_k - r_ik) . v_k / |p_k - r_ik|`, 3-D slant geometry, horizontal emitter velocity over ground (plus current with `motion="water"`), receivers static. Sign: positive = opening.
- Optional start prior on `Z[0, :2]` (`Config.start_prior = (x, y, sigma)`).
- Missed detections are simply missing factors (epochs with no receiver at all are fine); multipath and wrong correlation peaks are handled by the robust loss.

Measurement dict (`meas`), all per epoch:

| Key | Shape | |
|---|---|---|
| `toa`, `toa_mask`, `toa_sig` (opt.) | (K, nr) | phase 1, nr = nb + 1, shore last |
| `pairs`, `tdoa`, `tdoa_mask` | list, (K, npairs) | phase 2 |
| `bearing`, `bearing_mask` | (K,) | optional |
| `mp_range`, `mp_range_mask`, `mp_range_sig` (opt.) | (K, nr) | optional, `Config.sig_mp` default 5 m |
| `rr`, `rr_mask`, `rr_sig` (opt.) | (K, nr) | optional, `Config.sig_rr` default 0.1 m/s |

Robust loss: Huber (3 sigma) on measurement and motion rows first, then Cauchy on the measurement rows (`Config.loss = "cauchy"`, the default; `"huber"` gives the old behaviour). Priors, the ping chain and buoy drift are always Gaussian. Both changes matter: a Huber prior grows only linearly beyond 3 sigma, so it could not hold the sound speed (one run drifted to c = 79 m/s), and Huber clips the pull of each outlier to a constant, so one-sided outliers (late multipath arrivals) still shifted the weakly observable "buoy line + track" translation by 30-40 m. Cauchy lets their pull fade.

## Initialisation

`initial_guess` makes single-epoch fixes with nominal buoys and prior globals, rejects jumps against a running median, interpolates gaps and differentiates a smoothed track for course and speed. Each fix is a small robust least squares from the previous fix. A coarse grid search over `Config.region` (default: receivers' bounding box +- 1.5 km, step `grid_step` 25 m, all sigmas inflated to the step) seeds the first fix, re-seeds after `reseed_gap` epochs without a fix, and replaces local fixes that fit badly or jump further than `u_max` allows. The grid cost has a soft pull to `start_prior` (first fix) or to the last fix (sigma = reachable distance), which resolves the two-solution ambiguity when only two constraints are present. Phase 1 runs twice: emission times profiled out, then a robust line through them gives the ping chain (skew bounded to 2 prior sigmas), and the fixes are redone with known emission times. The raw fixes are in `fg.init_fixes`.

## Results on simulated data (seed 1, 300 s track)

Simulated (`demo.py`): 2 % outliers (late arrivals for TOA, either sign for TDOA, uniform bearings) and 5 % missed detections, buoys 10 m off nominal plus +-3 m wander (200 s period), array rotated 4 deg, sound speed 1506 m/s vs 1500 assumed, shore clock offset 0.15 ms, pinger skew 5 ppm. Geometry as in `multipath/`: shore array at the origin (2 m deep), buoys nominally at (+-600, 250), track 350-800 m offshore. Phase 2's emitter is the propeller, 3 m behind the boat centre, at 1 m depth.

| | Single-epoch fixes | Smoother | Velocity RMSE | Mean of seeds 1-8 |
|---|---|---|---|---|
| Phase 1, pinger | 16.8 m | **12.6 m** | 0.24 m/s | 8.8 m, 0.25 m/s |
| + multipath ranges + range rates | 13.4 m | **7.7 m** | 0.17 m/s | 6.2 m, 0.13 m/s |
| Phase 2, noise | 44.5 m | **5.1 m** | 0.38 m/s | 6.6 m, 0.42 m/s |
| + multipath ranges | 21.5 m | **5.5 m** | 0.34 m/s | 6.1 m, 0.37 m/s |

Timing (one core-ish, includes init): K = 300 pinger 3 s, noise 1 s; K = 600 (0.5 s pings) pinger 8.5 s, noise 1.7 s.

The numbers differ from the earlier version of this table (7.7 m / 22.6 m), whose simulator was lost; `demo.py` is a reconstruction with the same stated error budget, so treat both as one realisation each. Seed 1 is one of the worse pinger runs (range 5.5-11.5 m over 8 seeds). The truth track is now kinematic (`motion_model/demo.py`, within 34 m of the earlier 3-DOF track); the seed-1 columns are rerun on it, the 8-seed means and the current results below are from the earlier track.

What limits accuracy is the receiver geometry, not the measurements. With the buoy positions known to 0.5 m, error drops to about 5 m (pinger) and 3-4.5 m (noise). The weakest mode is a common translation of the buoy line and the track: buoy-buoy ranges do not change, and only the shore TOA (through `b_S`, prior 0.2 ms = 30 m) and the bearing anchor it. Buoy position, sound speed, shore clock offset and drift trade off against each other (`b_S` came out -0.10 ms against 0.15 ms true). Getting GPS on the buoys, even only at deployment, is worth more than any estimator tuning. The graph's own 1-sigma (about 6-15 m) is consistent with the observed errors to within about 2x.

Shore clock drift: with an injected 40 ppm and no buoy wander, `d_S` is recovered as 40.6 +- 2.0 ppm (pinger) and 37 +- 16 ppm (noise). With the +-3 m / 200 s wander the pinger estimate scatters by +-15 ppm while the graph claims 2 ppm: the wander is faster than the buoy random walk (`q_buoy` 0.05 m/sqrt(s)) allows, so the drift absorbs part of it. `q_buoy` 0.15-0.3 gives honest sigmas (5-10 ppm) and slightly better pinger tracks; tune it on the real data.

## Water current

Current cannot be seen from position alone. The track gives velocity over ground = velocity through water + current, and only the motion model separates the two. It can do that only where the boat's speed through water is known not to change while its direction does, as in a constant-throttle circle or reciprocal legs at the same throttle. On the 200–240 s circle of the truth track, a circle fit recovers the exact current; on the zig-zag it gives nonsense.

Two equivalent ways to estimate it (`current.py`):

- **A. Current as a graph variable.** One solve gives the estimate, and its covariance comes from J^T J.
- **B. Sweep.** Fix `(cx, cy)` on a grid, re-solve everything else (warm start), record the minimum cost `J(c)`. That is the profile likelihood: `delta chi2 = 2 (J(c) - J_min)`. The best value is the argmin, the 1-sigma region is `delta chi2 < 2.30` and the 95 % region is `< 5.99` (2 dof). The curvature gives the covariance.

A and B agree to within 0.01 m/s, in the estimate and in the sigma. A is one solve; B costs one solve per grid point but stays valid when the cost surface is non-quadratic, and it shows the whole surface.

How to do it: estimate current with a **stiff water-speed model** (`sigma_a` 0.05 m/s², with Huber letting the few real throttle changes through as jumps), then fix it and re-smooth the track with the normal model. With the default loose model (`sigma_a` 0.5) the estimate is biased by about +0.5 m/s because speed changes masquerade as current. The sweep does not fix that bias; both methods inherit whatever the motion model assumes.

Results (truth 0.20, -0.10 m/s):

| | A: graph variable | B: sweep (63 solves) | Track after stage 2 |
|---|---|---|---|
| Phase 1, pinger | (0.25, -0.03) ± 0.05 | (0.25, -0.03) ± 0.05 | 11.4 m, 0.26 m/s |
| Phase 2, noise | (0.22, +0.08) ± 0.06 | (0.22, +0.08) ± 0.07 | 4.5 m, 0.38 m/s |

(Reconstructed `demo.py`, Cauchy loss; the earlier table had (0.25, -0.04) and (0.21, -0.02).) The east component is within 1 sigma of truth. North is 1.5 sigma off for the pinger and 2.8 sigma for the noise run, because the stiff model is still slightly wrong. Treat the stated sigma as optimistic by roughly 1.5x until the motion noise is tuned on real data. Plots: `current_pinger.png`, `current_noise.png`.

For the field test, one or two **constant-throttle circles** (or an out-and-back leg at fixed throttle) make current observable. That is cheap to add to the run plan.

## Open choices

- Motion model noise is hand-tuned. It could be estimated in the same way, by sweeping it or by matching chi2/dof to 1.
- Doppler range rate is in (optional factor); it halves the pinger velocity error in simulation.
- Buoy drift is a random walk. If buoy GPS logs exist, even intermittently, they go in as position priors on `B[k]`.
- The emitter in phase 1 is the pinger, not the boat. `motion_model.boat_model.boat_from_towed_body` inverts the tow model (tow point = pinger + L along the pinger velocity, then the stern offset along the tow point's course); about 0.1 m on straight legs, up to 0.8 m in tight turns on the demo track (more with real sideslip).
- Buoy drift `q_buoy` is a placeholder; see the drift paragraph above.
