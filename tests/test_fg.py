"""Factor-graph smoother: sparsity completeness, demo regression, new factors,
shore clock drift, grid-search initialisation. Thresholds are loose (about
1.5x the values seen on seed 1) so they catch breakage, not noise."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from boat_model import ct_predict
from fg_tracker import FactorGraph, initial_guess, ground_velocity

# estimation/demo.py by path: motion_model/demo.py shadows the name on sys.path
_spec = importlib.util.spec_from_file_location(
    "estimation_demo", Path(__file__).resolve().parent.parent / "estimation" / "demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def _solve(cfg, geo, meas, tr):
    fg = FactorGraph(cfg, geo, len(tr["te"]), meas)
    v0 = initial_guess(fg)
    return fg, v0, fg.solve(v0, max_iter=100)


def _rmse(fg, res, tr, ok=None):
    K = fg.K
    if ok is None:
        ok = np.zeros(K, bool); ok[15:K - 2] = True
    Z = res.x[fg.iZ]
    return demo.rmse(Z[ok, :2], tr["pos"][ok]), demo.rmse(ground_velocity(fg, res.x)[ok], tr["vel"][ok])


@pytest.mark.parametrize("mode,motion", [("pinger", "ground"), ("pinger", "water"), ("noise", "water")])
def test_sparsity_pattern_complete(mode, motion):
    """Every nonzero of a dense numerical Jacobian lies inside the declared
    sparsity, with all optional factors on; the grouped FD matches it."""
    cfg, geo, meas, tr = demo.simulate(mode, np.random.default_rng(0), motion=motion,
                                       extras=True, t_end=30.0)
    cfg.start_prior = (tr["pos"][0, 0] + 10, tr["pos"][0, 1] - 5, 30.0)
    cfg.pos_priors = ((5, tr["pos"][5, 0] - 3, tr["pos"][5, 1] + 4, 10.0),
                      (20, tr["pos"][20, 0] + 2, tr["pos"][20, 1], 5.0))
    fg = FactorGraph(cfg, geo, len(tr["te"]), meas)
    v = initial_guess(fg)
    v[fg.iZ[:, 4]] = 0.05                     # away from omega = 0
    Jd = approx_derivative(fg.residuals, v, method="2-point")
    assert Jd.shape == fg.sparsity.shape == (fg.n_res, fg.n)
    outside = np.abs(Jd) * (fg.sparsity.toarray() == 0)
    assert outside.max() < 1e-6 * np.abs(Jd).max()
    Js = fg.jacobian(v).toarray()
    assert np.allclose(Js, Jd, rtol=1e-4, atol=1e-6 * np.abs(Jd).max())


def test_ct_predict_vectorized_and_smooth():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 5)) * [100, 100, 3, 5, 0.5]
    Z[:5, 4] = 0.0
    P = ct_predict(Z, 1.0)
    assert np.allclose(P, np.array([ct_predict(z, 1.0) for z in Z]))
    x, y, chi, U, w = Z[10]
    ref = [x + U / w * (np.sin(chi + w) - np.sin(chi)), y + U / w * (np.cos(chi) - np.cos(chi + w))]
    assert np.allclose(P[10, :2], ref)
    # d x+/d omega at omega = 0 is -U dt^2/2 sin(chi), not 0
    z0 = np.array([0, 0, 0.7, 5.0, 0.0]); h = 1e-6
    d = (ct_predict(z0 + [0, 0, 0, 0, h], 1.0) - ct_predict(z0, 1.0))[0] / h
    assert abs(d + 0.5 * 5.0 * np.sin(0.7)) < 1e-4


@pytest.mark.parametrize("mode,pos_max,vel_max", [("pinger", 15.0, 0.4), ("noise", 12.0, 0.7)])
def test_demo_regression(mode, pos_max, vel_max):
    r = demo.run(mode, plot=False, verbose=False)
    print(f"{mode}: fixes {r['fix']:.1f} m, smoother {r['pos']:.1f} m, vel {r['vel']:.2f} m/s, "
          f"{r['nit']} it, {r['t_init'] + r['t_solve']:.1f} s")
    assert r["pos"] < pos_max and r["vel"] < vel_max
    assert r["pos"] < r["fix"]                                    # smoother beats single fixes
    g, gt = r["g"], r["g_true"]
    assert abs(g[3] - gt[3]) < 8.0                                # sound speed
    assert abs(np.rad2deg(g[1] - gt[1])) < 2.0                   # array orientation


@pytest.mark.parametrize("mode", ["pinger", "noise"])
def test_multipath_and_range_rate_factors_help(mode):
    """Same base measurements (extras are drawn after them) with and without
    multipath ranges (+ range rates for the pinger)."""
    base = demo.run(mode, plot=False, verbose=False)
    ext = demo.run(mode, extras=True, plot=False, verbose=False)
    print(f"{mode}: pos {base['pos']:.1f} -> {ext['pos']:.1f} m, vel {base['vel']:.2f} -> {ext['vel']:.2f} m/s")
    assert ext["pos"] < 1.1 * base["pos"] + 0.5
    assert ext["vel"] < 1.1 * base["vel"] + 0.02
    if mode == "pinger":
        assert ext["vel"] < 0.8 * base["vel"]                     # Doppler: direct velocity info


@pytest.mark.parametrize("mode,tol_ppm", [("pinger", 8.0), ("noise", 35.0)])
def test_shore_clock_drift_recovered(monkeypatch, mode, tol_ppm):
    monkeypatch.setattr(demo, "WANDER", 0.0)    # wander vs random-walk mismatch biases d_S
    drift = 40e-6
    r = demo.run(mode, drift=drift, plot=False, verbose=False)
    fg, res = r["fg"], r["res"]
    sd = fg.marginal_std(res, fg.iG[[6]])[0]
    dS = res.x[fg.iG[6]]
    print(f"{mode}: d_S {dS * 1e6:.1f} +- {sd * 1e6:.1f} ppm (true {drift * 1e6:.0f})")
    assert abs(dS - drift) < max(3 * sd, 1e-6) and abs(dS - drift) < tol_ppm * 1e-6
    # ignoring the drift (d_S pinned at 0) must fit worse
    fg.cfg.g_sig = fg.cfg.g_sig[:6] + (1e-9,)
    assert fg.solve(initial_guess(fg), max_iter=100).cost > res.cost


def test_grid_seed_far_start_with_dropouts(monkeypatch):
    """Track ~1.1 km from the old hard-coded seed (shore + (0, 400)), first 20
    epochs and a 50 s stretch with no detections at all: the grid-seeded
    init must land in the same basin as a solve started from the truth."""
    monkeypatch.setattr(demo, "OFFSET", np.array([700.0, 1250.0]))
    cfg, geo, meas, tr = demo.simulate("pinger", np.random.default_rng(1))
    K = len(tr["te"])
    gap = np.zeros(K, bool); gap[:20] = True; gap[120:170] = True
    meas["toa_mask"][gap] = False; meas["bearing_mask"][gap] = False
    fg, v0, res = _solve(cfg, geo, meas, tr)
    ok = ~gap; ok[-2:] = False
    fix_err = np.hypot(*(fg.init_fixes - tr["pos"]).T)
    assert np.all(np.isnan(fg.init_fixes[gap]))
    assert np.nanmedian(fix_err) < 100
    vt = res.x.copy()
    Zt = vt[fg.iZ]
    Zt[:, :2] = tr["pos"]
    Zt[:, 2] = np.unwrap(np.arctan2(tr["vel"][:, 1], tr["vel"][:, 0]))
    Zt[:, 3] = np.hypot(*tr["vel"].T); Zt[:, 4] = 0
    vt[fg.iZ] = Zt; vt[fg.iB] = tr["buoys"]; vt[fg.iG] = tr["g"]; vt[fg.iT] = tr["te"]
    rt = fg.solve(vt, max_iter=100)
    e, et = _rmse(fg, res, tr, ok)[0], _rmse(fg, rt, tr, ok)[0]
    print(f"far start: fixes median {np.nanmedian(fix_err):.1f} m, smoother {e:.1f} m, from truth {et:.1f} m")
    # Outside the gaps the tracks agree; inside the 50 s gap only the motion
    # model acts and the path through it has several local minima, so the
    # costs may differ somewhat (seed 1: 869 vs 759).
    assert e < et + 2.0 and res.cost < 1.2 * rt.cost


def test_start_prior_and_noise_dropouts():
    """Phase 2 with the first 30 windows empty and a start prior (100 m off,
    sigma 200 m): still converges to a good track."""
    cfg, geo, meas, tr = demo.simulate("noise", np.random.default_rng(1))
    meas["tdoa_mask"][:30] = False; meas["bearing_mask"][:30] = False
    p0 = tr["pos"][0]
    cfg.start_prior = (p0[0] + 100.0, p0[1], 200.0)
    fg, v0, res = _solve(cfg, geo, meas, tr)
    ok = np.zeros(fg.K, bool); ok[30:-2] = True
    pos, vel = _rmse(fg, res, tr, ok)
    print(f"noise, 30 empty windows + start prior: {pos:.1f} m, {vel:.2f} m/s")
    assert pos < 12.0 and vel < 0.7


def test_noise_per_epoch_tdoa_sigma():
    """Phase 2 with per-epoch TDOA sigmas: a third of the windows get 2 ms
    extra noise and a matching tdoa_sig; the track stays good."""
    rng = np.random.default_rng(2)
    cfg, geo, meas, tr = demo.simulate("noise", rng)
    K = len(meas["tdoa"])
    bad = rng.random(K) < 1 / 3
    meas["tdoa"][bad] += rng.normal(0, 2e-3, (bad.sum(), 3))
    meas["tdoa_sig"] = np.where(bad[:, None], np.hypot(2e-3, cfg.sig_tdoa), cfg.sig_tdoa) * np.ones((1, 3))
    fg, v0, res = _solve(cfg, geo, meas, tr)
    assert fg.m["tdoa_sig"].shape == (K, 3)
    ok = np.zeros(fg.K, bool); ok[2:-2] = True
    pos, vel = _rmse(fg, res, tr, ok)
    print(f"noise, per-epoch tdoa_sig: {pos:.1f} m, {vel:.2f} m/s")
    assert pos < 12.0 and vel < 0.7


def test_conditional_std_matches_inverse_of_reduced_information():
    from types import SimpleNamespace
    from fg_tracker import FactorGraph
    rng = np.random.default_rng(1)
    J = rng.normal(size=(40, 6))
    res = SimpleNamespace(jac=J)
    obj = SimpleNamespace(free=np.arange(6))
    H = J.T @ J
    given = np.array([4, 5])
    ref = np.sqrt(np.diag(np.linalg.inv(H[:4, :4]))[[0, 2]])
    got = FactorGraph.conditional_std(obj, res, np.array([0, 2]), given)
    assert np.allclose(got, ref)
    assert np.all(got <= FactorGraph.marginal_std(obj, res, np.array([0, 2])) + 1e-12)
