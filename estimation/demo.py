"""
Simulated phase 1 (towed pinger, TOA) and phase 2 (radiated noise, TDOA)
on the motion-model truth track, solved with the factor-graph smoother.

Geometry (same as multipath/multipath_demo.py): shore array at the origin,
buoys nominally at (-600, 250) and (600, 250) on a line parallel to the
coast, truth track shifted by (-150, 700) so the boat runs 350-800 m
offshore. Simulated errors, so the estimator has something to do:
  buoys 10 m off nominal plus +-3 m slow wander, shore array rotated 4 deg,
  sound speed 1506 m/s (1500 assumed), shore clock offset 0.15 ms (and an
  optional drift), pinger clock skew 5 ppm, 2 % outliers (late multipath
  arrivals / wrong correlation peaks), 5 % missed detections.
Optional extras: single-buoy multipath horizontal ranges (sigma 1 m + 1 %,
5 % outliers) and, for the pinger, Doppler range rates (0.05 m/s).

simulate(mode, rng, motion="ground", drift=0.0, extras=False)
    -> cfg, geo, meas, tr      tr: te, pos, vel (truth per epoch), g, buoys

Run: uv run python estimation/demo.py
     -> prints RMSEs, writes demo_<mode>.png and demo_<mode>_estimate.csv
"""
import time
from pathlib import Path

import numpy as np

from fg_tracker import Config, Geometry, FactorGraph, initial_guess, ground_velocity

HERE = Path(__file__).resolve().parent
TRUTH = HERE.parent / "motion_model" / "demo_truth.csv"
OFFSET = np.array([-150.0, 700.0])
SHORE = np.array([0.0, 0.0, -2.0])
BUOYS_NOM = np.array([[-600.0, 250.0], [600.0, 250.0]])
BUOYS_OFF = np.array([[8.0, -5.0], [-6.0, 7.0]])       # ~10 m off nominal
C_TRUE, ALPHA_TRUE, BS_TRUE, SKEW_TRUE = 1506.0, np.deg2rad(4.0), 0.15e-3, 5e-6
P_OUT, P_MISS = 0.02, 0.05
STERN = 3.0                                            # boat centre -> propeller [m]
WANDER, WANDER_PERIOD = 3.0, 200.0                    # buoy wander amplitude [m], period [s]


def load_truth():
    d = np.genfromtxt(TRUTH, delimiter=",", names=True)
    t = d["t"]
    pinger = np.column_stack([d["pinger_x"], d["pinger_y"]]) + OFFSET
    # phase 2 emitter: propeller / engine noise centre near the stern
    noise = np.column_stack([d["x"] - STERN * np.cos(d["psi"]),
                             d["y"] - STERN * np.sin(d["psi"])]) + OFFSET
    return t, pinger, noise


def _interp(t, tq, X):
    return np.column_stack([np.interp(tq, t, X[:, d]) for d in range(X.shape[1])])


def simulate(mode, rng, motion="ground", drift=0.0, extras=False, t_end=300.0, Tp=1.0):
    t, Pp, Pn = load_truth()
    pinger = mode == "pinger"
    track = Pp if pinger else Pn
    Vt = np.gradient(track, t, axis=0)
    if pinger:   # emission times on the (skewed) pinger clock, unknown start
        te = 0.3 + np.arange(int((t_end - 1.3) / (Tp * (1 + SKEW_TRUE)))) * Tp * (1 + SKEW_TRUE)
    else:        # 1 s analysis windows
        te = 0.5 + np.arange(int((t_end - 1) / Tp)) * Tp
    K = len(te)
    pos, vel = _interp(t, te, track), _interp(t, te, Vt)
    zs = -10.0 if pinger else -1.0
    nb, nr = 2, 3

    # true receivers: drifting buoys, fixed shore
    ph = rng.uniform(0, 2 * np.pi, (nb, 2))
    B = (BUOYS_NOM + BUOYS_OFF)[None] + WANDER * np.sin(2 * np.pi * te[:, None, None] / WANDER_PERIOD + ph[None])
    R = np.empty((K, nr, 3)); R[:, :nb, :2] = B; R[:, :nb, 2] = -10.0; R[:, nb] = SHORE
    Dv = np.column_stack([pos, np.full(K, zs)])[:, None, :] - R
    rng3 = np.linalg.norm(Dv, axis=2)
    clk = np.zeros((K, nr)); clk[:, nb] = BS_TRUE + drift * (te - te[0])

    cfg = Config(mode=mode, motion=motion, Tp=Tp)
    geo = Geometry(shore=SHORE.copy(), buoys_nominal=BUOYS_NOM.copy(), src_depth=zs)
    meas = {}
    if pinger:
        toa = te[:, None] + rng3 / C_TRUE + clk + rng.normal(0, cfg.sig_toa, (K, nr))
        out = rng.random((K, nr)) < P_OUT
        toa[out] += rng.uniform(0.5e-3, 10e-3, out.sum())          # late multipath
        meas.update(toa=toa, toa_mask=rng.random((K, nr)) > P_MISS)
    else:
        pairs = [(0, 1), (0, 2), (1, 2)]
        tdoa = np.column_stack([(rng3[:, i] - rng3[:, j]) / C_TRUE + clk[:, i] - clk[:, j]
                                for i, j in pairs]) + rng.normal(0, cfg.sig_tdoa, (K, 3))
        out = rng.random((K, 3)) < P_OUT
        tdoa[out] += rng.uniform(-5e-3, 5e-3, out.sum())
        meas.update(pairs=pairs, tdoa=tdoa, tdoa_mask=rng.random((K, 3)) > P_MISS)
    brg = np.arctan2(pos[:, 1] - SHORE[1], pos[:, 0] - SHORE[0]) - ALPHA_TRUE \
        + rng.normal(0, cfg.sig_bearing, K)
    out = rng.random(K) < P_OUT
    brg[out] = rng.uniform(-np.pi, np.pi, out.sum())
    meas.update(bearing=brg, bearing_mask=rng.random(K) > P_MISS)

    if extras:
        rh = np.hypot(Dv[..., 0], Dv[..., 1])
        sig = 1.0 + (0.01 if pinger else 0.03) * rh
        mp = rh + rng.normal(0, 1, (K, nr)) * sig
        out = rng.random((K, nr)) < 0.05
        mp[out] *= 1 + rng.uniform(-0.3, 0.3, out.sum())
        mmask = rng.random((K, nr)) > P_MISS
        mmask[:, nb] = False                       # buoys only
        meas.update(mp_range=mp, mp_range_mask=mmask, mp_range_sig=sig)
        if pinger:
            rr = (Dv[..., 0] * vel[:, None, 0] + Dv[..., 1] * vel[:, None, 1]) / rng3
            rr += rng.normal(0, 0.05, (K, nr))
            out = rng.random((K, nr)) < P_OUT
            rr[out] += rng.uniform(-2, 2, out.sum())
            meas.update(rr=rr, rr_mask=meas["toa_mask"].copy(), rr_sig=np.full((K, nr), 0.05))

    g = np.array([BS_TRUE, ALPHA_TRUE, SKEW_TRUE if pinger else 0.0, C_TRUE, 0.2, -0.1, drift])
    tr = dict(te=te, pos=pos, vel=vel, g=g, buoys=B)
    return cfg, geo, meas, tr


def rmse(a, b):
    e = np.sum((a - b) ** 2, axis=1)
    return np.sqrt(np.nanmean(e))


def run(mode, rng=None, extras=False, plot=True, verbose=True, **kw):
    """Simulate + solve one phase. Returns a dict of metrics and the solve."""
    rng = np.random.default_rng(1) if rng is None else rng
    cfg, geo, meas, tr = simulate(mode, rng, extras=extras, **kw)
    K = len(tr["te"])
    t0 = time.perf_counter()
    fg = FactorGraph(cfg, geo, K, meas)
    v0 = initial_guess(fg)
    t1 = time.perf_counter()
    res = fg.solve(v0, max_iter=100)
    t2 = time.perf_counter()
    Z = res.x[fg.iZ]
    V = ground_velocity(fg, res.x)
    sl = slice(15, K - 2)          # skip the start-up transient and the last epoch
    out = dict(
        fix=rmse(fg.init_fixes[sl], tr["pos"][sl]),
        pos=rmse(Z[sl, :2], tr["pos"][sl]),
        vel=rmse(V[sl], tr["vel"][sl]),
        g=res.x[fg.iG], g_true=tr["g"], nit=res.nit, t_init=t1 - t0, t_solve=t2 - t1,
        chi2dof=2 * res.cost / (res.fun.size - len(fg.free)), K=K)
    if verbose:
        g, gt = out["g"], tr["g"]
        print(f"=== {mode}{' + multipath ranges' if extras else ''}"
              f"{' + range rates' if extras and mode == 'pinger' else ''}  (K = {K})")
        print(f"  position RMSE: single-epoch fixes {out['fix']:.1f} m, smoother {out['pos']:.1f} m;"
              f"  velocity RMSE {out['vel']:.2f} m/s")
        print(f"  b_S {g[0]*1e3:.3f} ms (true {gt[0]*1e3:.3f}), d_S {g[6]*1e6:+.1f} ppm "
              f"(true {gt[6]*1e6:+.1f}), alpha {np.rad2deg(g[1]):.2f} deg (true {np.rad2deg(gt[1]):.1f}), "
              f"c {g[3]:.1f} (true {gt[3]:.0f})")
        print(f"  {res.nit} iterations, init {out['t_init']:.1f} s, solve {out['t_solve']:.1f} s, "
              f"cost/dof {out['chi2dof']:.2f}")
    if plot:
        save(mode, fg, res, tr)
    out.update(fg=fg, res=res, tr=tr)
    return out


def save(mode, fg, res, tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = fg.K
    Z = res.x[fg.iZ]
    V = ground_velocity(fg, res.x)
    sd = fg.marginal_std(res, fg.iZ[:, :2].ravel()).reshape(K, 2)
    np.savetxt(HERE / f"demo_{mode}_estimate.csv",
               np.column_stack([tr["te"], Z[:, :2], V, sd, tr["pos"], tr["vel"]]),
               delimiter=",", fmt="%.4f", comments="",
               header="t,x,y,vx,vy,sx,sy,x_true,y_true,vx_true,vy_true")
    err = np.hypot(*(Z[:, :2] - tr["pos"]).T)
    fix_err = np.hypot(*(fg.init_fixes - tr["pos"]).T)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    a = ax[0]
    a.plot(*tr["pos"].T, "k", lw=2, alpha=.4, label="truth")
    a.plot(*fg.init_fixes.T, ".", ms=2, c="C1", label="single-epoch fixes")
    a.plot(*Z[:, :2].T, "C0", lw=1, label="smoother")
    B = res.x[fg.iB]
    a.plot(*fg.geo.buoys_nominal.T, "ks", mfc="none", label="buoys nominal")
    a.plot(*B[0].T, "C3s", ms=4, label="buoys estimated")
    a.plot(*tr["buoys"][0].T, "k+", ms=8, label="buoys true")
    a.plot(*fg.geo.shore[:2], "k^", label="shore array")
    a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=7); a.set_title(f"{mode}: track")
    a = ax[1]
    a.semilogy(tr["te"], fix_err, ".", ms=2, c="C1", label="single-epoch fix")
    a.semilogy(tr["te"], err, "C0", label="smoother")
    a.semilogy(tr["te"], np.hypot(*sd.T), "C0--", lw=1, label="smoother 1-sigma (graph)")
    a.set_xlabel("t [s]"); a.set_ylabel("position error [m]"); a.grid(alpha=.3, which="both")
    a.legend(fontsize=7); a.set_title("position error")
    a = ax[2]
    a.plot(tr["te"], np.hypot(*tr["vel"].T), "k", lw=2, alpha=.4, label="truth")
    a.plot(tr["te"], np.hypot(*V.T), "C0", lw=1, label="smoother")
    a.set_xlabel("t [s]"); a.set_ylabel("speed over ground [m/s]"); a.grid(alpha=.3)
    a.legend(fontsize=7); a.set_title("speed")
    fig.tight_layout()
    fig.savefig(HERE / f"demo_{mode}.png", dpi=100)
    plt.close(fig)


def main():
    for mode in ("pinger", "noise"):
        run(mode)
        run(mode, extras=True, plot=False)


if __name__ == "__main__":
    main()
