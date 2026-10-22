"""
Water current (cx, cy) two ways, on the simulated pinger and noise data.

A) Current as a variable in the graph (motion="water"): one solve, the
   estimate and its covariance come from the Gauss-Newton information.
B) Sweep: fix (cx, cy) on a grid, re-solve everything else (warm start),
   record the minimum cost J(c). This is the profile likelihood:
       -2 log L(c) = 2 J(c) + const,  delta chi2(c) = 2 (J(c) - J_min)
   with 2 dof: 1-sigma region delta chi2 < 2.30, 95 % region < 5.99.
   Best value = argmin J. Same answer as A when the problem is close to
   linear; the sweep stays valid when it isn't (banana-shaped regions,
   several local minima) and needs nothing but a solver that can hold
   parameters fixed.

Run: python3 current.py   -> prints both results, writes current_<mode>.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demo
from fg_tracker import FactorGraph, initial_guess, ground_velocity

HERE = Path(__file__).resolve().parent
TRUE_CURRENT = np.array([0.2, -0.1])   # what the motion-model demo used
GRID = np.arange(-0.2, 0.61, 0.1), np.arange(-0.4, 0.21, 0.1)
# Current is only observable where the boat's speed THROUGH WATER is known to
# stay constant while its direction changes (circles, reciprocal legs at the
# same throttle). So stage 1 uses a stiff water-speed model; the Huber loss
# lets the few real throttle changes through as sparse jumps. A loose model
# (sigma_a 0.5) lets speed changes masquerade as current: bias ~ +0.5 m/s.
STIFF = dict(sigma_a=0.05, sig_U=0.05)


def run(mode):
    rng = np.random.default_rng(1)
    cfg, geo, meas, tr = demo.simulate(mode, rng, motion="water")
    cfg.g_sig = cfg.g_sig[:4] + (10.0, 10.0) + cfg.g_sig[6:]   # flat current prior: let the data speak
    cfg.sigma_a = STIFF["sigma_a"]
    q = list(cfg.q_reg); q[3] = STIFF["sig_U"]; cfg.q_reg = tuple(q)
    K = len(tr["te"])
    fg = FactorGraph(cfg, geo, K, meas)
    ic = fg.iG[4:6]

    # A) current as a graph variable
    res = fg.solve(initial_guess(fg), max_iter=100)
    vA = res.x.copy()
    cA = vA[ic]
    P = fg.marginal_cov(res, ic)
    sl = slice(15, K - 2)
    Z = vA[fg.iZ]
    epos = np.sqrt(np.mean(np.sum((Z[sl, :2] - tr["pos"][sl]) ** 2, 1)))
    evel = np.sqrt(np.mean(np.sum((ground_velocity(fg, vA)[sl] - tr["vel"][sl]) ** 2, 1)))
    print(f"\n=== {mode}")
    print(f"A) graph variable: c = ({cA[0]:+.3f}, {cA[1]:+.3f}) m/s, "
          f"1-sigma ({np.sqrt(P[0,0]):.3f}, {np.sqrt(P[1,1]):.3f}), corr {P[0,1]/np.sqrt(P[0,0]*P[1,1]):+.2f}"
          f"   [truth {TRUE_CURRENT}]")
    print(f"   position RMSE {epos:.2f} m, ground-velocity RMSE {evel:.2f} m/s")

    # B) sweep, warm-started from the free solution
    gx, gy = GRID
    J = np.full((len(gy), len(gx)), np.nan)
    for i, cy in enumerate(gy):
        for j, cx in enumerate(gx):
            v0 = vA.copy(); v0[ic] = (cx, cy)
            J[i, j] = fg.solve(v0, fixed=ic, max_iter=25).cost
    Jmin_grid = np.nanmin(J)
    # refine: quadratic fit of J around the grid minimum -> best value + curvature
    i0, j0 = np.unravel_index(np.nanargmin(J), J.shape)
    X, Y = np.meshgrid(gx, gy)
    m = (np.abs(X - gx[j0]) < 0.25) & (np.abs(Y - gy[i0]) < 0.25)
    A = np.column_stack([np.ones(m.sum()), X[m], Y[m], X[m]**2, X[m]*Y[m], Y[m]**2])
    a = np.linalg.lstsq(A, J[m], rcond=None)[0]
    Hq = np.array([[2*a[3], a[4]], [a[4], 2*a[5]]])
    cB = np.linalg.solve(Hq, -a[1:3])
    Jmin = a[0] + a[1:3] @ cB / 2
    PB = np.linalg.inv(Hq)        # delta chi2 = 2 dJ  ->  cov = (d2J)^-1
    print(f"B) sweep ({J.size} solves): best c = ({cB[0]:+.3f}, {cB[1]:+.3f}) m/s, "
          f"1-sigma ({np.sqrt(PB[0,0]):.3f}, {np.sqrt(PB[1,1]):.3f}) from the cost curvature")
    # stage 2: hold the current, re-smooth the track with the normal (loose) model
    fg.cfg.sigma_a = 0.5; fg.cfg.q_reg = (0.3, 0.3, np.deg2rad(1.0), 0.05, np.deg2rad(0.2))
    r2 = fg.solve(vA, fixed=ic, max_iter=100)
    Z2 = r2.x[fg.iZ]
    e2 = np.sqrt(np.mean(np.sum((Z2[sl, :2] - tr["pos"][sl]) ** 2, 1)))
    ev2 = np.sqrt(np.mean(np.sum((ground_velocity(fg, r2.x)[sl] - tr["vel"][sl]) ** 2, 1)))
    print(f"   stage 2 (current fixed, loose model): position RMSE {e2:.2f} m, velocity RMSE {ev2:.2f} m/s")
    fg.cfg.sigma_a = STIFF["sigma_a"]
    dof = res.fun.size - len(fg.free)
    print(f"   cost/dof at optimum {2*res.cost/dof:.2f} (about 1 if the sigmas are right)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    a = ax[0]
    dchi = 2 * (J - min(Jmin, Jmin_grid))
    cs = a.contourf(gx, gy, np.sqrt(np.maximum(dchi, 0)), levels=20, cmap="viridis_r")
    fig.colorbar(cs, ax=a, label="sqrt(delta chi2)  (~ sigmas from best)")
    a.contour(gx, gy, dchi, levels=[2.30, 5.99], colors=["w", "w"], linestyles=["-", "--"])
    t = np.linspace(0, 2 * np.pi, 100)
    for c0, Pc, col, lab in [(cA, P, "C3", "A) graph variable, 1/2.45 sigma"),
                             (cB, PB, "C1", "B) sweep fit, 1/2.45 sigma")]:
        Lc = np.linalg.cholesky(Pc)
        for s_, ls in [(1.0, "-"), (2.45, "--")]:
            e = c0[:, None] + s_ * Lc @ np.vstack([np.cos(t), np.sin(t)])
            a.plot(*e, col, ls=ls, lw=1.2, label=lab if s_ == 1 else None)
        a.plot(*c0, "o", color=col)
    a.plot(*TRUE_CURRENT, "k*", ms=14, label="truth")
    a.set_xlabel("current east cx [m/s]"); a.set_ylabel("current north cy [m/s]")
    a.set_title(f"{mode}: profile cost over current (white: 1-sigma / 95 %)")
    a.legend(fontsize=7, loc="lower left"); a.set_aspect("equal")
    a = ax[1]
    a.plot(*tr["pos"].T, "k", lw=2, alpha=.4, label="truth")
    a.plot(*Z[:, :2].T, "C0", lw=1, label="estimate (water-relative model)")
    a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_title("track")
    fig.tight_layout()
    fig.savefig(HERE / f"current_{mode}.png", dpi=100)


if __name__ == "__main__":
    import sys
    for mode in (sys.argv[1:] or ["pinger", "noise"]):
        run(mode)
