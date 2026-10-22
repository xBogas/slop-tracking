"""
Truth-free validation of the phase 1 and phase 2 solves.

No GPS truth exists for the boat, so the checks are internal consistency:

1. Normalized residuals per factor type and receiver (should be ~N(0,1)
   apart from a few robust-loss outliers), and chi2/dof of the inliers.
2. Held-out pings: every 5th ping's TOAs are removed, the graph is
   re-solved, and the held-out TDOAs (Rx01-Rx902, USBL-Rx902) are predicted
   from the smoothed track. TDOA cancels the emission time, whose 0.13 ms
   jitter would otherwise dominate. The prediction error is compared with
   the TOA sigma (sqrt(2) sigma_toa per TDOA) plus the graph's own position
   uncertainty.
3. Residual trends over time per receiver (a slow trend means an
   unmodelled clock drift or buoy motion).

Run from run_phase1.py (validate.report), or standalone after it.
"""
import numpy as np

SPLIT = 5   # hold out every SPLIT-th ping


def factor_residuals(fg, res):
    """Whitened TOA residuals (K, nr) at the solution, NaN where masked."""
    Z, T, B, g = fg.unpack(res.x)
    rng = fg.ranges(Z, B)
    clk = fg.clock(g)
    pred = T[:, None] + rng / g[3] + clk
    r = (fg.m["toa"] - pred) / fg.m.get("toa_sig", fg.cfg.sig_toa)
    return np.where(fg.m["toa_mask"], r, np.nan), pred


def tdoa(a, ref=0):
    return a - a[:, [ref]]


def held_out(fg, res, info):
    """Re-solve without every SPLIT-th ping's TOAs; predict their TDOAs."""
    from fg_tracker import FactorGraph, initial_guess
    K = fg.K
    hold = np.zeros(K, bool); hold[SPLIT // 2::SPLIT] = True
    m = dict(fg.m)
    m["toa_mask"] = fg.m["toa_mask"] & ~hold[:, None]
    fg2 = FactorGraph(fg.cfg, fg.geo, K, m)
    r2 = fg2.solve(res.x.copy(), max_iter=100)
    _, pred = factor_residuals(fg2, r2)
    obs = fg.m["toa"]
    e = tdoa(obs)[:, 1:] - tdoa(pred)[:, 1:]                # (K, 2): Rx01-Rx902, USBL-Rx902
    ok = fg.m["toa_mask"][:, [0]] & fg.m["toa_mask"][:, 1:] & hold[:, None]
    sig = fg.m["toa_sig"] if "toa_sig" in fg.m else np.full(obs.shape, fg.cfg.sig_toa)
    s_meas = np.hypot(sig[:, 1:], sig[:, [0]])
    return e, ok, s_meas


def report(fg, res, info, out):
    lines = []
    z, _ = factor_residuals(fg, res)
    names = ["Rx902", "Rx01", "USBL"]
    lines.append("1) normalized TOA residuals at the solution (inliers |z| < 5)")
    for i, nm in enumerate(names):
        zi = z[:, i][~np.isnan(z[:, i])]
        inl = np.abs(zi) < 5
        lines.append(f"   {nm:6s} n {zi.size:4d}  median {np.median(zi):+.2f}  rms(inl) {np.sqrt(np.mean(zi[inl]**2)):.2f}"
                     f"  outliers {np.mean(~inl)*100:4.1f} %")
    dof = res.fun.size - len(fg.free)
    lines.append(f"   total 2*cost/dof = {2 * res.cost / dof:.2f} (robust cost; ~1 if sigmas are right)")

    e, ok, s = held_out(fg, res, info)
    lines.append(f"2) held-out pings (every {SPLIT}th), TDOA prediction error vs Rx902")
    for j, nm in enumerate(["Rx01", "USBL"]):
        ej = e[ok[:, j], j]
        if ej.size == 0:
            continue
        zj = ej / s[ok[:, j], j]
        med = np.median(np.abs(ej))
        lines.append(f"   {nm:5s} n {ej.size:3d}  median |e| {med*1e6:6.1f} us ({med*1500:.2f} m)"
                     f"  rms z (|z|<5) {np.sqrt(np.mean(zj[np.abs(zj) < 5]**2)):.2f}  >5 sigma {np.mean(np.abs(zj) >= 5)*100:.1f} %")

    lines.append("3) residual trend over time (median z per 60 pings)")
    for i, nm in enumerate(names):
        seg = [np.nanmedian(z[a:a + 60, i]) for a in range(0, fg.K, 60)]
        lines.append(f"   {nm:6s} " + " ".join(f"{s_:+.1f}" for s_ in seg))
    text = "\n".join(lines)
    (out / "validation.txt").write_text(text + "\n")
    print(text)
    plot(fg, z, e, ok, s, info, out)


def plot(fg, z, e, ok, s, info, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    t = np.arange(fg.K)
    for i, nm in enumerate(["Rx902", "Rx01", "USBL"]):
        ax[0].plot(t, z[:, i], ".", ms=2.5, label=nm)
    ax[0].set_ylim(-8, 8); ax[0].set_ylabel("TOA residual [sigma]"); ax[0].grid(alpha=.3); ax[0].legend()
    for j, nm in enumerate(["Rx01", "USBL"]):
        ax[1].plot(t[ok[:, j]], e[ok[:, j], j] * 1e6, "o", ms=3, label=f"{nm}-Rx902 held-out")
    ax[1].set_ylabel("TDOA prediction error [us]"); ax[1].set_xlabel("ping"); ax[1].grid(alpha=.3); ax[1].legend()
    ax[1].set_ylim(-1000, 1000)
    fig.tight_layout(); fig.savefig(out / "validation.png", dpi=110)


# ------------------------------------------------------------------ phase 2
BLOCK, EVERY = 10, 40     # phase 2: hold out 10-epoch blocks, one per 40 epochs


def residuals_phase2(fg, res):
    """Whitened TDOA residuals (K, npairs) and bearing residuals (K,), NaN where masked,
    plus the predicted TDOAs and bearings."""
    from fg_tracker import wrap
    Z, _, B, g = fg.unpack(res.x)
    rng = fg.ranges(Z, B)
    clk = fg.clock(g)
    m = fg.m
    pred = np.column_stack([(rng[:, i] - rng[:, j]) / g[3] + clk[:, i] - clk[:, j] for i, j in m["pairs"]])
    zt = np.where(m["tdoa_mask"], (m["tdoa"] - pred) / m["tdoa_sig"], np.nan)
    sh = fg.geo.shore
    th = np.arctan2(Z[:, 1] - sh[1], Z[:, 0] - sh[0]) - g[1]
    sb = m.get("bearing_sig", fg.cfg.sig_bearing)
    zb = np.where(m["bearing_mask"], wrap(m["bearing"] - th) / sb, np.nan)
    return zt, zb, pred, th


def held_out_phase2(fg, res, win=4.0):
    """Drop blocks of BLOCK epochs (all measurements), re-solve, predict them.
    Only epochs at least win/2 inside a block are scored: the measurement
    windows overlap, so the block edges share data with the kept epochs."""
    from fg_tracker import FactorGraph, wrap
    K = fg.K
    ks = np.arange(K)
    hold = (ks % EVERY) >= EVERY - BLOCK
    edge = int(np.ceil(win / 2 / fg.cfg.Tp))
    score = hold.copy()
    for d in range(1, edge + 1):
        score &= np.roll(hold, d) & np.roll(hold, -d)
    m = dict(fg.m)
    m["tdoa_mask"] = fg.m["tdoa_mask"] & ~hold[:, None]
    m["bearing_mask"] = fg.m["bearing_mask"] & ~hold
    fg2 = FactorGraph(fg.cfg, fg.geo, K, m)
    r2 = fg2.solve(res.x.copy(), max_iter=100)
    _, _, pred, th = residuals_phase2(fg2, r2)
    et = np.where(fg.m["tdoa_mask"] & score[:, None], fg.m["tdoa"] - pred, np.nan)
    eb = np.where(fg.m["bearing_mask"] & score, wrap(fg.m["bearing"] - th), np.nan)
    return et, eb, score


def report_phase2(fg, res, info, out):
    names = ["Rx902", "Rx01", "USBL"]
    pn = [f"{names[i]}-{names[j]}" for i, j in fg.m["pairs"]]
    zt, zb, _, _ = residuals_phase2(fg, res)
    lines = ["1) normalized residuals at the solution (inliers |z| < 5)"]
    for q, nm in enumerate(pn + ["bearing"]):
        zi = zt[:, q] if q < len(pn) else zb
        zi = zi[~np.isnan(zi)]
        if zi.size == 0:
            lines.append(f"   {nm:11s} n    0"); continue
        inl = np.abs(zi) < 5
        lines.append(f"   {nm:11s} n {zi.size:4d}  median {np.median(zi):+.2f}  rms(inl) {np.sqrt(np.mean(zi[inl]**2)):.2f}"
                     f"  outliers {np.mean(~inl)*100:4.1f} %")
    dof = res.fun.size - len(fg.free)
    lines.append(f"   total 2*cost/dof = {2 * res.cost / dof:.2f}")

    et, eb, score = held_out_phase2(fg, res)
    lines.append(f"2) held-out blocks ({BLOCK} of every {EVERY} epochs, all measurements dropped, {score.sum()} scored epochs)")
    for q, nm in enumerate(pn):
        e = et[:, q][~np.isnan(et[:, q])]
        if e.size == 0:
            continue
        s = fg.m["tdoa_sig"][:, q][~np.isnan(et[:, q])]
        lines.append(f"   {nm:11s} n {e.size:3d}  median |e| {np.median(np.abs(e))*1e3:6.2f} ms"
                     f" ({np.median(np.abs(e))*fg.cfg.g_mean[3]:.1f} m)  median |e|/sigma {np.median(np.abs(e)/s):.2f}")
    e = eb[~np.isnan(eb)]
    if e.size:
        lines.append(f"   bearing     n {e.size:3d}  median |e| {np.rad2deg(np.median(np.abs(e))):.2f} deg")
    chk = out / "tdoa_pinger_check.txt"
    if chk.exists():
        lines.append("3) envelope-TDOA method check on the pinger band, runs 1-2 (noise_tdoa.py --pinger):")
        lines += ["   " + ln for ln in chk.read_text().strip().splitlines()]
    text = "\n".join(lines)
    (out / "validation.txt").write_text(text + "\n")
    print(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = info["t"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for q, nm in enumerate(pn):
        ax[0].plot(t, zt[:, q], ".", ms=3, label=nm)
    ax[0].plot(t, zb, ".", ms=3, label="bearing")
    ax[0].set_ylim(-8, 8); ax[0].set_ylabel("residual [sigma]"); ax[0].grid(alpha=.3); ax[0].legend()
    for q, nm in enumerate(pn):
        ax[1].plot(t, et[:, q] * 1e3, "o", ms=3, label=f"{nm} held-out")
    ax[1].set_ylabel("TDOA prediction error [ms]"); ax[1].set_xlabel("buoy time [s]"); ax[1].grid(alpha=.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "validation.png", dpi=110)
