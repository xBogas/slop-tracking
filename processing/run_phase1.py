"""
Phase 1 end to end: TOA table (+ optional USBL bearings and multipath
ranges) -> factor-graph smoother -> pinger and boat tracks.

Inputs (all produced by the other processing steps):
  out/toa_phase1.npz                 detect.py + associate.py (required)
  out/usbl/bearings_phase1.npz       usbl.py           (optional)
  out/multipath/mp_range_phase1.npz  multipath_range.py (only with --mp; the ranges are empty)

The pinger ran continuously across the run 1/2 boundary with no phase jump,
so both runs are solved as one graph (one ping chain, one set of clock and
buoy unknowns); the output is split per run afterwards.

Outputs in out/phase1/: track.csv (per ping: time, pinger and boat E/N and
lat/lon, ground velocity, 1-sigma), globals.txt, and plots. Validation
(truth-free) is in validate.py and runs at the end.

Run: uv run python processing/run_phase1.py [processing/forfun.toml] [--no-bearing] [--mp] [--fixes]
--fixes adds the run start fixes ([[run]] lat/lon, no time stamp) as priors and writes
to out/phase1_fixes/ instead (see fix_priors). --no-bearing (TOA only) writes to
out/phase1_toa_only/, the reference track usbl.py calibrates the array orientation on.
"""
import sys
import time
from dataclasses import replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for d in ("estimation", "motion_model"):
    sys.path.insert(0, str(ROOT / d))
import config                                                           # noqa: E402
from fg_tracker import (Config, Geometry, FactorGraph, initial_guess,   # noqa: E402
                        ground_velocity, GLOBALS)
from boat_model import boat_from_towed_body                             # noqa: E402

OUT = ROOT / "out" / "phase1"


def dense(tab, key, K, k, fill):
    """Rows of the associator's table (indexed by ping k) on the dense grid 0..K-1."""
    a = tab[key]
    d = np.full((K,) + a.shape[1:], fill, dtype=a.dtype)
    d[k] = a
    return d


def build(cfg, use_bearing=True, use_mp=False, q_buoy=0.2):
    tab = np.load(ROOT / "out" / "toa_phase1.npz")
    k = tab["k"] - tab["k"].min()
    K = int(k.max()) + 1
    names = list(tab["receivers"])
    assert names == ["Rx902", "Rx01", "USBL"], names
    buoys = [cfg.receiver(n) for n in names[:2]]
    usbl = cfg.receiver("USBL")

    meas = dict(toa=dense(tab, "toa", K, k, np.nan),
                toa_mask=dense(tab, "toa_mask", K, k, False))
    meas["toa_sig"] = np.broadcast_to(tab["toa_sigma"], (K, 3)).copy()
    meas["bearing"] = np.full(K, np.nan)
    meas["bearing_mask"] = np.zeros(K, bool)
    run = dense(tab, "run", K, k, -1)
    # pings missing from the table inherit the run of the previous ping
    for i in range(1, K):
        if run[i] < 0:
            run[i] = run[i - 1]

    notes = []
    alpha0 = None
    bp = ROOT / "out" / "usbl" / "bearings_phase1.npz"
    if use_bearing and bp.exists():
        b = np.load(bp)
        kb = b["k"] - tab["k"].min()
        ok = (kb >= 0) & (kb < K) & b["mask"]
        # The graph models bearing = atan2(p - usbl) - alpha. usbl.py found the
        # array frame is mirrored (az_enu = alpha - az_array), so feed -az_array
        # and let the graph estimate alpha (orientation) itself.
        s = float(b["handedness"])
        meas["bearing"][kb[ok]] = s * b["az_array"][ok]
        meas["bearing_mask"][kb[ok]] = True
        meas["bearing_sig"] = np.full(K, np.deg2rad(1.0))
        meas["bearing_sig"][kb[ok]] = np.maximum(b["sigma"][ok], np.deg2rad(0.5))
        # alpha = ENU azimuth of the array axis: the operator's compass bearing
        # when given, else usbl.py's fit to the TOA-only track
        if usbl.axis_enu is not None:
            alpha0, alpha_sig = usbl.axis_enu, usbl.axis_bearing_sigma
        else:
            alpha0, alpha_sig = float(b["alpha"]), 10.0
        notes.append(f"bearings: {ok.sum()} pings (alpha prior {np.rad2deg(alpha0):.1f} +- {alpha_sig:g} deg, "
                     f"usbl.py {np.rad2deg(float(b['alpha'])):.1f})")
    mp = ROOT / "out" / "multipath" / "mp_range_phase1.npz"
    if use_mp and mp.exists():
        m = np.load(mp)
        for key in ("mp_range", "mp_range_mask", "mp_range_sig"):
            meas[key] = dense(m, key, K, k, False if key.endswith("mask") else np.nan)
        notes.append(f"multipath ranges: {meas['mp_range_mask'].sum(0).tolist()} per receiver")

    pr = cfg.priors
    g_mean = [0.0, alpha0 or 0.0, 0.0, pr["sound_speed"], 0.0, 0.0, 0.0]
    g_sig = [float(tab["usbl_offset_halfwidth"]),   # residual USBL offset after the coarse shift
             np.deg2rad(180.0 if alpha0 is None else alpha_sig),   # array orientation
             50e-6,                                  # pinger clock skew around the measured Tp
             pr["sound_speed_sigma"], 1.0, 1.0,
             100e-6]                                 # USBL clock drift
    fcfg = Config(mode="pinger", motion="ground", Tp=float(tab["Tp"]),
                  sig_chain=float(tab["emission_jitter"]) * np.sqrt(2),
                  sig_buoy0=max(b.pos_sigma for b in buoys), q_buoy=q_buoy,
                  g_mean=tuple(g_mean), g_sig=tuple(g_sig), nb=2)
    geo = Geometry(shore=np.r_[usbl.en, -usbl.depth],
                   buoys_nominal=np.array([b.en for b in buoys]),
                   buoy_depth=-np.array([b.depth for b in buoys]), src_depth=-cfg.pinger.depth)
    ef = dense(tab, "emission_fix", K, k, np.nan)        # NaN where a ping had no fix
    t_ping = np.nanmedian(ef - np.arange(K) * fcfg.Tp) + np.arange(K) * fcfg.Tp
    return FactorGraph(fcfg, geo, K, meas), dict(run=run, t=t_ping, notes=notes, tab=tab)


def solve(fg, verbose=False):
    t0 = time.time()
    v0 = initial_guess(fg)
    res = fg.solve(v0, max_iter=150, verbose=verbose)
    return res, time.time() - t0


FIX_MARGIN = 60.0      # [s] fix search before the run start


def fix_priors(cfg, fg, x, t, sigma):
    """Run start fixes (boat positions without a time stamp) as priors on the
    pinger: each fix is assigned to the epoch where the current boat track
    passes closest to it inside that run's window (from FIX_MARGIN before the
    run start, which is itself the data's estimate of the fix passage), and
    moved to the pinger by the current boat-to-pinger offset at that epoch."""
    Z, T, _, _ = fg.unpack(x)
    t = T if T is not None else t
    _, C = boat_from_towed_body(Z[:, :2], ground_velocity(fg, x), t, L=cfg.pinger.tow_length)
    out = []
    for r in cfg.runs:
        w = np.flatnonzero((t >= r.start - FIX_MARGIN) & (t < r.end))
        if r.en is None or not len(w):
            continue
        k = int(w[np.argmin(np.hypot(*(C[w] - r.en).T))])
        p = r.en - (C[k] - Z[k, :2])
        out.append((k, float(p[0]), float(p[1]), sigma))
    return tuple(out)


def solve_with_fixes(cfg, fg, res, info, sigma, n_iter=5):
    """Re-solve with fix_priors, re-assigning the epochs until they settle."""
    names = [r.name for r in cfg.runs if r.en is not None]
    pp = fix_priors(cfg, fg, res.x, info["t"], sigma)
    for it in range(n_iter):
        fg = FactorGraph(replace(fg.cfg, pos_priors=pp), fg.geo, fg.K, fg.m)
        res = fg.solve(res.x, max_iter=150)
        new = fix_priors(cfg, fg, res.x, info["t"], sigma)
        done = [q[0] for q in new] == [q[0] for q in pp]
        pp = new
        if done:
            break
    Z, T, _, _ = fg.unpack(res.x)
    t = T if T is not None else info["t"]
    for nm, (k, x0, y0, s0) in zip(names, pp):
        info["notes"].append(f"fix {nm}: epoch {k} (t = {t[k]:.0f} s), pinger residual "
                             f"({Z[k, 0] - x0:+.1f}, {Z[k, 1] - y0:+.1f}) m, sigma {s0} m, {it + 1} re-assignments")
    return fg, res


def outputs(cfg, fg, res, info):
    OUT.mkdir(parents=True, exist_ok=True)
    K = fg.K
    Z, T, B, g = fg.unpack(res.x)
    V = ground_velocity(fg, res.x)
    sd = fg.marginal_std(res, fg.iZ[:, :2].ravel()).reshape(K, 2)
    sdg = fg.marginal_std(res, fg.iG[np.r_[0, 2:4, 6]])
    t = T if T is not None else info["t"]
    Q, C = boat_from_towed_body(Z[:, :2], V, t, L=cfg.pinger.tow_length)
    plat, plon = cfg.to_llh(Z[:, 0], Z[:, 1])
    blat, blon = cfg.to_llh(C[:, 0], C[:, 1])
    has = fg.m["toa_mask"].sum(1)
    hdr = ("ping,run,t_emit_buoy_s,pinger_e,pinger_n,pinger_lat,pinger_lon,"
           "boat_e,boat_n,boat_lat,boat_lon,v_e,v_n,speed,sig_e,sig_n,n_receivers")
    rows = np.column_stack([np.arange(K), info["run"], t, Z[:, 0], Z[:, 1], plat, plon,
                            C[:, 0], C[:, 1], blat, blon, V[:, 0], V[:, 1],
                            np.hypot(*V.T), sd[:, 0], sd[:, 1], has])
    fmt = ["%d", "%d", "%.6f", "%.2f", "%.2f", "%.7f", "%.7f", "%.2f", "%.2f", "%.7f", "%.7f",
           "%.3f", "%.3f", "%.3f", "%.2f", "%.2f", "%d"]
    np.savetxt(OUT / "track.csv", rows, delimiter=",", header=hdr, comments="", fmt=fmt)

    lines = [f"K = {K} pings, {res.nit} iterations, cost {res.cost:.1f}"]
    for i, nm in enumerate(GLOBALS):
        lines.append(f"{nm:6s} = {g[i]: .6g}")
    lines.append(f"sigma b_S {sdg[0]*1e3:.3f} ms, skew {sdg[1]*1e6:.1f} ppm, "
                 f"c {sdg[2]:.2f} m/s, d_S {sdg[3]*1e6:.1f} ppm")
    lines.append(f"USBL offset total = {float(info['tab']['usbl_offset']) + g[0]:.6f} s")
    for j, nm in enumerate(["Rx902", "Rx01"]):
        b0 = fg.geo.buoys_nominal[j]
        mv = B[:, j] - b0
        lines.append(f"{nm} position shift from GPS: mean ({mv[:,0].mean():+.1f}, {mv[:,1].mean():+.1f}) m, "
                     f"wander rms {np.std(np.hypot(*(B[:, j] - B[:, j].mean(0)).T)):.1f} m")
    lines += info["notes"]
    (OUT / "globals.txt").write_text("\n".join(lines) + "\n")
    # posterior summary for phase 2 (priors on the globals and buoys, start state)
    free = np.isin(fg.iG, fg.free)                         # cx, cy are fixed in motion="ground"
    gsig = np.full(len(fg.iG), np.nan)
    gsig[free] = fg.marginal_std(res, fg.iG[free])
    # TOA the solution predicts per epoch and receiver (per-epoch buoys and clock), rows as track.csv
    toa_pred = T[:, None] + fg.ranges(Z, B) / g[3] + fg.clock(g)
    # emission-time uncertainty for export.py: the full posterior covariance of each T_k with the
    # USBL clock offset, sound speed and drift (b_S, c, d_S), and its sigma with the globals and the
    # buoy positions held fixed (the part that varies from ping to ping)
    gi = fg.iG[[0, 3, 6]]
    Cv = fg.marginal_cov(res, np.r_[fg.iT, gi])
    K_ = len(fg.iT)
    t_emit_sig = np.sqrt(np.diag(Cv)[:K_])
    t_emit_cov = Cv[:K_, K_:]                                 # (K, 3): cov(T_k, [b_S, c, d_S])
    g_cov = Cv[K_:, K_:]                                      # (3, 3)
    t_emit_sig_cond = fg.conditional_std(res, fg.iT, np.r_[fg.iG, fg.iB.ravel()])
    np.savez(OUT / "state.npz", globals=g, globals_sig=gsig, names=np.array(GLOBALS),
             usbl_offset=float(info["tab"]["usbl_offset"]), t0=t[0], Tp=fg.cfg.Tp,
             buoys=B.mean(0), buoys_end=B[-1], t_end=t[-1], boat_end=C[-1], v_end=V[-1], sig_end=sd[-1],
             toa_pred=toa_pred, t_emit_sig=t_emit_sig, t_emit_sig_cond=t_emit_sig_cond, t_emit_cov=t_emit_cov,
             g_cov=g_cov, g_cov_names=np.array(["b_S", "c", "d_S"]))
    print("\n".join(lines))
    plot(cfg, fg, Z, C, B, sd, info)
    return dict(Z=Z, V=V, C=C, B=B, g=g, sd=sd)


def plot(cfg, fg, Z, C, B, sd, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    a = ax[0]
    for r in np.unique(info["run"]):
        m = info["run"] == r
        a.plot(*Z[m, :2].T, lw=1.2, label=f"pinger run {r + 1}")
        a.plot(*C[m].T, lw=0.6, alpha=0.6, label=f"boat run {r + 1}")
    for j, nm in enumerate(["Rx902", "Rx01"]):
        a.plot(*fg.geo.buoys_nominal[j], "k^", ms=9)
        a.plot(*B[:, j].T, "r-", lw=2)
        a.annotate(nm, fg.geo.buoys_nominal[j], xytext=(6, 6), textcoords="offset points")
    a.plot(*fg.geo.shore[:2], "ks", ms=9); a.annotate("USBL", fg.geo.shore[:2], xytext=(6, 6), textcoords="offset points")
    a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_xlabel("east [m]"); a.set_ylabel("north [m]"); a.set_title("phase 1 track (ENU, origin USBL)")
    a = ax[1]
    tt = info["t"] - info["t"][0]
    a.set_ylim(0, np.nanpercentile(sd, 99) * 1.2)
    a.plot(tt, sd[:, 0], label="sigma east"); a.plot(tt, sd[:, 1], label="sigma north")
    a.plot(tt, fg.m["toa_mask"].sum(1) * 2, "k.", ms=2, label="2 x receivers with TOA")
    a.set_xlabel("time since first ping [s]"); a.set_ylabel("m"); a.grid(alpha=.3); a.legend()
    a.set_title("1-sigma position (graph)")
    fig.tight_layout(); fig.savefig(OUT / "track.png", dpi=110)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    cfg = config.load(*args)
    global OUT
    if "--no-bearing" in flags:
        OUT = ROOT / "out" / "phase1_toa_only"
    fg, info = build(cfg, use_bearing="--no-bearing" not in flags, use_mp="--mp" in flags)
    print(f"K = {fg.K}, TOAs per receiver {fg.m['toa_mask'].sum(0).tolist()}; " + "; ".join(info["notes"]))
    res, dt = solve(fg, verbose="--verbose" in flags)
    print(f"solved in {dt:.1f} s")
    if "--fixes" in flags:
        OUT = ROOT / "out" / "phase1_fixes"
        fg, res = solve_with_fixes(cfg, fg, res, info, cfg.priors.get("fix_sigma", 10.0))
    out = outputs(cfg, fg, res, info)
    if "--no-validate" not in flags:
        import validate
        validate.report(fg, res, info, OUT)
    return fg, res, out


if __name__ == "__main__":
    main(sys.argv[1:])
