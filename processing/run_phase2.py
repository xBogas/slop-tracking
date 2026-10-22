"""
Phase 2 end to end: noise TDOAs (+ USBL noise bearings) -> factor-graph
smoother -> boat track from its radiated noise.

Inputs (produced by the other processing steps):
  out/phase1/state.npz            run_phase1.py: posterior globals, buoys, last boat state
  out/phase2/tdoa_noise.npz       noise_tdoa.py    (required)
  out/phase2/bearings_noise.npz   noise_bearing.py (optional)

The emitter is the boat itself (the noise centre, near the propeller), so no
tow conversion. The calibrations come from phase 1 as tight priors: sound
speed, USBL clock (the TDOAs already have the phase 1 clock model removed,
so b_S and d_S here are the residual offset and drift), array orientation
alpha, and the buoy positions. The track starts where the phase 1 boat
track ends (795.4 s), which is the start prior.

Bearings after the buoy files end (USBL only) carry no range, so they are
written as bearings only (bearings_only.csv), not as positions.

Outputs in out/phase2/: track.csv, bearings_only.csv, globals.txt, track.png.

Run: uv run python processing/run_phase2.py [processing/forfun.toml] [--no-bearing] [--no-validate] [--verbose]
"""
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for d in ("estimation", "motion_model"):
    sys.path.insert(0, str(ROOT / d))
import config                                                           # noqa: E402
from fg_tracker import (Config, Geometry, FactorGraph, initial_guess,   # noqa: E402
                        ground_velocity, GLOBALS)

OUT = ROOT / "out" / "phase2"
NAMES = ["Rx902", "Rx01", "USBL"]
WEAK_SIGMA = 30.0          # [m] 1-sigma above which an epoch is flagged as weak geometry


def on_grid(t_meas, t_grid, step):
    """Index into t_meas of the window nearest to each epoch (-1 if none within step/2)."""
    j = np.clip(np.searchsorted(t_meas, t_grid), 1, len(t_meas) - 1)
    j = np.where(np.abs(t_meas[j - 1] - t_grid) < np.abs(t_meas[j] - t_grid), j - 1, j)
    return np.where(np.abs(t_meas[j] - t_grid) <= step / 2 + 1e-9, j, -1)


def build(cfg, use_bearing=True, q_buoy=0.2):
    nz = cfg.noise
    st = np.load(ROOT / "out" / "phase1" / "state.npz")
    g1, s1 = st["globals"], st["globals_sig"]
    t = np.arange(nz["t0"], nz["t1"] + 1e-9, nz["step"])
    K = len(t)
    buoys = [cfg.receiver(n) for n in NAMES[:2]]
    usbl = cfg.receiver("USBL")

    td = np.load(OUT / "tdoa_noise.npz")
    names = [str(n) for n in td["names"]] if "names" in td else NAMES
    assert names == NAMES, names
    pairs = [tuple(int(i) for i in p) for p in td["pair_idx"]]
    j = on_grid(td["t"], t, nz["step"])
    has = j >= 0
    meas = dict(pairs=pairs, tdoa=np.full((K, len(pairs)), np.nan),
                tdoa_mask=np.zeros((K, len(pairs)), bool), tdoa_sig=np.full((K, len(pairs)), 1.0))
    meas["tdoa"][has] = td["tdoa"][j[has]]
    meas["tdoa_mask"][has] = td["tdoa_mask"][j[has]]
    meas["tdoa_sig"][has] = td["tdoa_sig"][j[has]]
    meas["tdoa_mask"] &= np.isfinite(meas["tdoa"]) & np.isfinite(meas["tdoa_sig"])
    meas["bearing"] = np.full(K, np.nan)
    meas["bearing_mask"] = np.zeros(K, bool)
    notes = [f"TDOA windows per pair {dict(zip([f'{NAMES[a]}-{NAMES[b]}' for a, b in pairs], meas['tdoa_mask'].sum(0).tolist()))}"]

    bp = OUT / "bearings_noise.npz"
    brg = None
    if use_bearing and bp.exists():
        brg = np.load(bp)
        jb = on_grid(brg["t"], t, nz["step"])
        ok = jb >= 0
        ok[ok] &= brg["mask"][jb[ok]]
        s = float(brg["handedness"])
        meas["bearing"][ok] = s * brg["az_array"][jb[ok]]
        meas["bearing_mask"][ok] = True
        meas["bearing_sig"] = np.full(K, np.deg2rad(2.0))
        meas["bearing_sig"][ok] = np.maximum(brg["sigma"][jb[ok]], np.deg2rad(0.5))
        notes.append(f"bearings: {ok.sum()} epochs")

    # Residual USBL clock after the phase 1 model: the offset is extrapolated
    # from the pings, so its sigma grows with the drift sigma over the gap.
    gap = t[0] - float(st["t0"])
    g_mean = [0.0, g1[1], 0.0, g1[3], 0.0, 0.0, 0.0]
    g_sig = [float(np.hypot(s1[0], s1[6] * gap)),
             float(np.hypot(s1[1], np.deg2rad(1.0))),   # alpha: phase 1 + 1 deg for a broadband-noise bias
             1e-9,                                       # no pinger clock in phase 2
             float(cfg.priors["sound_speed_sigma"]), 1.0, 1.0,
             float(s1[6])]
    be = st["boat_end"]
    fcfg = Config(mode="noise", motion="ground", Tp=nz["step"],
                  sig_buoy0=3.0, q_buoy=q_buoy,
                  g_mean=tuple(g_mean), g_sig=tuple(g_sig), nb=2,
                  start_prior=(float(be[0]), float(be[1]), float(nz["start_sigma"])),
                  region=(-1500.0, 1000.0, -1500.0, 1000.0))
    geo = Geometry(shore=np.r_[usbl.en, -usbl.depth],
                   buoys_nominal=np.array(st["buoys"]),
                   buoy_depth=-np.array([b.depth for b in buoys]), src_depth=-nz["src_depth"])
    info = dict(t=t, notes=notes, state=st, brg=brg)
    return FactorGraph(fcfg, geo, K, meas), info


def solve(fg, verbose=False):
    t0 = time.time()
    v0 = initial_guess(fg)
    res = fg.solve(v0, max_iter=150, verbose=verbose)
    return res, time.time() - t0


def outputs(cfg, fg, res, info):
    OUT.mkdir(parents=True, exist_ok=True)
    K, t, m = fg.K, info["t"], fg.m
    Z, _, B, g = fg.unpack(res.x)
    V = ground_velocity(fg, res.x)
    sd = fg.marginal_std(res, fg.iZ[:, :2].ravel()).reshape(K, 2)
    lat, lon = cfg.to_llh(Z[:, 0], Z[:, 1])
    ntd = m["tdoa_mask"].sum(1)
    weak = (np.hypot(*sd.T) > WEAK_SIGMA) | ((ntd + m["bearing_mask"]) < 2)
    hdr = ("t_buoy_s,boat_e,boat_n,boat_lat,boat_lon,v_e,v_n,speed,sig_e,sig_n,"
           "n_tdoa,has_bearing,weak")
    rows = np.column_stack([t, Z[:, 0], Z[:, 1], lat, lon, V[:, 0], V[:, 1], np.hypot(*V.T),
                            sd[:, 0], sd[:, 1], ntd, m["bearing_mask"], weak])
    fmt = ["%.3f", "%.2f", "%.2f", "%.7f", "%.7f", "%.3f", "%.3f", "%.3f", "%.2f", "%.2f", "%d", "%d", "%d"]
    np.savetxt(OUT / "track.csv", rows, delimiter=",", header=hdr, comments="", fmt=fmt)

    # USBL-only bearings after the buoys stop: ENU direction from the USBL
    brg = info["brg"]
    if brg is not None:
        after = (brg["t"] > t[-1]) & brg["mask"]
        th = float(brg["handedness"]) * brg["az_array"][after] + g[1]   # graph: bearing = atan2(p - usbl) - alpha
        az_math = np.rad2deg(np.mod(th, 2 * np.pi))
        compass = np.mod(90.0 - az_math, 360.0)
        np.savetxt(OUT / "bearings_only.csv",
                   np.column_stack([brg["t"][after], az_math, compass, np.rad2deg(brg["sigma"][after])]),
                   delimiter=",", header="t_buoy_s,bearing_enu_deg_ccw_from_east,bearing_deg_true_north,sigma_deg",
                   comments="", fmt=["%.3f", "%.2f", "%.2f", "%.2f"])

    st = info["state"]
    lines = [f"K = {K} epochs ({t[0]:.0f}-{t[-1]:.0f} s), {res.nit} iterations, cost {res.cost:.1f}"]
    for i, nm in enumerate(GLOBALS):
        lines.append(f"{nm:6s} = {g[i]: .6g}   (prior {fg.cfg.g_mean[i]: .6g} +- {fg.cfg.g_sig[i]:.3g})")
    for jb, nm in enumerate(NAMES[:2]):
        mv = B[:, jb] - fg.geo.buoys_nominal[jb]
        lines.append(f"{nm} shift from phase 1 estimate: ({mv[:, 0].mean():+.1f}, {mv[:, 1].mean():+.1f}) m")
    d0 = np.hypot(*(Z[0, :2] - st["boat_end"]))
    lines.append(f"start: {d0:.1f} m from the phase 1 end position ({st['boat_end'][0]:.1f}, {st['boat_end'][1]:.1f})")
    lines.append(f"1-sigma median {np.median(np.hypot(*sd.T)):.1f} m; weak epochs {weak.sum()}/{K}")
    lines += info["notes"]
    (OUT / "globals.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(cfg, fg, Z, V, B, sd, weak, info)
    return dict(Z=Z, V=V, B=B, g=g, sd=sd, weak=weak)


def plot(cfg, fg, Z, V, B, sd, weak, info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = info["t"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
    a = ax[0]
    p1 = ROOT / "out" / "phase1" / "track.csv"
    if p1.exists():
        tr = np.genfromtxt(p1, delimiter=",", names=True)
        a.plot(tr["boat_e"], tr["boat_n"], color="0.7", lw=0.8, label="phase 1 boat (pinger)")
    sc = a.scatter(Z[:, 0], Z[:, 1], c=t, s=6, cmap="viridis", label="phase 2 boat (noise)")
    a.plot(*Z[weak, :2].T, "rx", ms=4, label="weak geometry")
    plt.colorbar(sc, ax=a, label="buoy time [s]")
    for jb, nm in enumerate(NAMES[:2]):
        a.plot(*B[:, jb].mean(0), "k^", ms=9); a.annotate(nm, B[:, jb].mean(0), xytext=(6, 6), textcoords="offset points")
    a.plot(*fg.geo.shore[:2], "ks", ms=9); a.annotate("USBL", fg.geo.shore[:2], xytext=(6, 6), textcoords="offset points")
    a.set_aspect("equal"); a.grid(alpha=.3); a.legend(fontsize=8)
    a.set_xlabel("east [m]"); a.set_ylabel("north [m]"); a.set_title("phase 2 track (ENU, origin USBL)")
    a = ax[1]
    a.plot(t, np.hypot(*sd.T), label="1-sigma position [m]")
    a.plot(t, np.hypot(*V.T) * 10, label="speed x10 [m/s]")
    a.plot(t, fg.m["tdoa_mask"].sum(1) * 5 + fg.m["bearing_mask"] * 2, "k.", ms=2, label="5 x TDOAs + 2 x bearing")
    a.set_ylim(0, max(60, np.nanpercentile(np.hypot(*sd.T), 95) * 1.2))
    a.set_xlabel("buoy time [s]"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "track.png", dpi=110)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    cfg = config.load(*args)
    fg, info = build(cfg, use_bearing="--no-bearing" not in flags)
    print(f"K = {fg.K}; " + "; ".join(info["notes"]))
    res, dt = solve(fg, verbose="--verbose" in flags)
    print(f"solved in {dt:.1f} s")
    out = outputs(cfg, fg, res, info)
    if "--no-validate" not in flags:
        import validate
        validate.report_phase2(fg, res, info, OUT)
    return fg, res, out


if __name__ == "__main__":
    main(sys.argv[1:])
