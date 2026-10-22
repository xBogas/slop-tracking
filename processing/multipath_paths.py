"""
Multipath paths on the buoys predicted from the bathymetry, and the per-ping
multipath measurements they give (phase 1 pinger). The receiver design of
notes/D2_multipath.md, without the tracker:

1. `multipath_predict.predict`: per ping and buoy, each path's delay after
   the direct path, launch and arrival angle and Doppler from the EMODnet
   bathymetry at the configured depths, and a search window per family from
   a grid of the unknown depths and tide (see that module).
2. Doppler first: the compressed pulses (multipath_range.extract) are
   already Doppler-matched. The replica is time-scaled by eta from the
   arrival sequence (detect.eta_track), which is the same as resampling the
   signal by 1/eta. `multipath_predict.doppler_budget` gives the delay bias
   without it, and the per-path residual bias left by one common scale.
3. `components`: free sparse path extraction (multipath_range.sparse_paths,
   OMP on the measured pulse model) per ping. Cached.
4. `measurements`: the false-alarm level is what the extractor finds before
   the direct-path window, where no path can arrive. The direct path is the
   earliest component above that level inside the window around the TOA the
   phase 1 track predicts (not the strongest one). Each later component above
   the level goes to the family whose window holds it (ambiguous if several
   do). It is a usable measurement when its family is unique, it is resolved
   (>= 1/B after the direct path, >= 1/(2B) from its neighbours) and it
   persists: the same family at the predicted delay change in >= PERSIST_MIN
   of the +-PERSIST_HALF neighbouring pings. sigma (every row) = CRLB (+)
   the ping-to-ping scatter of usable delays of that buoy and family (else
   that family on the other buoy, else 1/(B sqrt 12)); `sigma_basis` says which.

Measurements are delay differences tau_path - tau_direct on one buoy, so
they need no clock and no emission time.

Output: out/multipath/paths_phase1.npz (everything the notebook plots) and
out/multipath/paths_phase1.csv (the measurement list, one row per assigned
component, usable or not).

Run: uv run python processing/multipath_paths.py
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import multipath_range as mp
import stamp
from multipath_predict import FAMS, STATE, TOA, ZR_DELTA, ZS_GRID, geometry, predict, tide_offsets

OUT = config.ROOT / "out" / "multipath"

MAX_PATHS = 15                                # OMP components per ping
FA_QUANTILE = 0.95                            # false-alarm level: this quantile of the pre-arrival components
PERSIST_HALF, PERSIST_MIN, PERSIST_TOL = 2, 2, 30e-6

MEAS_DTYPE = [("row", "i4"), ("ping", "i4"), ("run", "i2"), ("t_emit", "f8"), ("buoy", "U5"),
              ("family", "U12"), ("tau_direct", "f8"), ("dtau", "f8"), ("sigma", "f8"), ("sigma_crlb", "f8"),
              ("sigma_basis", "U10"),
              ("level_db", "f4"), ("snr_db", "f4"), ("pred", "f8"), ("win_lo", "f8"), ("win_hi", "f8"),
              ("launch_deg", "f4"), ("arrival_deg", "f4"), ("doppler_hz", "f4"),
              ("unique", "?"), ("resolved", "?"), ("persistent", "?"), ("usable", "?")]


def resolution(cfg):
    return 1.0 / (cfg.pinger.f1 - cfg.pinger.f0)


# --------------------------------------------------------------------------
# 3. Path extraction
# --------------------------------------------------------------------------
def _components_job(args):
    name, rows = args
    d = mp.extract(name)
    pulse = mp.Pulse(d["X"])
    i0 = int(round(-mp.WIN[0] * d["fs"]))
    out = []
    for k in rows:
        taus, a = mp.sparse_paths(d["y"][k], pulse, d["noise"][k], max_paths=MAX_PATHS)
        snr = np.abs(a) ** 2 * pulse.e0 / d["noise"][k]
        out.append((np.full(len(taus), k), (taus - i0) / d["fs"], a, 10 * np.log10(snr)))
    return out


def components(name, nproc=16):
    """OMP components of every compressed pulse on buoy `name` (cached): flat
    arrays row, tau [s after the measured TOA], amp (complex), snr_db."""
    cache = OUT / f"paths_components_{name}.npz"
    key = stamp.key(code=__file__, arrivals=mp._arrivals_key(name), max_paths=MAX_PATHS)
    hit = stamp.load(cache, key)
    if hit is not None:
        return hit
    rows = np.flatnonzero(mp.extract(name)["ok"])
    with Pool(nproc) as pool:
        res = pool.map(_components_job, [(name, p) for p in np.array_split(rows, nproc * 4)])
    res = [r for part in res for r in part]
    out = {f: np.concatenate([r[i] for r in res]) for i, f in enumerate(("row", "tau", "amp", "snr_db"))}
    stamp.save(cache, key, **out)
    return out


# --------------------------------------------------------------------------
# 4. Direct path, association, measurements
# --------------------------------------------------------------------------
def crlb(snr_db, cfg):
    """Delay CRLB of one path [s]: 1 / (2 pi beta sqrt(SNR)), beta = B / sqrt(12)."""
    beta = (cfg.pinger.f1 - cfg.pinger.f0) / np.sqrt(12)
    return 1.0 / (2 * np.pi * beta * np.sqrt(10 ** (np.asarray(snr_db) / 10)))


def pick_direct(cmp, centre, half, thr_db):
    """Per row: index into cmp of the direct path (earliest component >= thr_db
    rel. the strongest of that row, inside centre +- half), or -1; also the
    level of every component rel. its row's strongest [dB]."""
    rows = cmp["row"]
    lvl = 20 * np.log10(np.abs(cmp["amp"]))
    top = np.full(len(centre), -np.inf)
    np.maximum.at(top, rows, lvl)
    rel = lvl - top[rows]
    direct = np.full(len(centre), -1)
    ok = (np.abs(cmp["tau"] - centre[rows]) <= half) & (rel >= thr_db)
    for i in np.flatnonzero(ok):
        if direct[rows[i]] < 0 or cmp["tau"][i] < cmp["tau"][direct[rows[i]]]:
            direct[rows[i]] = i
    return direct, rel


def measurements(cfg=None, verbose=True):
    """Everything for both buoys: {name: dict(pred, cmp, direct, rel, thr,
    centre, ...)} and the measurement table (MEAS_DTYPE)."""
    cfg = cfg or config.load()
    g = geometry(cfg)
    T = g["T"]
    toa_pred = np.load(STATE)["toa_pred"][T["k"] - T["k"].min()]
    res_ = resolution(cfg)
    per, table = {}, []
    for name in mp.BUOYS:
        j = list(T["receivers"]).index(name)
        pred, cmp = predict(name, cfg), components(name)
        centre = toa_pred[:, j] - T["toa"][:, j]               # predicted direct path on the TOA-centred grid
        rows = cmp["row"]
        # false alarms: components before the direct-path window (nothing can arrive there)
        _, rel = pick_direct(cmp, centre, res_, 0.0)
        early = cmp["tau"] < centre[rows] - res_
        thr = float(np.quantile(rel[early], FA_QUANTILE))
        direct, rel = pick_direct(cmp, centre, res_, thr)
        strongest = np.full(len(T["k"]), -1)
        for i in np.argsort(rel):
            strongest[rows[i]] = i                            # the last one written is the strongest
        lvl = 20 * np.log10(np.abs(cmp["amp"]))
        cand = []                                             # (row, comp index, families)
        lo, hi = pred["lo"] - res_ / 2, pred["hi"] + res_ / 2
        for k in np.flatnonzero(direct >= 0):
            iD = direct[k]
            idx = np.flatnonzero((rows == k) & (cmp["tau"] > cmp["tau"][iD]) & (rel >= thr))
            for i in idx:
                dt = cmp["tau"][i] - cmp["tau"][iD]
                fams = [f for q, f in enumerate(FAMS) if lo[k, q] <= dt <= hi[k, q]]
                if fams:
                    cand.append((k, i, fams, dt))
        # uniqueness: one family, and the only candidate in that family's window
        count = {}
        for k, i, fams, dt in cand:
            for f in fams:
                count[(k, f)] = count.get((k, f), 0) + 1
        by_fam = {}
        for k, i, fams, dt in cand:
            if len(fams) == 1:
                by_fam.setdefault(fams[0], {}).setdefault(k, []).append(dt)
        kk = T["k"]
        for k, i, fams, dt in cand:
            iD = direct[k]
            others = np.flatnonzero((rows == k) & (rel >= thr))
            gap = np.min(np.abs(cmp["tau"][others[others != i]] - cmp["tau"][i]), initial=np.inf)
            f = fams[0]
            q = FAMS.index(f)
            unique = len(fams) == 1 and count[(k, f)] == 1
            resolved = dt >= res_ and gap >= res_ / 2
            n_pers = 0
            if unique:
                for k2 in range(k - PERSIST_HALF, k + PERSIST_HALF + 1):
                    if k2 == k or not 0 <= k2 < len(kk) or abs(kk[k2] - kk[k]) > PERSIST_HALF:
                        continue
                    step = pred["delay"][k2, q] - pred["delay"][k, q]
                    step = 0.0 if not np.isfinite(step) else step
                    if any(abs(d2 - dt - step) < PERSIST_TOL for d2 in by_fam.get(f, {}).get(k2, [])):
                        n_pers += 1
            snr = cmp["snr_db"][i]
            table.append((k, kk[k], g["run"][k], g["t_emit"][k], name, "|".join(fams), cmp["tau"][iD], dt, np.nan,
                          np.hypot(crlb(snr, cfg), crlb(cmp["snr_db"][iD], cfg)), "",
                          lvl[i] - lvl[iD], snr, pred["delay"][k, q], pred["lo"][k, q], pred["hi"][k, q],
                          pred["launch"][k, q + 1], pred["arrival"][k, q + 1],
                          pred["doppler"][k, q + 1] - pred["doppler"][k, 0],
                          unique, resolved, n_pers >= PERSIST_MIN, False))
        per[name] = dict(pred=pred, cmp=cmp, direct=direct, strongest=strongest, rel=rel, thr=thr,
                         centre=centre, early=early)
    M = np.array(table, MEAS_DTYPE)
    M["usable"] = M["unique"] & M["resolved"] & M["persistent"]
    # empirical sigma: ping-to-ping scatter of usable delays, after the predicted change
    for name in mp.BUOYS:
        for q, f in enumerate(FAMS):
            m = M[M["usable"] & (M["buoy"] == name) & (M["family"] == f)]
            per[name].setdefault("sigma_emp", {})[f] = scatter(m, per[name]["pred"]["delay"][:, q])
    # sigma of every row (usable or not): CRLB (+) the scatter of its buoy and family ("scatter"),
    # else of that family on the other buoy ("pooled"), else a uniform spread over one resolution cell
    # ("resolution": S, B and ambiguous rows, which never pass the persistence test)
    for i, r in enumerate(M):
        own = per[r["buoy"]]["sigma_emp"].get(r["family"], np.nan)
        other = np.nanmax([per[n]["sigma_emp"].get(r["family"], np.nan) for n in mp.BUOYS] + [-1.0])
        if np.isfinite(own):
            emp, basis = own, "scatter"
        elif other > 0:
            emp, basis = other, "pooled"
        else:
            emp, basis = resolution(cfg) / np.sqrt(12), "resolution"
        M["sigma"][i], M["sigma_basis"][i] = np.hypot(r["sigma_crlb"], emp), basis
    if verbose:
        print(summary(M, per, cfg))
    return per, M


def scatter(m, pred_delay):
    """Robust ping-to-ping scatter of delays m["dtau"] on consecutive pings, net of
    the predicted change: 1.4826 MAD(diff) / sqrt(2); NaN with < 5 pairs."""
    o = np.argsort(m["ping"])
    p, d, r = m["ping"][o], m["dtau"][o], m["row"][o]
    pair = np.diff(p) == 1
    if pair.sum() < 5:
        return np.nan
    step = np.nan_to_num(pred_delay[r[1:]] - pred_delay[r[:-1]])
    x = (np.diff(d) - step)[pair]
    return 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(2)


def depth_fit(M, per, usable=True):
    """Misfit of one common (pinger depth, hydrophone depth, tide) for all
    (usable) assigned delays, per buoy: {name: (nzs, nzr, ntide) median |dtau -
    grid delay| [s]} and the number of measurements used."""
    out = {}
    for name in mp.BUOYS:
        m = M[(M["buoy"] == name) & M["unique"] & (M["usable"] if usable else True)]
        G = per[name]["pred"]["grid"]
        q = np.array([FAMS.index(f) for f in m["family"]], int)
        r = np.abs(m["dtau"][:, None, None, None] - G[m["row"], ..., q])      # (n, nzs, nzr, ntide)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = (np.nanmedian(r, 0), len(m))
    return out


def summary(M, per, cfg):
    T = np.load(TOA)
    lines = [f"resolution 1/B = {resolution(cfg) * 1e3:.2f} ms; windows span pinger depth "
             f"{ZS_GRID[0]:g}-{ZS_GRID[-1]:g} m, hydrophone depth {ZR_DELTA[0]:+g}..{ZR_DELTA[-1]:+g} m, "
             f"water level tide {cfg.site.get('tide', 0.0):+g} m {tide_offsets(cfg)[0]:+g}..{tide_offsets(cfg)[-1]:+g} m"]
    for name in mp.BUOYS:
        p = per[name]
        j = list(T["receivers"]).index(name)
        n_ok = int(mp.extract(name)["ok"].sum())
        has = p["direct"] >= 0
        not_strong = np.sum(has & (p["direct"] != p["strongest"]))
        m = M[M["buoy"] == name]
        u = m[m["usable"]]
        lines.append(f"{name}: {n_ok} pings; false-alarm level {p['thr']:.1f} dB rel. the strongest; "
                     f"direct path found in {has.sum()}, not the strongest component in {not_strong} "
                     f"({100 * not_strong / max(has.sum(), 1):.0f} %)")
        lines.append(f"   {len(m)} components in a window: {m['unique'].sum()} unique, "
                     f"{(m['unique'] & m['resolved']).sum()} unique+resolved, {len(u)} usable "
                     f"(also persistent) on {len(np.unique(u['row']))} pings")
        for f in FAMS:
            uf = u[u["family"] == f]
            lines.append(f"   {f:3s} usable {len(uf):4d}  sigma ping-to-ping {per[name]['sigma_emp'][f] * 1e6:6.1f} us"
                         f"  median CRLB {np.median(uf['sigma_crlb']) * 1e6 if len(uf) else np.nan:5.1f} us")
    u = M[M["usable"]]
    per_ping = {}
    for r, f in zip(u["ping"], u["family"]):
        per_ping.setdefault(r, set()).add(f)
    n2 = sum(len(v) >= 2 for v in per_ping.values())
    lines.append(f"pings with >= 1 usable multipath delay (either buoy): {len(per_ping)} of {len(T['k'])}; "
                 f"with >= 2 different families: {n2}")
    return "\n".join(lines)


def write_csv(M, path):
    hdr = ("ping,run,t_emit_buoy_s,buoy,family,dtau_us,sigma_us,sigma_crlb_us,sigma_basis,level_db,snr_db,pred_us,"
           "win_lo_us,win_hi_us,launch_deg,arrival_deg,doppler_hz,unique,resolved,persistent,usable")
    us = lambda v: "" if not np.isfinite(v) else f"{v * 1e6:.1f}"
    lines = [hdr]
    for r in M:
        lines.append(",".join([str(r["ping"]), str(r["run"]), f"{r['t_emit']:.6f}", r["buoy"], r["family"],
                               us(r["dtau"]), us(r["sigma"]), us(r["sigma_crlb"]), r["sigma_basis"], f"{r['level_db']:.1f}",
                               f"{r['snr_db']:.1f}", us(r["pred"]), us(r["win_lo"]), us(r["win_hi"]),
                               f"{r['launch_deg']:.2f}", f"{r['arrival_deg']:.2f}", f"{r['doppler_hz']:.3f}",
                               *(str(int(r[f])) for f in ("unique", "resolved", "persistent", "usable"))]))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    tic = time.time()
    cfg = config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    per, M = measurements(cfg)
    arrays = dict(meas=M, fams=np.array(FAMS), zs_grid=ZS_GRID, zr_delta=ZR_DELTA, tide=tide_offsets(cfg),
                  resolution=resolution(cfg))
    for name, p in per.items():
        for k, v in p["pred"].items():
            arrays[f"{name}_pred_{k}"] = v
        for k, v in p["cmp"].items():
            arrays[f"{name}_cmp_{k}"] = v
        for k in ("direct", "strongest", "rel", "thr", "centre", "early"):
            arrays[f"{name}_{k}"] = p[k]
        arrays[f"{name}_sigma_emp"] = np.array([p["sigma_emp"][f] for f in FAMS])
    np.savez(OUT / "paths_phase1.npz", **arrays)
    write_csv(M, OUT / "paths_phase1.csv")
    print(f"wrote {OUT / 'paths_phase1.npz'} and paths_phase1.csv ({len(M)} rows, "
          f"{M['usable'].sum()} usable) in {time.time() - tic:.0f} s")


if __name__ == "__main__":
    main()
