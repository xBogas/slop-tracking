"""
Phase 1 measurements per receiver as CSV, on one clock (buoy time), with a
sigma on every reading. out/export/:

  first_arrival_<rx>.csv   per chirp heard by <rx> (Rx902, Rx01, USBL): the
                           first-arrival time synced to buoy time, its sigma,
                           and the one-way range c (toa - emission time)
  usbl_bearing_range.csv   per chirp heard by the USBL: range (as above) and
                           bearing to the pinger, with sigmas
  multipath_<buoy>.csv     per buoy: every component assigned to a path family
                           (multipath_paths.py), its delay after the direct
                           path, its absolute arrival time and sigma, and flags
  README.md                the column definitions (written from COLUMNS)

Clock: the buoys share one clock (buoy time = seconds since the buoy files'
start); the USBL's TOAs are mapped onto it with the phase 1 clock model
(usbl_offset + b_S + d_S t). Emission times, sound speed, clock and array
orientation are the phase 1 posterior (out/phase1/state.npz, track.csv).

Every reading has two sigmas. `*_sigma` is the total, from the full phase 1
posterior with its correlations (emission time, sound speed, USBL clock,
array orientation). `*_sigma_random` is the part that varies from ping to
ping, with those globals and the buoy positions held fixed; it averages
down. The difference is shared by all pings and does not average down.

Run: uv run python processing/export.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ROOT = config.ROOT
OUT = ROOT / "out" / "export"
TOA = ROOT / "out" / "toa_phase1.npz"
STATE = ROOT / "out" / "phase1" / "state.npz"
TRACK = ROOT / "out" / "phase1" / "track.csv"
BEARINGS = ROOT / "out" / "usbl" / "bearings_phase1.npz"
PATHS = ROOT / "out" / "multipath" / "paths_phase1.npz"
OUTLIER_Z = 5.0                                    # |TOA residual| / sigma above which a reading is flagged

COLUMNS = {
    "first_arrival": [
        ("ping", "ping index (emission number since the first ping)"),
        ("run", "run index (0 = run 1, 1 = run 2; pings before run 1 count as run 1)"),
        ("t_emit_buoy_s", "emission time of the chirp, buoy time [s] (phase 1 solution)"),
        ("t_emit_sigma_s", "its total sigma [s] (phase 1 marginal)"),
        ("t_emit_sigma_random_s", "its ping-to-ping sigma [s] (globals and buoy positions held fixed)"),
        ("toa_buoy_s", "first-arrival time (start of the chirp), synced to buoy time [s]"),
        ("toa_sigma_s", "total sigma of toa_buoy: toa_sigma_random (+) sync_sigma [s]"),
        ("toa_sigma_random_s", "receiver's empirical TOA noise (+) per-ping CRLB [s]"),
        ("toa_sigma_crlb_s", "per-ping CRLB alone [s]"),
        ("sync_offset_s", "USBL only: clock correction applied, toa_buoy = t_usbl_clock - sync_offset [s]"),
        ("sync_sigma_s", "USBL only: sigma of that correction, shared by all pings [s] "
                         "(0 for the buoys: they define the clock)"),
        ("t_rx_clock_s", "arrival time on the receiver's own clock [s] (file time)"),
        ("snr_db", "matched-filter output SNR of the first path [dB]"),
        ("range_m", "one-way range to the pinger, c (toa_buoy - t_emit) [m]"),
        ("range_sigma_m", "total sigma of range [m]: full posterior of (t_emit, c, USBL clock) (+) toa noise"),
        ("range_sigma_random_m", "ping-to-ping sigma of range [m]: c (toa_sigma_random (+) t_emit_sigma_random)"),
        ("toa_resid_us", "toa minus the phase 1 prediction [us]"),
        ("outlier", f"1 if |toa_resid| > {OUTLIER_Z:g} toa_sigma_random"),
    ],
    "usbl_bearing_range": [
        ("ping", "ping index"), ("run", "run index"),
        ("t_emit_buoy_s", "emission time, buoy time [s]"),
        ("toa_buoy_s", "first arrival at the array centre, synced to buoy time [s] (blank if no TOA)"),
        ("range_m", "one-way range to the pinger [m] (as in first_arrival_USBL.csv)"),
        ("range_sigma_m", "total sigma [m]"), ("range_sigma_random_m", "ping-to-ping sigma [m]"),
        ("bearing_true_deg", "bearing from the array to the pinger, clockwise from true north [deg]"),
        ("az_enu_deg", "the same as an ENU azimuth: counter-clockwise from east [deg] (repo convention)"),
        ("bearing_sigma_deg", "total sigma: bearing_sigma_random (+) array orientation sigma (phase 1 alpha) [deg]"),
        ("bearing_sigma_random_deg", "ping-to-ping sigma (usbl.py per-ping sigma, floor 0.5 deg as in phase 1) [deg]"),
        ("bearing_resid_deg", "bearing minus the bearing to the phase 1 track [deg]"),
        ("bearing_used", "1 if used by the phase 1 solve (usbl.py mask: good delays, no wall overlap)"),
        ("wall_overlap", "1 if the wall echo overlaps the direct arrival"),
    ],
    "multipath": [
        ("ping", "ping index"), ("run", "run index"),
        ("t_emit_buoy_s", "emission time, buoy time [s]"),
        ("family", "path family (S surface, B bottom, SB, BS; 'S|B' = ambiguous: several windows hold it)"),
        ("t_direct_buoy_s", "arrival time of the direct path picked by multipath_paths.py, buoy time [s]"),
        ("t_path_buoy_s", "arrival time of this path, buoy time [s]"),
        ("dtau_us", "delay after the direct path [us] (clock-free measurement)"),
        ("sigma_us", "sigma of dtau [us]: CRLB (+) ping-to-ping scatter (see sigma_basis)"),
        ("sigma_crlb_us", "CRLB alone [us]"),
        ("sigma_basis", "scatter: this buoy and family; pooled: same family, other buoy; "
                        "resolution: 1/(B sqrt 12) (no persistent detections of the family)"),
        ("level_db", "amplitude relative to the direct path [dB]"), ("snr_db", "SNR of the component [dB]"),
        ("pred_us", "predicted delay at the configured depths and tide [us] (first family if ambiguous)"),
        ("win_lo_us", "search window, low end [us]"), ("win_hi_us", "search window, high end [us]"),
        ("launch_deg", "predicted launch angle at the pinger [deg, + down]"),
        ("arrival_deg", "predicted arrival angle at the buoy [deg, + down]"),
        ("doppler_hz", "predicted Doppler minus that of the direct path [Hz]"),
        ("unique", "1 if one family window holds it and no other component"),
        ("resolved", "1 if >= 1/B after the direct path and >= 1/(2B) from other components"),
        ("persistent", "1 if found on >= 2 of the +-2 neighbouring pings at the predicted delay change"),
        ("usable", "unique and resolved and persistent: the recommended measurements"),
    ],
}


def _fmt(v, nd):
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, str):
        return v
    return "" if not np.isfinite(v) else f"{v:.{nd}f}"


def write(path, kind, rows, digits):
    names = [c for c, _ in COLUMNS[kind]]
    lines = [",".join(names)]
    for r in rows:
        lines.append(",".join(_fmt(r[c], digits.get(c, 6)) for c in names))
    Path(path).write_text("\n".join(lines) + "\n")


def load(cfg):
    T, S = np.load(TOA), np.load(STATE)
    tr = np.genfromtxt(TRACK, delimiter=",", names=True)
    d = T["k"] - T["k"].min()                                 # rows of state / track (dense ping grid)
    g = dict(zip([str(n) for n in S["names"]], S["globals"]))
    gs = dict(zip([str(n) for n in S["names"]], S["globals_sig"]))
    return T, S, tr, d, g, gs


def first_arrivals(cfg):
    """{receiver: list of row dicts} (COLUMNS["first_arrival"])."""
    T, S, tr, d, g, gs = load(cfg)
    c = g["c"]
    tk = d * float(S["Tp"])                                  # epoch time of the USBL drift (fg_tracker.clock)
    t_emit, s_emit, s_emit_r = tr["t_emit_buoy_s"][d], S["t_emit_sig"][d], S["t_emit_sig_cond"][d]
    ct, G = S["t_emit_cov"][d], S["g_cov"]                    # cov(T_k, [b_S, c, d_S]), cov([b_S, c, d_S])
    pred = S["toa_pred"][d]
    out = {}
    for j, name in enumerate(T["receivers"]):
        name = str(name)
        rec = cfg.receiver(name)
        usbl = rec.kind == "usbl"
        sync = g["b_S"] + g["d_S"] * tk if usbl else np.zeros(len(d))
        s_sync = np.hypot(gs["b_S"], tk * gs["d_S"]) if usbl else np.zeros(len(d))
        rows = []
        for i in np.flatnonzero(T["toa_mask"][:, j]):
            toa_tab = T["toa"][i, j]                          # USBL: t_usbl - usbl_offset (associate.py)
            toa = toa_tab - sync[i]
            s_toa = np.hypot(T["toa_sigma"][j], T["toa_sigma_crlb"][i, j])
            rng = c * (toa - t_emit[i])
            resid = toa_tab - pred[i, j]
            # range = c (toa_tab - b_S - d_S tk - T_k): gradient on (T_k, b_S, c, d_S), full covariance
            a = np.array([-c, -c if usbl else 0.0, rng / c, -c * tk[i] if usbl else 0.0])
            P = np.empty((4, 4))
            P[0, 0], P[0, 1:], P[1:, 0], P[1:, 1:] = s_emit[i] ** 2, ct[i], ct[i], G
            rows.append(dict(
                ping=int(T["k"][i]), run=int(tr["run"][d[i]]), t_emit_buoy_s=t_emit[i], t_emit_sigma_s=s_emit[i],
                t_emit_sigma_random_s=s_emit_r[i], toa_buoy_s=toa, toa_sigma_s=np.hypot(s_toa, s_sync[i]),
                toa_sigma_random_s=s_toa, toa_sigma_crlb_s=T["toa_sigma_crlb"][i, j],
                sync_offset_s=float(T["usbl_offset"]) + sync[i] if usbl else 0.0, sync_sigma_s=s_sync[i],
                t_rx_clock_s=toa_tab + (float(T["usbl_offset"]) if usbl else rec.clock_offset_guess),
                snr_db=T["snr"][i, j], range_m=rng, range_sigma_m=np.sqrt(a @ P @ a + (c * s_toa) ** 2),
                range_sigma_random_m=c * np.hypot(s_toa, s_emit_r[i]), toa_resid_us=resid * 1e6,
                outlier=bool(abs(resid) > OUTLIER_Z * s_toa)))
        out[name] = rows
    return out


def usbl_bearing_range(cfg, fa):
    """Rows of COLUMNS["usbl_bearing_range"]: USBL range (from fa) and bearing."""
    T, S, tr, d, g, gs = load(cfg)
    b = np.load(BEARINGS)
    p0 = cfg.receiver("USBL").en
    by_ping = {r["ping"]: r for r in fa["USBL"]}
    bi = {int(k): i for i, k in enumerate(b["k"])}
    row_of = {int(k): i for i, k in enumerate(T["k"])}
    rows = []
    for k in sorted(set(by_ping) | set(bi)):
        if k not in row_of:
            continue
        dk = d[row_of[k]]
        r = by_ping.get(k, {})
        az = sb = res = np.nan
        used = wall = False
        if k in bi:
            i = bi[k]
            # the graph models bearing = atan2(pinger - usbl) - alpha with bearing = handedness * az_array
            az = float(b["handedness"]) * b["az_array"][i] + g["alpha"]
            sb = max(b["sigma"][i], np.deg2rad(0.5))
            p = np.r_[tr["pinger_e"][dk], tr["pinger_n"][dk]] - p0
            res = np.angle(np.exp(1j * (az - np.arctan2(p[1], p[0]))))
            used, wall = bool(b["mask"][i]), bool(b["wall_overlap"][i])
        az_deg = np.degrees(np.angle(np.exp(1j * az))) if np.isfinite(az) else np.nan
        rows.append(dict(
            ping=k, run=int(tr["run"][dk]), t_emit_buoy_s=tr["t_emit_buoy_s"][dk],
            toa_buoy_s=r.get("toa_buoy_s", np.nan), range_m=r.get("range_m", np.nan),
            range_sigma_m=r.get("range_sigma_m", np.nan), range_sigma_random_m=r.get("range_sigma_random_m", np.nan),
            bearing_true_deg=(90.0 - az_deg) % 360.0 if np.isfinite(az_deg) else np.nan, az_enu_deg=az_deg,
            bearing_sigma_deg=np.degrees(np.hypot(sb, gs["alpha"])), bearing_sigma_random_deg=np.degrees(sb),
            bearing_resid_deg=np.degrees(res), bearing_used=used, wall_overlap=wall))
    return rows


def multipath(cfg):
    """{buoy: rows of COLUMNS["multipath"]} from out/multipath/paths_phase1.npz."""
    T = np.load(TOA)
    M = np.load(PATHS)["meas"]
    out = {}
    for name in np.unique(M["buoy"]):
        j = list(T["receivers"]).index(name)
        rows = []
        for r in M[M["buoy"] == name]:
            t_d = T["toa"][r["row"], j] + r["tau_direct"]      # buoy clock (multipath is buoys only)
            rows.append(dict(
                ping=int(r["ping"]), run=int(r["run"]), t_emit_buoy_s=r["t_emit"], family=str(r["family"]),
                t_direct_buoy_s=t_d, t_path_buoy_s=t_d + r["dtau"], dtau_us=r["dtau"] * 1e6,
                sigma_us=r["sigma"] * 1e6, sigma_crlb_us=r["sigma_crlb"] * 1e6, sigma_basis=str(r["sigma_basis"]),
                level_db=float(r["level_db"]), snr_db=float(r["snr_db"]), pred_us=r["pred"] * 1e6,
                win_lo_us=r["win_lo"] * 1e6, win_hi_us=r["win_hi"] * 1e6, launch_deg=float(r["launch_deg"]),
                arrival_deg=float(r["arrival_deg"]), doppler_hz=float(r["doppler_hz"]),
                unique=bool(r["unique"]), resolved=bool(r["resolved"]), persistent=bool(r["persistent"]),
                usable=bool(r["usable"])))
        out[str(name)] = rows
    return out


def readme(cfg, fa, ub, mpth):
    T, S, tr, d, g, gs = load(cfg)
    lines = ["# Phase 1 measurements per receiver (out/export/)", "",
             "Written by `processing/export.py` (re-run by `processing/pipeline.py`). Clock: buoy time, seconds "
             f"since the buoy files' start ({cfg.site.get('buoy_t0_local')} local, {cfg.site.get('date')}). "
             f"Phase 1 posterior: c = {g['c']:.1f} ± {gs['c']:.1f} m/s, USBL clock b_S = {g['b_S'] * 1e3:+.2f} ± "
             f"{gs['b_S'] * 1e3:.2f} ms, drift {g['d_S'] * 1e6:+.1f} ± {gs['d_S'] * 1e6:.1f} ppm, array orientation "
             f"{np.degrees(g['alpha']):.1f} ± {np.degrees(gs['alpha']):.1f} deg (ENU).", "",
             "`*_sigma` is the total sigma of a reading (full phase 1 posterior, with correlations); "
             "`*_sigma_random` is its ping-to-ping part (globals and buoy positions held fixed), which averages down. "
             "The rest is shared by every ping (clock offset, sound speed, buoy positions, orientation). Ranges and "
             "bearings are to the **pinger** (towed behind the boat), not the boat.", ""]
    files = [(f"first_arrival_{n}.csv", "first_arrival", len(r)) for n, r in fa.items()]
    files += [("usbl_bearing_range.csv", "usbl_bearing_range", len(ub))]
    files += [(f"multipath_{n}.csv", "multipath", len(r)) for n, r in mpth.items()]
    lines += ["| file | rows |", "|---|---|"] + [f"| `{f}` | {n} |" for f, _, n in files] + [""]
    for kind, title in (("first_arrival", "first_arrival_<rx>.csv"), ("usbl_bearing_range", "usbl_bearing_range.csv"),
                        ("multipath", "multipath_<buoy>.csv")):
        lines += [f"## {title}", "", "| column | meaning |", "|---|---|"]
        lines += [f"| `{c}` | {m} |" for c, m in COLUMNS[kind]] + [""]
    (OUT / "README.md").write_text("\n".join(lines))


def main():
    cfg = config.load()
    OUT.mkdir(parents=True, exist_ok=True)
    fa = first_arrivals(cfg)
    dig = dict(toa_sigma_s=7, toa_sigma_random_s=7, toa_sigma_crlb_s=9, t_emit_sigma_s=7, t_emit_sigma_random_s=7,
               sync_sigma_s=7, snr_db=1, range_m=2, range_sigma_m=2, range_sigma_random_m=3, toa_resid_us=1)
    for name, rows in fa.items():
        write(OUT / f"first_arrival_{name}.csv", "first_arrival", rows, dig)
        z = np.array([r["outlier"] for r in rows])
        med = lambda f: np.median([r[f] for r in rows])
        print(f"first_arrival_{name}.csv: {len(rows)} chirps, median toa sigma {med('toa_sigma_s') * 1e6:.1f} us "
              f"(random {med('toa_sigma_random_s') * 1e6:.1f}), range sigma {med('range_sigma_m'):.2f} m "
              f"(random {med('range_sigma_random_m'):.3f}), {z.sum()} flagged outliers")
    ub = usbl_bearing_range(cfg, fa)
    write(OUT / "usbl_bearing_range.csv", "usbl_bearing_range", ub,
          dict(range_m=2, range_sigma_m=2, range_sigma_random_m=3, bearing_true_deg=2, az_enu_deg=2,
               bearing_sigma_deg=2, bearing_sigma_random_deg=2, bearing_resid_deg=2))
    nb = sum(np.isfinite(r["bearing_true_deg"]) for r in ub)
    print(f"usbl_bearing_range.csv: {len(ub)} chirps, {nb} with a bearing "
          f"({sum(r['bearing_used'] for r in ub)} used by phase 1), median bearing sigma "
          f"{np.nanmedian([r['bearing_sigma_deg'] for r in ub]):.2f} deg")
    mpth = multipath(cfg)
    for name, rows in mpth.items():
        write(OUT / f"multipath_{name}.csv", "multipath", rows,
              dict(dtau_us=1, sigma_us=1, sigma_crlb_us=2, level_db=1, snr_db=1, pred_us=1, win_lo_us=1,
                   win_hi_us=1, launch_deg=2, arrival_deg=2, doppler_hz=3))
        print(f"multipath_{name}.csv: {len(rows)} components on {len({r['ping'] for r in rows})} chirps, "
              f"{sum(r['usable'] for r in rows)} usable")
    readme(cfg, fa, ub, mpth)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
