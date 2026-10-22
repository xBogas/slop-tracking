"""
Multipath paths predicted from the bathymetry, per phase 1 ping and buoy
(step 1 of multipath_paths.py; a module of its own so that its cache, ~8 min
on 16 cores, does not depend on the extraction and association code).

`predict`: the eigenrays (bathy.eigenrays, at most bathy.MAX_BOUNCES
reflections) over the EMODnet profile pinger -> buoy, isovelocity at the
phase 1 sound speed, at the phase 1 track positions and the graph buoy
positions. Per path: delay after the direct path, launch and arrival
angle, and Doppler from the track velocity, at the configured depths
("nominal"). The water depth is the charted depth plus the configured tide
(bathy.Bathymetry.depth). The delays also on a grid of pinger depth,
hydrophone depth (both unknown; PLACEHOLDER spans below) and water level
(the configured tide +- [site] tide_sigma, `tide_offsets`): their range is
each family's search window. Also the flat-bottom image method and the
out-of-plane (cross-slope) delay change of the bottom path, to show whether
either simplification would do.

`doppler_budget`: the LFM range-Doppler coupling bias without Doppler
matching, and per path after one common time scale.
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bathy
import config
import detect
import multipath_range as mp
import posterior
import stamp

ROOT = config.ROOT
OUT = ROOT / "out" / "multipath"
TRACK = ROOT / "out" / "phase1" / "track.csv"
STATE = ROOT / "out" / "phase1" / "state.npz"
TOA = ROOT / "out" / "toa_phase1.npz"
DTM = bathy.BATHY_DIR / "emodnet_mean_vodice.tif"

FAMS = ("S", "B", "SB", "BS")                 # every path with <= bathy.MAX_BOUNCES reflections
PATHS = ("D",) + FAMS
ZS_GRID = np.arange(1.0, 5.01, 1.0)           # PLACEHOLDER pinger depths spanned by the windows [m]
ZR_DELTA = np.arange(-2.0, 2.01, 1.0)         # PLACEHOLDER hydrophone depth about the configured one [m]


def tide_offsets(cfg):
    """Water-level offsets of the window grid about the configured tide [m]."""
    s = float(cfg.site.get("tide_sigma", 0.0))
    return np.array([-s, 0.0, s]) if s > 0 else np.zeros(1)


def geometry(cfg):
    """Per toa_phase1 row: pinger ENU, emission time, run and ground velocity
    (phase 1 track), the graph buoy positions and the sound speed."""
    T = np.load(TOA)
    pe = mp.load_track(T, TRACK)
    tr = np.genfromtxt(TRACK, delimiter=",", names=True)
    i = T["k"] - T["k"].min()                                 # track.csv rows: the dense ping grid
    vel = np.c_[tr["v_e"], tr["v_n"]][i]
    return dict(T=T, pe=pe, vel=vel, t_emit=tr["t_emit_buoy_s"][i], run=tr["run"][i].astype(int),
                buoys=mp.graph_buoys(STATE), c=posterior.phase1(STATE, cfg)["c"])


def _bounce_out_of_plane(Bm, src, rx, zs, zr, x, zb, launch):
    """The bottom path B with the local seabed plane at its bounce point:
    (delay change when the cross-path slope is included [s], cross-path and
    along-path slope [deg]). Exact image source in that plane, in path
    coordinates (along, across, depth)."""
    v = bathy.ray_path(zs, launch, x, zb)
    xb_, zb_ = v[1]                                           # first vertex after the source: the bounce
    R = x[-1]
    u = (np.asarray(rx) - src) / R
    w = np.array([-u[1], u[0]])
    p = src + xb_ * u
    h = 5.0
    g = np.array([Bm.depth(p[0] + h, p[1]) - Bm.depth(p[0] - h, p[1]),
                  Bm.depth(p[0], p[1] + h) - Bm.depth(p[0], p[1] - h)]).ravel() / (2 * h)
    gu, gw = g @ u, g @ w

    def length(gw_):
        nrm = np.array([-gu, -gw_, 1.0]) / np.sqrt(gu ** 2 + gw_ ** 2 + 1)
        s = np.array([0.0, 0.0, zs])
        img = s - 2 * ((s - np.array([xb_, 0.0, zb_])) @ nrm) * nrm
        return np.linalg.norm(img - np.array([R, 0.0, zr]))
    return length(gw) - length(0.0), np.degrees(np.arctan(gw)), np.degrees(np.arctan(gu))


def _predict_job(args):
    name, rows, pe, vel, buoy, zs0, zr0, c, tides = args
    cfg = config.load()
    Bm = bathy.Bathymetry(cfg, DTM)
    nf, nz, nr, nt = len(FAMS), len(ZS_GRID), len(ZR_DELTA), len(tides)
    out = dict(delay=np.full((len(rows), nf), np.nan), launch=np.full((len(rows), nf + 1), np.nan),
               arrival=np.full((len(rows), nf + 1), np.nan), image=np.full((len(rows), nf), np.nan),
               grid=np.full((len(rows), nz, nr, nt, nf), np.nan), dtau3d=np.full(len(rows), np.nan),
               slope_across=np.full(len(rows), np.nan), slope_along=np.full(len(rows), np.nan),
               R=np.full(len(rows), np.nan), D_src=np.full(len(rows), np.nan), D_mean=np.full(len(rows), np.nan))
    for q, k in enumerate(rows):
        x, zb = Bm.profile(pe[k], buoy)
        out["R"][q], out["D_src"][q], out["D_mean"][q] = x[-1], zb[0], zb.mean()
        rays = bathy.eigenrays(zs0, zr0, x, zb)
        fam = {}
        for r in rays:
            fam.setdefault(bathy.family(r), r)
        if "D" not in fam:
            continue
        for j, f in enumerate(PATHS):
            if f in fam:
                out["launch"][q, j], out["arrival"][q, j] = fam[f]["angle"], fam[f]["arrival"]
                if j:
                    out["delay"][q, j - 1] = (fam[f]["L"] - fam["D"]["L"]) / c
        out["image"][q] = mp.rel_delays(x[-1], zs0, zb.mean(), zr0, c)[1:5]
        if "B" in fam:
            d3, sa, sl = _bounce_out_of_plane(Bm, pe[k], buoy, zs0, zr0, x, zb, fam["B"]["angle"])
            out["dtau3d"][q], out["slope_across"][q], out["slope_along"][q] = d3 / c, sa, sl
        for a, zs in enumerate(ZS_GRID):
            for b, dz in enumerate(ZR_DELTA):
                for t_, tide in enumerate(tides):
                    zr = zr0 + dz
                    if zs < zb[0] + tide - 0.2 and zr < zb[-1] + tide - 0.2:
                        out["grid"][q, a, b, t_] = bathy.family_delays(zs, zr, x, zb + tide, FAMS, c, n_rays=3001)
    return out


def _predict_key(name, cfg):
    return stamp.key(["site", "receiver", "pinger"], files=[TOA, TRACK, STATE, DTM], code=__file__,
                     name=name, grids=(ZS_GRID, ZR_DELTA, tide_offsets(cfg)), fams=FAMS,
                     max_bounces=bathy.MAX_BOUNCES)


def predict(name, cfg=None, nproc=16):
    """Predicted paths for every toa_phase1 row on buoy `name` (cached). Keys:
    delay (K, F) [s after D], launch / arrival (K, 1 + F) [deg, positive
    down], doppler (K, 1 + F) [Hz], grid (K, nzs, nzr, ntide, F) [s], lo / hi
    (K, F) window [s], exist (K, F) fraction of grid nodes where the family
    exists, image (K, F) flat-bottom image delays, dtau3d (K,) out-of-plane
    delay change of B, slope_across / slope_along (K,) at its bounce [deg],
    v_r (K,) closing speed [m/s], R, D_src, D_mean (K,) [m]."""
    cfg = cfg or config.load()
    cache, key = OUT / f"paths_predict_{name}.npz", _predict_key(name, cfg)
    hit = stamp.load(cache, key)
    if hit is not None:
        return hit
    g = geometry(cfg)
    rows = np.flatnonzero(np.isfinite(g["pe"][:, 0]))
    zs0, zr0 = cfg.pinger.depth, cfg.receiver(name).depth
    parts = np.array_split(rows, nproc * 4)
    with Pool(nproc) as pool:
        res = pool.map(_predict_job, [(name, p, g["pe"], g["vel"], g["buoys"][name], zs0, zr0, g["c"],
                                       tide_offsets(cfg)) for p in parts])
    K = len(g["pe"])
    out = {}
    for key_ in res[0]:
        a = np.concatenate([r[key_] for r in res])
        full = np.full((K,) + a.shape[1:], np.nan)
        full[rows] = a
        out[key_] = full
    G = out["grid"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # families absent at every node
        out["lo"], out["hi"] = np.nanmin(G, (1, 2, 3)), np.nanmax(G, (1, 2, 3))
    out["exist"] = np.isfinite(G).mean((1, 2, 3))
    # Doppler of each path: the source's velocity along the ray's launch direction
    h = g["buoys"][name] - g["pe"]
    out["v_r"] = np.sum(g["vel"] * h, 1) / np.hypot(*h.T)    # closing speed (positive: towards the buoy)
    fc = 0.5 * (cfg.pinger.f0 + cfg.pinger.f1)
    out["doppler"] = fc * out["v_r"][:, None] * np.cos(np.radians(out["launch"])) / g["c"]
    stamp.save(cache, key, **out)
    return out


def doppler_budget(pred, cfg, c):
    """Delay bias of the LFM range-Doppler coupling per row [s]: `common`
    without Doppler matching (closing speed v_r on a zero-Doppler replica),
    and `residual` (K, F) per path after one common time scale (each path
    sees v_r cos(launch angle))."""
    Cc = detect.coupling(cfg.pinger)                          # peak shift per unit time-scale error [s]
    cos = np.cos(np.radians(pred["launch"]))
    return dict(common=Cc * pred["v_r"] / c,
                residual=Cc * pred["v_r"][:, None] * (cos[:, 1:] - cos[:, :1]) / c)
