"""
Bathymetry and range-dependent multipath prediction.

`Bathymetry`: the EMODnet DTM around the site (GeoTIFF from the EMODnet WCS,
cached in data/bathymetry/), queried in the local ENU frame. Depth is
positive down; land comes out negative. `depth` is the water depth during
the runs: charted depth plus the configured tide ([site] tide).

`eigenrays`: isovelocity rays in the vertical plane between source and
receiver over a piecewise-linear bottom (the sloping-bottom version of the
image method in multipath_range.py, which it reproduces on a flat bottom).
Returns every path family (bounce sequence) that connects source and
receiver, with its travel time and bounce counts. No absorption or
reflection loss: geometry only. Paths with more than MAX_BOUNCES reflections
(surface + bottom) are not computed.
"""
import sys
import urllib.request
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

BATHY_DIR = config.ROOT / "data" / "bathymetry"
MAX_BOUNCES = 2                     # reflections per path (surface + bottom): D, S, B, SB, BS
WCS = ("https://ows.emodnet-bathymetry.eu/wcs?SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
       "&COVERAGEID=emodnet__mean&FORMAT=image/tiff&SUBSET=Lat({0},{1})&SUBSET=Long({2},{3})")


def download_emodnet(bbox=(43.735, 43.765, 15.765, 15.800), path=BATHY_DIR / "emodnet_mean_vodice.tif"):
    """Fetch the EMODnet mean-depth DTM (1/16 arcmin) for bbox = (lat0, lat1, lon0, lon1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WCS.format(*bbox), path)
    return path


class Bathymetry:
    def __init__(self, cfg, path=BATHY_DIR / "emodnet_mean_vodice.tif"):
        from PIL import Image
        im = Image.open(path)
        g = im.tag_v2[34264]                          # ModelTransformation: lon/lat of the NW corner
        dlon, lon_w, dlat, lat_n = g[0], g[3], -g[5], g[7]
        self.depth_grid = -np.array(im, float)        # EMODnet stores elevation (negative at sea)
        self.lat = lat_n - (np.arange(self.depth_grid.shape[0]) + 0.5) * dlat
        self.lon = lon_w + (np.arange(self.depth_grid.shape[1]) + 0.5) * dlon
        self._f = RegularGridInterpolator((self.lat[::-1], self.lon), self.depth_grid[::-1],
                                          bounds_error=False)
        self.cfg = cfg
        self.tide = float(cfg.site.get("tide", 0.0))  # water level above the DTM datum [m]

    def depth_ll(self, lat, lon):
        """Charted depth [m, positive down] at (lat, lon), bilinear in the DTM (no tide)."""
        lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
        return self._f(np.c_[lat.ravel(), lon.ravel()]).reshape(lat.shape)

    def depth(self, e, n):
        """Water depth [m, positive down] at local ENU (e, n): charted depth + tide."""
        e, n = np.broadcast_arrays(np.asarray(e, float), np.asarray(n, float))
        lat, lon = self.cfg.to_llh(e.ravel(), n.ravel())       # to_llh takes 1-D arrays
        return self.depth_ll(lat, lon).reshape(e.shape) + self.tide

    def profile(self, src, rx, step=10.0):
        """Bottom profile along the straight line src -> rx: (x [m from src], depth)."""
        R = float(np.hypot(*(np.asarray(rx) - np.asarray(src))))
        x = np.linspace(0.0, R, max(int(np.ceil(R / step)), 2) + 1)
        s = x / R
        return x, self.depth(src[0] + s * (rx[0] - src[0]), src[1] + s * (rx[1] - src[1]))


def _shoot(zs, th, xb, zb, max_bounces, record=False):
    """March rays launched at angles th (rad, positive down) from (0, zs) to
    the receiver plane x = xb[-1], reflecting at z = 0 and on the bottom.
    Returns per ray: depth and path length at the receiver plane (NaN if the
    ray turns back or needs more bounces), bounce counts, first boundary
    (0 none, 1 surface, 2 bottom), direction at the receiver plane [rad,
    positive down] and, if record, the vertex list per step."""
    R = xb[-1]
    n = len(th)
    x, z = np.zeros(n), np.full(n, float(zs))
    dx, dz = np.cos(th), np.sin(th)
    L = np.zeros(n)
    ns, nb, first = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    zR, LR, thR = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    live = np.ones(n, bool)
    slope = np.diff(zb) / np.diff(xb)
    verts = [np.c_[x, z].copy()] if record else None
    eps = 1e-9
    for _ in range(max_bounces + 1):
        i = np.flatnonzero(live)
        if not len(i):
            break
        xi, zi, dxi, dzi = x[i], z[i], dx[i], dz[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            tR = np.where(dxi > eps, (R - xi) / dxi, np.inf)
            tS = np.where(dzi < -eps, -zi / dzi, np.inf)
            # bottom: every segment, keep the nearest valid hit
            den = dzi[:, None] - dxi[:, None] * slope[None]
            tB = (zb[None, :-1] + (xi[:, None] - xb[None, :-1]) * slope[None] - zi[:, None]) / den
            xh = xi[:, None] + tB * dxi[:, None]
            okb = (tB > 1e-6) & (xh >= xb[None, :-1] - 1e-9) & (xh <= xb[None, 1:] + 1e-9) & (den > eps)
        tB = np.where(okb, tB, np.inf)
        seg = np.argmin(tB, 1)
        tB = tB[np.arange(len(i)), seg]
        t = np.minimum(np.minimum(tR, tS), tB)
        dead = ~np.isfinite(t)
        hitR = (tR <= tS) & (tR <= tB) & ~dead
        x[i] += np.where(dead, 0, t) * dxi
        z[i] += np.where(dead, 0, t) * dzi
        L[i] += np.where(dead, 0, t)
        zR[i[hitR]], LR[i[hitR]] = z[i[hitR]], L[i[hitR]]
        thR[i[hitR]] = np.arctan2(dz[i[hitR]], dx[i[hitR]])
        live[i[hitR | dead]] = False
        if record:
            verts.append(np.c_[x, z].copy())
        s_hit = ~hitR & ~dead & (tS <= tB)
        b_hit = ~hitR & ~dead & ~s_hit
        js, jb = i[s_hit], i[b_hit]
        dz[js] = -dz[js]
        ns[js] += 1
        first[js] = np.where(first[js] == 0, 1, first[js])
        # specular reflection off the sloping bottom facet
        sl = slope[seg[b_hit]]
        nx, nz = -sl / np.hypot(sl, 1), 1 / np.hypot(sl, 1)
        dot = dx[jb] * nx + dz[jb] * nz
        dx[jb] -= 2 * dot * nx
        dz[jb] -= 2 * dot * nz
        nb[jb] += 1
        first[jb] = np.where(first[jb] == 0, 2, first[jb])
    zR[live], LR[live] = np.nan, np.nan             # ran out of bounces
    return zR, LR, ns, nb, first, thR, verts


def eigenrays(zs, zr, xb, zb, max_bounces=MAX_BOUNCES, n_rays=6001, max_angle=40.0):
    """Eigenrays from (0, zs) to (xb[-1], zr) in isovelocity water, surface at
    z = 0, bottom z = zb(x) piecewise linear on nodes xb (z positive down).

    Returns a list of dicts with keys `L` (path length [m]), `ns`, `nb`
    (surface / bottom bounces), `first` ('s', 'b' or '-'), `angle`
    (launch angle [deg], positive down) and `arrival` (direction of travel at
    the receiver [deg], positive down: a ray from the surface arrives at a
    positive angle), sorted by L. Rays that turn back
    or need more than max_bounces reflections are dropped."""
    xb, zb = np.asarray(xb, float), np.asarray(zb, float)
    th = np.radians(np.linspace(-max_angle, max_angle, n_rays))
    zR, LR, ns, nb, first, thR, _ = _shoot(zs, th, xb, zb, max_bounces)
    # eigenrays: zero crossings of zR - zr between neighbouring rays of one family
    fam = first * 10000 + ns * 100 + nb
    good = np.isfinite(zR)
    a, b = np.arange(n_rays - 1), np.arange(1, n_rays)
    m = good[a] & good[b] & (fam[a] == fam[b]) & (np.sign(zR[a] - zr) != np.sign(zR[b] - zr))
    out = {}
    for k in np.flatnonzero(m):
        w = (zr - zR[k]) / (zR[k + 1] - zR[k])
        r = dict(L=LR[k] + w * (LR[k + 1] - LR[k]), ns=int(ns[k]), nb=int(nb[k]),
                 first="-sb"[first[k]], angle=float(np.degrees(th[k] + w * (th[k + 1] - th[k]))),
                 arrival=float(np.degrees(thR[k] + w * (thR[k + 1] - thR[k]))))
        out.setdefault((fam[k], round(r["L"], 3)), r)   # a crossing on a facet node shows up twice
    return sorted(out.values(), key=lambda r: r["L"])


def ray_path(zs, angle, xb, zb, max_bounces=MAX_BOUNCES):
    """Vertices (x, z) of one ray launched at `angle` [deg] (for plotting eigenrays)."""
    xb, zb = np.asarray(xb, float), np.asarray(zb, float)
    *_, verts = _shoot(zs, np.radians([angle]), xb, zb, max_bounces, record=True)
    v = np.array([p[0] for p in verts])
    return v[np.r_[True, np.any(np.diff(v, axis=0) != 0, 1)]]


def family_delays(zs, zr, xb, zb, fams, c=1520.0, **kw):
    """Delay [s] after the direct path of each family in `fams` (NaN if absent)."""
    rays = eigenrays(zs, zr, xb, zb, **kw)
    L = {}
    for r in rays:
        L.setdefault(family(r), r["L"])
    if "D" not in L:
        return np.full(len(fams), np.nan)
    return np.array([(L[f] - L["D"]) / c if f in L else np.nan for f in fams])


def _grid_job(args):
    xb, zb, zs_grid, zr_grid, fams, c = args
    out = np.full((len(zs_grid), len(zr_grid), len(fams)), np.nan)
    for a, zs in enumerate(zs_grid):
        for b, zr in enumerate(zr_grid):
            if zs < zb[0] - 0.2 and zr < zb[-1] - 0.2:
                out[a, b] = family_delays(zs, zr, xb, zb, fams, c, n_rays=3001)
    return out


def family_delay_grid(profiles, zs_grid, zr_grid, fams, c=1520.0, nproc=16):
    """family_delays for every bottom profile (list of (xb, zb)) on a
    (zs, zr) grid; returns (n_profiles, n_zs, n_zr, n_fams). Parallel."""
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        return np.array(pool.map(_grid_job, [(x, z, zs_grid, zr_grid, fams, c) for x, z in profiles]))


def family(r):
    """Short label of a path family, e.g. 'D', 'S', 'B', 'SB', 'BS', 'SBS'."""
    if r["ns"] + r["nb"] == 0:
        return "D"
    a, b = ("S", "B") if r["first"] == "s" else ("B", "S")
    n = r["ns"] + r["nb"]
    return "".join(a if k % 2 == 0 else b for k in range(n))
