"""
Multipath ranging on the real pinger recordings (phase 1, buoys).

Real-data version of multipath/multipath_demo.py part 1. Every arrival is a
sum of image paths (direct, surface, bottom, surface-bottom, bottom-surface,
surface-bottom-surface); their delays relative to the direct path depend
only on horizontal range R, pinger depth zs, hydrophone depth zr, water
depth D and sound speed c, not on the emission time or the clocks.

Method (and what the data showed; numbers in notes/D_multipath.md)
------
1. `extract`: per ping and buoy, the baseband matched-filter output on
   -0.6..+9 ms around agent A's TOA (eta-matched replica), resampled onto a
   TOA-centred grid; the pulse model X is A's cross-ambiguity of the
   measured per-receiver pulse shape (`detect.MatchedFilter.xamb`).
2. Coherent image-model fit (`grid_energy`, `ping_profile`, `profiles`):
   the direct multi-path delay fit on the compressed pulse. For a delay
   pattern (R, zs, D) the complex path amplitudes are nuisances solved
   linearly (variable projection via the pulse autocorrelation, so one
   pattern costs a 6x6 solve), with bounds on |a_p / a_direct|. This is the
   exact Gaussian likelihood of the multipath model, so it uses everything
   the demo's power-spectrum fit does, without needing the received
   spectrum W. It recovers R, zs, D on synthetic data (tests), but on the
   real data it FAILS: the arrivals in 0.05-0.6 ms behind the direct path
   are a diffuse rough-surface cluster, not one specular surface echo, and
   the fit puts crowded image paths there (R 1.5-2.5 x the TOA ranges,
   water depth running to the edge of the grid).
3. What the data does support (`echogram`, `track_ridge`): one discrete
   late arrival per buoy (0.5-1.7 ms, intermittent), tracked through the
   echogram by Viterbi without any geometry. In the far field every image
   delay is a quadratic form in the depths over c R (surface 2 zs zr/(cR),
   bottom 2 (D - zs)(D - zr)/(cR), ...), so with only the surface cluster
   and this one ridge the range is known only up to a scale: (zs, D, zr, R)
   and (l zs, l D, l zr, l^2 R) give the same delays, and with zr fixed a
   1-D family (`equivalent_geometry`) remains. The ridge gives K = tau R,
   i.e. (D - zs)(D - zr) given the range, which is a water-depth estimate
   when the range comes from the TOA track, not a range.

Output (`main`): out/multipath/mp_range_phase1.npz aligned to toa_phase1
rows. mp_range is NaN / mp_range_mask False everywhere (no independent
range is good enough for the graph); the ridge delays, its constant
K = tau R against the TOA track for the GPS and the graph buoy positions,
and the implied water depths are stored as diagnostics.

Run: python3 multipath_range.py [--full]   (--full recomputes the coherent
fit profiles, ~25 min on 16 cores; otherwise cached ones are used if they
match the current arrivals, zr and c, and the fit is skipped if not)
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import detect
import recordings
import stamp

C_SOUND = 1520.0
OUT = config.ROOT / "out" / "multipath"
BUOYS = ("Rx902", "Rx01")
WIN = (-0.6e-3, 9.0e-3)          # extraction window around the TOA [s]
UPS = 4                          # oversampling of the pulse table


# --------------------------------------------------------------------------
# Image-source geometry
# --------------------------------------------------------------------------
PATHS = ("direct", "surface", "bottom", "surf-bottom", "bottom-surf", "surf-bottom-surf")


def image_depths(zs, D):
    """Image source depths (trailing axis = PATHS)."""
    zs, D = np.broadcast_arrays(np.asarray(zs, float), np.asarray(D, float))
    return np.stack([zs, -zs, 2 * D - zs, 2 * D + zs, zs - 2 * D, -zs - 2 * D], -1)


def path_delays(R, zs, D, zr, c=C_SOUND):
    """Travel times of the image paths; arguments broadcast, trailing axis = PATHS."""
    R = np.asarray(R, float)
    zi = image_depths(zs, D)
    return np.sqrt(R[..., None] ** 2 + (np.asarray(zr, float)[..., None] - zi) ** 2) / c


def rel_delays(R, zs, D, zr, c=C_SOUND):
    t = path_delays(R, zs, D, zr, c)
    return t - t[..., :1]


def equivalent_geometry(R, zs, D, zr, zs2):
    """The far-field degeneracy: for pinger depth zs2, the (R2, D2) that give
    the same surface and bottom delays as (R, zs, D) at hydrophone depth zr.
    surface: zs zr / R; bottom: (D - zs)(D - zr) / R."""
    R2 = R * zs2 / zs
    q = (D - zs) * (D - zr) * R2 / R                       # (D2 - zs2)(D2 - zr)
    b = zs2 + zr
    D2 = 0.5 * (b + np.sqrt(b * b - 4 * (zs2 * zr - q)))
    return R2, D2


# --------------------------------------------------------------------------
# Data: pulse-compressed arrival windows per ping
# --------------------------------------------------------------------------
def _extract_job(args):
    name, rows, shapes = args
    cfg = config.load()
    rec = recordings.load(cfg.receiver(name))
    fs = rec.fs
    mf = detect.MatchedFilter(cfg.pinger, fs, int(0.25 * fs), shapes=shapes)
    n0, n1 = int(round(WIN[0] * fs)), int(round(WIN[1] * fs))
    lag = np.arange(n0, n1 + 1)
    ys, nv = [], []
    for t_rx, eta in rows:
        x, tx0 = rec.segment(t_rx - 0.01, t_rx + 0.03 + 1.05 * cfg.pinger.duration, clock="rx")
        c = mf.out(mf.xspec(x[:, :1]), eta)[:, 0]
        ts = tx0 + np.arange(len(c)) / fs - t_rx               # time rel. TOA per sample
        i0 = int(np.floor(-ts[0] * fs))                       # sample just before the TOA
        frac = -ts[0] * fs - i0
        # resample on the TOA-centred grid (fractional shift by FFT phase ramp)
        C = np.fft.fft(c)
        f = np.fft.fftfreq(len(c))
        cs = np.fft.ifft(C * np.exp(2j * np.pi * f * frac))      # cs[i] = c(ts[i] + frac/fs)
        y = cs[i0 + lag] * np.exp(-2j * np.pi * mf.fc * lag / fs)  # baseband, t = lag/fs
        # noise: MF output power before the pulse (10 -> 2 ms before the TOA)
        pre = cs[max(i0 - int(0.010 * fs), 0):i0 - int(0.002 * fs)]
        ys.append(y)
        nv.append(np.median(np.abs(pre) ** 2) / np.log(2) if len(pre) > 10 else np.nan)
    return np.array(ys), np.array(nv)


def _arrivals_key(name):
    cfg = config.load()
    return stamp.key(["pinger"], files=[config.ROOT / "out" / "toa_phase1.npz",
                                        config.ROOT / "out" / f"detections_{name}.npz", cfg.receiver(name).file],
                     code=__file__, win=WIN, ups=UPS)


def extract(name, nproc=12):
    """Baseband matched-filter output on WIN around each TOA of toa_phase1
    (rows aligned to it; NaN rows where the receiver has no detection).
    Also returns the pulse model table. Cached in out/multipath/ (recomputed
    when the TOA table, detections, recording, pinger config or code change)."""
    cache, key = OUT / f"arrivals_{name}.npz", _arrivals_key(name)
    hit = stamp.load(cache, key)
    if hit is not None:
        return hit
    T = np.load(config.ROOT / "out" / "toa_phase1.npz")
    Dd = np.load(config.ROOT / "out" / f"detections_{name}.npz")
    det, shapes = Dd["det"], Dd["shapes"]
    j = list(T["receivers"]).index(name)
    di = T["det_index"][:, j]
    ok = (di >= 0) & T["toa_mask"][:, j]
    ok[ok] &= np.isfinite(det["t_rx"][di[ok]])
    rows = [(det["t_rx"][i], det["eta"][i]) for i in di[ok]]
    parts = np.array_split(np.arange(len(rows)), nproc)
    with Pool(nproc) as pool:
        res = pool.map(_extract_job, [(name, [rows[i] for i in p], shapes) for p in parts])
    fs = int(Dd["fs"])
    nl = int(round(WIN[1] * fs)) - int(round(WIN[0] * fs)) + 1
    y = np.full((len(ok), nl), np.nan + 0j, complex)
    y[ok] = np.concatenate([r[0] for r in res])
    nv = np.full(len(ok), np.nan)
    nv[ok] = np.concatenate([r[1] for r in res])
    t = (np.arange(nl) + int(round(WIN[0] * fs))) / fs
    # pulse model: baseband cross-ambiguity of the measured shape, eta ~ 1
    cfg = config.load()
    mf = detect.MatchedFilter(cfg.pinger, fs, int(0.25 * fs), shapes=shapes)
    eta = float(np.median(det["eta"]))
    half = int(0.015 * fs)
    sp = mf.xamb(eta, eta, half, 0)
    tx = np.arange(-half * UPS, half * UPS + 1) / UPS
    X = sp(tx)
    out = dict(y=y, ok=ok, noise=nv, t=t, fs=fs, X=X, tx=tx / fs, snr=T["snr"][:, j])
    stamp.save(cache, key, **out)
    return out


# --------------------------------------------------------------------------
# Coherent multipath fit on the compressed pulse
# --------------------------------------------------------------------------
class Pulse:
    """Pulse model X(t) (baseband, tabulated at 1/UPS sample) with its
    autocorrelation Q(d) = sum_u conj(X(u + d)) X(u), both on the fine grid."""

    def __init__(self, X, ups=UPS):
        self.X, self.ups = X, ups
        self.half = (len(X) - 1) // 2
        n = 1 << int(np.ceil(np.log2(2 * len(X))))
        F = np.fft.fft(X, n)
        q = np.fft.ifft(np.abs(F) ** 2) / ups                 # q[m] = sum_u X(u + m) conj(X(u))
        m = np.arange(-2 * self.half, 2 * self.half + 1)
        self.Q = np.conj(q[m % n])                           # Q(d) on m = d * ups
        self.e0 = self.Q[2 * self.half].real                 # energy of one unit path

    def gram(self, dly):
        """Gram matrices for fine-grid delays dly (..., P) (integer, fine units)."""
        d = dly[..., None, :] - dly[..., :, None]             # tau_q - tau_p
        return self.Q[np.clip(d + 2 * self.half, 0, len(self.Q) - 1)]

    def zcorr(self, y, t0_samp, n_tau):
        """z(tau) = sum_t conj(X(t - tau)) y(t) for tau = t0_samp + j / ups, j < n_tau
        (t: integer sample index 0..len(y)-1)."""
        tau = t0_samp * self.ups + np.arange(n_tau)
        idx = np.arange(len(y))[None, :] * self.ups - tau[:, None] + self.half
        valid = (idx >= 0) & (idx < len(self.X))
        g = np.where(valid, self.X[np.clip(idx, 0, len(self.X) - 1)], 0)
        return np.conj(g) @ y


def model_columns(pulse, t_samp, taus):
    """Columns X(t - tau_p) for integer sample times t and delays taus (samples)."""
    u = (t_samp[:, None] - np.asarray(taus)[None, :]) * pulse.ups + pulse.half
    i = np.floor(u).astype(int)
    w = u - i
    ok = (i >= 0) & (i + 1 < len(pulse.X))
    i = np.clip(i, 0, len(pulse.X) - 2)
    return np.where(ok, (1 - w) * pulse.X[i] + w * pulse.X[i + 1], 0)


def grid_energy(y, pulse, rel, t0_grid, ridge=0.1, amax=None, z=None, lo=None):
    """Explained energy b^H A^-1 b of the image-path model for every delay
    pattern rel (n_pat, P) [samples, rel. to the direct path] and every
    direct-path delay in t0_grid [samples, y index]. Complex path
    amplitudes are solved linearly (variable projection) with a small
    ridge; amax (P,) bounds |a_p / a_direct| (patterns that need larger
    amplitudes, e.g. near-coincident paths fitting pulse-shape mismatch,
    get zero). Returns (n_pat, n_t0)."""
    ups = pulse.ups
    if z is None:
        lo = int(np.floor(t0_grid.min()))
        z = pulse.zcorr(y, lo, int((len(y) + 1 - lo) * ups))
    n_tau = len(z)
    P = rel.shape[1]
    fine = np.rint(rel * ups).astype(int)                     # (n_pat, P)
    inwin = (t0_grid.max() + rel) < len(y) - 2                 # paths beyond the window: dropped
    A = pulse.gram(fine) + ridge * pulse.e0 * np.eye(P)
    A = np.where(inwin[:, :, None] & inwin[:, None, :], A, np.eye(P) * pulse.e0)
    Ai = np.linalg.inv(A)
    t0f = np.rint((t0_grid - lo) * ups).astype(int)
    idx = np.clip(t0f[None, :, None] + fine[:, None, :], 0, n_tau - 1)
    b = z[idx] * inwin[:, None, :]                            # (n_pat, n_t0, P)
    a = np.einsum("npq,ntq->ntp", Ai, b)
    e = np.einsum("ntp,ntp->nt", b.conj(), a).real
    if amax is not None:
        bad = (np.abs(a[..., 1:]) > amax[1:] * np.abs(a[..., :1])).any(-1)
        e = np.where(bad, 0.0, e)
    return e


def sparse_paths(y, pulse, noise, max_paths=10, kappa=25.0, min_frac=0.003, min_sep=0.25):
    """Free arrival structure: orthogonal matching pursuit on the fine delay
    grid (delays of any value, complex amplitudes). Paths are added while
    each explains > max(kappa noise, min_frac |y|^2); no two closer than
    min_sep samples. Returns (taus [samples, y index], amps), sorted."""
    ups = pulse.ups
    n_tau = len(y) * ups
    z = pulse.zcorr(y, 0, n_tau)
    E = np.sum(np.abs(y) ** 2)
    fine, a = np.zeros(0, int), np.zeros(0, complex)
    zr = z.copy()
    grid = np.arange(n_tau)
    for _ in range(max_paths):
        g = np.abs(zr) ** 2 / pulse.e0
        if len(fine):
            g[np.abs(grid[:, None] - fine[None, :]).min(1) < min_sep * ups] = 0
        j = int(np.argmax(g))
        if g[j] < max(kappa * noise, min_frac * E):
            break
        fine = np.r_[fine, j]
        a = np.linalg.solve(pulse.gram(fine), z[fine])
        # residual correlation z_res(tau) = z(tau) - sum_p a_p Q(tau_p - tau)
        d = np.clip(fine[None, :] - grid[:, None] + 2 * pulse.half, 0, len(pulse.Q) - 1)
        zr = z - pulse.Q[d] @ a
    o = np.argsort(fine)
    return fine[o] / ups, a[o]


# grids of the coarse search (range is log-spaced: delays scale with 1/R)
R_GRID = np.exp(np.arange(np.log(100.0), np.log(1500.0), 0.005))
ZS_GRID = np.arange(1.0, 16.01, 0.5)
D_GRID = np.arange(12.0, 45.01, 1.0)
T0_GRID = np.arange(-6.0, 3.01, 0.5)               # direct path rel. TOA [samples]
AMAX = np.array([1.0, 2.0, 2.0, 1.5, 1.5, 1.0])    # |a_p / a_direct| bounds


def ping_profile(y, pulse, fs, n0, zr, Rg=R_GRID, zg=ZS_GRID, Dg=D_GRID, c=C_SOUND, amax=AMAX):
    """Profile of the explained energy fraction over (zs, D), maximised over
    R and the direct-path delay; also the maximising R. Returns (e, R) (nz, nD)."""
    E = np.sum(np.abs(y) ** 2)
    t0g = -n0 + T0_GRID
    lo = int(np.floor(t0g.min()))
    z = pulse.zcorr(y, lo, int((len(y) + 1 - lo) * pulse.ups))
    RR, ZZ = np.meshgrid(Rg, zg, indexing="ij")
    best = np.zeros((zg.size, Dg.size), np.float32)
    bR = np.zeros_like(best)
    for b, D in enumerate(Dg):
        rel = rel_delays(RR.ravel(), ZZ.ravel(), D, zr, c) * fs
        e = grid_energy(y, pulse, rel, t0g, amax=amax, z=z, lo=lo).max(1).reshape(RR.shape) / E
        best[:, b] = e.max(0)
        bR[:, b] = Rg[np.argmax(e, 0)]
    return best, bR


def _profile_job(args):
    name, ks, zr, c = args
    d = extract(name)
    pulse = Pulse(d["X"])
    n0 = int(round(WIN[0] * d["fs"]))
    return [ping_profile(d["y"][k], pulse, d["fs"], n0, zr, c=c) for k in ks]


def _profile_path(name, zr):
    return OUT / f"profile_{name}_zr{zr:g}.npz"


def _profile_key(name, zr, c):
    return stamp.key(extra=dict(arrivals=_arrivals_key(name), zr=zr, c=c, grids=(R_GRID, ZS_GRID, D_GRID, AMAX)))


def profiles(name, zr, c, nproc=16):
    """Coarse (zs, D) profiles for every ping of one buoy (cached; recomputed
    when the arrivals, zr, c or the grids change)."""
    cache, key = _profile_path(name, zr), _profile_key(name, zr, c)
    hit = stamp.load(cache, key)
    if hit is not None:
        return hit["e"], hit["R"]
    d = extract(name)
    ks = np.flatnonzero(d["ok"])
    parts = np.array_split(ks, nproc * 4)
    with Pool(nproc) as pool:
        res = pool.map(_profile_job, [(name, p, zr, c) for p in parts])
    res = [r for part in res for r in part]
    e = np.full((len(d["ok"]), ZS_GRID.size, D_GRID.size), np.nan, np.float32)
    R = np.full_like(e, np.nan)
    e[ks] = np.array([r[0] for r in res])
    R[ks] = np.array([r[1] for r in res])
    stamp.save(cache, key, e=e, R=R)
    return e, R


# --------------------------------------------------------------------------
# Bottom-ridge tracking on the echogram (what the data does support)
# --------------------------------------------------------------------------
def echogram(d, smooth=5):
    """Per-ping power of the compressed pulse, normalised to the direct-path
    peak, and an incoherent running mean over `smooth` pings."""
    from scipy.ndimage import uniform_filter1d
    i0 = int(round(-WIN[0] * d["fs"]))
    A = np.abs(d["y"]) ** 2
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        A = A / np.nanmax(A[:, i0 - 8:i0 + 12], 1, keepdims=True)
    A = np.where(np.isfinite(A), A, 0.0)
    return A, uniform_filter1d(A, smooth, axis=0)


def track_ridge(As, t, ok, band=(0.45e-3, 2.6e-3), step=0.02, jump=0.025, miss=1.0):
    """Viterbi track of the strongest late arrival (the bottom-path ridge)
    through the smoothed echogram As: log-delay grid with `step`, at most
    `jump` relative change per ping, score = log(peak / local background).
    Uses no geometry. Returns (tau [s], contrast) per ping (NaN if no data)."""
    lg = np.arange(np.log(band[0]), np.log(band[1]), step / 8)
    G = np.array([np.interp(np.exp(lg), t, a) for a in As])                  # (K, n)
    # background: the reverberation tail decays with delay, so normalise each
    # delay by its median over the run (ok pings) and each ping by its median
    bg = np.median(G[ok], 0)[None, :] + 1e-12
    S = np.log(np.maximum(G, 0) / bg + 1e-12)
    S -= np.median(S, 1, keepdims=True)
    S[~ok] = 0.0
    n, K = len(lg), len(As)
    w = int(round(jump / (step / 8)))
    acc = S[0].copy()
    back = np.zeros((K, n), int)
    for k in range(1, K):
        pad = np.pad(acc, w, constant_values=-np.inf)
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1)
        j = np.argmax(win, 1)
        back[k] = np.arange(n) + j - w
        acc = win[np.arange(n), j] + S[k]
    path = np.zeros(K, int)
    path[-1] = int(np.argmax(acc))
    for k in range(K - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    tau = np.exp(lg[path])
    con = np.exp(S[np.arange(K), path])
    # sub-grid refinement: parabola on the single-ping-smoothed power
    for k in range(K):
        i = path[k]
        if 0 < i < n - 1 and ok[k]:
            a, b, c = G[k, i - 1:i + 2]
            den = a - 2 * b + c
            if den < 0:
                tau[k] = np.exp(lg[i] + np.clip(0.5 * (a - c) / den, -0.5, 0.5) * (step / 8))
    tau[~ok] = np.nan
    return tau, con


def ridge_segments(good, min_len=15, gap=3):
    """Segment id per ping for runs of `good` pings (gaps <= gap), -1 elsewhere."""
    seg, cur, out, last = np.full(len(good), -1), [], [], -99
    for k in np.flatnonzero(good):
        if k - last > gap and cur:
            out.append(cur)
            cur = []
        cur.append(k)
        last = k
    if cur:
        out.append(cur)
    for i, c in enumerate([c for c in out if len(c) >= min_len]):
        seg[c] = i
    return seg


def scale_free_cost(tau, seg, R):
    """Robust scatter of log(tau R) about its median per ridge segment: if the
    ridge is one image path with constant depths, tau R is constant, so this
    tests a buoy position (through R) without zs, D, zr or c."""
    lk = np.log(tau * R)
    r = [lk[q] - np.median(lk[q]) for q in (np.isfinite(lk) & (seg == s) for s in range(seg.max() + 1))
         if q.sum() > 3]
    r = np.concatenate(r)
    return 1.4826 * np.median(np.abs(r)), r


def full_fit_ranges(E, R, half=3, d_min=18.0, zs_max=13.0):
    """Ranges from the coherent fit profiles with zs shared by both buoys and
    over +/- half pings, D per buoy and window (as in the demo)."""
    zm, dm = ZS_GRID <= zs_max, D_GRID >= d_min
    K = len(E[BUOYS[0]])
    out = {n: np.full(K, np.nan) for n in BUOYS}
    zs = np.full(K, np.nan)
    for k in range(K):
        s = slice(max(k - half, 0), k + half + 1)
        prof = {n: np.nan_to_num(E[n][s][:, zm][:, :, dm]).sum(0) for n in BUOYS}
        iz = int(np.argmax(sum(p.max(1) for p in prof.values())))
        zs[k] = ZS_GRID[zm][iz]
        for n in BUOYS:
            out[n][k] = R[n][k][zm][iz][dm][int(np.argmax(prof[n][iz]))]
    return out, zs


def load_track(T, path=config.ROOT / "out" / "phase1" / "track.csv"):
    """Pinger ENU per toa_phase1 row from the graph solve (NaN if absent)."""
    pe = np.full((len(T["k"]), 2), np.nan)
    if not Path(path).exists():
        return pe
    tr = np.genfromtxt(path, delimiter=",", names=True)
    i = np.searchsorted(T["k"], tr["ping"].astype(int))
    ok = (i < len(T["k"]))
    ok[ok] &= T["k"][i[ok]] == tr["ping"][ok]
    pe[i[ok]] = np.c_[tr["pinger_e"], tr["pinger_n"]][ok]
    return pe


def graph_buoys(path=config.ROOT / "out" / "phase1" / "state.npz"):
    """Buoy positions of the phase 1 graph solve, {} before the first solve."""
    if not Path(path).exists():
        return {}
    return dict(zip(BUOYS, np.load(path)["buoys"]))


def main(full=False, zr=None, zs_ref=6.8, c=None):
    """zr: hydrophone depth (default: the configured buoy depth); c: sound
    speed (default: the configured prior)."""
    tic = time.time()
    cfg = config.load()
    zr = cfg.receiver(BUOYS[0]).depth if zr is None else zr
    c = float(cfg.priors["sound_speed"]) if c is None else c
    OUT.mkdir(parents=True, exist_ok=True)
    T = np.load(config.ROOT / "out" / "toa_phase1.npz")
    K = len(T["k"])
    pe = load_track(T)
    lines = []
    log = lambda s: (print(s), lines.append(s))
    data = {n: extract(n) for n in BUOYS}
    res = dict(k=T["k"], receivers=T["receivers"], zr=zr, sound_speed=c,
               mp_range=np.full((K, 3), np.nan), mp_range_mask=np.zeros((K, 3), bool),
               mp_range_sig=np.full((K, 3), np.nan), zs=np.full(K, np.nan),
               mp_tau_ridge=np.full((K, 3), np.nan), mp_ridge_contrast=np.full((K, 3), np.nan),
               mp_ridge_seg=np.full((K, 3), -1), K_track_gps=np.full((K, 3), np.nan),
               K_track_graph=np.full((K, 3), np.nan))
    # coherent image-model fit (the full method); evidence that it fails here
    cached = all(stamp.load(_profile_path(n, zr), _profile_key(n, zr, c)) is not None for n in BUOYS)
    if not (full or cached):
        log(f"coherent fit skipped: no profiles for the current arrivals, zr {zr:g} m, c {c:g} (--full, ~25 min)")
    if full or cached:
        E, R = {}, {}
        for n in BUOYS:
            E[n], R[n] = profiles(n, zr, c)
        Rfit, zs_fit = full_fit_ranges(E, R)
        res["full_fit_range"] = np.c_[Rfit[BUOYS[0]], Rfit[BUOYS[1]], np.full(K, np.nan)]
        res["full_fit_zs"] = zs_fit
        log(f"coherent fit: zs median {np.nanmedian(zs_fit):.1f} m, IQR "
            f"{np.nanpercentile(zs_fit, 25):.1f}-{np.nanpercentile(zs_fit, 75):.1f} m")
        for n in BUOYS:
            j = BUOYS.index(n)
            pD = np.nanmean(E[n].max(1), 0)
            log(f"  {n}: D profile best {D_GRID[np.argmax(pD)]:.0f} m (grid edge 12-45 m; "
                f"monotone above 16 m: {np.all(np.diff(pD[D_GRID >= 16]) < 0)})")
            Rt = np.hypot(*(pe - cfg.receiver(n).en).T)
            r = Rfit[n] / Rt - 1
            log(f"  {n}: R_fit / R_track - 1: median {np.nanmedian(r) * 100:+.0f} %, "
                f"within 10 %: {np.nanmean(np.abs(r[np.isfinite(r)]) < 0.1) * 100:.0f} % of pings")
    # bottom ridge (geometry-free tracking), scale-free tests against the track
    for n in BUOYS:
        j = list(T["receivers"]).index(n)
        d = data[n]
        _, As = echogram(d)
        tau, con = track_ridge(As, d["t"], d["ok"])
        good = d["ok"] & (con > 3)
        seg = ridge_segments(good)
        res["mp_tau_ridge"][:, j] = np.where(seg >= 0, tau, np.nan)
        res["mp_ridge_contrast"][:, j] = con
        res["mp_ridge_seg"][:, j] = seg
        for r_ in (0, 1):
            m = T["run"] == r_
            log(f"{n} run{r_ + 1}: ridge tracked on {np.mean(seg[m & d['ok']] >= 0) * 100:.0f} % "
                f"of detected pings ({len(np.unique(seg[m & (seg >= 0)]))} segments)")
        if not np.isfinite(pe).any():
            continue
        gb = graph_buoys()
        for lab, pos in [("gps", cfg.receiver(n).en)] + ([("graph", gb[n])] if gb else []):
            Rt = np.hypot(*(pe - pos).T)
            res[f"K_track_{lab}"][:, j] = res["mp_tau_ridge"][:, j] * Rt
            s, _ = scale_free_cost(tau, seg, Rt)
            Kt = res[f"K_track_{lab}"][:, j]
            log(f"  {n} {lab} buoy ({pos[0]:.1f}, {pos[1]:.1f}): scatter of tau R about its segment "
                f"median {s * 100:.2f} % ({s * np.nanmedian(Rt):.0f} m at the median range "
                f"{np.nanmedian(Rt):.0f} m); K = tau R per segment: "
                + ", ".join(f"{np.nanmedian(Kt[seg == q]):.3f}" for q in range(seg.max() + 1)))
        # water depth implied by K, for the reference pinger depth
        Kt = res["K_track_gps"][:, j]
        q = Kt * c / 2
        Dimp = 0.5 * (zs_ref + zr + np.sqrt((zs_ref + zr) ** 2 - 4 * (zs_ref * zr - q)))
        res[f"D_implied_{n}"] = Dimp
        log(f"  {n}: implied water depth (bottom path, zs {zs_ref} m, zr {zr} m, GPS buoy): "
            f"median {np.nanmedian(Dimp):.1f} m, p10-p90 {np.nanpercentile(Dimp, 10):.1f}-"
            f"{np.nanpercentile(Dimp, 90):.1f} m")
    res["D"] = np.array([np.nanmedian(res.get(f"D_implied_{n}", np.nan)) for n in BUOYS])
    res["runtime_s"] = time.time() - tic
    np.savez(OUT / "mp_range_phase1.npz", **res)
    (OUT / "mp_log.txt").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'mp_range_phase1.npz'} ({time.time() - tic:.0f} s)")


if __name__ == "__main__":
    main(full="--full" in sys.argv)
