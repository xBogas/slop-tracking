"""
Inter-receiver TDOAs from the boat's radiated noise (phase 2), by envelope
cross-correlation, plus a method check of the same code on the pinger band.

Convention
----------
tdoa(i, j) = arrival time at receiver i minus arrival time at receiver j, in
buoy time [s] = (|p - r_i| - |p - r_j|) / c. The USBL clock is removed with
the phase 1 model usbl_time = buoy_time + b + d_S (t - t_ref) (posterior.py,
from out/phase1/state.npz), so the
returned TDOAs carry no clock terms. Pairs (into NAMES = [Rx902, Rx01, USBL]):
(Rx902, Rx01), (Rx902, USBL), (Rx01, USBL). Closure: tdoa01 = tdoa02 - tdoa12.

Why envelopes
-------------
Carrier-phase GCC-PHAT between receivers 400-700 m apart fails: the
waveguide decorrelates the fine structure and snapping-shrimp clicks give
strong spurious peaks. The slow (< 2 kHz) amplitude modulation of the boat
noise (propeller blade rate, cavitation bursts) survives the channel, so its
envelope correlates between receivers.

Method
------
1. Streams (`envelope_stream`): each receiver is read by buoy time (the
   USBL through the clock model; its 4 channel envelopes averaged), in 10 s
   chunks in parallel. Band-pass (noise 2-20 kHz: Rx01 has strong
   self-noise below 2 kHz), clicks = full-band envelope > CLICK_K x its
   20 ms running median, dilated +-2.5 ms and marked invalid; Hilbert
   envelopes of 6 log-spaced sub-bands, block-averaged to FS_ENV = 4 kHz.
2. Correlograms (`correlogram`): per window (WIN = 4 s every STEP = 1 s)
   and sub-band, valid samples mean-removed, cross-spectrum kept in the
   modulation band 2-200 Hz and partially whitened (|A||B|^-0.5,
   smoothed), normalised per band and averaged over bands; z-scored by
   median / MAD over lag. Windows with a > GATE_K power jump are gated.
3. Detection (`stack`, `viterbi`, `extract`): track-before-detect stack,
   the best mean of z along straight lines in (window, lag) over +-4
   windows with |slope| <= 2 u_max / c, re-z-scored; then a max-sum path
   over lag with the same slew limit per step and emission max(z - Z0, 0)
   (a window without a peak is a free "no detection", so the path coasts
   through gaps and clutter). Detection = stacked z at the path >= Z_DET
   for >= MIN_RUN consecutive windows.
4. Closure (`closure_search`): the USBL pairs are weak. Where the buoy
   pair is detected, the two USBL stacks are added along the closure line
   tau12 = tau02 - tau01 and traced jointly (associates the right peaks).
   USBL pairs without a buoy detection are taken from their own trace with
   a stricter threshold (Z_DET + Z_EXTRA).
5. Measurement (`measure_warped`): per selected window, envelope b is
   resampled at n - s (n - m/2) with s the local delay rate of the trace
   (2 passes), so a moving delay correlates like a fixed one; the lag is
   the parabolic peak of that single-window correlogram within +-REFINE
   bins of the path (+-CLOSE_WIN of the joint lag for the USBL pairs, so
   their closure residual is a real check). Closure gate: where all three
   are measured, the USBL pairs are dropped if the residual is > 4 MAD
   (and > 1 ms) off the median. Note every per-receiver error (path
   structure, the USBL's own noise) cancels in the closure; it checks
   association and the correlation noise, not receiver timing.
6. Sigma: raw w / z (w = curvature width of the peak) times a per-pair
   scale calibrated truth-free from the scatter of each value about a
   quadratic through detections >= WIN away (independent windows); that
   random part is `tdoa_sig_rand`. `tdoa_sig` adds SIG_SYS = 2 ms in
   quadrature: the pinger check shows slowly varying +-2-3 ms errors
   (multipath: the envelope peak sits at the energy centroid of the
   arrivals, not the direct path) on top of ~1 ms random.
7. Method check (`--pinger`): the same code on 30-40 kHz (3 sub-bands)
   over runs 1-2, against TDOAs predicted from the phase 1 pinger track.
   The pinger is pulsed at ~1 s, so the envelope correlation repeats every
   1 s in lag; all |TDOA| < 0.46 s here and the lag search is capped at
   +-0.5 s.

Run: uv run python processing/noise_tdoa.py [--pinger]
     -> out/phase2/tdoa_noise.npz, tdoa_noise.png
        (--pinger: out/phase2/tdoa_pinger_check.txt/.png/.npz)
"""
import functools
import sys
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.fft import next_fast_len
from scipy.ndimage import maximum_filter1d, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import posterior
import recordings

OUT = config.ROOT / "out" / "phase2"
NAMES = ["Rx902", "Rx01", "USBL"]
PAIRS = [(0, 1), (0, 2), (1, 2)]



@functools.cache
def search_geometry():
    """Receiver xyz (GPS, z = -depth) and the prior sound speed from the config:
    only for the lag search window and the slew limit, which need no posterior."""
    cfg = config.load()
    pos = {n: np.r_[cfg.receiver(n).en, -cfg.receiver(n).depth] for n in NAMES}
    return pos, float(cfg.priors["sound_speed"])


@functools.cache
def clock():
    """USBL clock (b, d_S, t_ref) of the phase 1 graph."""
    p1 = posterior.phase1()
    return p1["b"], p1["d_S"], p1["t_ref"]


_E = np.geomspace(2e3, 20e3, 7)
BANDS_NOISE = tuple((float(a), float(b)) for a, b in zip(_E[:-1], _E[1:]))   # 6 log bands


@dataclass(frozen=True)
class Params:
    bands: tuple = BANDS_NOISE  # sub-bands for envelopes [Hz]
    fs_env: float = 4000.0      # envelope sample rate [Hz]
    win: float = 4.0            # correlation window [s]
    step: float = 1.0           # window step [s]
    env_band: tuple = (2.0, 200.0)    # envelope modulation band kept [Hz]
    beta: float = 0.5           # partial whitening exponent of the envelope cross-spectrum
    click_k: float = 4.0        # click = envelope > click_k x running median
    click_hw: float = 2.5e-3    # blank +- this around a click [s]
    gate_k: float = 10.0        # gate windows whose envelope power > gate_k x running median
    lag_margin: float = 0.05    # lag search = baseline / c + margin [s]
    max_lag: float = 0.5        # hard cap (pinger period ambiguity) [s]
    u_max: float = 15.0         # max source speed for the slew limit [m/s]
    stack_m: int = 4            # track-before-detect stack over +- this many windows
    slope_step: float = 0.5e-3  # stack slope grid [s/s]
    z0: float = 3.0             # Viterbi emission floor (stacked z)
    z_det: float = 6.0          # detection threshold (stacked z at the path)
    z_extra: float = 2.0        # extra threshold for USBL pairs without the closure
    min_run: int = 5            # min consecutive detections
    refine: int = 8             # re-peak within +- this many bins of the path
    warp: bool = True           # re-correlate with the local delay rate removed
    close_win: float = 5e-3     # USBL re-peak half-width around the joint lag [s]
    closure_k: float = 4.0      # closure gate: |residual - median| > closure_k x MAD ...
    closure_min: float = 1e-3   # ... and > this [s]
    sig_floor: float = 50e-6    # sigma floor [s]
    sig_sys: float = 2e-3       # slowly varying error (multipath), from the pinger check [s]
    usbl_all_ch: bool = True    # average the envelopes of the 4 USBL channels


NOISE = Params()
PINGER = replace(NOISE, bands=((30e3, 33.4e3), (33.3e3, 36.7e3), (36.6e3, 40e3)), u_max=5.0)



# --------------------------------------------------------------------------- reading
def usbl_offset(t):
    """usbl_time - buoy_time at buoy time t [s]."""
    b, d_S, t_ref = clock()
    return b + d_S * (np.asarray(t) - t_ref)


def read(rec, t0, n, chans=(0,)):
    """n samples from buoy time t0 (USBL through the clock model), float64 (n, len(chans)).
    The USBL start index is rounded (+-5 us). Zero-padded outside the file."""
    off = usbl_offset(t0) if rec.name == "USBL" else 0.0
    i0 = int(round((t0 + off) * rec.fs))
    a, b = max(i0, 0), min(i0 + n, rec.data.shape[0])
    x = np.zeros((n, len(chans)))
    if b > a:
        x[a - i0:b - i0] = rec.data[a:b][:, list(chans)]
        if rec.data.dtype == np.int16:
            x /= 32768.0
    return x


# --------------------------------------------------------------------------- envelopes
def _sos(band, fs):
    return signal.butter(6, band, "bandpass", fs=fs, output="sos")


def _block_mean(e, dec):
    m = len(e) // dec * dec
    return e[:m].reshape(-1, dec).mean(axis=1)


def envelopes(x, fs, p):
    """Sub-band envelopes of one channel at p.fs_env and their validity mask.
    x: (n,) with n a multiple of fs / fs_env. Returns (nb, m) float, (m,) bool."""
    dec = int(round(fs / p.fs_env))
    lo, hi = min(b[0] for b in p.bands), max(b[1] for b in p.bands)
    full = _block_mean(np.abs(signal.hilbert(signal.sosfiltfilt(_sos((lo, hi), fs), x))), dec)
    med = median_filter(full, size=int(0.02 * p.fs_env) | 1, mode="nearest")
    click = full > p.click_k * med
    bad = maximum_filter1d(click.astype(np.uint8), 2 * int(p.click_hw * p.fs_env) + 1) > 0
    env = np.array([_block_mean(np.abs(signal.hilbert(signal.sosfiltfilt(_sos(b, fs), x))), dec)
                    for b in p.bands])
    return env, ~bad


def _stream_job(args):
    name, t0, n_env, p = args
    rec = recordings.load(config.load().receiver(name))
    dec = int(round(rec.fs / p.fs_env))
    pad = int(0.25 * p.fs_env)               # filter edge margin [envelope samples]
    chans = range(rec.n_ch) if (rec.name == "USBL" and p.usbl_all_ch) else (0,)
    x = read(rec, t0 - pad / p.fs_env, (n_env + 2 * pad) * dec, chans)
    envs, oks = zip(*(envelopes(x[:, c], rec.fs, p) for c in range(x.shape[1])))
    env = np.mean(envs, axis=0)[:, pad:pad + n_env]
    ok = np.all(oks, axis=0)[pad:pad + n_env] if len(oks) == 1 else \
        (np.mean(oks, axis=0) >= 0.5)[pad:pad + n_env]
    ok &= np.any(x[pad * dec:(pad + n_env) * dec].reshape(n_env, dec, -1) != 0, axis=(1, 2))
    return env, ok


def envelope_stream(name, t0, t1, p, chunk=10.0, nproc=16):
    """Envelope stream of a receiver over buoy time [t0, t1): env (nb, m),
    valid (m,), sample k at buoy time t0 + k / fs_env."""
    n = int(round((t1 - t0) * p.fs_env))
    nc = int(round(chunk * p.fs_env))
    starts = np.arange(0, n, nc)
    jobs = [(name, t0 + s / p.fs_env, min(nc, n - s), p) for s in starts]
    with Pool(min(nproc, len(jobs))) as pool:
        res = pool.map(_stream_job, jobs)
    return np.concatenate([r[0] for r in res], axis=1), np.concatenate([r[1] for r in res])


# --------------------------------------------------------------------------- correlograms
def correlogram(ea, eb, oka, okb, p, n_lag):
    """Envelope cross-correlation of one window, averaged over sub-bands.
    ea, eb: (nb, m); returns (2 n_lag + 1,) with lag l/fs_env = tdoa(a, b)."""
    m = ea.shape[1]
    nfft = next_fast_len(2 * m)
    f = np.fft.rfftfreq(nfft, 1 / p.fs_env)
    keep = (f >= p.env_band[0]) & (f <= p.env_band[1])
    out = np.zeros(2 * n_lag + 1)
    for a, b in zip(ea, eb):
        za = np.where(oka, a - a[oka].mean(), 0.0) if oka.any() else np.zeros(m)
        zb = np.where(okb, b - b[okb].mean(), 0.0) if okb.any() else np.zeros(m)
        A, B = np.fft.rfft(za, nfft), np.fft.rfft(zb, nfft)
        G = A * np.conj(B) * keep
        if p.beta:
            w = np.abs(A) * np.abs(B)
            G = G / (signal.savgol_filter(w, 31, 1).clip(1e-30) ** p.beta)
        cc = np.fft.irfft(G, nfft)
        cc = np.r_[cc[-n_lag:], cc[:n_lag + 1]]
        out += cc / (np.std(cc) + 1e-30)
    return out / len(ea)


def window_power(env, ok, idx, m):
    """Median valid envelope power per window (for transient gating)."""
    e = env.sum(axis=0)
    return np.array([np.median(e[i:i + m][ok[i:i + m]]) if ok[i:i + m].any() else np.nan for i in idx])


def gate(power, p, span=31):
    """True where the window power is below gate_k x its running median."""
    pw = np.nan_to_num(power, nan=np.nanmedian(power))
    med = median_filter(pw, size=span, mode="nearest")
    return (pw < p.gate_k * med) & np.isfinite(power)


def zscore(C):
    med = np.median(C, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(C - med), axis=1, keepdims=True)
    return (C - med) / np.maximum(mad, 1e-30)


def max_lag_bins(pair, p):
    pos, c = search_geometry()
    a, b = (pos[NAMES[i]] for i in pair)
    return int(np.ceil(min(np.linalg.norm(a - b) / c + p.lag_margin, p.max_lag) * p.fs_env))


def correlograms(streams, t0, tc, p):
    """Per-pair z-scored correlograms on window centres tc (buoy time).
    streams: {name: (env, ok)} sampled from buoy time t0.
    Returns {pair: (Z (K, L), lags (L,) [s], gate (K,) bool)}."""
    m = int(round(p.win * p.fs_env))
    idx = np.round((tc - p.win / 2 - t0) * p.fs_env).astype(int)
    out = {}
    g = {n: gate(window_power(*streams[n], idx, m), p) for n in NAMES}
    for pr in PAIRS:
        na, nb = NAMES[pr[0]], NAMES[pr[1]]
        (ea, oa), (eb, ob) = streams[na], streams[nb]
        L = max_lag_bins(pr, p)
        C = np.array([correlogram(ea[:, i:i + m], eb[:, i:i + m], oa[i:i + m], ob[i:i + m], p, L)
                      for i in idx])
        out[pr] = (zscore(C), np.arange(-L, L + 1) / p.fs_env, g[na] & g[nb])
    return out


# --------------------------------------------------------------------------- trace
def stack(Z, valid, p):
    """Track-before-detect: for each window and lag, the best mean of Z along a
    straight line lag + s (m step) over windows m in [-M, M], |s| <= SLEW,
    re-z-scored. Returns (K, L)."""
    M = p.stack_m
    if not M:
        return Z
    K, L = Z.shape
    dl = 1 / p.fs_env
    Zp = np.zeros((K + 2 * M, L))
    Zp[M:M + K] = np.where(valid[:, None], Z, 0.0)
    n = np.convolve(valid.astype(float), np.ones(2 * M + 1), "same")[:, None]
    best = np.full((K, L), -np.inf)
    smax = 2 * p.u_max / search_geometry()[1]
    for s in np.arange(-smax, smax + 1e-12, p.slope_step):
        acc = np.zeros((K, L))
        for m in range(-M, M + 1):
            sh = int(round(s * m * p.step / dl))
            row = Zp[M + m:M + m + K]
            if sh > 0:
                acc[:, :L - sh] += row[:, sh:]
            elif sh < 0:
                acc[:, -sh:] += row[:, :L + sh]
            else:
                acc += row
        np.maximum(best, acc, out=best)
    return zscore(best / np.maximum(n, 1.0))


def viterbi(Z, valid, z0, max_step):
    """Max-sum lag path through Z (K, L) with |step| <= max_step bins per window,
    emission max(Z - z0, 0) (0 where not valid). Returns path (K,) bin indices."""
    K, L = Z.shape
    em = np.where(valid[:, None], np.maximum(Z - z0, 0.0), 0.0)
    D = int(max_step)
    dp = em[0].copy()
    back = np.zeros((K, L), np.int32)
    offs = np.arange(-D, D + 1)
    for k in range(1, K):
        pad = np.r_[np.full(D, -np.inf), dp, np.full(D, -np.inf)]
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * D + 1)   # (L, 2D+1)
        j = np.argmax(win, axis=1)
        back[k] = np.arange(L) + offs[j]
        dp = win[np.arange(L), j] + em[k]
    path = np.zeros(K, int)
    path[-1] = int(np.argmax(dp))
    for k in range(K - 1, 0, -1):
        path[k - 1] = back[k, path[k]]
    return path


def refine_peak(z, i0, r):
    """Local max of z within +-r of i0, parabolic sub-bin offset and curvature width.
    Returns (bin position (float), peak z, width [bins])."""
    lo, hi = max(i0 - r, 1), min(i0 + r, len(z) - 2)
    i = lo + int(np.argmax(z[lo:hi + 1]))
    y0, y1, y2 = z[i - 1], z[i], z[i + 1]
    curv = y0 - 2 * y1 + y2
    d = float(np.clip(0.5 * (y0 - y2) / curv, -0.5, 0.5)) if curv < 0 else 0.0
    peak = y1 - 0.25 * (y0 - y2) * d
    width = np.sqrt(max(peak, 1e-9) / -curv) if curv < 0 else np.inf
    return i + d, peak, width


def min_run(det, n):
    """Drop detection runs shorter than n windows."""
    out = det.copy()
    k = 0
    while k < len(det):
        if det[k]:
            j = k
            while j < len(det) and det[j]:
                j += 1
            if j - k < n:
                out[k:j] = False
            k = j
        else:
            k += 1
    return out


def max_step(p):
    return np.ceil(2 * p.u_max / search_geometry()[1] * p.step * p.fs_env)


def measure(Z, lags, path, p, r=None):
    """Refined lag [s], peak z and raw sigma (w / z) [s] around a path."""
    dl = lags[1] - lags[0]
    pos, zp, w = np.array([refine_peak(z, i, p.refine if r is None else r) for z, i in zip(Z, path)]).T
    return lags[0] + pos * dl, zp, w * dl / np.maximum(zp, 1.0)


def extract(Z, lags, valid, p):
    """Trace one pair. Viterbi on the stacked correlogram; detection = stacked z at
    the path >= z_det for >= min_run windows; lag re-peaked on the single-window
    correlogram near the path. Returns a dict."""
    Zs = stack(Z, valid, p)
    path = viterbi(Zs, valid, p.z0, max_step(p))
    zs = Zs[np.arange(len(path)), path]
    tau, zp, sig = measure(Z, lags, path, p)
    det = min_run(valid & (zs >= p.z_det), p.min_run)
    return dict(Zs=Zs, path=path, zs=zs, tau=tau, z=zp, sig=sig, det=det)


def local_slope(t, lag, ok, half=3.0):
    """Delay rate [s/s]: LS slope of lag over the ok windows within +-half s (0 if < 3)."""
    out = np.zeros(len(t))
    for k in np.flatnonzero(ok):
        nb = ok & (np.abs(t - t[k]) <= half) & np.isfinite(lag)
        if nb.sum() >= 3:
            out[k] = np.polyfit(t[nb] - t[k], lag[nb], 1)[0]
    return out


def remeasure(sa, sb, idx, lag_path, sel, slope, r, n_lag, p):
    """Lag, z and raw sigma per selected window: correlate envelope a over the window
    with envelope b resampled at n - slope (n - m/2) (so a moving delay correlates
    like a fixed one at the window centre), z-score, re-peak within +-r bins of the path."""
    (ea, oa), (eb, ob) = sa, sb
    m = int(round(p.win * p.fs_env))
    n = np.arange(m)
    N = eb.shape[1]
    K = len(idx)
    tau, zp, sig = np.full(K, np.nan), np.full(K, np.nan), np.full(K, np.nan)
    rr = np.broadcast_to(r, (K,))
    for k in np.flatnonzero(sel):
        i = idx[k]
        pos = i + n - slope[k] * (n - m / 2)
        ebw = np.array([np.interp(pos, np.arange(N), band) for band in eb])
        obw = ob[np.clip(np.round(pos).astype(int), 0, N - 1)]
        z = zscore(correlogram(ea[:, i:i + m], ebw, oa[i:i + m], obw, p, n_lag)[None])[0]
        i0 = int(np.clip(np.round(lag_path[k] * p.fs_env) + n_lag, 0, 2 * n_lag))
        q, zp[k], w = refine_peak(z, i0, int(rr[k]))
        tau[k], sig[k] = (q - n_lag) / p.fs_env, w / p.fs_env / max(zp[k], 1.0)
    return tau, zp, sig


def measure_warped(sa, sb, idx, tc, lag_path, sel, r, n_lag, p, passes=2):
    """remeasure with the delay rate from the lag path, then (passes - 1) times with
    the rate from the previous measurement. Returns tau, z, raw sigma, slope."""
    slope = local_slope(tc, lag_path, sel) if p.warp else np.zeros(len(tc))
    tau, z, sig = remeasure(sa, sb, idx, lag_path, sel, slope, r, n_lag, p)
    for _ in range(passes - 1 if p.warp else 0):
        ok = sel & np.isfinite(tau)
        slope = local_slope(tc, tau, ok)
        tau, z, sig = remeasure(sa, sb, idx, np.where(ok, tau, lag_path), sel, slope, r, n_lag, p)
    return tau, z, sig, slope


def scatter_scale(t, tau, sig, det, gap, half=10.0):
    """Truth-free sigma scale: residual of each detection about a quadratic fit
    to detections at gap <= |dt| <= gap + half (errors there are independent of
    this window's), normalised by sig. Returns (robust std of tau / sig, of tau [s])."""
    r = np.full(len(t), np.nan)
    for k in np.flatnonzero(det):
        dt = t - t[k]
        nb = det & (np.abs(dt) >= gap) & (np.abs(dt) <= gap + half)
        if (dt[nb] < 0).sum() < 3 or (dt[nb] > 0).sum() < 3:
            continue
        c = np.polyfit(dt[nb], tau[nb], 2)
        r[k] = tau[k] - c[-1]
    ok = np.isfinite(r)
    if ok.sum() < 5:
        return np.nan, np.nan
    return mad(r[ok] / sig[ok]), mad(r[ok])


def mad(v):
    v = np.asarray(v)
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def closure_search(tau01, det01, Zs02, lags02, Zs12, lags12, valid, p):
    """Joint USBL trace along the closure line tau12 = tau02 - tau01:
    S(tau02) = (zs02(tau02) + zs12(tau02 - tau01)) / sqrt(2), on windows where the
    buoy pair is detected. Returns the joint tau02 [s] (NaN elsewhere) and its z."""
    K = Zs02.shape[0]
    S = np.zeros_like(Zs02)
    for k in np.flatnonzero(det01):
        S[k] = (Zs02[k] + np.interp(lags02 - tau01[k], lags12, Zs12[k], left=0, right=0)) / np.sqrt(2)
    v = valid & det01
    path = viterbi(S, v, p.z0, max_step(p))
    zj = S[np.arange(K), path]
    return np.where(v, lags02[path], np.nan), np.where(v, zj, np.nan)


# --------------------------------------------------------------------------- pipeline
def run(t0, t1, p, nproc=16):
    """Correlograms + traces over buoy time [t0, t1]. Returns a dict (npz-ready)."""
    streams = {n: envelope_stream(n, t0, t1, p, nproc=nproc) for n in NAMES}
    return solve(streams, t0, t1, p)


def solve(streams, t0, t1, p):
    """Correlograms + traces from envelope streams {name: (env, valid)} sampled from
    buoy time t0 (see envelope_stream)."""
    tc = np.arange(t0 + p.win / 2, t1 - p.win / 2 + 1e-9, p.step)
    cg = correlograms(streams, t0, tc, p)
    K = len(tc)
    ind = {pr: extract(*cg[pr], p) for pr in PAIRS}
    tdoa, zz, sraw = (np.full((K, 3), np.nan) for _ in range(3))
    mask = np.zeros((K, 3), bool)
    src = np.zeros((K, 3), np.int8)          # 0 none, 1 own trace, 2 closure-joint trace
    # which windows to measure on which lag path: the buoy pair on its own trace; the
    # USBL pairs on the joint trace along the closure line (where the buoy pair is
    # detected), else on their own trace with a stricter threshold
    e0 = ind[PAIRS[0]]
    (Z02, l02, g02), (Z12, l12, g12) = cg[PAIRS[1]], cg[PAIRS[2]]
    tau01 = np.where(e0["det"], e0["tau"], np.nan)
    tj, zj = closure_search(tau01, e0["det"], ind[PAIRS[1]]["Zs"], l02, ind[PAIRS[2]]["Zs"], l12,
                            g02 & g12, p)
    jdet = min_run(np.nan_to_num(zj) >= p.z_det, p.min_run)
    path_lag = np.full((K, 3), np.nan)
    path_lag[:, 0] = cg[PAIRS[0]][1][e0["path"]]
    src[e0["det"], 0], zz[e0["det"], 0] = 1, e0["zs"][e0["det"]]
    for j, pred in ((1, tj), (2, tj - tau01)):
        e = ind[PAIRS[j]]
        own = e["det"] & ~jdet & ~e0["det"] & (e["zs"] >= p.z_det + p.z_extra)
        path_lag[jdet, j], src[jdet, j], zz[jdet, j] = pred[jdet], 2, zj[jdet]
        path_lag[own, j], src[own, j], zz[own, j] = cg[PAIRS[j]][1][e["path"][own]], 1, e["zs"][own]
    # measure: re-correlate each selected window with the second envelope time-warped by
    # the local slope of the lag path (no smear of a moving delay), re-peak near the path
    idx = np.round((tc - p.win / 2 - t0) * p.fs_env).astype(int)
    zw = np.full((K, 3), np.nan)
    for j, (a, b) in enumerate(PAIRS):
        sel = src[:, j] > 0
        r = np.where(src[:, j] == 2, int(round(p.close_win * p.fs_env)), p.refine)
        tdoa[:, j], zw[:, j], sraw[:, j], _ = measure_warped(
            streams[NAMES[a]], streams[NAMES[b]], idx, tc, path_lag[:, j], sel, r,
            max_lag_bins((a, b), p), p)
        mask[:, j] = sel
    mask &= np.isfinite(tdoa) & np.isfinite(sraw)
    # closure gate: where all three are measured, the USBL pairs (the uncertain ones)
    # are dropped when tdoa01 - (tdoa02 - tdoa12) is off the closure core
    all3 = mask.all(axis=1)
    rc = tdoa[:, 0] - (tdoa[:, 1] - tdoa[:, 2])
    n_rej = 0
    if all3.sum() >= 10:
        c0, cs = np.median(rc[all3]), max(mad(rc[all3]), p.sig_floor)
        bad = all3 & (np.abs(rc - c0) > max(p.closure_k * cs, p.closure_min))
        mask[bad, 1:] = False
        n_rej = int(bad.sum())
    # sigma calibration (truth-free), per pair
    gap = p.win
    ksig, rsig = np.ones(3), np.full(3, np.nan)
    for j in range(3):
        k_, r_ = scatter_scale(tc, tdoa[:, j], sraw[:, j], mask[:, j], gap)
        if np.isfinite(k_):
            ksig[j], rsig[j] = k_, r_
    sig_rand = np.where(mask, np.maximum(sraw * ksig, p.sig_floor), np.nan)
    sig = np.hypot(sig_rand, p.sig_sys)
    tdoa[~mask] = np.nan
    src[~mask] = 0
    return dict(t=tc, names=np.array(NAMES), pairs=np.array([[NAMES[a], NAMES[b]] for a, b in PAIRS]),
                pair_idx=np.array(PAIRS), tdoa=tdoa, tdoa_sig=sig, tdoa_mask=mask, snr=zz, snr_window=zw, source=src,
                tdoa_sig_rand=sig_rand, sig_raw=sraw, sig_scale=ksig, scatter_rms=rsig,
                tdoa_indep=np.array([ind[pr]["tau"] for pr in PAIRS]).T,
                snr_indep=np.array([ind[pr]["zs"] for pr in PAIRS]).T,
                mask_indep=np.array([ind[pr]["det"] for pr in PAIRS]).T,
                tdoa_joint02=tj, snr_joint=zj, gate=np.array([cg[pr][2] for pr in PAIRS]).T,
                **{f"corr_{a}{b}": cg[(a, b)][0].astype(np.float32) for a, b in PAIRS},
                **{f"stack_{a}{b}": ind[(a, b)]["Zs"].astype(np.float32) for a, b in PAIRS},
                **{f"lags_{a}{b}": cg[(a, b)][1] for a, b in PAIRS},
                closure_rejected=n_rej, params=np.array(repr(p)), clock=np.array(clock()))


def closure_stats(res, indep=False):
    """Closure residual tdoa01 - (tdoa02 - tdoa12) where all three are detected, for
    the final TDOAs or (indep=True) the three pairs' own traces (no joint search).
    Returns (residuals [s], combined sigma [s])."""
    if indep:
        m, d = res["mask_indep"].all(axis=1), res["tdoa_indep"]
        s = np.full(m.sum(), np.nan)
    else:
        m, d = res["tdoa_mask"].all(axis=1), res["tdoa"]
        s = np.sqrt((res["tdoa_sig_rand"][m] ** 2).sum(axis=1))
    return d[m, 0] - (d[m, 1] - d[m, 2]), s


def closure_lines(res):
    out = []
    for lab, ind in (("final (joint closure search)", False), ("own traces (independent)", True)):
        r, s = closure_stats(res, ind)
        if len(r):
            out.append(f"closure {lab}: n={len(r)}, median {np.median(r) * 1e3:.3f} ms, MAD-sig "
                       f"{mad(r) * 1e3:.3f} ms, |r| > 10 ms: {np.sum(np.abs(r) > 0.01)}")
    return out


# --------------------------------------------------------------------------- pinger check
def predicted_tdoa(tc, track_csv=config.ROOT / "out" / "phase1" / "track.csv"):
    """TDOA per pair at window centres tc, from the phase 1 pinger track (mean
    over the pings whose mid-pair arrival time falls in each window), with the
    phase 1 receiver positions and sound speed and the configured pinger depth."""
    d = np.genfromtxt(track_csv, delimiter=",", names=True)
    p1 = posterior.phase1()
    c = p1["c"]
    P = np.c_[d["pinger_e"], d["pinger_n"], np.full(len(d), -config.load().pinger.depth)]
    R = {n: np.linalg.norm(P - p1["pos"][n], axis=1) for n in NAMES}
    out = np.full((len(tc), 3), np.nan)
    for j, (a, b) in enumerate(PAIRS):
        ra, rb = R[NAMES[a]], R[NAMES[b]]
        tm = d["t_emit_buoy_s"] + (ra + rb) / (2 * c)
        td = (ra - rb) / c
        for k, t in enumerate(tc):
            sel = np.abs(tm - t) < 2.0
            if sel.sum() >= 2:
                out[k, j] = td[sel].mean()
    return out


def summary_lines(res):
    out = []
    for j, (a, b) in enumerate(PAIRS):
        m = res["tdoa_mask"][:, j]
        src = res["source"][m, j]
        out.append(f"{NAMES[a]:5s}-{NAMES[b]:5s}: {m.sum():4d} / {len(m)} windows detected "
                   f"({np.sum(src == 2)} via closure), median sigma {np.nanmedian(res['tdoa_sig'][m, j]) * 1e3:.2f} ms "
                   f"(random {np.nanmedian(res['tdoa_sig_rand'][m, j]) * 1e3:.2f} ms), scale {res['sig_scale'][j]:.2f}, "
                   f"scatter {res['scatter_rms'][j] * 1e3:.2f} ms")
    out.append(f"closure gate rejected the USBL pairs in {int(res['closure_rejected'])} windows")
    return out + closure_lines(res)


def pinger_check(t0=52.0, t1=796.0, p=PINGER, nproc=16):
    """Run the envelope TDOA on the pinger band and compare with the phase 1 track."""
    res = run(t0, t1, p, nproc)
    pred = predicted_tdoa(res["t"])
    lines = [f"Envelope-TDOA method check on the pinger band, buoy time {t0}-{t1} s",
             f"{p}", "",
             f"error = measured - predicted from out/phase1/track.csv (pinger {config.load().pinger.depth:g} m deep, "
             f"c = {posterior.phase1()['c']:.1f})",
             "pair         n_pred  det%  bias[ms]  MAD[ms]  random[ms]  >10ms  z(sig)  z(sig_rand)"]
    for j, (a, b) in enumerate(PAIRS):
        ok = np.isfinite(pred[:, j])
        m = ok & res["tdoa_mask"][:, j]
        e = np.where(m, res["tdoa"][:, j] - pred[:, j], np.nan)
        ev = e[m]
        bias = np.median(ev)
        sm = median_filter(np.nan_to_num(e, nan=bias), 31, mode="nearest")
        rnd = mad((e - sm)[m])
        zs = mad((ev - bias) / res["tdoa_sig"][m, j])
        zr = mad((e - sm)[m] / res["tdoa_sig_rand"][m, j])
        lines.append(f"{NAMES[a]:5s}-{NAMES[b]:5s}  {ok.sum():6d}  {100 * m.sum() / ok.sum():4.1f}  {bias * 1e3:8.2f}"
                     f"  {mad(ev) * 1e3:7.2f}  {rnd * 1e3:10.2f}  {np.sum(np.abs(ev - bias) > 0.01):5d}"
                     f"  {zs:6.2f}  {zr:11.2f}")
    lines += ["", "random = MAD about a 31-window running median of the error (the rest varies slowly)",
              "z(sig) = MAD of (error - bias) / tdoa_sig; z(sig_rand) = MAD of random part / tdoa_sig_rand", ""]
    lines += summary_lines(res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tdoa_pinger_check.txt").write_text("\n".join(lines) + "\n")
    np.savez(OUT / "tdoa_pinger_check.npz", pred=pred, **res)
    plot(res, OUT / "tdoa_pinger_check.png", pred=pred, title="pinger band 30-40 kHz, runs 1-2")
    print("\n".join(lines))
    return res, pred


# --------------------------------------------------------------------------- plot
def plot(res, path, pred=None, title="boat noise 2-20 kHz"):
    """Stacked correlograms with the detected trace (and the phase 1 prediction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = res["t"]
    n = 4 if pred is not None else 3
    fig, ax = plt.subplots(n, 1, figsize=(14, 3.6 * n), sharex=True)
    for j, (a, b) in enumerate(PAIRS):
        Z, lags = res[f"stack_{a}{b}"], res[f"lags_{a}{b}"]
        ax[j].imshow(Z.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=12,
                     extent=[t[0] - 0.5, t[-1] + 0.5, lags[0] * 1e3, lags[-1] * 1e3])
        m = res["tdoa_mask"][:, j]
        ax[j].plot(t[m], res["tdoa"][m, j] * 1e3, ".", color="tab:red", ms=2.5, label="detected TDOA")
        if pred is not None:
            ax[j].plot(t, pred[:, j] * 1e3, "-", color="tab:cyan", lw=0.8, label="phase 1 track")
            e = np.where(m, res["tdoa"][:, j] - pred[:, j], np.nan)
            ax[-1].plot(t, e * 1e3, ".", ms=2, label=f"{NAMES[a]}-{NAMES[b]}")
        ax[j].set_ylabel(f"{NAMES[a]}-{NAMES[b]} [ms]")
        ax[j].set_title(f"{NAMES[a]}-{NAMES[b]}: {m.sum()} / {len(t)} windows detected", fontsize=10)
        ax[j].legend(loc="upper right", fontsize=8)
    if pred is not None:
        ax[-1].set_ylim(-15, 15)
        ax[-1].grid(alpha=0.3)
        ax[-1].set_ylabel("measured - predicted [ms]")
        ax[-1].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("buoy time [s]")
    fig.suptitle(f"Envelope cross-correlation, track-stacked z-score, {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv):
    if "--pinger" in argv:
        pinger_check()
        return
    res = run(796.0, 1020.0, NOISE)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tdoa_noise.npz", **res)
    plot(res, OUT / "tdoa_noise.png")
    print("\n".join(summary_lines(res)))


if __name__ == "__main__":
    main(sys.argv[1:])
