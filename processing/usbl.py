"""
USBL (4-channel shore array): ping snippets, inter-channel delays, array
self-calibration, wall-echo analysis and per-ping bearing.

Method
------
1. Ping times: agent A's table (out/toa_phase1.npz, shared ping index k,
   Doppler scale eta) when present, else `find_pings`: a complex-replica
   matched filter over the file in 10 s chunks, cleaned against the smooth
   1 s arrival track, missing slots re-searched around the track, Doppler
   scale from the slope of the arrival track.
2. Snippets: `snippets` pulse-compresses each channel with the Doppler
   matched replica (complex output = analytic compressed pulse).
3. Delays: the direct pulse of each channel (first arrival inside a common
   region, so all channels lock on the same path) is gated +-0.1 ms around
   its own envelope peak; the gated pulses are cross-correlated; the
   envelope of the upsampled cross-correlation picks the lobe and its carrier
   phase refines the delay. The channel spacing is several wavelengths, so
   the lobe can slip by 1/fc: `resolve_cycles` tests integer shifts and
   scores the plane-wave fit plus the distance to the envelope-only delays.
4. Geometry: pair delays tau_ij = t_j - t_i = (r_i - r_j) . u / c. The delay
   vectors (rel. channel 0) are rank ~2 here (sources near horizontal), so
   the fit is 2-D, in the plane of the arrivals. Closed form (d^T M d = 1
   with |u| = 1) for a start, then a joint LM fit of element positions + one
   azimuth per ping (Schur complement over the pings). The scale comes from
   |u| = 1 through the small off-line element offset, so it is checked with a
   profile over the array length and a block bootstrap. Per-channel timing
   biases are not identifiable with half-plane coverage (they trade with the
   scale) and are assumed zero.
5. Wall: a plane reflector at distance h_i from element i adds a copy
   delayed by 2 h_i (n . u) / c. `fit_wall` grid-searches (h0, normal phi)
   maximizing the stacked direct-aligned envelopes along the predicted lags.
6. Bearing: plane-wave LS on the pair delays with the calibrated geometry ->
   array-frame azimuth (no elevation: 2-D), the mirror solution on the
   other side of the array axis for comparison, formal sigma from the delay
   residuals, and the truth-free scatter about a local quadratic in time
   (which dominates and is used as the sigma floor).
7. Orientation: az_enu = alpha + s az_array (s = +-1 handedness, since a 2-D
   frame does not know up from down) fitted to the bearing of a reference
   pinger track (out/phase1_toa_only/track.csv: run_phase1.py --no-bearing).

Run: uv run python processing/usbl.py [forfun.toml]
     -> out/usbl/pings.npz, out/usbl/bearings_phase1.npz, plots in out/usbl/
"""
import sys
from pathlib import Path
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import recordings
import stamp

OUT = config.ROOT / "out" / "usbl"
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# --------------------------------------------------------------------------- replica / MF
def lfm_replica(fs, f0=30e3, f1=40e3, T=0.1, scale=1.0, taper=0.1):
    """Complex LFM replica as received with time scale `scale` (x(t) = p(scale t))."""
    t = np.arange(int(round(T / scale * fs))) / fs
    k = (f1 - f0) / T
    return np.exp(2j * np.pi * (f0 * scale * t + 0.5 * k * scale ** 2 * t ** 2)) \
        * signal.windows.tukey(len(t), taper)


def compress(x, rep):
    """Complex matched filter; output index n = replica start at sample n."""
    x = np.atleast_2d(x.T).T
    y = signal.fftconvolve(x, np.conj(rep[::-1])[:, None], mode="full", axes=0)
    return y[len(rep) - 1:len(rep) - 1 + x.shape[0]]


# --------------------------------------------------------------------------- ping finder
def _mf_peak(rec, tc, half, pg, scale, nref_lag=0.4):
    """Strongest first arrival within tc +- half (rx clock): (t, snr[ch], sum_snr)."""
    fs = rec.fs
    rep = lfm_replica(fs, pg.f0, pg.f1, pg.duration, scale)
    x, ts = rec.segment(tc - half - 0.005, tc + half + 0.005 + pg.duration, clock="rx")
    xn, _ = rec.segment(tc + nref_lag, tc + nref_lag + 0.15 + pg.duration, clock="rx")
    e = np.abs(compress(x, rep))[: int((2 * half + 0.01) * fs)]
    nref = np.median(np.abs(compress(xn, rep))[: int(0.15 * fs)], 0)
    s = (e / nref).mean(1)
    i = first_arrival(s, back=int(0.004 * fs))
    snr = 20 * np.log10(e[max(i - 5, 0):i + 6].max(0) / nref)
    return ts + i / fs, snr, 20 * np.log10(s[i])


def _grid_clean(t, interval, tol, support=2, win=6):
    """Slot index per detection and a mask of those on the smooth arrival
    track (within tol of the median of >= support neighbours in +- win slots)."""
    k = np.round((t - t[0]) / interval).astype(int)
    ph = t - k * interval
    ok = np.ones(len(t), bool)
    for _ in range(3):
        new = np.zeros(len(t), bool)
        for i, kk in enumerate(k):
            nb = ok & (np.abs(k - kk) <= win)
            nb[i] = False
            new[i] = nb.sum() >= support and abs(ph[i] - np.median(ph[nb])) < tol
        ok = new
    return k, ok


def find_pings(rec, t0, t1, pg, thr_db=14.0, fill_db=7.0, chunk=10.0, tol=0.01):
    """Standalone ping finder on the USBL file (rx clock).

    Coarse pass (one MF peak per interval above thr_db), clean against the
    smooth arrival track, then re-search every missing slot around the
    interpolated track (+- 4 ms + 1 ms per slot of gap) with the lower
    fill_db threshold. The Doppler time scale comes from the slope of the
    arrival track (t_k - k T)' = Rdot / c, which is far smoother than a
    per-ping scale search on a multipath-spread pulse.
    Returns dict: t (first arrival of the replica start, rx clock), slot,
    scale, snr[K, ch] (per channel, dB re median MF noise), snr_sum."""
    fs = rec.fs
    rep = lfm_replica(fs, pg.f0, pg.f1, pg.duration)
    cand, t = [], t0
    while t < t1:
        x, ts = rec.segment(t, min(t + chunk, t1) + pg.duration, clock="rx")
        e = np.abs(compress(x, rep))[: int(chunk * fs)]
        s = (e / np.median(e, 0)).mean(1)
        pk, _ = signal.find_peaks(s, height=10 ** (thr_db / 20), distance=int(0.7 * pg.interval * fs))
        cand += list(ts + pk / fs)
        t += chunk
    t = np.array(cand)
    k, ok = _grid_clean(t, pg.interval, tol)
    k, t = k[ok], t[ok]
    ph = t - k * pg.interval
    slots = np.arange(k[0], k[-1] + 1)
    php = np.interp(slots, k, ph)
    gap = np.minimum(np.abs(slots[:, None] - k[None, :]).min(1), 50)
    sm = np.convolve(np.pad(php, 4, mode="edge"), np.ones(9) / 9, mode="valid")
    slope = np.clip(np.gradient(sm, slots) / pg.interval, -0.01, 0.01)
    out = dict(t=[], slot=[], scale=[], snr=[], snr_sum=[])
    for j, sl in enumerate(slots):
        tc = php[j] + sl * pg.interval
        tt, snr, ss = _mf_peak(rec, tc, 0.004 + 0.001 * gap[j], pg, 1.0 - slope[j])
        if ss >= (fill_db if gap[j] else 0.0):
            for key, v in zip(out, (tt, sl, 1.0 - slope[j], snr, ss)):
                out[key].append(v)
    out = {key: np.array(v) for key, v in out.items()}
    _, ok = _grid_clean(out["t"], pg.interval, 0.003)
    return {key: v[ok] for key, v in out.items()}


def first_arrival(env, drop_db=6.0, back=None):
    """Earliest local maximum within drop_db of the global maximum (direct path
    rule); `back` limits the search before the max (samples)."""
    im = int(np.argmax(env))
    lo = 0 if back is None else max(im - back, 0)
    pk, _ = signal.find_peaks(env[lo:im + 1])
    pk = np.r_[pk + lo, im]
    ok = pk[env[pk] >= env[im] * 10 ** (-drop_db / 20)]
    return int(ok.min())


# --------------------------------------------------------------------------- snippets / delays
def snippets(rec, t_arr, scale, pg, pre=0.003, post=0.006):
    """Pulse-compressed complex snippets (K, n_ch, n) around each first arrival
    t_arr (rx clock), with the Doppler-matched replica. Sample 0 is
    t_arr - pre (rounded to the sample grid; returned as t0[K])."""
    fs = rec.fs
    n = int(round((pre + post) * fs))
    out = np.zeros((len(t_arr), rec.n_ch, n), complex)
    t0 = np.zeros(len(t_arr))
    for k, (t, sc) in enumerate(zip(t_arr, scale)):
        rep = lfm_replica(fs, pg.f0, pg.f1, pg.duration, sc)
        x, ts = rec.segment(t - pre, t - pre + (n + 1) / fs + len(rep) / fs, clock="rx")
        out[k] = compress(x, rep)[:n].T
        t0[k] = ts
    return out, t0


def upsample(y, up):
    """FFT interpolation of complex snippets along the last axis."""
    return signal.resample(y, y.shape[-1] * up, axis=-1)


def channel_toa(y, fs, fc=35e3, up=16, half=0.1e-3, drop_db=6.0, search=None, spread=0.35e-3):
    """Direct-path arrival per channel of one compressed snippet y (n_ch, n).

    The arrival region comes from the channel-summed envelope smoothed over
    `spread` (> the largest inter-channel delay), first-arrival rule (earliest
    peak within drop_db of the max; only `search` = (i0, i1) samples). Each
    channel then takes its own first-arrival peak within +- spread of it, so
    all channels lock on the same (direct) path.
    Returns (t_env[ch] in s from sample 0, gated upsampled snippets, fs_up)."""
    yu = upsample(y, up)
    fu = fs * up
    e = np.abs(yu)
    lo, hi = (0, e.shape[1]) if search is None else (search[0] * up, search[1] * up)
    w = int(spread * fu)
    es = np.convolve((e / e[:, lo:hi].max(1, keepdims=True)).sum(0), np.ones(w) / w, mode="same")
    j = lo + first_arrival(es[lo:hi], drop_db, back=int(0.003 * fu))
    pk = np.zeros(len(y), int)
    for c in range(len(y)):
        a = max(j - w, 1)
        pk[c] = a + first_arrival(e[c, a:j + w], drop_db)
    # sub-sample envelope peak (parabola on the upsampled envelope)
    a, b, d = (e[np.arange(len(y)), pk + o] for o in (-1, 0, 1))
    frac = 0.5 * (a - d) / (a - 2 * b + d)
    t_env = (pk + frac) / fu
    h = int(round(half * fu))
    n = np.arange(yu.shape[1])
    w = np.clip(1 - ((n[None, :] - pk[:, None]) / h) ** 2, 0, None)   # Welch gate on each direct pulse
    return t_env, yu * w, fu


def pair_delays(y, fs, fc=35e3, up=16, half=0.1e-3, search=None, pairs=PAIRS):
    """Inter-channel delays tau_ij = t_j - t_i [s] of one snippet y (n_ch, n).

    Each channel's direct pulse is gated around its own envelope peak (so the
    echoes ~0.15 ms and later stay out), the gated pulses are cross-correlated,
    the envelope of the upsampled cross-correlation picks the lobe and its
    carrier phase refines the delay inside that lobe.
    Returns tau[npair], quality[npair] (normalized correlation, 0..1),
    coarse (envelope-only) tau[npair], t_env[ch]."""
    t_env, g, fu = channel_toa(y, fs, fc, up, half, search=search)
    G = np.fft.fft(g, 2 * g.shape[1], axis=1)
    en = np.sqrt((np.abs(g) ** 2).sum(1))
    tau, q, tc = np.zeros(len(pairs)), np.zeros(len(pairs)), np.zeros(len(pairs))
    L = G.shape[1]
    for p, (i, j) in enumerate(pairs):
        r = np.fft.ifft(G[j] * np.conj(G[i]))        # r[l] = sum g_j(n + l) g_i*(n): peak at l = t_j - t_i
        lag0 = int(round((t_env[j] - t_env[i]) * fu))
        ls = np.arange(lag0 - int(0.03e-3 * fu), lag0 + int(0.03e-3 * fu) + 1)
        ra = np.abs(r[ls % L])
        m = int(np.argmax(ra))
        if 0 < m < len(ls) - 1:
            fr = 0.5 * (ra[m - 1] - ra[m + 1]) / (ra[m - 1] - 2 * ra[m] + ra[m + 1])
        else:
            fr = 0.0
        tc[p] = (ls[m] + fr) / fu
        # carrier phase at the envelope peak: r(l) ~ |r| exp(j 2 pi fc (l/fu - tau))
        ph = np.angle(r[ls[m] % L])
        tau[p] = ls[m] / fu - ph / (2 * np.pi * fc)
        q[p] = ra[m] / (en[i] * en[j])
    return tau, q, tc, t_env


# --------------------------------------------------------------------------- geometry
def plane_delays(pos, az, c, pairs=PAIRS):
    """tau_ij = t_j - t_i = (r_i - r_j) . u / c for unit u = (cos az, sin az)
    (2-D array frame). pos (n_ch, 2), az (K,) -> (K, npair)."""
    u = np.stack([np.cos(az), np.sin(az)], -1)
    i, j = np.array(pairs).T
    return u @ (pos[i] - pos[j]).T / c


def geometry_closed_form(tau, c, pairs=PAIRS):
    """Closed-form array geometry from delays alone (step 4 of the module doc):
    d_k = delays rel. channel 0 = B u_k, |u_k| = 1 -> d^T M d = 1, linear LS for
    M on the dominant 2-D subspace of the data. Returns pos (n_ch, 2) in an
    arbitrary rotation, the singular values of D, and M's eigenvalues."""
    i0 = [p for p, (i, j) in enumerate(pairs) if i == 0]
    D = tau[:, i0]
    _, sv, Vt = np.linalg.svd(D, full_matrices=False)
    P = Vt[:2].T
    z = D @ P
    A = np.c_[z[:, 0] ** 2, 2 * z[:, 0] * z[:, 1], z[:, 1] ** 2]
    m = np.linalg.lstsq(A, np.ones(len(z)), rcond=None)[0]
    M = np.array([[m[0], m[1]], [m[1], m[2]]])
    w, V = np.linalg.eigh(M)
    C = V @ np.diag(np.abs(w) ** -0.5)
    pos = np.vstack([np.zeros(2), -(P @ C) * c])
    return pos, sv, w


def canonical_frame(pos, az):
    """Array frame: origin channel 0, +x from channel 0 to the last channel,
    +y toward the side where the sources are (median sin az > 0)."""
    v = pos[-1] - pos[0]
    a = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    p = (pos - pos[0]) @ R.T
    az = np.angle(np.exp(1j * (az - a)))
    if np.median(np.sin(az)) < 0:
        p[:, 1] *= -1
        az = -az
    return p, az


def fit_geometry(tau, c, pos0, w=None, pairs=PAIRS, f_scale=3e-6, length=None, bias=False, n_iter=50):
    """Joint LS fit of 2-D element positions + per-ping azimuth to the pair
    delays tau (K, npair) [s]. Gauge: channel 0 at the origin, the last
    channel on +x, sources on +y. `length` fixes the 0-to-last distance (for
    a profile of the scale); `bias` adds a constant timing bias per channel
    (channel 0 = 0), which trades with the small off-line offsets.

    Levenberg-Marquardt with the azimuths eliminated per ping (Schur
    complement; the ping blocks are 1-D) and Huber weights (f_scale): the
    geometry/azimuth coupling makes a generic trust-region solver crawl.
    Returns pos (n_ch, 2), az (K,), residual (K, npair) [s], bias (n_ch,) [s],
    cost (Huber chi2 with f_scale as sigma)."""
    K, npair = tau.shape
    nch = len(pos0)
    i, j = np.array(pairs).T
    w = np.ones_like(tau) if w is None else w
    pos0, az = canonical_frame(pos0, bearing_ls(tau, pos0, c, pairs=pairs)[0])
    if length is not None:
        pos0 = pos0 * length / pos0[-1, 0]
        az = bearing_ls(tau, pos0, c, pairs=pairs)[0]
    free = [(n, d) for n in range(1, nch) for d in range(2)
            if not (n == nch - 1 and (d == 1 or length is not None))]
    nb = nch - 1 if bias else 0
    ng = len(free) + nb

    def unpack(x):
        p = pos0.copy()
        for v, (n, d) in zip(x[:len(free)], free):
            p[n, d] = v
        bb = np.r_[0.0, x[len(free):] * 1e-6] if bias else np.zeros(nch)
        return p, bb

    def resid(x, az):
        p, bb = unpack(x)
        return (plane_delays(p, az, c, pairs) + bb[j] - bb[i] - tau) * w / f_scale

    def huber(r):
        a = np.abs(r)
        rho = np.where(a <= 1, a ** 2, 2 * a - 1)
        return rho.sum(), np.where(a <= 1, 1.0, 1 / np.maximum(a, 1e-12))

    x = np.r_[[pos0[n, d] for n, d in free], np.zeros(nb)]
    r = resid(x, az)
    cost, wt = huber(r)
    lam = 1e-3
    for _ in range(n_iter):
        p, _ = unpack(x)
        Jb = plane_delays(p, az + np.pi / 2, c, pairs) * w / f_scale            # (K, npair) d r / d az_k
        eps = np.r_[[1e-5] * len(free), [1e-3] * nb]
        Jg = np.stack([(resid(x + e * np.eye(ng)[m], az) - r) / e for m, e in enumerate(eps)], -1)  # (K, npair, ng)
        sw = np.sqrt(wt)
        Jb, Jg, rr = Jb * sw, Jg * sw[..., None], r * sw
        bb_ = (Jb * Jb).sum(1)                                                   # (K,)
        bg = np.einsum("kp,kpg->kg", Jb, Jg)
        br = (Jb * rr).sum(1)
        H = np.einsum("kpg,kph->gh", Jg, Jg) - np.einsum("kg,kh->gh", bg / bb_[:, None], bg)
        gvec = np.einsum("kpg,kp->g", Jg, rr) - (bg * (br / bb_)[:, None]).sum(0)
        while True:
            dg = -np.linalg.solve(H + lam * np.diag(np.diag(H)), gvec)
            daz = -(br + bg @ dg) / (bb_ * (1 + lam))
            r2 = resid(x + dg, az + daz)
            c2, wt2 = huber(r2)
            if c2 < cost:
                x, az, r, cost, wt = x + dg, az + daz, r2, c2, wt2
                lam = max(lam / 3, 1e-9)
                break
            lam *= 10
            if lam > 1e8:
                break
        if lam > 1e8 or np.abs(dg).max() < 1e-7:
            break
    p, bb = unpack(x)
    p2, az2 = canonical_frame(p, az)            # (channel biases are frame independent)
    return p2, az2, plane_delays(p2, az2, c, pairs) + bb[j] - bb[i] - tau, bb, cost


# --------------------------------------------------------------------------- bearing
def bearing_ls(tau, pos, c, w=None, pairs=PAIRS, n_grid=720):
    """Per-ping azimuth (array frame) by grid + Gauss-Newton on the pair
    delays. Returns az (K,), rms residual (K,) [s], az of the best solution on
    the other side of the array axis (mirror) and its rms (K,), and the
    sensitivity |d tau / d az| (K,) [s/rad] for the sigma."""
    tau = np.atleast_2d(tau)
    w = np.ones_like(tau) if w is None else np.atleast_2d(w)
    g = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    G = plane_delays(pos, g, c, pairs)                              # (n_grid, npair)
    cost = (((tau[:, None, :] - G[None]) * w[:, None, :]) ** 2).sum(-1)
    axis = np.arctan2(*(pos[-1] - pos[0])[::-1])
    side = np.sin(g - axis) >= 0

    def refine(az):
        for _ in range(8):
            r = (plane_delays(pos, az, c, pairs) - tau) * w
            J = plane_delays(pos, az + np.pi / 2, c, pairs) * w       # d u / d az = u(az + pi/2)
            az = az - (J * r).sum(1) / (J * J).sum(1)
        r = (plane_delays(pos, az, c, pairs) - tau) * w
        J = plane_delays(pos, az + np.pi / 2, c, pairs)
        return az, np.sqrt((r ** 2).mean(1)), np.sqrt((J ** 2).sum(1))

    k = np.argmin(cost, 1)
    az, rms, sens = refine(g[k])
    same = side[k]
    cm = np.where(side[None, :] != same[:, None], cost, np.inf)
    azm, rmsm, _ = refine(g[np.argmin(cm, 1)])
    return np.angle(np.exp(1j * az)), rms, np.angle(np.exp(1j * azm)), rmsm, sens


def resolve_cycles(tau, tc, pos, c, fc=35e3, sig_fine=1.5e-6, sig_coarse=6e-6, pairs=PAIRS, n=1):
    """Pick the carrier cycle of each fine delay. The fine delays are exact
    modulo 1/fc and consistent around triangles; their absolute lobe is
    chosen by testing integer shifts (n_1..n_3 in -n..n) of the channel-0
    delays and scoring the plane-wave fit residual on `pos` plus the distance
    to the envelope-only delays tc. Returns tau (K, npair), score margin to
    the second-best combination (chi2 units)."""
    Tc = 1 / fc
    i0 = {j: p for p, (i, j) in enumerate(pairs) if i == 0}
    nch = len(pos)
    # channel offsets rel. ch0 from the fine 0-j delays, then all pairs from them
    base = np.c_[np.zeros(len(tau)), tau[:, [i0[j] for j in range(1, nch)]]]
    combos = np.array(np.meshgrid(*[np.arange(-n, n + 1)] * (nch - 1), indexing="ij")).reshape(nch - 1, -1).T
    i, j = np.array(pairs).T
    best = np.full(len(tau), np.inf)
    second = np.full(len(tau), np.inf)
    out = tau.copy()
    for cmb in combos:
        b = base + np.r_[0, cmb] * Tc
        cand = tau + np.round((b[:, j] - b[:, i] - tau) / Tc) * Tc   # each pair keeps its own phase
        az, rms, *_ = bearing_ls(cand, pos, c, pairs=pairs)
        chi = len(pairs) * rms ** 2 / sig_fine ** 2 + (((cand - tc) / sig_coarse) ** 2).sum(1)
        better = chi < best
        second = np.where(better, best, np.minimum(second, chi))
        out[better] = cand[better]
        best = np.where(better, chi, best)
    return out, second - best


# --------------------------------------------------------------------------- wall echo
def echo_stack(Y, t_env, fs, lags, up=8):
    """Envelopes |MF| (K, ch, nlag), each channel aligned on its own direct
    arrival t_env (s from snippet start) and normalized to its direct peak."""
    E = np.zeros((len(Y), Y.shape[1], len(lags)))
    for k, y in enumerate(Y):
        e = np.abs(upsample(y, up))
        tt = np.arange(e.shape[1]) / (fs * up)
        for c in range(len(y)):
            E[k, c] = np.interp(t_env[k, c] + lags, tt, e[c]) / e[c].max()
    return E


def wall_lags(pos, az, h0, phi, c):
    """Echo lag behind the direct path (K, ch) for a plane reflector with unit
    normal n = (cos phi, sin phi) pointing from the wall to the array, at
    distance h0 from channel 0: lag_i = 2 (h0 + n . r_i) (n . u) / c, with u
    the unit vector to the source; NaN where the source is behind the wall."""
    n = np.array([np.cos(phi), np.sin(phi)])
    nu = np.cos(az - phi)
    lag = 2 * (h0 + pos @ n)[None, :] * nu[:, None] / c
    return np.where(nu[:, None] > 0, lag, np.nan)


def fit_wall(E, lags, pos, az, c, h_grid=np.arange(0.1, 3.0, 0.01), phi_grid=np.radians(np.arange(0, 360, 1.0)),
             min_lag=0.35e-3):
    """Plane reflector (h0, phi) maximizing the mean stacked envelope along the
    predicted echo lags (only lags > min_lag, clear of the direct pulse and its
    sidelobes). Returns dict h0, phi, h (per element distance to the wall),
    score map, the per-ping/channel envelope at the predicted lag (echo
    amplitude re direct) and the background level for the same lags."""
    dl = lags[1] - lags[0]
    S = np.full((len(h_grid), len(phi_grid)), np.nan)
    K, nch, _ = E.shape
    kk, cc = np.meshgrid(np.arange(K), np.arange(nch), indexing="ij")

    def score(h0, phi):
        L = wall_lags(pos, az, h0, phi, c)
        m = np.isfinite(L) & (L > min_lag) & (L < lags[-1])
        if m.sum() < 0.2 * L.size:
            return np.nan, None
        idx = np.round((L[m] - lags[0]) / dl).astype(int)
        return E[kk[m], cc[m], idx].mean(), (m, idx)

    for a, h0 in enumerate(h_grid):
        for b, phi in enumerate(phi_grid):
            S[a, b] = score(h0, phi)[0]
    a, b = np.unravel_index(np.nanargmax(S), S.shape)
    h0, phi = h_grid[a], phi_grid[b]
    n = np.array([np.cos(phi), np.sin(phi)])
    L = wall_lags(pos, az, h0, phi, c)
    amp = np.full(L.shape, np.nan)
    m = np.isfinite(L) & (L < lags[-1])
    amp[m] = E[kk[m], cc[m], np.round((L[m] - lags[0]) / dl).astype(int)]
    bg = np.nanmedian(E[:, :, lags > min_lag])
    return dict(h0=h0, phi=phi, h=h0 + pos @ n, score=S, h_grid=h_grid, phi_grid=phi_grid,
                lag=L, amp=amp, background=bg)


# --------------------------------------------------------------------------- pipeline
def load_pings(cfg, rec, source="auto", cache=True):
    """USBL ping arrivals (rx clock) + Doppler scale. source "toa" uses agent
    A's table (out/toa_phase1.npz + out/detections_USBL.npz, gives the shared
    ping index k); "own" the standalone finder (k = -1, cached in
    out/usbl/pings.npz); "auto" = toa if present."""
    ft, fd = config.ROOT / "out" / "toa_phase1.npz", config.ROOT / "out" / "detections_USBL.npz"
    if source in ("auto", "toa") and ft.exists() and fd.exists():
        T, D = np.load(ft), np.load(fd)["det"]
        u = list(T["receivers"]).index("USBL")
        di = T["det_index"][:, u]
        m = di >= 0
        m[m] = np.isfinite(D["t_rx"][di[m]]) & np.isfinite(D["eta"][di[m]])
        return dict(t=D["t_rx"][di[m]], scale=D["eta"][di[m]], k=T["k"][m], b=float(T["usbl_offset"]),
                    source="toa_phase1")
    f = OUT / "pings.npz"
    key = stamp.key(["pinger"], files=[cfg.receiver("USBL").file], code=__file__, offset=rec.offset)
    p = stamp.load(f, key) if cache else None
    if p is None:
        p = find_pings(rec, 100.0, 960.0, cfg.pinger)
        stamp.save(f, key, **p)
    return dict(t=p["t"], scale=p["scale"], k=-np.ones(len(p["t"]), int), b=rec.offset, source="own")


def measure(rec, pings, pg, pre=0.003, post=0.006, search=1.5e-3, half=0.1e-3):
    """Snippets + per-ping fine/coarse pair delays, per-channel direct arrival
    (rx clock), per-channel direct amplitude and MF noise."""
    fs = rec.fs
    Y, t0 = snippets(rec, pings["t"], pings["scale"], pg, pre, post)
    i0 = int(pre * fs)
    R = [pair_delays(y, fs, 0.5 * (pg.f0 + pg.f1) * sc, half=half,
                     search=(i0 - int(search * fs), i0 + int(search * fs)))
         for y, sc in zip(Y, pings["scale"])]
    tau, q, tc, te = (np.array([r[n] for r in R]) for n in range(4))
    amp = np.abs(Y).max(-1)
    noise = np.zeros_like(amp)              # MF noise floor between pings (0.45-0.6 s after the arrival)
    for k, (t, sc) in enumerate(zip(pings["t"], pings["scale"])):
        rep = lfm_replica(fs, pg.f0, pg.f1, pg.duration, sc)
        x, _ = rec.segment(t + 0.45, t + 0.6 + pg.duration, clock="rx")
        noise[k] = np.median(np.abs(compress(x, rep)[: int(0.15 * fs)]), 0)
    return dict(Y=Y, t0=t0, tau=tau, q=q, tc=tc, t_env=te, t_ch=t0[:, None] + te, amp=amp,
                snr=20 * np.log10(amp / noise))


def calibrate(tau, tc, c, fc=35e3, n_iter=3, ok_rms=4e-6):
    """Self-calibration loop: clean subset (cycle closure + temporal smoothness)
    -> closed-form geometry -> joint fit -> resolve carrier cycles of all pings
    on that geometry -> refit on the pings that fit (rms < ok_rms)."""
    from scipy.ndimage import median_filter
    med = median_filter(tau, size=(9, 1), mode="nearest")
    cl = np.c_[tau[:, 0] + tau[:, 3] - tau[:, 1], tau[:, 0] + tau[:, 4] - tau[:, 2], tau[:, 1] + tau[:, 5] - tau[:, 2]]
    ok = (np.abs(tau - med).max(1) < 15e-6) & (np.abs(cl).max(1) < 3e-6)
    pos, sv, _ = geometry_closed_form(tau[ok], c)
    pos = fit_geometry(tau[ok], c, pos)[0]
    for _ in range(n_iter):
        tr, margin = resolve_cycles(tau, tc, pos, c, fc)
        rms = bearing_ls(tr, pos, c)[1]
        ok = rms < ok_rms
        pos = fit_geometry(tr[ok], c, pos)[0]
    return dict(pos=pos, tau=tr, margin=margin, ok=ok, sv=sv)


def bootstrap_geometry(tau, c, pos, n_boot=40, block=30, seed=1):
    """Block bootstrap (blocks of consecutive pings, the delay errors are
    correlated in time) of the joint geometry fit -> positions (n_boot, ch, 2)."""
    rng = np.random.default_rng(seed)
    K = len(tau)
    nb = int(np.ceil(K / block))
    out = []
    for _ in range(n_boot):
        st = rng.integers(0, K - block, nb)
        idx = (st[:, None] + np.arange(block)).ravel()[:K]
        out.append(fit_geometry(tau[idx], c, pos)[0])
    return np.array(out)


def bearings(tau, pos, c, sig_floor=0.5e-6, pos_boot=None):
    """Per-ping azimuth (array frame) + sigma. sigma_tau per ping from the fit
    residual (rms * sqrt(npair / (npair - 1)), floored), mapped through the
    azimuth sensitivity; sigma_sys from the geometry bootstrap (az with each
    bootstrap geometry; correlated over pings)."""
    az, rms, azm, rmsm, sens = bearing_ls(tau, pos, c)
    npair = tau.shape[1]
    st = np.maximum(rms * np.sqrt(npair / (npair - 1)), sig_floor)
    sig = st / sens
    sys_ = np.zeros_like(az)
    if pos_boot is not None:
        A = np.array([bearing_ls(tau, p, c)[0] for p in pos_boot])
        sys_ = np.std(np.angle(np.exp(1j * (A - az))), 0)
    return dict(az=az, sigma=sig, sigma_sys=sys_, rms=rms, az_mirror=azm, rms_mirror=rmsm, sigma_tau=st)


def bearing_scatter(t, az, ok, win=7.5, min_n=8):
    """Truth-free bearing noise: residual of each ok ping about a local
    quadratic in time (+- win s of ok pings, the ping itself included).
    Returns residual (K,) [rad] (NaN where not computed), robust std
    (corrected for the fit's own dof), lag-1 correlation of consecutive pings."""
    res = np.full(len(t), np.nan)
    idx = np.where(ok)[0]
    for k in idx:
        m = np.abs(t[idx] - t[k]) <= win
        if m.sum() >= min_n:
            p = np.polyfit(t[idx][m] - t[k], np.unwrap(az[idx][m]), 2)
            res[k] = np.angle(np.exp(1j * (az[k] - np.polyval(p, 0))))
    n = np.median([(np.abs(t[idx] - t[k]) <= win).sum() for k in idx])
    std = 1.4826 * np.nanmedian(np.abs(res)) * np.sqrt(n / (n - 3))
    a, b = res[:-1], res[1:]
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(np.diff(t) - np.median(np.diff(t))) < 0.05)
    return res, std, np.corrcoef(a[m], b[m])[0, 1]


def calibrate_orientation(az, ok, k, track_csv):
    """Array-frame azimuth -> ENU bearing az_enu = alpha + s * az_array, with
    the handedness s = +-1 (the 2-D array frame does not know up from down,
    i.e. the mirror of the whole frame) and the rotation alpha fitted to the
    bearing atan2(n, e) of a reference pinger track (graph solve, TOA only)
    at the same ping index k. Robust (circular median) fit per s; the s with
    the smaller residual wins. Returns alpha, s, residual rms per s, residuals."""
    import csv
    with open(track_csv) as f:
        rows = list(csv.DictReader(f))
    kt = np.array([int(r["ping"]) for r in rows])
    ref = np.array([np.arctan2(float(r["pinger_n"]), float(r["pinger_e"])) for r in rows])
    m = ok & np.isin(k, kt)
    rmap = dict(zip(kt, ref))
    rf = np.array([rmap[kk] for kk in k[m]])
    out = {}
    for sgn in (1, -1):
        d = np.angle(np.exp(1j * (rf - sgn * az[m])))
        al = np.angle(np.exp(1j * d).mean())
        for _ in range(5):                                   # circular median by iterated shrink
            al = al + np.median(np.angle(np.exp(1j * (d - al))))
        r = np.angle(np.exp(1j * (d - al)))
        out[sgn] = (al, 1.4826 * np.median(np.abs(r)), r)
    sgn = min(out, key=lambda v: out[v][1])
    return dict(alpha=out[sgn][0], s=sgn, rms={v: out[v][1] for v in out}, resid=out[sgn][2], mask=m)


def noise_psd(rec, t0, t1, nperseg=4096):
    """Welch PSD per channel of a ping-free stretch (rx clock)."""
    x, _ = rec.segment(t0, t1, clock="rx")
    return signal.welch(x, rec.fs, nperseg=nperseg, axis=0)


def main(cfg_path=None, ping_source="auto"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = config.load(*([cfg_path] if cfg_path else []))
    rx, pg = cfg.receiver("USBL"), cfg.pinger
    rec = recordings.load(rx)
    c = cfg.priors.get("sound_speed", 1520.0)
    c_sig = cfg.priors.get("sound_speed_sigma", 10.0)
    fc = 0.5 * (pg.f0 + pg.f1)
    OUT.mkdir(parents=True, exist_ok=True)

    P = load_pings(cfg, rec, ping_source)
    M = measure(rec, P, pg)
    K = len(P["t"])
    print(f"{K} pings from {P['source']}, USBL clock offset b = {P['b']:.5f} s")

    # channel health: MF SNR, direct amplitude ratios, broadband noise
    f, psd = noise_psd(rec, 1000.0, 1010.0)
    band = (f > pg.f0) & (f < pg.f1)
    nb_db = 10 * np.log10(psd[band].mean(0))
    print("in-band noise PSD [dB re 1/Hz]", nb_db.round(1), " MF SNR median", np.median(M["snr"], 0).round(1),
          " p10", np.percentile(M["snr"], 10, 0).round(1))
    print("direct amplitude re ch0 (median dB)", np.median(20 * np.log10(M["amp"] / M["amp"][:, :1]), 0).round(1))

    # geometry
    G = calibrate(M["tau"], M["tc"], c, fc)
    pos, ok, tau = G["pos"], G["ok"], G["tau"]
    i0 = [p for p, (i, j) in enumerate(PAIRS) if i == 0]
    sv = np.linalg.svd(tau[ok][:, i0], compute_uv=False)
    boot = bootstrap_geometry(tau[ok], c, pos)
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    dist_b = np.linalg.norm(boot[:, :, None] - boot[:, None], axis=-1)
    dsig = np.sqrt(dist_b.std(0) ** 2 + (dist * c_sig / c) ** 2)
    # per-channel timing skew (e.g. a multiplexed ADC: 1/(fs n_ch) = 2.6 us per channel) vs the scale;
    # free per-channel biases are not identifiable with this half-plane coverage (they trade with the scale)
    i_, j_ = np.array(PAIRS).T
    skew = np.array([-2.6e-6, 2.6e-6])
    L_skew = [fit_geometry(tau[ok] - (np.arange(4) * s)[j_] + (np.arange(4) * s)[i_], c, pos)[0][-1, 0] for s in skew]
    prof_L = np.linspace(0.44, 0.60, 17)
    prof = np.array([fit_geometry(tau[ok], c, pos, length=L)[4] for L in prof_L])
    print(f"ok pings {ok.sum()}/{K}; delay SVs (rel ch0) [us] {np.round(sv / np.sqrt(ok.sum()) * 1e6, 2)}")
    print("element positions [m] (array frame)\n", pos.round(4))
    for (i, j) in PAIRS:
        print(f"  d{i}{j} = {dist[i, j]:.3f} +- {dsig[i, j]:.3f} m")
    print(f"  |r3 - r0| with a +-2.6 us/channel sampling skew: {np.round(L_skew, 3)} m")
    print(f"  endfire bound |r3 - r0| >= max|tau03| c = {np.abs(tau[ok][:, 2]).max() * c:.3f} m")

    # bearings
    B = bearings(tau, pos, c, pos_boot=boot)
    B["quality"] = np.where(ok, np.exp(-0.5 * (B["rms"] / 2e-6) ** 2), 0.0)
    print("bearing sigma [deg] median/p90 (ok)", np.degrees(np.percentile(B["sigma"][ok], [50, 90])).round(2),
          " systematic (geometry) median", np.degrees(np.median(B["sigma_sys"][ok])).round(2))
    az_ok = np.degrees(B["az"][ok])
    print("azimuth coverage [deg] (ok)", np.percentile(az_ok, [0, 5, 50, 95, 100]).round(1))
    print("mirror rejection: rms best/mirror median [us]", np.median(B["rms"][ok]) * 1e6, np.median(B["rms_mirror"][ok]) * 1e6)
    # truth-free noise: scatter about a local quadratic in time; outliers (wrong lobe) > 5 sigma dropped
    res, s_emp, rho1 = bearing_scatter(t_usbl := M["t_ch"].mean(1), B["az"], ok)
    out_l = ok & ((np.abs(np.nan_to_num(res)) > 5 * s_emp) | (np.sin(B["az"]) <= 0) | np.isnan(res))
    ok &= ~out_l
    res, s_emp, rho1 = bearing_scatter(t_usbl, B["az"], ok)
    B["sigma_noise"] = B["sigma"]
    B["sigma"] = np.sqrt(B["sigma_noise"] ** 2 + s_emp ** 2)
    B["quality"] = np.where(ok, B["quality"], 0.0)
    print(f"bearing scatter about a local quadratic (+-7.5 s): {np.degrees(s_emp):.2f} deg rms, lag-1 corr {rho1:.2f}"
          f" (formal from the delay fit: {np.degrees(np.median(B['sigma_noise'][ok])):.2f} deg); {out_l.sum()} outliers dropped")
    for lo, hi in [(0, 60), (60, 120), (120, 145), (145, 180)]:
        m = ok & (np.degrees(B["az"]) >= lo) & (np.degrees(B["az"]) < hi)
        if m.sum() > 10:
            print(f"   az {lo:3d}-{hi:3d} deg: n {m.sum():3d}, scatter {np.degrees(1.4826 * np.nanmedian(np.abs(res[m]))):.2f} deg,"
                  f" formal {np.degrees(np.median(B['sigma_noise'][m])):.2f}, geometry {np.degrees(np.median(B['sigma_sys'][m])):.2f}")

    # wall echo
    lags = np.arange(-0.5e-3, 3e-3, 1 / (rec.fs * 8))
    E = echo_stack(M["Y"], M["t_env"], rec.fs, lags)
    W = fit_wall(E[ok], lags, pos, B["az"][ok], c, h_grid=np.arange(0.1, 3.0, 0.02),
                 phi_grid=np.radians(np.arange(0, 360, 2.0)))
    W = fit_wall(E[ok], lags, pos, B["az"][ok], c, h_grid=np.arange(W["h0"] - 0.1, W["h0"] + 0.1, 0.005),
                 phi_grid=W["phi"] + np.radians(np.arange(-6, 6, 0.25)))
    Wb = [fit_wall(E[ok][ib], lags, pos, B["az"][ok][ib], c, h_grid=np.arange(W["h0"] - 0.15, W["h0"] + 0.15, 0.01),
                   phi_grid=W["phi"] + np.radians(np.arange(-10, 10, 0.5)))
          for ib in np.random.default_rng(2).integers(0, ok.sum(), (20, ok.sum()))]
    hb = np.array([w["h"] for w in Wb])
    phib = np.degrees([w["phi"] for w in Wb])
    lag_all = wall_lags(pos, B["az"], W["h0"], W["phi"], c)
    print(f"wall: normal {np.degrees(W['phi']):.1f} +- {phib.std():.1f} deg (array frame), distance per element "
          f"{W['h'].round(3)} +- {np.sqrt(hb.std(0) ** 2 + (W['h'] * c_sig / c) ** 2).round(3)} m")
    print(f"  echo amplitude re direct median {np.nanmedian(W['amp']):.2f} (background {W['background']:.2f});"
          f" lag range {np.nanmin(W['lag']) * 1e3:.2f}-{np.nanmax(W['lag']) * 1e3:.2f} ms;"
          f" pings with lag < 0.3 ms: {(np.nanmin(lag_all, 1) < 0.3e-3).sum()}")

    # orientation from the TOA-only graph track (independent of the USBL bearings)
    ftr = config.ROOT / "out" / "phase1_toa_only" / "track.csv"
    if ftr.exists() and (P["k"] >= 0).any():
        O = calibrate_orientation(B["az"], ok, P["k"], ftr)
        az_enu = np.angle(np.exp(1j * (O["alpha"] + O["s"] * B["az"])))
        print(f"orientation vs track.csv: alpha {np.degrees(O['alpha']):.2f} deg, handedness s = {O['s']:+d}; "
              f"robust resid rms s=+1 {np.degrees(O['rms'][1]):.2f} / s=-1 {np.degrees(O['rms'][-1]):.2f} deg "
              f"(n = {O['mask'].sum()})")
        alpha, hand = O["alpha"], O["s"]
    else:
        az_enu, alpha, hand = np.full(K, np.nan), np.nan, 0
        print("no TOA-only track.csv / ping index: az_enu not calibrated (NaN)")

    # outputs
    out = dict(k=P["k"], mask=ok, az_enu=az_enu, az_array=B["az"], alpha=alpha, handedness=hand,
               t_usbl=t_usbl, t_buoy=t_usbl - P["b"], t_ch=M["t_ch"], az=B["az"],
               el=np.full(K, np.nan), sigma=B["sigma"], sigma_noise=B["sigma_noise"], sigma_emp=s_emp,
               sigma_sys=B["sigma_sys"], az_resid=res, wall_overlap=np.nanmin(lag_all, 1) < 0.3e-3, quality=B["quality"], ok=ok,
               rms=B["rms"], az_mirror=B["az_mirror"], rms_mirror=B["rms_mirror"], tau=tau, tau_raw=M["tau"],
               tau_coarse=M["tc"], cycle_margin=G["margin"], snr=M["snr"], scale=P["scale"],
               pairs=np.array(PAIRS), pos=pos, pos_boot=boot, dist=dist, dist_sigma=dsig, sound_speed=c,
               usbl_offset=P["b"], ping_source=P["source"], wall_h0=W["h0"], wall_phi=W["phi"], wall_h=W["h"],
               wall_h_sigma=hb.std(0), wall_lag=lag_all, wall_amp=W["amp"], prof_L=prof_L, prof_cost=prof,
               L_skew=np.array(L_skew))
    np.savez(OUT / "bearings_phase1.npz", **out)
    print("wrote", OUT / "bearings_phase1.npz")

    # ---------------------------------------------------------------- plots
    t = t_usbl
    fig, ax = plt.subplots(6, 1, figsize=(10, 13), sharex=True)
    for p, (i, j) in enumerate(PAIRS):
        ax[p].plot(t[~ok], M["tau"][~ok, p] * 1e6, "x", color="0.6", ms=3, label="rejected (raw)")
        ax[p].plot(t[ok], tau[ok, p] * 1e6, ".", ms=2, label="resolved")
        ax[p].set_ylabel(f"tau{i}{j} [us]")
        ax[p].set_ylim(-400, 400)
        ax[p].grid(alpha=0.3)
    ax[0].legend(loc="upper left", fontsize=8)
    ax[-1].set_xlabel("USBL time [s]")
    fig.suptitle("USBL inter-channel delays tau_ij = t_j - t_i")
    fig.tight_layout()
    fig.savefig(OUT / "delays.png", dpi=90)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for b_ in boot:
        ax[0].plot(b_[:, 0], b_[:, 1], ".", color="0.7", ms=2)
    ax[0].plot(pos[:, 0], pos[:, 1], "ko")
    for i, p in enumerate(pos):
        ax[0].annotate(f"ch{i}", p, textcoords="offset points", xytext=(4, 6))
    n = np.array([np.cos(W["phi"]), np.sin(W["phi"])])
    wp = pos[0] - W["h0"] * n
    tv = np.array([-n[1], n[0]])
    ax[0].plot(*np.c_[wp - 0.8 * tv, wp + 0.8 * tv], "brown", lw=3, label="wall (fit)")
    for a in np.radians([np.percentile(az_ok, 1), np.percentile(az_ok, 99)]):
        ax[0].plot([0.25, 0.25 + 0.5 * np.cos(a)], [0, 0.5 * np.sin(a)], "b--", lw=1)
    ax[0].set_aspect("equal")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[0].set_title("array frame: elements (bootstrap grey), wall, az coverage (blue)")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[1].plot(prof_L, prof - prof.min(), "o-")
    ax[1].set_xlabel("fixed |r3 - r0| [m]")
    ax[1].set_ylabel("cost - min (chi2, sigma 3 us)")
    ax[1].set_title("scale profile of the self-calibration")
    ax[1].grid(alpha=0.3)
    ax[2].hist(az_ok, bins=np.arange(0, 181, 2))
    ax[2].set_xlabel("azimuth in array frame [deg]")
    ax[2].set_title("bearing coverage (ok pings)")
    fig.tight_layout()
    fig.savefig(OUT / "geometry.png", dpi=90)

    o = np.argsort(B["az"][ok])
    fig, ax = plt.subplots(1, 4, figsize=(16, 7), sharey=True)
    for ch in range(4):
        ax[ch].pcolormesh(lags * 1e3, np.degrees(B["az"][ok][o]), 20 * np.log10(E[ok][o, ch] + 1e-3),
                          vmin=-25, vmax=0, shading="nearest")
        ax[ch].plot(W["lag"][o, ch] * 1e3, np.degrees(B["az"][ok][o]), "r--", lw=0.8)
        ax[ch].set_title(f"ch{ch}: |MF| re direct [dB]")
        ax[ch].set_xlabel("lag after direct [ms]")
    ax[0].set_ylabel("azimuth (array frame) [deg]")
    fig.suptitle(f"wall echo: h = {W['h'].round(2)} m, normal {np.degrees(W['phi']):.0f} deg (red: model)")
    fig.tight_layout()
    fig.savefig(OUT / "wall_echo.png", dpi=90)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].errorbar(t[ok], np.degrees(B["az"][ok]), np.degrees(B["sigma"][ok]), fmt=".", ms=2, elinewidth=0.5)
    ax[0].set_ylabel("az (array frame) [deg]")
    ax[1].semilogy(t[ok], np.degrees(B["sigma"][ok]), ".", ms=2, label="sigma (noise)")
    ax[1].semilogy(t[ok], np.degrees(B["sigma_sys"][ok]), ".", ms=2, label="sigma_sys (geometry)")
    ax[1].set_ylabel("deg")
    ax[1].legend()
    for ch in range(4):
        ax[2].plot(t, M["snr"][:, ch], ".", ms=2, label=f"ch{ch}")
    ax[2].set_ylabel("MF SNR [dB]")
    ax[2].legend(ncol=4)
    ax[2].set_xlabel("USBL time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bearing.png", dpi=90)

    fig, ax = plt.subplots(figsize=(9, 4))
    for ch in range(4):
        ax.plot(f / 1e3, 10 * np.log10(psd[:, ch]), lw=0.8, label=f"ch{ch}")
    ax.axvspan(pg.f0 / 1e3, pg.f1 / 1e3, color="0.9")
    ax.set_xlabel("kHz")
    ax.set_ylabel("PSD [dB re FS^2/Hz]")
    ax.set_title("USBL noise spectrum (1000-1010 s, no pinger)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "noise_psd.png", dpi=90)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--pings", default="auto", choices=["auto", "toa", "own"])
    a = ap.parse_args()
    main(a.config, a.pings)
