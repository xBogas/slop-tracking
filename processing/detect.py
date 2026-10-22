"""
Pinger detection, TOA and range-rate estimation on the raw recordings.

Signal: LFM up-sweep f0 -> f1 (30 -> 40 kHz), T = 100 ms, rectangular
envelope (measured: notes/A_detection.md). TOA is the arrival time of the
pulse START (the f0 edge) on every receiver.

Method
------
1. Coarse pass (chunked). Chunks of a few seconds (plus a pulse of
   overlap) are matched-filtered in the frequency domain with the analytic
   replica; bins outside [f0 - 1 kHz, f1 + 1 kHz] are zeroed, so the filter
   is also the band-pass. The complex output c(t) peaks at the pulse start.
   Channels are summed incoherently, P = sum |c_ch|^2. The noise floor is
   the median of P per 1 s block (P/sigma^2 ~ Gamma(n_ch) in noise), so
   the threshold follows the noise. Local maxima above it are clustered
   (+/- 0.3 s, pings are 1 s apart); the strongest one is the anchor.
2. Doppler. Motion time-scales the pulse, x(t) = a s(eta (t - t0)),
   eta ~ 1 - rr/c (rr > 0 opening). For an LFM, a scale mismatch d_eta
   moves the matched-filter peak by about -T f1/B * d_eta (0.4 s per unit,
   i.e. 0.26 ms per m/s), so the replica must be Doppler-matched for an
   unbiased TOA. On these recordings, the pulse's own Doppler estimate
   (the peak of power vs eta) is not reliable: the ridge is flat and
   overlapping multipath tilts it by several m/s. So:
   - the per-ping Doppler `rr_dop` is the profile-likelihood eta of the
     multipath fit of step 4 (`profile_eta`; a joint least squares over
     eta and delays gets stuck on the ridge) and is reported as a separate
     measurement with its scatter;
   - the reference eta for the TOA comes from the receiver's own arrival
     sequence: the arrival interval is Tp/eta, so a robust local quadratic
     fit of TOA vs ping number gives eta_k (`eta_track`). That is iterated
     (TOA -> eta -> TOA). With a nominal Tp, a pinger clock error biases
     eta by the same amount on every receiver, so it shifts all TOAs of a
     ping equally and is absorbed by the emission time.
3. First arrival. On the eta-matched output (channel sum), the earliest
   local maximum within `first_db` of the strongest, at most `pre_win`
   before it, is the candidate (echoes come later: surface/bottom on the
   buoys, the wall 0.4-1.1 ms behind the USBL).
4. Multipath fit. The baseband MF output in a window around the
   candidate is fitted with c(t) = sum_i a_i X(t - tau_i; eta), with X
   the cross-ambiguity of s(eta t) with the reference replica (complex a_i
   solved linearly, tau_i by least squares: variable projection). Paths are added greedily while each removes a significant
   part of the window energy. The TOA is the earliest path within
   `first_db` of the strongest. This resolves echoes closer than 1/B that
   would otherwise pull the peak. Multi-channel: candidate from the
   channel sum, fits per channel, t_rx = mean over channels (array centre,
   missing channels filled by their smoothed offset, `array_toa`).
5. Pulse shape. The received pulse is not the ideal LFM: the transmit
   envelope and each receiver's response tilt the amplitude across the
   sweep by up to 10 dB and add a quadratic phase (~0.5-1 m/s of apparent
   Doppler). `pulse_shape` measures the complex shape w(u) per channel by
   averaging dechirped first arrivals over ~400 pings (other paths average
   out); the model X then uses w(eta t) s(eta t). The pipeline
   (`detect_receiver`) runs: ideal model -> shape -> shaped model -> shape
   -> shaped model with Doppler.

Times: t_rx is the receiver clock, t_buoy = t_rx - rec.offset (the coarse
guess; associate.py resolves the USBL offset). snr_db is the MF output SNR
of the first path (in-band SNR + 10 log10(B T) = + 30 dB). sigma_toa and
sigma_rr are CRLB-style; associate.py gives empirical ones.

Run: python3 detect.py [rx ...]   -> out/detections_<rx>.npz (pinger runs)
"""
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import optimize, stats
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import recordings

C_SOUND = 1520.0

DET_DTYPE = [("t_buoy", "f8"), ("t_rx", "f8"), ("snr_db", "f4"), ("snr_peak_db", "f4"),
             ("eta", "f8"), ("rr_dop", "f4"), ("rr_track", "f4"), ("amp", "f4"),
             ("n_paths", "i2"), ("dt_strong", "f4"), ("echo_db", "f4"), ("echo_dt", "f4"),
             ("sigma_toa", "f4"), ("sigma_rr", "f4"), ("t_coarse", "f8"), ("k", "i4")]


def replica(pinger, fs, eta=1.0):
    """Analytic LFM s(eta t), t in [0, T/eta), unit amplitude."""
    T, k = pinger.duration, (pinger.f1 - pinger.f0) / pinger.duration
    u = eta * np.arange(int(np.ceil(T / eta * fs))) / fs     # emitter time
    s = np.exp(2j * np.pi * (pinger.f0 * u + 0.5 * k * u ** 2))
    s[u >= T] = 0
    return s


def coupling(pinger):
    """d(peak time)/d(eta mismatch) [s] of the LFM ridge (magnitude)."""
    return pinger.duration * pinger.f1 / (pinger.f1 - pinger.f0)


class MatchedFilter:
    """FFT matched filter (and cross-ambiguity) for segments up to `nseg`."""

    def __init__(self, pinger, fs, nseg, guard=1000.0, shapes=None):
        self.p, self.fs = pinger, fs
        self.shapes = shapes                         # (n_u, n_ch) complex pulse shapes or None
        self.nrep = int(np.ceil(pinger.duration * 1.02 * fs))
        self.nfft = 1 << int(np.ceil(np.log2(nseg + self.nrep)))
        f = np.fft.fftfreq(self.nfft, 1 / fs)
        self.band = (f > pinger.f0 - guard) & (f < pinger.f1 + guard)
        self.fc = 0.5 * (pinger.f0 + pinger.f1)
        self._S, self._X = {}, {}

    def spec(self, eta):
        """FFT of s(eta t) / sum|s|, so a matching pulse of amplitude A gives |c| = A."""
        key = round(eta, 9)
        S = self._S.get(key)
        if S is None:
            s = replica(self.p, self.fs, eta)
            S = np.fft.fft(s, self.nfft) * self.band / np.sum(np.abs(s))
            if len(self._S) > 256:
                self._S.clear()
            self._S[key] = S
        return S

    def xspec(self, x):
        return np.fft.fft(x, self.nfft, axis=0)

    def out(self, X, eta=1.0, n=None):
        """Analytic MF output for spectra X (nfft, n_ch); lag 0 = segment start."""
        c = np.fft.ifft(2 * X * np.conj(self.spec(eta))[:, None], axis=0)
        return c if n is None else c[:n]

    def xamb(self, eta, eta_r, half, ch=0):
        """Baseband cross-ambiguity of the received pulse w(eta t) s(eta t)
        (w: measured pulse shape of channel ch, 1 if none) against the replica
        s(eta_r t), on lags -half..half samples, as a CubicSpline."""
        key = (round(eta, 9), round(eta_r, 9), half, ch if self.shapes is not None else 0)
        sp = self._X.get(key)
        if sp is None:
            s = replica(self.p, self.fs, eta)
            if self.shapes is not None:
                w = self.shapes[:, ch]
                u = eta * np.arange(len(s)) / self.fs / self.p.duration
                ug = (np.arange(len(w)) + 0.5) / len(w)
                s = s * (np.interp(u, ug, w.real) + 1j * np.interp(u, ug, w.imag))
            Sx = np.fft.fft(s, self.nfft) * self.band
            r = np.fft.ifft(Sx * np.conj(self.spec(eta_r)))
            lag = np.arange(-half, half + 1)
            r = r[lag % self.nfft] * np.exp(-2j * np.pi * self.fc * lag / self.fs)
            sp = CubicSpline(lag.astype(float), r)
            if len(self._X) > 256:
                self._X.clear()
            self._X[key] = sp
        return sp


def _noise_floor(P, fs, n_ch, block=1.0):
    """sigma^2 per sample of P (sum of n_ch |c|^2), from per-block medians."""
    nb = max(int(block * fs), 1)
    m = max(len(P) // nb, 1)
    med = np.array([np.median(P[i * nb:(i + 1) * nb if i < m - 1 else len(P)]) for i in range(m)])
    sig = np.repeat(med, nb)[:len(P)]
    sig = np.concatenate([sig, np.full(len(P) - len(sig), med[-1])])
    return sig / stats.gamma.median(n_ch)


def _local_max(P):
    return np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])) + 1


def _parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return 0.0, y[i]
    a, b, c = y[i - 1], y[i], y[i + 1]
    d = a - 2 * b + c
    if d >= 0:
        return 0.0, b
    o = 0.5 * (a - c) / d
    return o, b - 0.25 * (a - c) * o


def coarse(rec, t0, t1, pinger, chans, thr_db=20.0, chunk=8.0, sep=0.3):
    """Coarse anchors (eta = 1) between buoy times t0..t1:
    list of (t_rx of the strongest peak, snr_db, noise var per channel)."""
    fs = rec.fs
    mf = MatchedFilter(pinger, fs, int(chunk * fs) + int(1.1 * pinger.duration * fs))
    ta, tb = t0 + rec.offset, t1 + rec.offset
    anchors = []
    gap = int(sep * fs)
    for c0 in np.arange(ta, tb, chunk):
        x, tx0 = rec.segment(c0, min(c0 + chunk, tb) + 1.05 * pinger.duration, clock="rx")
        x = x[:, chans]
        nval = min(int(round(chunk * fs)), len(x) - mf.nrep)
        if nval <= 0:
            continue
        Pch = np.abs(mf.out(mf.xspec(x), 1.0, nval)) ** 2
        sig_ch = np.median(Pch, axis=0) / np.log(2)
        P = Pch.sum(1)
        snr = P / _noise_floor(P, fs, len(chans))
        idx = _local_max(snr)
        idx = idx[snr[idx] > 10 ** (thr_db / 10)]
        idx = idx[np.argsort(-snr[idx])]
        taken = np.zeros(len(snr), bool)
        for i in idx:
            if not taken[i]:
                taken[max(i - gap, 0):i + gap] = True
                anchors.append((tx0 + i / fs, 10 * np.log10(snr[i]), sig_ch))
    anchors.sort(key=lambda a: a[0])
    out = []                                         # merge across chunk borders
    for a in anchors:
        if out and a[0] - out[-1][0] < sep:
            if a[1] > out[-1][1]:
                out[-1] = a
            continue
        out.append(a)
    return out


def fit_paths(y, tt, model, eta, max_paths=4, min_frac=0.01, kappa=25.0, sig2=0.0, taus0=None):
    """Variable-projection fit y(t) ~ sum a_i X(t - tau_i; eta) on sample times tt,
    model(eta) -> spline of the baseband cross-ambiguity against the replica.
    Paths are added greedily while each removes > max(kappa sig2, min_frac E)
    of the window energy, or exactly len(taus0) paths are fitted from taus0.
    Returns (taus, amps, rss), sorted by tau."""
    tc = 0.5 * (tt[0] + tt[-1])
    g_of = model(eta)

    def solve(th):
        g = g_of(tt[:, None] - (th + tc)[None, :])
        a = np.linalg.lstsq(g, y, rcond=None)[0]
        return a, y - g @ a

    def res(th):
        r = solve(th)[1]
        return np.concatenate([r.real, r.imag])

    lo, hi = tt[0] - tc - 0.004 * 96000, tt[-1] - tc + 0.004 * 96000
    fit = lambda th: optimize.least_squares(res, th, bounds=(lo, hi), diff_step=1e-6, x_scale=0.2).x
    e0 = np.sum(np.abs(y) ** 2)
    if taus0 is not None:
        th = fit(np.asarray(taus0, float) - tc)
    else:
        th, rss, r = np.zeros(0), e0, y
        for _ in range(max_paths):
            th_new = fit(np.r_[th, tt[np.argmax(np.abs(r))] - tc])
            a, r_new = solve(th_new)
            rss_new = np.sum(np.abs(r_new) ** 2)
            if len(th) and (rss - rss_new < max(kappa * sig2, min_frac * e0)
                            or np.min(np.diff(np.sort(th_new))) < 0.3):
                break
            th, rss, r = th_new, rss_new, r_new
    a, r = solve(th)
    o = np.argsort(th)
    return th[o] + tc, a[o], float(np.sum(np.abs(r) ** 2))


def profile_eta(y, tt, model, eta_r, taus_r, Cc_fs, span=1.5e-3, step=2.5e-4, **kw):
    """Doppler by profile likelihood: minimise over eta the residual of the
    multipath fit at that eta (path delays re-fitted, warm-started along the
    LFM ridge tau(eta) = tau(eta_r) + Cc (eta - eta_r)). A joint least-squares
    over (eta, taus) gets stuck on the ridge; this does not.
    Returns (eta, taus, amps)."""
    def rss(e, taus0):
        return fit_paths(y, tt, model, e, taus0=taus0 + Cc_fs * (e - eta_r))

    grid = eta_r + np.arange(-span, span + 1e-12, step)
    prof = [rss(e, taus_r)[2] for e in grid]
    j = int(np.argmin(prof))
    e1 = grid[j]
    taus1 = fit_paths(y, tt, model, e1, **kw)[0]        # path count at the best grid eta
    taus1_r = taus1 - Cc_fs * (e1 - eta_r)
    r = optimize.minimize_scalar(lambda e: rss(e, taus1_r)[2], method="bounded",
                                 bounds=(e1 - step, e1 + step), options={"xatol": 2e-7})
    taus, a, _ = rss(r.x, taus1_r)
    return r.x, taus, a


def _search_eta(X, n, mf, lo_s, hi_s, eta_max, step=1e-3):
    """Peak power vs eta (grid + Brent); used only without a reference eta."""
    def peak(e):
        P = (np.abs(mf.out(X, e, n)) ** 2).sum(1)
        i = lo_s + np.argmax(P[lo_s:hi_s])
        return _parabolic(P, i)[1]
    grid = 1.0 + np.arange(-eta_max, eta_max + 1e-12, step)
    g = np.array([peak(e) for e in grid])
    j = int(np.argmax(g))
    r = optimize.minimize_scalar(lambda e: -peak(e), method="bounded", options={"xatol": 1e-7},
                                 bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]))
    return r.x


def refine(rec, t_anchor, pinger, chans, sig_ch, mf, eta_ref=None, eta_max=0.013,
           pre_win=0.012, first_db=10.0, win=(-0.6e-3, 1.6e-3), max_paths=4,
           doppler=True, aperture=0.25e-3, c=C_SOUND):
    """TOA and Doppler around one coarse anchor (receiver clock).
    eta_ref: reference time scale (from the arrival track); None -> found
    from the pulse itself (peak power vs eta, then the joint fit), which is
    the single-ping mode. Returns a dict (see DET_DTYPE) plus t_ch."""
    fs = rec.fs
    Cc = coupling(pinger)
    span = Cc * eta_max + pre_win + 0.003
    x, tx0 = rec.segment(t_anchor - span, t_anchor + span + 1.05 * pinger.duration / (1 - eta_max),
                         clock="rx")
    x = x[:, chans]
    X = mf.xspec(x)
    n = len(x) - int(np.ceil(pinger.duration / (1 - eta_max) * fs))
    ia = int(round((t_anchor - tx0) * fs))
    # the anchor is the eta = 1 peak; with a time scale eta it moves by +Cc (eta - 1)
    search = int((Cc * eta_max + 0.002) * fs)
    single = eta_ref is None
    if single:
        eta_ref = _search_eta(X, n, mf, max(ia - search, 1), min(ia + search, n - 2), eta_max)
    ic = ia + int(round(Cc * (eta_ref - 1.0) * fs))
    lo_s, hi_s = max(ic - int(0.002 * fs), 1), min(ic + int(0.002 * fs), n - 2)

    C = mf.out(X, eta_ref, n)
    P = (np.abs(C) ** 2).sum(1)
    iS = lo_s + int(np.argmax(P[lo_s:hi_s]))
    sig_tot = float(np.sum(sig_ch))
    lm = _local_max(P)
    lm = lm[(lm >= iS - int(pre_win * fs)) & (lm <= iS) & (P[lm] >= P[iS] * 10 ** (-first_db / 10))
            & (P[lm] >= 10 ** 1.5 * sig_tot)]
    iF = int(lm[0]) if len(lm) else iS

    half = int(0.004 * fs)
    w0, w1 = max(iF + int(win[0] * fs), 0), min(iF + int(win[1] * fs) + 1, n)
    tt = np.arange(w0, w1).astype(float)
    ph = np.exp(-2j * np.pi * mf.fc * tt / fs)
    toas, amps, npaths, echo, edt, etas = [], [], [], [], [], []
    for j in range(len(chans)):
        y = C[w0:w1, j] * ph
        model = lambda e, j=j: mf.xamb(e, eta_ref, half, j)
        taus, a, _ = fit_paths(y, tt, model, eta_ref, max_paths, sig2=sig_ch[j])
        if doppler or single:
            eta_f, taus_f, a_f = profile_eta(y, tt, model, eta_ref, taus, Cc * fs,
                                             max_paths=max_paths, sig2=sig_ch[j])
            etas.append(eta_f)
            if single:
                taus, a = taus_f, a_f
        aa = np.abs(a)
        ok = (aa >= aa.max() * 10 ** (-first_db / 20)) & (taus >= iF - 0.35e-3 * fs)
        i1 = int(np.flatnonzero(ok)[0]) if ok.any() else int(np.argmax(aa))
        toas.append(taus[i1])
        amps.append(aa[i1])
        npaths.append(len(taus))
        later = (taus > taus[i1] + 0.3)
        if later.any():
            m = np.argmax(np.where(later, aa, -1))
            echo.append(20 * np.log10(aa[m] / aa[i1]))
            edt.append((taus[m] - taus[i1]) / fs)
        else:
            echo.append(np.nan)
            edt.append(np.nan)
    toas = np.array(toas)
    if len(toas) > 1:                                # drop channels off the array aperture
        toas = np.where(np.abs(toas - np.median(toas)) > aperture * fs, np.nan, toas)
    amps = np.array(amps)
    snr_first = np.sum(amps ** 2) / sig_tot
    beta = (pinger.f1 - pinger.f0) / np.sqrt(12)
    sig_tau = 1.0 / (2 * np.pi * beta * np.sqrt(max(snr_first, 1.0)))
    # CRLB-style Doppler sigma: the scale information is ~ (pi B T / sqrt(45))^2 SNR / f1^2 ... use
    # the quadratic-phase curvature 2 pi (B T / 4) per unit eta, divided by sqrt(SNR)
    sig_eta = 1.0 / (2 * np.pi * (pinger.f1 - pinger.f0) * pinger.duration / np.sqrt(180) * np.sqrt(max(snr_first, 1.0)))
    eta_d = float(np.mean(etas)) if etas else np.nan
    i_e = int(np.nanargmax(echo)) if np.isfinite(echo).any() else 0
    return dict(t_rx=tx0 + np.nanmean(toas) / fs, t_ch=tx0 + toas / fs,
                snr_db=10 * np.log10(snr_first), snr_peak_db=10 * np.log10(P[iS] / sig_tot),
                eta=eta_d if single else eta_ref, rr_dop=c * (1 - eta_d), amp=float(np.mean(amps)),
                n_paths=int(np.max(npaths)), dt_strong=(iS - iF) / fs, echo_db=echo[i_e], echo_dt=edt[i_e],
                sigma_toa=np.hypot(sig_tau, Cc * sig_eta) if single else sig_tau, sigma_rr=c * sig_eta)


def eta_track(t, k, Tp, half=6, c=C_SOUND):
    """Time scale per detection from the arrival sequence: robust local
    quadratic fit of t vs k over +/- half pings; eta = Tp / (dt/dk).
    Returns (eta, ok)."""
    eta, ok = np.ones(len(t)), np.zeros(len(t), bool)
    for i in range(len(t)):
        m = np.abs(k - k[i]) <= half
        if m.sum() < 4:
            continue
        kk, tt = (k[m] - k[i]).astype(float), t[m] - t[i]
        w = np.ones(len(kk))
        for _ in range(4):                           # bisquare reweighting
            A = np.c_[np.ones_like(kk), kk, kk ** 2] if m.sum() >= 5 else np.c_[np.ones_like(kk), kk]
            p = np.linalg.lstsq(A * w[:, None], tt * w, rcond=None)[0]
            r = tt - A @ p
            s = max(1.4826 * np.median(np.abs(r)), 20e-6)
            u = r / (4.685 * s)
            w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() >= 3 and abs(p[1] / Tp - 1) < 0.02:
            eta[i], ok[i] = Tp / p[1], True
    return eta, ok


def _refine_job(args):
    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN channel rows -> NaN, flagged
    name, anchors, etas, shapes, kw = args
    cfg = config.load()
    rec = recordings.load(cfg.receiver(name))
    mf = MatchedFilter(cfg.pinger, rec.fs, int(0.25 * rec.fs), shapes=shapes)
    chans = list(range(rec.n_ch))
    out = []
    for (ta, snr, sig_ch), e in zip(anchors, etas):
        try:
            out.append(refine(rec, ta, cfg.pinger, chans, sig_ch, mf, eta_ref=e, **kw))
        except Exception as ex:                      # keep going; flagged by NaN
            print(f"{name} refine failed at {ta:.3f}: {ex}")
            out.append(None)
    return out


def refine_all(name, anchors, etas, shapes=None, nproc=12, **kw):
    """Parallel refine over anchors (etas: reference eta per anchor or None)."""
    parts = np.array_split(np.arange(len(anchors)), max(1, min(nproc, len(anchors) // 8 + 1)))
    jobs = [(name, [anchors[i] for i in p], [etas[i] for i in p], shapes, kw) for p in parts if len(p)]
    with Pool(len(jobs)) as pool:
        res = pool.map(_refine_job, jobs)
    return [r for part in res for r in part]


def _table(rec, anchors, res, k, rr_track):
    n_ch = rec.n_ch
    det = np.zeros(len(anchors), DET_DTYPE)
    tch = np.full((len(anchors), n_ch), np.nan)
    for i, ((ta, snr, sig), r) in enumerate(zip(anchors, res)):
        det["t_coarse"][i], det["k"][i], det["rr_track"][i] = ta, k[i], rr_track[i]
        if r is None:
            det["t_rx"][i] = np.nan
            continue
        for f in det.dtype.names:
            if f in r:
                det[f][i] = r[f]
        tch[i] = r["t_ch"]
    det["t_buoy"] = det["t_rx"] - rec.offset
    return det, tch


def detect_receiver(name, t0, t1, thr_db=20.0, verbose=True, **kw):
    """Full pipeline for one receiver: coarse anchors -> ping indices ->
    eta track -> refine with the ideal LFM -> measured pulse shape ->
    refine with the shape (twice, re-estimating shape and eta track; the
    last pass also estimates the per-ping Doppler). Returns (det, t_ch, info)."""
    import associate
    cfg = config.load()
    rec = recordings.load(cfg.receiver(name))
    p = cfg.pinger
    tic = time.time()
    anchors = coarse(rec, t0, t1, p, list(range(rec.n_ch)), thr_db)
    t_c = np.array([a[0] for a in anchors])
    k, keep = associate.index_pings(t_c, p.interval)
    anchors = [a for a, m in zip(anchors, keep) if m]
    k, t = k[keep], t_c[keep]
    info = dict(n_anchors=len(t_c), n_rejected=int((~keep).sum()), t_coarse_s=time.time() - tic)
    if verbose:
        print(f"{name}: {len(t_c)} anchors, {info['n_rejected']} off-sequence, {info['t_coarse_s']:.0f} s")
    eta, ok = eta_track(t, k, p.interval)
    shapes = None
    for it in range(3):
        e_ref = [float(e) if o else 1.0 for e, o in zip(eta, ok)]
        res = refine_all(name, anchors, e_ref, shapes, doppler=(it == 2), **kw)
        t_new = np.array([r["t_rx"] if r is not None else np.nan for r in res])
        good = np.isfinite(t_new)
        eta_n, ok_n = np.ones(len(t)), np.zeros(len(t), bool)
        eta_n[good], ok_n[good] = eta_track(t_new[good], k[good], p.interval)
        d = np.nanmedian(np.abs(C_SOUND * (eta_n - eta))[ok_n & ok])
        eta, ok = eta_n, ok_n
        det, tch = _table(rec, anchors, res, k, np.where(ok, C_SOUND * (1 - eta), np.nan))
        if verbose:
            print(f"{name}: pass {it}: median |d rr_track| {d:.3f} m/s, {time.time() - tic:.0f} s")
        if it < 2:
            shapes = pulse_shape(rec, det, p)
    if rec.n_ch > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            det["t_rx"] = array_toa(tch, k)
        det["t_buoy"] = det["t_rx"] - rec.offset
    info["runtime_s"] = time.time() - tic
    info["shapes"] = shapes
    return det, tch, info


def array_toa(tch, k, half=10):
    """Array-centre TOA from per-channel TOAs with missing channels: each
    channel's offset from the channel mean (it varies slowly with bearing)
    is smoothed over +/- half pings (median of complete rows) and removed
    before averaging, so a missing channel does not move the mean."""
    full = np.all(np.isfinite(tch), axis=1)
    dev = tch - tch.mean(1, keepdims=True)
    t = np.full(len(tch), np.nan)
    for i in range(len(tch)):
        m = full & (np.abs(k - k[i]) <= half)
        if m.sum() < 2:
            m = full
        off = np.median(dev[m], axis=0) if m.any() else np.zeros(tch.shape[1])
        t[i] = np.nanmean(tch[i] - off)
    return t


def detect(rec, t0, t1, pinger, ch=None, thr_db=20.0, **kw):
    """Single-pass detection between buoy times t0..t1, Doppler from each
    pulse (no arrival track). ch: channel index/list or None (all).
    Returns (structured DET_DTYPE array, per-channel TOAs (rx clock))."""
    chans = list(range(rec.n_ch)) if ch is None else list(np.atleast_1d(ch))
    anchors = coarse(rec, t0, t1, pinger, chans, thr_db)
    mf = MatchedFilter(pinger, rec.fs, int(0.25 * rec.fs))
    det = np.zeros(len(anchors), DET_DTYPE)
    tch = np.full((len(anchors), len(chans)), np.nan)
    for i, (ta, snr, sig_ch) in enumerate(anchors):
        r = refine(rec, ta, pinger, chans, sig_ch, mf, **kw)
        for f in det.dtype.names:
            if f in r:
                det[f][i] = r[f]
        det["t_coarse"][i] = ta
        tch[i] = r["t_ch"]
    det["t_buoy"] = det["t_rx"] - rec.offset
    det["rr_track"] = np.nan
    return det, tch


def main(names=("Rx902", "Rx01", "USBL")):
    cfg = config.load()
    out = config.ROOT / "out"
    out.mkdir(exist_ok=True)
    runs = [r for r in cfg.runs if r.pinger]
    # from the start of the files: the pinger is on before run 1 starts (first ping 52.4 s)
    t0, t1 = 0.0, max(r.end for r in runs)
    for name in names:
        det, tch, info = detect_receiver(name, t0, t1)
        rec = recordings.load(cfg.receiver(name))
        np.savez(out / f"detections_{name}.npz", det=det, t_ch=tch, offset=rec.offset, fs=rec.fs,
                 t0=t0, t1=t1, **info)
        print(f"{name}: {len(det)} detections, median SNR {np.nanmedian(det['snr_db']):.1f} dB, "
              f"{info['runtime_s']:.0f} s")




def pulse_shape(rec, det, pinger, n_out=200, min_snr=30.0, max_pings=400, smooth=0.002):
    """Complex pulse shape w(u), u in [0, 1] over the pulse, per channel:
    mean over pings of the dechirped first arrival x_a(t) conj(s_eta(t - toa)),
    each ping normalised to unit mean and with its linear phase (sub-sample
    delay) removed. Other paths beat at k * delay and average out over pings
    whose multipath delays differ. Captures the transmit envelope and the
    receiver response (incl. dispersion) that the ideal LFM model misses.
    Returns (n_out, n_ch) complex."""
    fs = rec.fs
    d = det[np.isfinite(det["t_rx"]) & (det["snr_db"] >= min_snr)]
    if len(d) > max_pings:
        d = d[np.linspace(0, len(d) - 1, max_pings).astype(int)]
    T = pinger.duration
    nb = int(smooth * fs)
    u_out = (np.arange(n_out) + 0.5) / n_out
    acc = np.zeros((n_out, rec.n_ch), complex)
    for r in d:
        x, tx0 = rec.segment(r["t_rx"] - 0.01, r["t_rx"] + T + 0.01, clock="rx")
        X = np.fft.fft(x, axis=0)
        f = np.fft.fftfreq(len(x), 1 / fs)
        X[(f < pinger.f0 - 1000) | (f > pinger.f1 + 1000)] = 0
        xa = 2 * np.fft.ifft(X, axis=0)
        t = tx0 + np.arange(len(x)) / fs - r["t_rx"]
        u = r["eta"] * t / T
        m = (u >= 0) & (u < 1)
        k = (pinger.f1 - pinger.f0) / T
        s = np.exp(2j * np.pi * (pinger.f0 * T * u[m] + 0.5 * k * (T * u[m]) ** 2))
        dd = xa[m] * np.conj(s)[:, None]
        box = np.ones(nb)
        dd = np.apply_along_axis(lambda v: np.convolve(v, box, "same"), 0, dd) \
            / np.convolve(np.ones(len(dd)), box, "same")[:, None]     # no edge droop
        for j in range(rec.n_ch):
            v = dd[:, j]
            ph = np.unwrap(np.angle(v))
            core = (u[m] > 0.05) & (u[m] < 0.95)
            p = np.polyfit(u[m][core], ph[core], 1, w=np.abs(v[core]))
            v = v * np.exp(-1j * np.polyval(p, u[m]))
            v = v / np.mean(np.abs(v[core]))
            acc[:, j] += np.interp(u_out, u[m], v.real) + 1j * np.interp(u_out, u[m], v.imag)
    w = acc / max(len(d), 1)
    for j in range(rec.n_ch):                        # drop the remaining linear phase, unit mean
        core = (u_out > 0.05) & (u_out < 0.95)
        p = np.polyfit(u_out[core], np.unwrap(np.angle(w[core, j])), 1)
        w[:, j] *= np.exp(-1j * np.polyval(p, u_out)) / np.mean(np.abs(w[core, j]))
    return w


if __name__ == "__main__":
    main(*([sys.argv[1:]] if len(sys.argv) > 1 else []))
