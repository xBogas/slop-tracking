"""
USBL bearings from the boat's radiated noise (phase 2, run 3).

Method
------
1. Windows: `win` s (default 1 s) on a `step` s grid of buoy time; the USBL
   samples are read on the USBL clock, usbl_time = buoy_time + b +
   d_S (buoy_time - t_ref), with b, d_S and t_ref from the phase 1 graph
   (out/phase1/state.npz, posterior.py). Only inter-channel delays matter for a bearing,
   so the clock only sets the time stamps.
2. Band-pass (Butterworth, zero phase; default 2-28 kHz, the pinger band
   29.5-40.5 kHz notched when the band reaches it).
3. Click blanking: snapping-shrimp clicks are short, loud and come from
   their own (often near-field) directions. The envelope is taken in 0.5 ms
   blocks; a block more than `click_k` times its 20 ms running median on any
   channel is zeroed on all channels (+-1 block). Continuous sources (boat,
   pings) follow their running median and are kept.
4. Cross-spectra: Welch averages (nperseg 1024, 94 Hz bins, ~190 segments
   per 1 s window) of all pairs in `usbl.PAIRS`.
5. Common-mode crosstalk: all channels share a weak broadband electrical
   component at exactly zero lag (coherency ~0.07 when the boat is quiet,
   stationary over the whole file, present in every band up to 47 kHz). It
   would pull weak-window delays to 0 (broadside). Its cross-spectrum is real
   and stationary, so it is estimated as the median over all windows of
   Re S_ij(f) (the boat's phase rotates over the run and does not survive
   the median), smoothed in frequency, and subtracted.
6. GCC: SCOT weighting S_ij / sqrt(P_ii P_jj) (coherency), evaluated
   directly on a 0.5 us lag grid. SCOT rather than PHAT: with averaged
   spectra it weights each bin by its coherence, so incoherent bins (most
   of the band when the boat is far) add little noise; PHAT gives them full
   weight (compared in the note). The SCOT peak value is the band-averaged
   coherency at the delay (0..1), a direct quality measure.
7. Delays: steered response (sum of the pair GCCs along the plane-wave
   delays of the calibrated geometry, front half-plane only: the wall and
   the shore are behind the array) gives a coarse azimuth; each pair's delay
   is then the GCC local maximum reached uphill from that prediction, with
   a parabolic sub-sample refinement. This keeps all pairs on the same
   lobe (the gaps are 3-5 wavelengths at 20 kHz) and away from the wall-echo
   cross terms (direct x echo peaks at tau + 0.2-1.6 ms, mostly outside the
   +-d/c physical range).
8. Bearing: `usbl.bearing_ls` on the pair delays with the calibrated element
   positions and the sound speed they were calibrated with (1520 m/s: only
   pos / c is determined, so the bearing does not depend on the true c) -> az_array, plane-wave rms, mirror solution, formal sigma
   (as usbl.bearings). The truth-free scatter about a local quadratic in
   time (`usbl.bearing_scatter`) is added in quadrature as the sigma floor.
9. Mask: coherency peak, plane-wave rms, steered-response contrast and
   consistency with the neighbouring windows.

Convention (same as out/usbl/bearings_phase1.npz): az_array is the
azimuth in the array frame of usbl.py (x from ch0 to ch3, CCW). The graph
measurement is handedness * az_array = atan2(p - usbl) - alpha, i.e.
az_enu = alpha + handedness * az_array (handedness = -1).

Run: uv run python processing/noise_bearing.py [--t0 796] [--t1 1238] [--band 2000 20000]
     [--no-check] -> out/phase2/bearings_noise.npz, out/phase2/bearings_noise.png
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy import signal, ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import posterior
import recordings
from usbl import PAIRS, bearing_ls, plane_delays, bearing_scatter

OUT = config.ROOT / "out" / "phase2"
P1 = config.ROOT / "out" / "usbl" / "bearings_phase1.npz"
PINGER_NOTCH = (29.5e3, 40.5e3)


# --------------------------------------------------------------------------- clock
def usbl_time(t_buoy, b, d_S, t_ref):
    return t_buoy + b + d_S * (t_buoy - t_ref)


# --------------------------------------------------------------------------- per window
def bandpass(fs, band, notch=PINGER_NOTCH, order=6):
    """Zero-phase band-pass SOS (list); the notch is added when it falls inside the band."""
    sos = [signal.butter(order, band, "bandpass", fs=fs, output="sos")]
    if notch is not None and band[0] < notch[1] and band[1] > notch[0]:
        sos.append(signal.butter(order, notch, "bandstop", fs=fs, output="sos"))
    return sos


def blank_clicks(x, fs, k=8.0, block=0.5e-3, med_win=20e-3, guard=1):
    """Zero short loud transients (clicks) on all channels. x (n, nch)
    band-passed. A block (`block` s) whose envelope maximum exceeds k times
    its running median (`med_win` s) on any channel is blanked, with `guard`
    blocks on each side. Returns blanked x and the blanked fraction."""
    if k is None:
        return x, 0.0
    nb = int(round(block * fs))
    n = (len(x) // nb) * nb
    env = np.abs(signal.hilbert(x[:n], axis=0)).reshape(-1, nb, x.shape[1]).max(1)
    med = ndimage.median_filter(env, size=(int(round(med_win / block)) | 1, 1), mode="nearest")
    bad = (env > k * med).any(1)
    bad = ndimage.binary_dilation(bad, iterations=guard) if guard else bad
    if bad.any():
        x = x.copy()
        x[:n][np.repeat(bad, nb)] = 0.0
    return x, bad.mean()


def cross_spectra(x, fs, nperseg=1024, fsel=None, pairs=PAIRS):
    """Welch auto/cross spectra of x (n, nch): f (nf,), S (npair, nf) with
    S_ij = E[conj(X_i) X_j] (phase -2 pi f tau_ij for tau_ij = t_j - t_i),
    P (nch, nf). fsel (bool over rfft bins) keeps a sub-band."""
    seg = signal.windows.hann(nperseg, sym=False)
    hop = nperseg // 2
    idx = np.arange(0, len(x) - nperseg + 1, hop)[:, None] + np.arange(nperseg)
    X = np.fft.rfft(x[idx] * seg[None, :, None], axis=1)          # (nseg, nf, nch)
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    if fsel is not None:
        X, f = X[:, fsel], f[fsel]
    i, j = np.array(pairs).T
    S = (np.conj(X[:, :, i]) * X[:, :, j]).mean(0).T
    P = (np.abs(X) ** 2).mean(0).T
    return f, S, P


def band_bins(fs, band, notch=PINGER_NOTCH, nperseg=1024):
    """rfft frequencies and the bins kept (inside band, outside notch)."""
    f = np.fft.rfftfreq(nperseg, 1 / fs)
    fsel = (f >= band[0]) & (f <= band[1])
    if notch is not None:
        fsel &= ~((f > notch[0]) & (f < notch[1]))
    return f, fsel


def _spectra(x, fs, band, notch, click_k, nperseg):
    rms = np.sqrt((x ** 2).mean(0))
    x, frac = blank_clicks(x, fs, click_k)
    _, S, P = cross_spectra(x, fs, nperseg, band_bins(fs, band, notch, nperseg)[1])
    return S.astype(np.complex64), P.astype(np.float32), frac, rms


def block_spectra(x, fs, win, band=(2e3, 28e3), notch=PINGER_NOTCH, click_k=8.0, nperseg=1024, pad=0.05):
    """Raw block x (n, nch) covering the window plus `pad` s each side ->
    band-pass, trim, click blanking, Welch spectra: S, P, blanked fraction, band rms."""
    for s in bandpass(fs, band, notch):
        x = signal.sosfiltfilt(s, x, axis=0)
    i0 = int(round(pad * fs))
    return _spectra(x[i0: i0 + int(round(win * fs))], fs, band, notch, click_k, nperseg)


def _window(args):
    """Worker: spectra of one window centred at USBL time tc (rx clock)."""
    rx, tc, win, band, notch, click_k, nperseg = args
    rec = _REC.get(str(rx.file))
    if rec is None:
        rec = _REC[str(rx.file)] = recordings.load(rx)
    pad = 0.05
    x, _ = rec.segment(tc - win / 2 - pad, tc + win / 2 + pad, clock="rx")
    return block_spectra(x, rec.fs, win, band, notch, click_k, nperseg, pad)


_REC = {}


def window_spectra(rx, t_usbl, win=1.0, band=(2e3, 28e3), notch=PINGER_NOTCH, click_k=8.0, nperseg=1024, workers=None):
    """Spectra of all windows centred at t_usbl (USBL clock). Returns f, S (K,
    npair, nf), P (K, nch, nf), blanked fraction (K,), band rms (K, nch)."""
    rec = recordings.load(rx)
    f, fsel = band_bins(rec.fs, band, notch, nperseg)
    jobs = [(rx, tc, win, band, notch, click_k, nperseg) for tc in t_usbl]
    workers = workers or min(16, os.cpu_count() or 1)
    if workers > 1 and len(jobs) > 8:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            res = pool.map(_window, jobs, chunksize=4)
    else:
        res = [_window(j) for j in jobs]
    S, P, frac, rms = (np.array(v) for v in zip(*res))
    return f[fsel], S, P, frac, rms


# --------------------------------------------------------------------------- crosstalk, GCC, delays
def crosstalk(S, f, smooth_hz=1000.0):
    """Stationary zero-lag common-mode cross-spectrum: median over windows of
    Re S_ij(f), running-median smoothed over smooth_hz. S (K, npair, nf) ->
    (npair, nf) real."""
    B = np.median(S.real, 0)
    n = max(int(round(smooth_hz / np.median(np.diff(f)))) | 1, 1)
    return ndimage.median_filter(B, size=(1, n), mode="nearest")


def gcc(S, P, f, lags, weight="scot", pairs=PAIRS, chunk=64):
    """Weighted GCC R (K, npair, nlag) on the lag grid (direct evaluation).
    weight 'scot': S / sqrt(P_i P_j) (coherency; R(tau) = mean over bins of
    Re(gamma e^{i 2 pi f tau}) is in [-1, 1], peak at tau_ij); 'phat': S / |S|."""
    i, j = np.array(pairs).T
    if weight == "scot":
        G = S / np.sqrt(P[:, i] * P[:, j] + 1e-300)
    elif weight == "phat":
        G = S / (np.abs(S) + 1e-300)
    else:
        raise ValueError(weight)
    E = np.exp(2j * np.pi * f[:, None] * lags[None]) / len(f)      # (nf, nlag)
    R = np.empty(G.shape[:2] + (len(lags),))
    for k in range(0, len(G), chunk):
        R[k:k + chunk] = (G[k:k + chunk] @ E).real
    return R


def srp_azimuth(R, lags, pos, c, pairs=PAIRS, n_az=720):
    """Steered response over the front half-plane (az in (0, pi), array
    frame): sum over pairs of R at the plane-wave delays. Returns az (K,),
    the response at it (K,), its contrast (peak / median over az) and the
    full response (K, n_az) with the grid. The azimuth is refined by a
    parabola through the grid maximum."""
    g = np.linspace(0, np.pi, n_az + 1)[:-1] + np.pi / n_az / 2
    T = plane_delays(pos, g, c, pairs)                                # (n_az, npair)
    dl = lags[1] - lags[0]
    q = np.clip((T - lags[0]) / dl, 0, len(lags) - 1.001)
    q0 = q.astype(int)
    a = q - q0
    resp = np.zeros((R.shape[0], n_az))
    for p in range(len(pairs)):
        r = R[:, p]
        resp += (1 - a[:, p]) * r[:, q0[:, p]] + a[:, p] * r[:, q0[:, p] + 1]
    resp /= len(pairs)
    k = np.clip(np.argmax(resp, 1), 1, n_az - 2)
    K = np.arange(len(k))
    y0, y1, y2 = resp[K, k - 1], resp[K, k], resp[K, k + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    floor = np.median(resp, 1)
    contrast = (y1 - floor) / (np.median(np.abs(resp - floor[:, None]), 1) + 1e-12)
    return g[k] + d * (g[1] - g[0]), y1, contrast, resp, g


def refine_delays(R, lags, tau0):
    """Per pair: hill-climb from the predicted delay tau0 (K, npair) to the
    local GCC maximum, then parabolic sub-sample refinement. Returns tau
    (K, npair) and the peak values (K, npair)."""
    K, npair, nl = R.shape
    dl = lags[1] - lags[0]
    m = np.clip(np.rint((tau0 - lags[0]) / dl).astype(int), 1, nl - 2)
    kk, pp = np.meshgrid(np.arange(K), np.arange(npair), indexing="ij")
    for _ in range(nl):
        up = R[kk, pp, np.minimum(m + 1, nl - 1)] > R[kk, pp, m]
        dn = R[kk, pp, np.maximum(m - 1, 0)] > R[kk, pp, m]
        step = np.where(up, 1, np.where(dn, -1, 0))
        if not step.any():
            break
        m = np.clip(m + step, 1, nl - 2)
    y0, y1, y2 = R[kk, pp, m - 1], R[kk, pp, m], R[kk, pp, m + 1]
    den = y0 - 2 * y1 + y2
    d = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    return lags[m] + d * dl, y1 - 0.25 * (y0 - y2) * d


def peak_snr(R, lags, lo=0.5e-3):
    """GCC peak contrast per pair: max over the physical lag range / rms of R
    at |lag| > lo (off-peak, incl. wall-echo cross terms: conservative)."""
    off = np.abs(lags) > lo
    return R[..., ~off].max(-1) / np.sqrt((R[..., off] ** 2).mean(-1))


# --------------------------------------------------------------------------- whole pipeline
def delays_and_bearings(S, P, f, pos, c, weight="scot", remove_crosstalk=True, pairs=PAIRS, lag_max=0.8e-3, dlag=0.5e-6):
    """Spectra -> GCC -> delays -> bearings. Returns a dict of per-window arrays."""
    if remove_crosstalk:
        S = S - crosstalk(S, f)[None]
    lags = np.arange(-lag_max, lag_max + dlag / 2, dlag)
    R = gcc(S, P, f, lags, weight, pairs)
    az0, srp, contrast, resp, grid = srp_azimuth(R, lags, pos, c, pairs)
    tau, pk = refine_delays(R, lags, plane_delays(pos, az0, c, pairs))
    az, rms, azm, rmsm, sens = bearing_ls(tau, pos, c, pairs=pairs)
    npair = len(pairs)
    st = np.maximum(rms * np.sqrt(npair / (npair - 1)), 0.5e-6)
    return dict(az=az, az_srp=az0, rms=rms, az_mirror=azm, rms_mirror=rmsm, sigma_formal=st / sens,
                tau=tau, peak=pk, coh=np.median(pk, 1), srp=srp, contrast=contrast,
                snr=np.median(peak_snr(R, lags), 1), resp=resp, az_grid=grid)


def neighbour_resid(t, az, ok, half=5.0):
    """Wrapped az minus the circular median of the other ok windows within
    +-half s (NaN where fewer than 3 neighbours)."""
    out = np.full(len(t), np.nan)
    idx = np.where(ok)[0]
    for k in range(len(t)):
        m = idx[(np.abs(t[idx] - t[k]) <= half) & (idx != k)]
        if len(m) >= 3:
            ref = np.angle(np.exp(1j * az[m]).sum())
            ref += np.median(np.angle(np.exp(1j * (az[m] - ref))))
            out[k] = np.angle(np.exp(1j * (az[k] - ref)))
    return out


def quality_mask(t, B, min_contrast=4.0, max_rms=15e-6, nb_half=5.0, nb_deg=5.0):
    """Good windows: steered-response contrast, plane-wave rms, front side of
    the array (sin az > 0: the wall and the shore are behind it; a residual
    zero-lag crosstalk peak fits best at az = -90 deg), then consistency with
    the neighbours (+-nb_half s, >= 3 good ones), twice. Returns mask and the
    neighbour residual [rad]."""
    ok0 = (B["contrast"] >= min_contrast) & (B["rms"] <= max_rms) & (np.sin(B["az"]) > 0)
    ok = ok0.copy()
    for _ in range(2):
        r = neighbour_resid(t, B["az"], ok, nb_half)
        ok = ok0 & np.isfinite(r) & (np.abs(np.nan_to_num(r, nan=np.inf)) <= np.radians(nb_deg))
    return ok, r


def run(rx, t_buoy, clock, pos, c, win=1.0, band=(2e3, 28e3), notch=PINGER_NOTCH, click_k=8.0,
        weight="scot", remove_crosstalk=True, nperseg=1024, workers=None):
    tu = usbl_time(t_buoy, clock["b"], clock["d_S"], clock["t_ref"])
    f, S, P, frac, rms = window_spectra(rx, tu, win, band, notch, click_k, nperseg, workers)
    B = delays_and_bearings(S, P, f, pos, c, weight, remove_crosstalk)
    B.update(t=t_buoy, t_usbl=tu, blanked=frac, band_rms=rms)
    return B, (f, S, P)


def calibrate_sigma(B, ok, win=7.5):
    """sigma = sqrt(formal^2 + s_emp^2), s_emp the scatter about a local
    quadratic in time (truth-free). Returns sigma (K,), s_emp, lag-1 corr, residuals."""
    res, s_emp, rho = bearing_scatter(B["t"], B["az"], ok, win=win)
    return np.sqrt(B["sigma_formal"] ** 2 + s_emp ** 2), s_emp, rho, res


# --------------------------------------------------------------------------- method check (pinger band)
def method_check(rx, clock, pos, c, p1, win=1.0, t0=52.5, t1=795.5, **kw):
    """Run the windowed method on the pinger band (30-40 kHz, no notch) over
    runs 1-2 and compare with the per-ping phase 1 az_array of the ping
    fully inside each window. Returns the comparison dict."""
    t = np.arange(t0, t1 + 1e-9, 1.0)
    B, _ = run(rx, t, clock, pos, c, win=win, band=(30e3, 40e3), notch=None, **kw)
    tu, pt, pm = B["t_usbl"], p1["t_usbl"], p1["mask"]
    dur = 0.1
    az1 = np.full(len(t), np.nan)
    for k in range(len(t)):
        m = pm & (pt >= tu[k] - win / 2 + 0.005) & (pt + dur <= tu[k] + win / 2 - 0.005)
        if m.any():
            az1[k] = p1["az_array"][m][0]
    d = np.angle(np.exp(1j * (B["az"] - az1)))
    ok = np.isfinite(d)
    return dict(t=t, az=B["az"], az_p1=az1, diff=d, n=int(ok.sum()), med_abs=np.median(np.abs(d[ok])),
                bias=np.median(d[ok]), robust_sig=1.4826 * np.median(np.abs(d[ok] - np.median(d[ok]))),
                frac_gt2=np.mean(np.abs(d[ok]) > np.radians(2)), coh=B["coh"], rms=B["rms"])


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--t0", type=float, default=796.0)
    ap.add_argument("--t1", type=float, default=1238.0)
    ap.add_argument("--win", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--band", type=float, nargs=2, default=[2e3, 28e3])
    ap.add_argument("--weight", default="scot", choices=["scot", "phat"])
    ap.add_argument("--click-k", type=float, default=8.0)
    ap.add_argument("--no-crosstalk", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--compare", action="store_true", help="also print band / weighting variants")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = config.load()
    rx = cfg.receiver("USBL")
    p1 = dict(np.load(P1))
    pos, hand, alpha_p1 = p1["pos"], int(p1["handedness"]), float(p1["alpha"])
    clock = posterior.phase1(cfg=cfg)
    # The self-calibration fixes pos / c (pos was fitted with c = sound_speed), so the
    # bearing does not depend on c as long as the same c is used with pos; another c
    # with the same pos would bend the bearings by up to 0.4 deg (cos az scales with c).
    c = float(p1["sound_speed"])
    alpha_g = clock["alpha"]
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.arange(a.t0, a.t1 + 1e-9, a.step)
    kw = dict(win=a.win, click_k=a.click_k, weight=a.weight, remove_crosstalk=not a.no_crosstalk)
    print(f"USBL clock b = {clock['b']:.6f} s, d_S = {clock['d_S'] * 1e6:.2f} ppm; c = {c}; "
          f"{len(t)} windows of {a.win} s, {a.t0:.0f}-{a.t1:.0f} s buoy time, band {a.band[0]/1e3:.0f}-{a.band[1]/1e3:.0f} kHz")

    B, _ = run(rx, t, clock, pos, c, band=tuple(a.band), **kw)
    ok, nres = quality_mask(t, B)
    sigma, s_emp, rho, res = calibrate_sigma(B, ok)
    print(f"good windows {ok.sum()}/{len(t)}; blanked fraction median {np.median(B['blanked']):.3f}"
          f" (p90 {np.percentile(B['blanked'], 90):.3f})")
    print(f"  coherency peak median {np.median(B['coh']):.3f}; plane-wave rms median (good) {np.median(B['rms'][ok]) * 1e6:.2f} us;"
          f" mirror rms median {np.median(B['rms_mirror'][ok]) * 1e6:.2f} us")
    print(f"  scatter about a local quadratic (+-7.5 s): {np.degrees(s_emp):.2f} deg, lag-1 corr {rho:.2f};"
          f" formal median {np.degrees(np.median(B['sigma_formal'][ok])):.2f} deg; sigma median {np.degrees(np.median(sigma[ok])):.2f} deg")
    spans = _spans(t[ok], 3 * a.step)
    print("  good spans [buoy s]:", ", ".join(f"{s:.0f}-{e:.0f} ({n})" for s, e, n in spans if n >= 5))

    if a.compare:
        for band in [(1e3, 6e3), (2e3, 10e3), (2e3, 20e3), (5e3, 20e3), (2e3, 28e3), (5e3, 28e3)]:
            for wgt, xt in [("scot", True), ("phat", True), ("scot", False)]:
                Bv, _ = run(rx, t, clock, pos, c, band=band, win=a.win, click_k=a.click_k, weight=wgt, remove_crosstalk=xt)
                okv, _ = quality_mask(t, Bv)
                _, sv, _, _ = calibrate_sigma(Bv, okv) if okv.sum() > 20 else (None, np.nan, None, None)
                dx = np.angle(np.exp(1j * (Bv["az"] - B["az"])))[okv & ok]
                print(f"  variant band {band[0]/1e3:.0f}-{band[1]/1e3:.0f} kHz {wgt} crosstalk-removed={xt}:"
                      f" good {okv.sum()}, scatter {np.degrees(sv):.2f} deg, median |az - default| {np.degrees(np.median(np.abs(dx))):.2f} deg")

    out = dict(t=t, t_usbl=B["t_usbl"], az_array=B["az"], sigma=sigma, sigma_formal=B["sigma_formal"], sigma_emp=s_emp,
               mask=ok, az_srp=B["az_srp"], coh=B["coh"], snr=B["snr"], contrast=B["contrast"], rms=B["rms"], az_mirror=B["az_mirror"],
               rms_mirror=B["rms_mirror"], az_resid=res, nb_resid=nres, tau=B["tau"], tau_peak=B["peak"],
               blanked=B["blanked"], band_rms=B["band_rms"], pairs=np.array(PAIRS), pos=pos, sound_speed=c,
               handedness=hand, alpha=alpha_p1, alpha_graph=alpha_g,
               az_enu=np.angle(np.exp(1j * (alpha_g + hand * B["az"]))),
               band=np.array(a.band), win=a.win, weight=a.weight, usbl_offset=clock["b"], usbl_drift=clock["d_S"], t_ref=clock["t_ref"])

    chk = None
    if not a.no_check:
        chk = method_check(rx, clock, pos, c, p1, win=a.win, click_k=a.click_k, weight=a.weight, remove_crosstalk=False)
        print(f"method check (pinger band 30-40 kHz, {a.win} s windows, runs 1-2) vs phase 1 az_array: n {chk['n']},"
              f" median |diff| {np.degrees(chk['med_abs']):.2f} deg, bias {np.degrees(chk['bias']):+.2f} deg,"
              f" robust sigma {np.degrees(chk['robust_sig']):.2f} deg, |diff| > 2 deg: {100 * chk['frac_gt2']:.1f} %")
        out.update(check_t=chk["t"], check_az=chk["az"], check_az_p1=chk["az_p1"], check_diff=chk["diff"])
    np.savez(OUT / "bearings_noise.npz", **out)
    print("wrote", OUT / "bearings_noise.npz")
    _plot(plt, out, B, chk, OUT / "bearings_noise.png")
    return out


def _spans(tg, gap):
    if len(tg) == 0:
        return []
    br = np.where(np.diff(tg) > gap)[0]
    s = np.r_[0, br + 1]
    e = np.r_[br, len(tg) - 1]
    return [(tg[a], tg[b], b - a + 1) for a, b in zip(s, e)]


def _plot(plt, o, B, chk, path):
    t, ok = o["t"], o["mask"]
    n = 5 if chk is not None else 4
    fig, ax = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=False)
    resp = B["resp"] / (np.abs(B["resp"]).max(1, keepdims=True) + 1e-12)
    ax[0].pcolormesh(t, np.degrees(B["az_grid"]), resp.T, vmin=0, vmax=1, shading="nearest", cmap="Greys")
    ax[0].plot(t[~ok], np.degrees(o["az_array"][~ok]), "x", color="tab:red", ms=3, label="rejected")
    ax[0].errorbar(t[ok], np.degrees(o["az_array"][ok]), np.degrees(o["sigma"][ok]), fmt=".", ms=3, color="tab:blue",
                   elinewidth=0.5, label="good")
    ax[0].set_ylabel("az_array [deg]")
    ax[0].set_title("noise bearings, array frame (grey: steered response, normalized)")
    ax[0].legend(fontsize=8, loc="upper left")
    ax[1].plot(t[ok], np.degrees(o["az_enu"][ok]) % 360 - 360, ".", ms=3)
    ax[1].set_ylabel("ENU bearing [deg] (-360..0)")
    ax[1].set_title(f"theta_enu = alpha + s az_array (alpha {np.degrees(o['alpha_graph']):.1f} deg, s = {o['handedness']:+d}),"
                    " atan2(n, e) from the USBL")
    ax[2].semilogy(t, o["coh"], ".", ms=3, label="SCOT peak (coherency)")
    ax[2].semilogy(t, o["contrast"] / 100, ".", ms=3, label="steered-response contrast / 100")
    ax[2].semilogy(t, o["blanked"] + 1e-4, ".", ms=2, label="blanked fraction")
    ax[2].legend(fontsize=8)
    ax[2].set_ylabel("quality")
    ax[3].plot(t, o["rms"] * 1e6, ".", ms=3, label="plane-wave rms")
    ax[3].plot(t, o["rms_mirror"] * 1e6, ".", ms=2, alpha=0.5, label="mirror rms")
    ax[3].set_ylim(0, 30)
    ax[3].set_ylabel("us")
    ax[3].legend(fontsize=8)
    for a_ in ax[:4]:
        a_.set_xlim(t[0], t[-1])
        a_.grid(alpha=0.3)
    ax[3].set_xlabel("buoy time [s]")
    if chk is not None:
        ax[4].plot(chk["t"], np.degrees(chk["az_p1"]), ".", ms=4, color="k", label="phase 1 per ping")
        ax[4].plot(chk["t"], np.degrees(chk["az"]), ".", ms=2, color="tab:orange", label="windowed GCC, pinger band")
        ax[4].set_ylabel("az_array [deg]")
        ax[4].set_xlabel("buoy time [s]")
        ax[4].set_title("method check (runs 1-2)")
        ax[4].legend(fontsize=8)
        ax[4].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    print("wrote", path)


if __name__ == "__main__":
    main()
