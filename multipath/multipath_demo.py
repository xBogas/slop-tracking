"""
Multipath as a measurement, not a nuisance.

Every arrival at a hydrophone is the same signal several times: direct path,
surface reflection (sign flipped), bottom reflection, surface-bottom, ...
In a spectrogram these are copies of one pattern shifted in time. Shifting
the record against itself and finding where it matches (the autocorrelation,
or equivalently aligning the reflections on the first arrival) gives the
differences between path delays. Those differences depend only on the
horizontal range, the source and receiver depths and the water depth. They
do NOT depend on the emission time, the clock offsets or the other receivers.

Estimation is model-based: the image-source model predicts the full set of
pairwise lags for a candidate (range, source depth, water depth), and a grid
search picks the candidate whose lags land on the bright ridges of the
autocorrelation. Picking single peaks does not work, because there are more
ridges than "the" reflection, and some cross-terms are stronger than the
bottom echo itself.

Part 1 (pinger): matched filter per ping -> autocorrelation of its output ->
range, pinger depth and water depth from ONE hydrophone. Two buoys -> a
position fix with no emission time and no sync.
Part 2 (boat noise): whitened autocorrelation of 1 s windows (cepstrum-like;
the striations in the spectrogram are the same thing seen in frequency) ->
range and water depth from ONE hydrophone.

Geometry and truth track as in estimation/demo.py. Numpy + matplotlib only.
Run: python3 /mnt/project-files/multipath/multipath_demo.py  (about 1 min)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = os.path.join(HERE, "..", "motion_model", "demo_truth.csv")
rng = np.random.default_rng(3)

C = 1500.0              # sound speed, m/s (isovelocity)
D_TRUE = 30.0           # water depth, flat bottom
ZR = 10.0               # buoy hydrophone depth
OFFSET = np.array([-150.0, 700.0])
BUOYS = np.array([[-600.0, 250.0], [600.0, 250.0]]) + np.array([[8.0, -5.0], [-6.0, 7.0]])
R_SURF, R_BOT = -0.9, 0.45     # reflection coefficients (rough surface, sandy bottom)
GAIN = np.array([1.0, R_SURF, R_BOT, R_SURF * R_BOT, R_SURF * R_BOT])
PATHS = ["direct", "surface", "bottom", "surf-bottom", "bottom-surf"]
II, JJ = np.triu_indices(5, 1)                 # the 10 path pairs
PAIR_W = np.abs(GAIN[II] * GAIN[JJ])

tr = np.genfromtxt(TRUTH, delimiter=",", names=True)


def path_delays(R, zs, D=D_TRUE, zr=ZR):
    """Travel times of the 5 image paths. Arguments broadcast; output has a
    trailing axis of 5."""
    R, zs, D = np.broadcast_arrays(*(np.asarray(a, float) for a in (R, zs, D)))
    zi = np.stack([zs, -zs, 2 * D - zs, 2 * D + zs, zs - 2 * D], -1)
    return np.sqrt(R[..., None] ** 2 + (zr - zi) ** 2) / C


def pair_lags(R, zs, D):
    t = path_delays(R, zs, D)
    return np.abs(t[..., JJ] - t[..., II])     # (..., 10)


def fit_multipath(env, lagax, Rg, zg, Dg, chunk=200):
    """Grid search for (R, zs, D) whose predicted pair lags best hit the
    autocorrelation envelope `env` (sampled on `lagax`)."""
    best, arg = -np.inf, None
    for i in range(0, Rg.size, chunk):
        R = Rg[i:i + chunk, None, None]
        L = pair_lags(R, zg[None, :, None], Dg[None, None, :])
        sc = (np.interp(L, lagax, env, right=0.0) * PAIR_W).sum(-1)
        k = np.argmax(sc)
        if sc.flat[k] > best:
            best = sc.flat[k]
            a, b, c = np.unravel_index(k, sc.shape)
            arg = (Rg[i + a], zg[b], Dg[c])
    return arg


def spectrum_profile(y, fb, W, Rg, zg, D, chunk=100):
    """Coherent fit for the pinger. The received power spectrum in the band,
    |X(f)|^2, is the chirp spectrum W times the multipath comb
    |sum_p g_p a_p exp(-2j pi f tau_p)|^2, which does not depend on the
    emission time. Returns, for each source depth in zg, the best correlation
    over range and the range that gives it.
    A coherent fit is needed because the surface echo sits within ~1/bandwidth
    of the direct path with the opposite sign, so envelope peaks move."""
    ym = y / y.mean()
    ym = ym - ym.mean()
    ym /= np.linalg.norm(ym)
    best = np.full(zg.size, -np.inf)
    Rbest = np.zeros(zg.size)
    for i in range(0, Rg.size, chunk):
        tau = path_delays(Rg[i:i + chunk, None], zg[None, :], D)       # (r, z, 5)
        amp = (GAIN * tau[..., :1] / tau).astype(np.float32)
        rel = (tau - tau[..., :1]).astype(np.float32)
        S = np.abs(np.einsum("rzp,rzpf->rzf", amp,
                             np.exp(-2j * np.pi * fb.astype(np.float32) * rel[..., None]))) ** 2 * W
        S -= S.mean(-1, keepdims=True)
        c = (S @ ym) / (np.linalg.norm(S, axis=-1) + 1e-12)            # (r, z)
        k = np.argmax(c, 0)
        cz = c[k, np.arange(zg.size)]
        better = cz > best
        best[better] = cz[better]
        Rbest[better] = Rg[i + k[better]]
    return best, Rbest


def autocorr_env(X, nfft, maxlag_n):
    """|autocorrelation| for lags 0..maxlag_n from a one-sided (analytic)
    spectrum X, normalised to 1 at lag 0, noise floor removed."""
    a = np.abs(np.fft.ifft(np.abs(X) ** 2, nfft)[:maxlag_n])
    a /= a[0]
    return np.clip(a - np.median(a), 0, None)


# --------------------------------------------------------------------------
# Part 1: pinger, 30-40 kHz LFM chirp, 5 ms, one ping per second
# --------------------------------------------------------------------------
def part1():
    fs = 192_000
    Tc = 5e-3
    tc = np.arange(int(Tc * fs)) / fs
    chirp = np.sin(2 * np.pi * (30e3 * tc + 0.5 * (10e3 / Tc) * tc ** 2)) * np.hanning(tc.size)
    t_ping = np.arange(5.0, 295.0, 1.0)
    K = t_ping.size
    px = np.interp(t_ping, tr["t"], tr["pinger_x"]) + OFFSET[0]
    py = np.interp(t_ping, tr["t"], tr["pinger_y"]) + OFFSET[1]
    spd = np.hypot(np.gradient(px, t_ping), np.gradient(py, t_ping))
    zs_true = np.clip(10.0 - 0.35 * spd, 5.0, 10.0)   # towed body rises with speed

    win = int(0.014 * fs)          # 14 ms window from 1 ms before the direct arrival
    pre = 0.001
    nfft = 1 << int(np.ceil(np.log2(win + chirp.size)))
    f = np.fft.fftfreq(nfft, 1 / fs)
    CH = np.fft.fft(chirp, nfft)
    inband = (f > 30.0e3) & (f < 40.0e3)
    inband &= np.cumsum(inband) % 2 == 0   # every 2nd bin (94 Hz) is enough for <5 ms delays
    fb = f[inband]
    W = np.abs(CH[inband]) ** 2
    lag = (np.arange(win) - int(pre * fs)) / fs
    Rg = np.arange(50.0, 2000.0, 2.0)

    out = {}
    spec_all, R_all, rel_all, echo_all = [], [], [], []
    for j, b in enumerate(BUOYS):
        R = np.hypot(px - b[0], py - b[1])
        tau = path_delays(R, zs_true)                 # (K, 5)
        rel = tau - tau[:, :1]
        echo = np.zeros((K, win))
        spec_b = np.zeros((K, inband.sum()))
        for k in range(K):
            g = GAIN.copy()
            g[[1, 3, 4]] *= rng.uniform(0.4, 1.0)     # surface bounces fade with waves
            amp = g * tau[k, 0] / tau[k]              # spherical spreading
            H = (amp[:, None] * np.exp(-2j * np.pi * f[None] * (rel[k][:, None] + pre))).sum(0)
            x = np.real(np.fft.ifft(CH * H))[:win]
            x += rng.normal(0, 0.05, win) * (R[k] / 300.0)   # SNR falls with range
            if j == 0 and k == 145:
                x_show = x.copy()
            X = np.fft.fft(x, nfft)
            Xm = X * np.conj(CH)
            Xm[f < 0] = 0                             # analytic matched-filter output
            echo[k] = np.abs(np.fft.ifft(Xm))[:win]
            spec_b[k] = np.abs(X[inband]) ** 2
        echo /= echo.max(1, keepdims=True)
        spec_all.append(spec_b)
        R_all.append(R)
        rel_all.append(rel)
        echo_all.append(echo)

    # The pinger depth is the same for both buoys at a given ping and changes
    # slowly, so it is estimated jointly: sum both buoys' depth profiles over
    # +-3 pings. Fitting depth per buoy and per ping alone trades depth against
    # range and gives 10-20 % range errors with outliers.
    zg = np.arange(3.0, 12.01, 0.5)
    sub = np.arange(0, K, 20)
    Dg = np.arange(27.0, 33.01, 0.5)
    prof_D = np.zeros(Dg.size)
    for d, D in enumerate(Dg):            # water depth: one value for the run
        for k in sub:
            prof_D[d] += max(spectrum_profile(spec_all[0][k], fb, W, Rg, zg, D)[0]
                             + spectrum_profile(spec_all[1][k], fb, W, Rg, zg, D)[0])
    D_hat = Dg[np.argmax(prof_D)]
    prof = np.zeros((2, K, zg.size))
    Rz = np.zeros((2, K, zg.size))
    for j in (0, 1):
        for k in range(K):
            prof[j, k], Rz[j, k] = spectrum_profile(spec_all[j][k], fb, W, Rg, zg, D_hat)
    P = prof.sum(0)
    Psm = np.array([P[max(k - 3, 0):k + 4].sum(0) for k in range(K)])
    iz = np.argmax(Psm, 1)
    # refine depth jointly (both buoys, local range windows), then range per buoy
    z_hat = np.zeros(K)
    R_fine = np.zeros((2, K))
    for k in range(K):
        zf = np.arange(zg[iz[k]] - 0.5, zg[iz[k]] + 0.5, 0.05)
        pr = [spectrum_profile(spec_all[j][k], fb, W,
                               np.arange(Rz[j, k, iz[k]] - 40, Rz[j, k, iz[k]] + 40, 0.5), zf, D_hat)
              for j in (0, 1)]
        m = np.argmax(pr[0][0] + pr[1][0])
        z_hat[k] = zf[m]
        R_fine[:, k] = pr[0][1][m], pr[1][1][m]
    for j, b in enumerate(BUOYS):
        # crude gate standing in for the tracker's robust loss: drop pings more
        # than 5 % off the running median of their neighbours (+-5 pings)
        R_hat = R_fine[j].copy()
        med = np.array([np.median(R_fine[j][max(k - 5, 0):k + 6]) for k in range(K)])
        R_hat[np.abs(R_hat - med) > 0.05 * med] = np.nan
        out[j] = dict(R=R_all[j], rel=rel_all[j], echo=echo_all[j], R_hat=R_hat, z_hat=z_hat, D_hat=D_hat)

    # position from the two multipath ranges (no emission time, no clock sync):
    # intersect the circles, keep the solution on the offshore side
    b0, b1 = BUOYS
    r0, r1 = out[0]["R_hat"], out[1]["R_hat"]
    d = np.linalg.norm(b1 - b0)
    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(np.clip(r0 ** 2 - a ** 2, 0, None))
    u = (b1 - b0) / d
    nrm = np.array([-u[1], u[0]])
    nrm = nrm if nrm[1] > 0 else -nrm
    fix = b0 + a[:, None] * u + h[:, None] * nrm
    err = np.hypot(fix[:, 0] - px, fix[:, 1] - py)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    a0 = ax[0, 0]
    nf, hop = 256, 16
    fr = np.lib.stride_tricks.sliding_window_view(x_show, nf)[::hop] * np.hanning(nf)
    Sg = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)) + 1e-9)
    fz = np.fft.rfftfreq(nf, 1 / fs)
    a0.imshow(Sg.T, aspect="auto", origin="lower", cmap="magma",
              extent=[lag[0] * 1e3, lag[-nf] * 1e3, 0, fz[-1] / 1e3], vmin=Sg.max() - 45, vmax=Sg.max())
    a0.set_ylim(27, 43)
    a0.set_xlim(-0.5, 9)
    for p in range(5):
        t0 = out[0]["rel"][145, p] * 1e3
        a0.plot([t0, t0 + Tc * 1e3], [30, 40], ":", lw=0.8, c="c")
    a0.set(xlabel="time after direct arrival (ms)", ylabel="frequency (kHz)",
           title=f"One ping at buoy 1 (t = {t_ping[145]:.0f} s): the same chirp\n"
                 "5 times, shifted in time (cyan: image-model paths)")
    a1 = ax[0, 1]
    o = out[0]
    a1.imshow(20 * np.log10(o["echo"].T + 1e-3), aspect="auto", origin="lower", cmap="magma",
              extent=[t_ping[0], t_ping[-1], lag[0] * 1e3, lag[-1] * 1e3], vmin=-30, vmax=0)
    a1.set_ylim(-0.3, 4.5)
    for p in range(1, 5):
        a1.plot(t_ping, o["rel"][:, p] * 1e3, "c:", lw=0.8)
    a1.set(xlabel="ping time (s)", ylabel="delay behind direct (ms)",
           title="Buoy 1 echogram: matched filter per ping, aligned on\n"
                 "the direct arrival (cyan: surface, bottom, s-b, b-s)")
    a2 = ax[0, 2]
    for j, c_ in zip((0, 1), ("C0", "C1")):
        a2.plot(t_ping, out[j]["R"], "-", c="k", lw=1)
        a2.plot(t_ping, out[j]["R_hat"], ".", ms=3, c=c_, label=f"buoy {j+1}, from multipath")
    a2.set(xlabel="ping time (s)", ylabel="horizontal range (m)",
           title="Range from ONE hydrophone each (black: truth)\nno emission time, no sync")
    a2.legend(fontsize=8)
    a3 = ax[1, 0]
    a3.plot(t_ping, zs_true, "k-", label="true pinger depth")
    a3.plot(t_ping, out[0]["z_hat"], ".", ms=3, c="C3", label="estimated (both buoys jointly)")
    a3.set(xlabel="ping time (s)", ylabel="pinger depth (m)", ylim=(0, 15),
           title="Pinger depth per ping (it rises when the boat speeds up)")
    a3.legend(fontsize=8)
    a4 = ax[1, 1]
    for j, c_ in zip((0, 1), ("C0", "C1")):
        rel_err = (out[j]["R_hat"] - out[j]["R"]) / out[j]["R"] * 100
        a4.plot(out[j]["R"], rel_err, ".", ms=3, c=c_, label=f"buoy {j+1}")
    a4.axhline(0, c="k", lw=0.5)
    a4.set(xlabel="true range (m)", ylabel="range error (%)", ylim=(-15, 15),
           title=f"Range error. Water depth estimated: {out[0]['D_hat']:.1f} / "
                 f"{out[1]['D_hat']:.1f} m (true {D_TRUE:.0f})")
    a4.legend(fontsize=8)
    a5 = ax[1, 2]
    a5.plot(px, py, "k-", lw=1, label="true pinger track")
    a5.plot(fix[:, 0], fix[:, 1], ".", ms=3, c="C3", label="2-buoy multipath fix")
    a5.plot(*BUOYS.T, "ks", mfc="none", ms=9, label="buoys")
    a5.plot(0, 0, "k^", ms=9, label="shore array")
    a5.set(xlabel="x (m)", ylabel="y (m)", aspect="equal",
           title=f"Fix from multipath ranges only: median err {np.nanmedian(err):.1f} m")
    a5.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pinger_multipath.png"), dpi=110)

    stats = {}
    for j in (0, 1):
        e = np.abs(out[j]["R_hat"] - out[j]["R"])
        stats[j] = (np.isfinite(e).mean(), np.nanmedian(e), np.nanpercentile(e, 90),
                    np.nanmedian(e / out[j]["R"]) * 100, out[j]["D_hat"])
    return stats, np.median(np.abs(z_hat - zs_true)), np.nanmedian(err), np.nanpercentile(err, 90)


# --------------------------------------------------------------------------
# Part 2: boat radiated noise, broadband, source at 1 m depth
# --------------------------------------------------------------------------
def part2():
    fs = 48_000
    zs = 1.0
    T0, T1 = 5.0, 295.0
    n = int((T1 - T0) * fs)
    t = T0 + np.arange(n) / fs
    # source: coloured broadband noise (cavitation) + blade-rate tonals
    ff = np.fft.rfftfreq(n, 1 / fs)
    shape = 1 / (1 + ff / 2000.0)
    shape[(ff < 150) | (ff > 16000)] = 0
    s = np.fft.irfft(np.fft.rfft(rng.normal(size=n)) * shape, n)
    s /= s.std()
    s += 0.3 * np.sin(2 * np.pi * 95 * t) + 0.2 * np.sin(2 * np.pi * 190 * t)

    bx = np.interp(t, tr["t"], tr["x"]) + OFFSET[0]
    by = np.interp(t, tr["t"], tr["y"]) + OFFSET[1]
    b = BUOYS[0]
    R = np.hypot(bx - b[0], by - b[1])
    x = np.zeros(n)
    tt = np.arange(n) / fs
    for p in range(5):
        tau_p = path_delays(R, zs)[:, p]
        tau_0 = path_delays(R, zs)[:, 0]
        x += GAIN[p] * (tau_0 / tau_p) * np.interp(tt - (tau_p - tau_0[0]), tt, s)
    x /= x.std()
    x += rng.normal(0, 0.3, n)      # ambient noise

    # spectrogram (Welch, 1 s) and whitened autocorrelation per 1 s window
    L = fs
    nseg = n // L
    tw = T0 + (np.arange(nseg) + 0.5)
    nf = 4096
    spec = np.zeros((nseg, nf // 2 + 1))
    maxlag = int(0.012 * fs)
    acg = np.zeros((nseg, maxlag))
    fx = np.fft.fftfreq(2 * L, 1 / fs)
    band = (fx > 500) & (fx < 15000)
    box = int(3000 / (fs / (2 * L)))                  # 3 kHz smoothing >> comb spacing
    for k in range(nseg):
        seg = x[k * L:(k + 1) * L]
        fr = seg[: (L // nf) * nf].reshape(-1, nf) * np.hanning(nf)
        spec[k] = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).mean(0)
        X = np.fft.fft(seg * np.hanning(L), 2 * L)
        P = np.abs(X) ** 2
        cs = np.concatenate([[0], np.cumsum(P)])
        idx = np.arange(P.size)
        lo, hi = np.clip(idx - box // 2, 0, P.size), np.clip(idx + box // 2, 0, P.size)
        Ps = (cs[hi] - cs[lo]) / (hi - lo) + 1e-12
        # whitened, one-sided -> analytic autocorrelation envelope
        acg[k] = autocorr_env(np.where(band & (fx > 0), np.sqrt(P / Ps), 0), 2 * L, maxlag)
    lagax = np.arange(maxlag) / fs
    acg[:, lagax < 0.25e-3] = 0      # inside the mainlobe for a ~15 kHz band
    Rw = np.interp(tw, t, R)
    tb_true = path_delays(Rw, zs)
    tb_true = tb_true[:, 2:] - tb_true[:, :1]

    # pass 1: water depth free -> global D; pass 2: range per window
    Rg = np.arange(50.0, 2500.0, 2.0)
    zg = np.array([zs])
    D_k = np.array([fit_multipath(acg[k], lagax, Rg, zg, np.arange(22.0, 38.01, 0.25))[2]
                    for k in range(nseg)])
    D_hat = np.median(D_k)
    R_hat = np.zeros(nseg)
    for k in range(nseg):
        r0 = fit_multipath(acg[k], lagax, Rg, zg, np.array([D_hat]))[0]
        R_hat[k] = fit_multipath(acg[k], lagax, np.arange(max(r0 - 3, 20), r0 + 3, 0.1),
                                 zg, np.array([D_hat]))[0]
    # sensitivity: same data, water depth assumed 1 m too deep
    R_D1 = np.array([fit_multipath(acg[k], lagax, Rg, zg, np.array([D_hat + 1.0]))[0]
                     for k in range(nseg)])

    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    fz = np.fft.rfftfreq(nf, 1 / fs)
    sel = fz < 6000
    a0 = ax[0, 0]
    S = 10 * np.log10(spec[:, sel] / np.median(spec[:, sel], 0) + 1e-6)
    a0.imshow(S.T, aspect="auto", origin="lower", cmap="viridis",
              extent=[tw[0], tw[-1], 0, fz[sel][-1] / 1e3], vmin=-6, vmax=4)
    a0.set(xlabel="time (s)", ylabel="frequency (kHz)",
           title="Buoy 1 spectrogram (each bin / its median): multipath\n"
                 "striations, fringe spacing = 1 / delay difference")
    a1 = ax[0, 1]
    a1.imshow(acg.T, aspect="auto", origin="lower", cmap="magma",
              extent=[tw[0], tw[-1], 0, lagax[-1] * 1e3], vmax=np.percentile(acg, 99.7))
    a1.plot(tw, tb_true[:, 0] * 1e3, "c:", lw=1, label="bottom - direct (truth)")
    a1.plot(tw, tb_true[:, 2] * 1e3, "w:", lw=1, label="bottom-surf - direct (truth)")
    a1.set(xlabel="time (s)", ylabel="lag (ms)", ylim=(0, 8),
           title="Whitened autocorrelation per 1 s (the signal shifted against\n"
                 "its own reflections): one ridge per pair of paths")
    a1.legend(fontsize=8)
    a2 = ax[1, 0]
    a2.plot(tw, Rw, "k-", label="true range")
    a2.plot(tw, R_hat, ".", ms=4, label=f"from multipath, D estimated ({D_hat:.2f} m)")
    a2.plot(tw, R_D1, ".", ms=3, alpha=0.6, label=f"same, D assumed {D_hat + 1:.2f} m")
    a2.set(xlabel="time (s)", ylabel="range to buoy 1 (m)",
           title="Range from ONE hydrophone, radiated noise only")
    a2.legend(fontsize=8)
    a3 = ax[1, 1]
    a3.plot(Rw, (R_hat - Rw) / Rw * 100, ".", ms=4, label="D estimated")
    a3.plot(Rw, (R_D1 - Rw) / Rw * 100, ".", ms=3, alpha=0.6, label="D off by +1 m")
    a3.axhline(0, c="k", lw=0.5)
    a3.set(xlabel="true range (m)", ylabel="range error (%)", ylim=(-15, 15),
           title=f"Error vs range. Water depth per window: median {D_hat:.2f} m, "
                 f"IQR {np.subtract(*np.percentile(D_k, [75, 25])):.2f} m (true {D_TRUE:.0f})")
    a3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "noise_multipath.png"), dpi=110)
    e = np.abs(R_hat - Rw)
    return (np.median(e), np.percentile(e, 90), np.median(e / Rw) * 100,
            np.median((R_D1 - Rw) / Rw) * 100, D_hat, Rw.min(), Rw.max())


if __name__ == "__main__":
    st, zerr, med, p90 = part1()
    for j, (ok, me, e90, mp, Dh) in st.items():
        print(f"pinger buoy {j+1}: {ok*100:.0f}% pings kept, range err median {me:.1f} m ({mp:.1f}%), "
              f"90th pct {e90:.1f} m; water depth est {Dh:.1f} m")
    print(f"pinger depth err median {zerr:.2f} m; 2-buoy multipath fix: median {med:.1f} m, 90th pct {p90:.1f} m")
    me, e90, mp, b1, Dh, rmin, rmax = part2()
    print(f"noise buoy 1 (range {rmin:.0f}-{rmax:.0f} m): range err median {me:.1f} m ({mp:.1f}%), "
          f"90th pct {e90:.1f} m; water depth est {Dh:.2f} m; D +1 m -> range bias {b1:+.1f}%")
