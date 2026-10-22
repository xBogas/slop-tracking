"""
Ping association across receivers -> TOA table (phase 1).

Input: out/detections_<rx>.npz from detect.py (per receiver: TOA in its own
clock, per-receiver ping index k from `index_pings`, SNR, Doppler).

Method
------
1. Per receiver, `index_pings` unwraps the arrival phase: consecutive
   arrivals differ by ~n Tp (the range changes by < ~10 ms per ping);
   detections inconsistent with their neighbours are dropped.
2. Cross-receiver index lags. A nominal lag comes from nearest arrivals
   (buoys are synchronized; the USBL uses the coarse offset 102.7 s); the
   candidates are nominal +/- 3. For the USBL a lag change of one ping is
   an offset change of one Tp: the integer ambiguity of the offset.
3. Lag choice: jitter fingerprint. The pinger has ~0.13 ms rms emission
   jitter common to all receivers, so the second differences of the TOA
   sequences of two receivers correlate (~0.8 here) only at the right lag
   (white jitter gives ~ -0.5 at +/- 1 and ~0 elsewhere). No geometry is
   needed; the margin is a z-score against the best non-adjacent lag.
4. Geometry check and fine offset. With 3 TOAs and unknown (x, y, emission
   time) each ping has an exact hyperbolic fix (closed form, known depths),
   so the redundancy is the pinger clock: emission times should follow a
   smooth clock. The cost is the robust RMS of the emission-time residuals
   about a quadratic clock per run (x c, metres), with a penalty for pings
   without a real fix. It is computed for all lag hypotheses (each with a
   1-D search of the USBL offset b) and gives the fine b at the chosen lags.
   On this data the cost does NOT separate the USBL lags (the fixes lie
   outside the receiver triangle where GDOP is large, and the buoy
   positions are uncertain), so step 3 decides; see notes/A_detection.md.
5. Output table on a common ping index (k = 0 at the first ping seen by two
   receivers), TOAs in buoy time with the USBL shifted by b, Tp from the
   long-baseline emission slope, per-receiver TOA sigmas from pairwise TDOA
   second differences (the common emission jitter cancels), the jitter
   sigma, and the per-ping hyperbolic fixes as a starting point.

Run: python3 associate.py   -> out/toa_phase1.npz (format: notes/A_detection.md)
"""
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

C_SOUND = 1520.0
NAMES = ("Rx902", "Rx01", "USBL")


def index_pings(t, Tp, tol=0.03, near=10):
    """Ping index per arrival on one receiver by unwrapping the arrival phase.

    Consecutive differences are close to integer multiples of the local
    arrival interval (Tp (1 + rr/c)). A detection is kept if it is
    consistent (|dt - n Tp| < tol + 2 ms n) with at least two of its
    neighbours within `near` pings (one if it has only one), which removes
    false alarms and isolated detections. Returns (k, keep)."""
    t = np.asarray(t, float)
    n = len(t)
    keep = np.zeros(n, bool)
    for i in range(n):
        j = np.flatnonzero(np.abs(t - t[i]) <= near * Tp)
        j = j[j != i]
        d = (t[j] - t[i]) / Tp
        m = np.round(d)
        good = (m != 0) & (np.abs(d - m) * Tp < tol + 0.002 * np.abs(m))
        keep[i] = good.sum() >= min(2, max(len(j), 1))
    # two kept arrivals within half a period: keep the one that fits its neighbours best
    while True:
        idx = np.flatnonzero(keep)
        close = np.flatnonzero(np.diff(t[idx]) < 0.5 * Tp)
        if not len(close):
            break
        a, b = idx[close[0]], idx[close[0] + 1]
        def misfit(i):
            nb = idx[(idx != a) & (idx != b)]
            nb = nb[np.argsort(np.abs(t[nb] - t[i]))[:6]]
            d = (t[nb] - t[i]) / Tp
            return np.sum(np.abs(d - np.round(d)))
        keep[a if misfit(a) > misfit(b) else b] = False
    k = np.zeros(n, int)
    idx = np.flatnonzero(keep)
    if len(idx):
        tk = t[idx]
        kk = np.zeros(len(idx), int)
        slope = Tp
        for a in range(1, len(idx)):
            d = tk[a] - tk[a - 1]
            step = max(int(np.round(d / slope)), 1)
            kk[a] = kk[a - 1] + step
            if step <= 3:                            # track the local arrival interval
                slope = min(max(0.7 * slope + 0.3 * d / step, 0.98 * Tp), 1.02 * Tp)
        k[idx] = kk
    return k, keep


def d2_series(t, k):
    """Second differences of t on the integer grid k (NaN where missing)."""
    full = np.full(k.max() - k.min() + 1, np.nan)
    full[k - k.min()] = t
    return full[2:] - 2 * full[1:-1] + full[:-2], k.min() + 1


def jitter_corr(ta, ka, tb, kb, lags):
    """Correlation of TOA second differences of receivers a, b for global
    index g = ka and g = kb + lag (each series clipped at 4 robust sigmas, so
    a few mis-picks do not dominate). Peaks (~ +0.8 here) at the right lag;
    white jitter gives ~ -0.5 at +/- 1 and ~0 elsewhere. Returns (corr, n)."""
    def clip(x):
        s = 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))
        return np.clip(x, -4 * s, 4 * s)
    da, a0 = d2_series(ta, ka)
    db, b0 = d2_series(tb, kb)
    da, db = clip(da), clip(db)
    out, nn = [], []
    for L in lags:
        i = np.arange(len(da))
        j = i + a0 - (b0 + L)
        m = (j >= 0) & (j < len(db))
        x, y = da[i[m]], db[j[m]]
        g = np.isfinite(x) & np.isfinite(y)
        out.append(np.corrcoef(x[g], y[g])[0, 1] if g.sum() > 10 else np.nan)
        nn.append(int(g.sum()))
    return np.array(out), np.array(nn)


def fingerprint_lag(corr, n, cand):
    """Best lag offset and its z-score against the best non-adjacent lag."""
    j = int(np.nanargmax(corr))
    others = [c for i, c in enumerate(corr) if abs(i - j) > 1 and np.isfinite(c)]
    z = (corr[j] - max(others)) * np.sqrt(max(n[j], 1)) if others else np.nan
    return int(cand[j]), float(z)


def nominal_lag(ta, ka, tb, kb, max_dt=0.5):
    """Median index lag (ka - kb) over nearest arrivals closer than max_dt."""
    j = np.clip(np.searchsorted(tb, ta), 1, len(tb) - 1)
    j = np.where(np.abs(tb[j - 1] - ta) < np.abs(tb[j] - ta), j - 1, j)
    m = np.abs(tb[j] - ta) < max_dt
    return int(np.round(np.median(ka[m] - kb[j[m]]))) if m.any() else 0


def hyperbolic_fix(toa, rx, z_src, c=C_SOUND, prior=None):
    """Closed-form position + emission time from 3 TOAs per ping.
    toa (K, 3) [s], rx (3, 3) receiver xyz (z up), source depth z_src (z up).
    Ranges R_i = R_1 + d_i, d_i = c (t_i - t_1). Subtracting the range
    equations gives p = A + B R_1 (linear); |p - r_1|^2 + dz_1^2 = R_1^2 is
    then a quadratic in R_1. Of two valid roots the one nearer `prior`
    wins. Returns (p (K, 2), e (K,), ok (K,))."""
    K = len(toa)
    r = rx[:, :2]
    dz2 = (rx[:, 2] - z_src) ** 2
    d = c * (toa - toa[:, :1])                       # (K, 3)
    M = -2 * (r[1:] - r[0])                          # (2, 2)
    Minv = np.linalg.inv(M)
    # -2 (r_i - r_1) p - 2 d_i R1 = d_i^2 - |r_i|^2 + |r_1|^2 - dz_i^2 + dz_1^2
    q0 = d[:, 1:] ** 2 - (r[1:] ** 2).sum(1) + (r[0] ** 2).sum() - dz2[1:] + dz2[0]
    A = q0 @ Minv.T                                  # (K, 2)
    B = (2 * d[:, 1:]) @ Minv.T                      # (K, 2)
    Ar = A - r[0]
    a2 = (B ** 2).sum(1) - 1
    a1 = 2 * (B * Ar).sum(1)
    a0 = (Ar ** 2).sum(1) + dz2[0]
    disc = a1 ** 2 - 4 * a2 * a0
    sq = np.sqrt(np.maximum(disc, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.stack([(-a1 + sq) / (2 * a2), (-a1 - sq) / (2 * a2)], 1)   # (K, 2)
    valid = (disc >= 0)[:, None] & np.isfinite(roots) & (roots > 0) \
        & np.all(roots[:, :, None] + d[:, None, :] > 0, axis=2)
    P = A[:, None, :] + B[:, None, :] * roots[:, :, None]                     # (K, 2, 2)
    pr = np.zeros(2) if prior is None else np.asarray(prior)
    dist = np.where(valid, np.linalg.norm(P - pr, axis=2), np.inf)
    pick = np.argmin(dist, axis=1)
    ok = np.isfinite(dist[np.arange(K), pick])
    # second pass: where both roots are valid, take the one nearer the
    # rolling median of the neighbouring fixes (track continuity)
    p1 = np.where(ok[:, None], P[np.arange(K), pick], np.nan)
    if K > 5 and (valid.sum(1) == 2).any():
        w = 5
        pad = np.pad(p1, ((w, w), (0, 0)), constant_values=np.nan)
        win = np.lib.stride_tricks.sliding_window_view(pad, 2 * w + 1, axis=0)   # (K, 2, 2w+1)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            med = np.nanmedian(win, axis=2)
        dist = np.where(valid, np.linalg.norm(P - med[:, None, :], axis=2), np.inf)
        both = valid.all(1) & np.isfinite(med).all(1)
        pick = np.where(both, np.argmin(dist, axis=1), pick)
    R1 = roots[np.arange(K), pick]
    p = P[np.arange(K), pick]
    e = toa[:, 0] - R1 / c
    p[~ok], e[~ok] = np.nan, np.nan
    return p, e, ok


def robust_poly(k, y, deg=2, n_iter=6):
    """Bisquare-weighted polynomial fit, started from a median line so that
    gross outliers cannot pull the first fit. Returns (coef, residuals, weights)."""
    kc = k - k.mean()
    o = np.argsort(kc)
    sl = np.median(np.diff(y[o]) / np.maximum(np.diff(kc[o]), 1e-9))
    y0 = y - sl * kc
    r0 = y0 - np.median(y0)
    s0 = max(1.4826 * np.median(np.abs(r0)), 1e-7)
    w = (np.abs(r0) < 5 * s0).astype(float)
    for _ in range(n_iter):
        coef = np.polyfit(kc, y, deg, w=np.sqrt(w))
        r = y - np.polyval(coef, kc)
        s = max(1.4826 * np.median(np.abs(r[w > 0])), 1e-7)
        u = r / (4.685 * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
    return coef, r, w


def clock_cost(k, toa, rx, z_src, seg, c=C_SOUND, prior=None, trim=0.9, miss_pen=30.0):
    """Hypothesis cost [m]: RMS of the lowest `trim` fraction of emission-time
    residuals about a smooth clock per segment (x c); pings without a real
    fix count with miss_pen each (they are not trimmed away)."""
    p, e, ok = hyperbolic_fix(toa, rx, z_src, c, prior)
    res = []
    n_bad = int((~ok).sum())
    for s in np.unique(seg):
        m = (seg == s) & ok
        if m.sum() < 10:
            n_bad += int(m.sum())
            continue
        _, r, _ = robust_poly(k[m].astype(float), e[m])
        res.append(np.abs(r) * c)
    res = np.sort(np.concatenate(res)) if res else np.zeros(0)
    r2 = np.minimum(res[: max(int(trim * len(res)), 1)], miss_pen) ** 2
    cost = np.sqrt((r2.sum() + n_bad * miss_pen ** 2) / (len(r2) + n_bad))
    return float(cost), p, e, ok


def build_table(D, lags):
    """Common index table: g = k_r + lag_r. Returns (g, toa (G, 3), idx (G, 3))."""
    gs = [D[n]["k"] + L for n, L in zip(NAMES, lags)]
    g = np.unique(np.concatenate(gs))
    toa = np.full((len(g), 3), np.nan)
    idx = np.full((len(g), 3), -1)
    for j, n in enumerate(NAMES):
        pos = np.searchsorted(g, gs[j])
        toa[pos, j] = D[n]["t"]
        idx[pos, j] = np.arange(len(gs[j]))
    return g, toa, idx


def search_offset(D, lags, rx, z_src, run_of, b0, span=1.0, step=0.004, c=C_SOUND, prior=None):
    """1-D search of the USBL offset b for fixed lags. D[n]['t'] for the USBL
    is its own clock. Returns (cost, b, extras)."""
    g, toa_raw, _ = build_table(D, lags)
    full = np.all(np.isfinite(toa_raw), axis=1)
    g3, toa3 = g[full], toa_raw[full]
    seg = run_of(toa3[:, 0])

    def cost(b):
        t = toa3.copy()
        t[:, 2] -= b
        return clock_cost(g3, t, rx, z_src, seg, c, prior)[0]

    grid = b0 + np.arange(-span, span + 1e-12, step)
    cg = np.array([cost(b) for b in grid])
    j = int(np.argmin(cg))
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(cost, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]),
                        method="bounded", options={"xatol": 1e-7})
    return r.fun, r.x, dict(grid=grid, cost_grid=cg, n=len(g3))


def load_detections(out_dir):
    D = {}
    for n in NAMES:
        z = np.load(out_dir / f"detections_{n}.npz", allow_pickle=True)
        det = z["det"]
        m = np.isfinite(det["t_rx"])
        D[n] = dict(det=det[m], t=det["t_rx"][m], k=det["k"][m].astype(int), t_ch=z["t_ch"][m],
                    offset=float(z["offset"]))
    return D


def _var_d2(x, g):
    """Robust variance of second differences / 6 (= per-sample variance for white noise)."""
    d, _ = d2_series(x, g)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return np.nan
    return (1.4826 * np.median(np.abs(d - np.median(d)))) ** 2 / 6


def pair_sigmas(toa, g):
    """Per-receiver TOA sigma from pairwise TDOA second differences (the
    common emission jitter cancels) and the common jitter sigma:
    var(d2 TDOA_ab)/6 = s_a^2 + s_b^2, var(d2 TOA_a)/6 = s_j^2 + s_a^2."""
    v = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        m = np.isfinite(toa[:, a]) & np.isfinite(toa[:, b])
        v[(a, b)] = _var_d2(toa[m, a] - toa[m, b], g[m])
    s2 = np.array([v[(0, 1)] + v[(0, 2)] - v[(1, 2)],
                   v[(0, 1)] + v[(1, 2)] - v[(0, 2)],
                   v[(0, 2)] + v[(1, 2)] - v[(0, 1)]]) / 2
    vj = [_var_d2(toa[np.isfinite(toa[:, a]), a], g[np.isfinite(toa[:, a])]) - s2[a] for a in range(3)]
    return np.sqrt(np.maximum(s2, 0)), float(np.sqrt(max(np.nanmedian(vj), 0)))


def usbl_cost_fn(D, lags, rx, z_src, run_of, c=C_SOUND, prior=None, run=None, t_ref=500.0):
    """cost(b, drift_ppm) of the emission-clock fit for fixed lags, with
    usbl_time = buoy_time + b + drift (usbl_time - t_ref); optionally one run."""
    g, toa_raw, _ = build_table(D, lags)
    full = np.all(np.isfinite(toa_raw), axis=1)
    g3, toa3 = g[full], toa_raw[full]
    seg = run_of(toa3[:, 0])
    if run is not None:
        g3, toa3, seg = g3[seg == run], toa3[seg == run], seg[seg == run]

    def cost(b, d=0.0):
        t = toa3.copy()
        t[:, 2] -= b + d * 1e-6 * (t[:, 2] - t_ref)
        return clock_cost(g3, t, rx, z_src, seg, c, prior)[0]
    return cost, len(g3)


def solve(D, rx, z_src, run_ends, b_guess, interval=1.0, lag_span=3, c=C_SOUND, verbose=True):
    """Association core. D[name] = dict(t (receiver clock), k (per-receiver
    index), det (structured, optional), t_ch (optional)) for NAMES; rx (3, 3)
    receiver xyz; run_ends: buoy-time end of each run. Returns the output dict.

    Integer lags: jitter fingerprint (geometry free); the emission-clock cost
    over all lag hypotheses is computed as a cross-check. The continuous USBL
    offset: 1-D search of the clock cost at the chosen lags (also per run,
    and with a clock drift, as consistency checks)."""
    from scipy.optimize import minimize, minimize_scalar
    run_of = lambda t: np.searchsorted(run_ends, t)
    prior = rx[:2, :2].mean(0)                       # sea side, between the buoys
    t902, k902 = D["Rx902"]["t"], D["Rx902"]["k"]
    L01 = nominal_lag(t902, k902, D["Rx01"]["t"], D["Rx01"]["k"])
    Lu = nominal_lag(t902, k902, D["USBL"]["t"] - b_guess, D["USBL"]["k"])
    cand = np.arange(-lag_span, lag_span + 1)

    # 1) jitter fingerprint
    jc01, n01 = jitter_corr(t902, k902, D["Rx01"]["t"], D["Rx01"]["k"], L01 + cand)
    jcu, nu = jitter_corr(t902, k902, D["USBL"]["t"], D["USBL"]["k"], Lu + cand)
    d01, z01 = fingerprint_lag(jc01, n01, cand)
    du, zu = fingerprint_lag(jcu, nu, cand)
    # cross-check the USBL against Rx01 too
    jcu1, nu1 = jitter_corr(D["Rx01"]["t"], D["Rx01"]["k"] + L01 + d01, D["USBL"]["t"], D["USBL"]["k"],
                            Lu + cand)
    du1, zu1 = fingerprint_lag(jcu1, nu1, cand)
    if verbose:
        print(f"nominal lags Rx01 {L01}, USBL {Lu} (coarse offset {b_guess} s)")
        print("fingerprint corr Rx01 vs Rx902:", dict(zip(cand.tolist(), np.round(jc01, 3).tolist())), f"z {z01:.1f}")
        print("fingerprint corr USBL vs Rx902:", dict(zip(cand.tolist(), np.round(jcu, 3).tolist())), f"z {zu:.1f}")
        print("fingerprint corr USBL vs Rx01: ", dict(zip(cand.tolist(), np.round(jcu1, 3).tolist())), f"z {zu1:.1f}")
    lags = (0, L01 + d01, Lu + du)

    # 2) emission-clock cost over all lag hypotheses (USBL offset searched for each)
    hyps = []
    for e01 in cand:
        for eu in cand:
            cst, b_h, ex = search_offset(D, (0, L01 + e01, Lu + eu), rx, z_src, run_of,
                                         b_guess - eu * interval, c=c, prior=prior)
            hyps.append((cst, int(e01), int(eu), b_h))
    hyps = np.array(hyps)
    h_sel = hyps[(hyps[:, 1] == d01) & (hyps[:, 2] == du)][0]
    h_oth = hyps[~((hyps[:, 1] == d01) & (hyps[:, 2] == du))]
    h_next = h_oth[np.argmin(h_oth[:, 0])]
    geo_best = hyps[np.argmin(hyps[:, 0])]
    b = float(h_sel[3])
    if verbose:
        for h in hyps[np.argsort(hyps[:, 0])][:6]:
            print(f"  clock cost d01 {int(h[1]):+d} du {int(h[2]):+d}: {h[0]:.3f} m at b {h[3]:.4f} s")
        print(f"chosen (fingerprint) d01 {d01:+d} du {du:+d}: cost {h_sel[0]:.3f} m, b {b:.5f} s; "
              f"best other d01 {int(h_next[1]):+d} du {int(h_next[2]):+d}: {h_next[0]:.3f} m")

    # 3) offset consistency: per run, and with a linear drift
    cost_fn, _ = usbl_cost_fn(D, lags, rx, z_src, run_of, c, prior)
    rd = minimize(lambda x: cost_fn(x[0], x[1]), [b, 0.0], method="Nelder-Mead",
                  options=dict(xatol=1e-6, fatol=1e-6, initial_simplex=[[b, 0], [b + 0.01, 0], [b, 20.0]]))
    b_runs = []
    for r in range(len(run_ends)):
        f_r, n_r = usbl_cost_fn(D, lags, rx, z_src, run_of, c, prior, run=r)
        if n_r > 30:
            rr = minimize_scalar(f_r, bounds=(b - 0.05, b + 0.05), method="bounded", options={"xatol": 1e-7})
            b_runs.append((r, rr.x, rr.fun, n_r))
    # sensitivity: cost increase of 10 % around the minimum
    db = np.linspace(-0.02, 0.02, 81)
    cv = np.array([cost_fn(b + x) for x in db])
    within = db[cv <= 1.1 * cv.min()]
    b_halfwidth = float((within.max() - within.min()) / 2) if len(within) else np.nan
    if verbose:
        print(f"USBL offset b {b:.5f} s (cost +10 % within +/- {b_halfwidth * 1e3:.1f} ms); "
              f"with drift: b {rd.x[0]:.5f} s at usbl t=500 s, drift {rd.x[1]:.1f} ppm, cost {rd.fun:.3f} m")
        print("  per run:", [(r, round(x, 5), round(f, 3), n) for r, x, f, n in b_runs])

    # 4) final table (rows seen by >= 2 receivers define the ping range)
    g, toa_raw, idx = build_table(D, lags)
    n_rx = np.isfinite(toa_raw).sum(1)
    g_first, g_last = g[n_rx >= 2].min(), g[n_rx >= 2].max()
    keep = (g >= g_first) & (g <= g_last)
    g, toa_raw, idx = g[keep], toa_raw[keep], idx[keep]
    toa = toa_raw.copy()
    toa[:, 2] -= b
    mask = np.isfinite(toa)
    full = mask.all(1)
    run = run_of(np.nanmin(toa, axis=1))
    _, pfix, efix, okfix = clock_cost(g[full], toa[full], rx, z_src, run[full], c, prior)
    clock, e_res = [], np.full(len(g), np.nan)
    rows = np.flatnonzero(full)
    for s_ in list(np.unique(run[full])) + ["all"]:
        m = okfix & ((run[full] == s_) if s_ != "all" else True)
        gk = g[full][m].astype(float)
        coef, r, w = robust_poly(gk, efix[m])
        clock.append(dict(run=-1 if s_ == "all" else int(s_), k_mean=gk.mean(), e_mean=coef[-1],
                          Tp=coef[1], quad=coef[0], jitter_rms=1.4826 * np.median(np.abs(r)), n=int(m.sum())))
        if s_ != "all":
            e_res[rows[m]] = r
    # Tp: emission-time slope between the first and the last 30 fixes (long
    # baseline; per-run polynomial slopes are spoilt by fixes with large GDOP)
    ge, ee = g[full][okfix], efix[okfix]
    nb = min(30, len(ge) // 3)
    Tp = float((np.median(ee[-nb:] - ge[-nb:]) - np.median(ee[:nb] - ge[:nb]))
               / (np.median(ge[-nb:]) - np.median(ge[:nb])) + 1.0) if nb >= 3 else interval
    sig, sig_j = pair_sigmas(toa, g)

    def col(field):
        a = np.full((len(g), 3), np.nan)
        for j, n in enumerate(NAMES):
            m = idx[:, j] >= 0
            if "det" in D[n]:
                a[m, j] = D[n]["det"][field][idx[m, j]]
        return a
    eta_used = 1 - col("rr_track") / C_SOUND
    rr_track = C_SOUND * (interval / (eta_used * Tp) - 1)    # nominal -> fitted Tp
    # the pulse shapes absorbed the nominal-Tp time scale, so the Doppler is
    # referred to the nominal Tp as well: same correction
    rr_dop = C_SOUND * (1 - (1 - col("rr_dop") / C_SOUND) * Tp / interval)
    pos, emis = np.full((len(g), 2), np.nan), np.full(len(g), np.nan)
    pos[full], emis[full] = pfix, efix
    tch = D["USBL"].get("t_ch", D["USBL"]["t"][:, None])
    usbl_ch = np.full((len(g), tch.shape[1]), np.nan)
    m = idx[:, 2] >= 0
    usbl_ch[m] = tch[idx[m, 2]] - b
    g0 = g.min()
    out = dict(receivers=np.array(NAMES), k=g - g0, toa=toa, toa_mask=mask, snr=col("snr_db"),
               range_rate=rr_dop, rr_mask=np.isfinite(rr_dop), rr_track=rr_track,
               rr_track_mask=np.isfinite(rr_track), run=run, Tp=Tp,
               usbl_offset=b, usbl_offset_margin=zu, usbl_offset_margin_rx01=zu1,
               usbl_offset_cost=h_sel[0], usbl_offset_cost_next=h_next[0],
               usbl_offset_cost_gap=h_next[0] - h_sel[0], usbl_offset_halfwidth=b_halfwidth,
               usbl_offset_drift=np.array([rd.x[0], rd.x[1], rd.fun]),
               usbl_offset_runs=np.array(b_runs), usbl_lag=lags[2], rx01_lag=lags[1],
               lag_offset_rx01=d01, lag_offset_usbl=du, rx01_margin=z01,
               geometry_choice_agrees=bool(geo_best[1] == d01 and geo_best[2] == du),
               toa_sigma=sig, toa_sigma_crlb=col("sigma_toa"), emission_jitter=sig_j,
               emission_fix=emis, emission_resid=e_res, pos_fix=pos, usbl_toa_ch=usbl_ch,
               det_index=idx, rx_xyz=rx, sound_speed=c, pinger_depth=-z_src, hypotheses=hyps,
               jitter_corr_rx01=jc01, jitter_corr_usbl=jcu, jitter_corr_usbl_rx01=jcu1,
               lag_candidates=cand,
               clock_runs=np.array([[c_["run"], c_["k_mean"] - g0, c_["e_mean"], c_["Tp"], c_["quad"],
                                     c_["jitter_rms"], c_["n"]] for c_ in clock]))
    if verbose:
        print(f"K = {len(g)} pings, all three {full.sum()}, per receiver {mask.sum(0)}")
        print(f"Tp {Tp:.7f} s; clock fits (run, Tp, resid rms us): "
              f"{[(c_['run'], round(c_['Tp'], 7), round(c_['jitter_rms'] * 1e6, 1)) for c_ in clock]}")
        print(f"TOA sigma per receiver [us] {np.round(sig * 1e6, 1)}, emission jitter {sig_j * 1e6:.1f} us")
    return out


def main(out_dir=config.ROOT / "out", verbose=True):
    tic = time.time()
    cfg = config.load()
    out_dir = Path(out_dir)
    D = load_detections(out_dir)
    rx = np.array([[*cfg.receiver(n).en, -cfg.receiver(n).depth] for n in NAMES])
    run_ends = [r.end for r in cfg.runs if r.pinger]
    out = solve(D, rx, -cfg.pinger.depth, run_ends, D["USBL"]["offset"], cfg.pinger.interval,
                c=float(cfg.priors["sound_speed"]), verbose=verbose)
    out["runtime_s"] = time.time() - tic
    np.savez(out_dir / "toa_phase1.npz", **out)
    if verbose:
        print(f"wrote {out_dir / 'toa_phase1.npz'} in {out['runtime_s']:.0f} s")
    return out


if __name__ == "__main__":
    main()
