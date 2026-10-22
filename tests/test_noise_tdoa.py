"""Synthetic tests for processing/noise_tdoa.py: envelope TDOA of modulated
broadband noise through multipath, clicks and clutter."""
from dataclasses import replace
import numpy as np
import pytest

import noise_tdoa as nt

FS = 16000
P = replace(nt.NOISE, bands=((2e3, 3e3), (3e3, 4.5e3), (4.5e3, 6.5e3)))


def _modulated_noise(rng, n, depth=0.9):
    """Broadband noise with a random 2-200 Hz amplitude modulation (oversampled
    by 2 so time-varying delays can be applied by interpolation)."""
    f = np.fft.rfftfreq(n, 1 / FS)
    M = np.fft.rfft(rng.standard_normal(n)) * ((f > 2) & (f < 200))
    m = np.fft.irfft(M, n)
    m = np.tanh(m / m.std())
    return rng.standard_normal(n) * (1 + depth * m)


def _receive(rng, src, delay, paths, snr_db, n_clicks):
    """src delayed by delay(t) (array), plus multipath copies (extra delay [s], gain),
    white noise at snr_db and impulsive clicks."""
    n = len(src)
    t = np.arange(n) / FS
    x = np.zeros(n)
    for d_extra, g in [(0.0, 1.0)] + paths:
        x += g * np.interp(t - delay - d_extra, t, src, left=0, right=0)
    x += rng.standard_normal(n) * x.std() * 10 ** (-snr_db / 20)
    for i in rng.integers(0, n - 40, n_clicks):           # 2.5 ms shrimp-like clicks
        x[i:i + 40] += 30 * x.std() * rng.standard_normal(40) * np.exp(-np.arange(40) / 8)
    return x


def _streams(xs, p=P):
    out = {}
    for name, x in zip(nt.NAMES, xs):
        env, ok = nt.envelopes(x, FS, p)
        out[name] = (env, ok)
    return out


@pytest.fixture(scope="module")
def constant_case():
    rng = np.random.default_rng(1)
    T = 30.0
    n = int(T * FS)
    src = _modulated_noise(rng, n)
    d = {"Rx902": 0.300, "Rx01": 0.180, "USBL": 0.050}     # tdoa01 0.12, tdoa02 0.25, tdoa12 0.13
    # surface/bottom-like copies, resolvable at the envelope bandwidth (>= 8 ms); closer
    # copies bias the envelope peak toward the energy centroid (see the module notes)
    paths = {"Rx902": [(0.009, 0.6), (0.021, 0.4)], "Rx01": [(0.012, 0.6), (0.026, 0.4)],
             "USBL": [(0.008, 0.5), (0.017, 0.5)]}
    xs = [_receive(rng, src, np.full(n, d[k]), paths[k], 0.0, 150) for k in nt.NAMES]
    res = nt.solve(_streams(xs), 0.0, T, P)
    truth = np.array([d["Rx902"] - d["Rx01"], d["Rx902"] - d["USBL"], d["Rx01"] - d["USBL"]])
    return res, truth


def test_constant_tdoa_all_pairs(constant_case):
    res, truth = constant_case
    m = res["tdoa_mask"]
    ok = (res["t"] > 4) & (res["t"] < 26)                  # full stack support
    assert m[ok].mean() > 0.9
    err = np.abs(res["tdoa"] - truth)[m & ok[:, None]]
    assert np.median(err) < 0.5e-3
    assert np.max(err) < 2e-3
    r, _ = nt.closure_stats(res)
    assert len(r) > 10 and np.max(np.abs(r)) < 2e-3
    assert np.all(res["tdoa_sig"][m] > 0)


def test_viterbi_follows_drift_through_clutter():
    """Buoy pair with a delay drifting at 3 ms/s, louder 0.6 s clutter bursts at random
    lags and a 5 s gap in the source: the trace stays on the drift."""
    rng = np.random.default_rng(2)
    T = 40.0
    n = int(T * FS)
    t = np.arange(n) / FS
    src = _modulated_noise(rng, n)
    src[(t > 18) & (t < 23)] *= 0.0                        # source silent
    xa = _receive(rng, src, np.full(n, 0.20), [(0.010, 0.5)], 3.0, 100)
    xb = _receive(rng, src, 0.20 - (-0.10 + 0.003 * t), [(0.014, 0.5)], 3.0, 100)
    # clutter: 0.6 s bursts of other, louder modulated sources at random lags
    for t0 in np.arange(3.0, 38.0, 4.5):
        lag = rng.uniform(-0.25, 0.25)
        b = _modulated_noise(rng, n) * 1.5 * ((t > t0) & (t < t0 + 0.6))
        xa += np.interp(t - 0.25, t, b, left=0)
        xb += np.interp(t - 0.25 + lag, t, b, left=0)
    ea, oa = nt.envelopes(xa, FS, P)
    eb, ob = nt.envelopes(xb, FS, P)
    tc = np.arange(P.win / 2, T - P.win / 2 + 1e-9, P.step)
    m = int(P.win * P.fs_env)
    L = int(0.3 * P.fs_env)
    idx = np.round((tc - P.win / 2) * P.fs_env).astype(int)
    Z = nt.zscore(np.array([nt.correlogram(ea[:, i:i + m], eb[:, i:i + m], oa[i:i + m], ob[i:i + m], P, L)
                            for i in idx]))
    lags = np.arange(-L, L + 1) / P.fs_env
    ex = nt.extract(Z, lags, np.ones(len(tc), bool), P)
    truth = -0.10 + 0.003 * tc
    # the clutter wins the single-window peak in some windows
    assert np.sum(np.abs(lags[np.argmax(Z, axis=1)] - truth) > 0.02) >= 3
    det = ex["det"]
    loud = (tc > 4) & (tc < 36) & ((tc < 16) | (tc > 25))  # windows fully on the source
    assert det[loud].mean() > 0.85
    tau, _, sig, slope = nt.measure_warped((ea, oa), (eb, ob), idx, tc, lags[ex["path"]], det,
                                           P.refine, L, P)
    assert np.median(np.abs(slope[det] - 0.003)) < 0.3e-3
    err = np.abs(tau - truth)
    assert np.median(err[det]) < 0.5e-3
    assert np.max(err[det]) < 3e-3
    # through the gap the hidden path coasts near the drift
    gap = (tc > 19) & (tc < 22)
    assert np.all(np.abs(lags[ex["path"][gap]] - truth[gap]) < 0.05)


def test_viterbi_slew_limit():
    """A synthetic correlogram with a ramp and a stronger jump: the path keeps the ramp."""
    K, L = 60, 401
    Z = np.random.default_rng(3).standard_normal((K, L))
    truth = (100 + 2 * np.arange(K)).astype(int)
    Z[np.arange(K), truth] += 8
    Z[30:34, 350] += 30                                     # strong, but out of slew reach
    path = nt.viterbi(Z, np.ones(K, bool), 3.0, 4)
    assert np.all(np.abs(path - truth) <= 1)


def test_refine_peak_subsample():
    x = np.arange(200)
    for true in (100.3, 87.75, 120.5):
        z = 10 * np.exp(-0.5 * ((x - true) / 6.0) ** 2)
        pos, peak, w = nt.refine_peak(z, int(true) + 3, 8)
        assert abs(pos - true) < 0.05
        assert 4 < w < 8
