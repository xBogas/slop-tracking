"""Detector tests on synthetic LFM pings (Doppler + multipath)."""
import numpy as np
import pytest

import config
import detect
import recordings
from associate import index_pings

P = config.Pinger(f0=30e3, f1=40e3, duration=0.1, interval=1.0, depth=10.0, tow_length=5.0)
FS = 96000


def synth(t0, rr, echoes, snr_inband_db=20.0, dur=1.0, seed=0, c=1520.0):
    """x(t) = sum a_i s(eta (t - t0 - d_i)) + white noise; returns a Recording."""
    rng = np.random.default_rng(seed)
    eta = 1 - rr / c
    t = np.arange(int(dur * FS)) / FS
    k = (P.f1 - P.f0) / P.duration
    x = np.zeros(len(t))
    for d, a in [(0.0, 1.0)] + list(echoes):
        u = eta * (t - t0 - d)
        x += a * np.cos(2 * np.pi * (P.f0 * u + 0.5 * k * u ** 2)) * ((u >= 0) & (u < P.duration))
    var = 0.5 / 10 ** (snr_inband_db / 10) / ((P.f1 - P.f0) / (FS / 2))
    x = 0.01 * (x + rng.normal(0, np.sqrt(var), len(t)))
    return recordings.Recording("syn", FS, x[:, None], 0.0)


# a sign-flipped echo inside 1/B (70 us < 100 us) and a later wall-like echo
ECHOES = [(70e-6, -0.5), (0.7e-3, 0.6)]


@pytest.mark.parametrize("rr", [-8.0, 0.0, 5.0, 12.0])
def test_single_ping_doppler_and_toa(rr):
    t0 = 0.4123456
    rec = synth(t0, rr, ECHOES, seed=int(rr) + 50)
    det, _ = detect.detect(rec, 0.0, 0.89, P)
    assert len(det) == 1
    assert abs(det["t_rx"][0] - t0) < 10e-6
    assert abs(det["rr_dop"][0] - rr) < 0.2


def test_eta_matched_toa_is_unbiased():
    """With the time scale known (arrival track), the TOA is the pulse start
    even at 10 dB in-band SNR, and the eta = 1 peak is biased by ~ Cc d_eta."""
    t0, rr = 0.3054321, 10.0
    rec = synth(t0, rr, ECHOES, snr_inband_db=10.0, seed=3)
    anchors = detect.coarse(rec, 0.0, 0.89, P, [0])
    assert len(anchors) == 1
    ta, _, sig = anchors[0]
    eta = 1 - rr / 1520.0
    mf = detect.MatchedFilter(P, FS, int(0.25 * FS))
    r = detect.refine(rec, ta, P, [0], sig, mf, eta_ref=eta, doppler=False)
    assert abs(r["t_rx"] - t0) < 10e-6
    # the eta = 1 peak is biased by ~ -Cc (eta - 1) (Cc = 0.4 s; ~0.45 s at 10 m/s)
    assert 0.38 < (ta - t0) / (-(eta - 1)) < 0.47


def test_coupling_small_mismatch():
    rr = 3.0
    eta = 1 - rr / 1520.0
    rec = synth(0.2, rr, [], snr_inband_db=30.0)
    ta = detect.coarse(rec, 0.0, 0.89, P, [0])[0][0]
    assert abs((ta - 0.2) - (-detect.coupling(P) * (eta - 1))) < 20e-6


def test_coarse_threshold_no_false_alarms():
    rng = np.random.default_rng(1)
    rec = recordings.Recording("noise", FS, 0.01 * rng.normal(size=(4 * FS, 1)), 0.0)
    assert len(detect.coarse(rec, 0.0, 3.8, P, [0])) == 0


def test_index_pings_misses_and_false_alarms():
    rng = np.random.default_rng(2)
    k_true = np.arange(300)
    rr = 8 * np.sin(k_true / 40)                     # m/s, slowly varying
    t = 5.0 + np.cumsum(1.0 * (1 + rr / 1520)) + rng.normal(0, 1e-4, 300)
    keep = rng.random(300) > 0.05
    t, k_true = t[keep], k_true[keep]
    t[100:130] = np.nan                              # a 30-ping gap
    m = np.isfinite(t)
    t, k_true = t[m], k_true[m]
    fa = rng.uniform(t[0], t[-1], 5)                 # false alarms
    tt = np.sort(np.r_[t, fa])
    is_true = np.isin(tt, t)
    k, ok = index_pings(tt, 1.0)
    assert np.all(ok[is_true]) and not np.any(ok[~is_true])
    assert np.all(np.diff(k[is_true] - k_true) == 0)
