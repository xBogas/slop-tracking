"""Multipath ranging tests on synthetic pulse-compressed arrivals."""
import numpy as np
import pytest

import config
import detect
import multipath_range as mp

P = config.Pinger(f0=30e3, f1=40e3, duration=0.1, interval=1.0, depth=10.0, tow_length=5.0)
FS = 96000
N0 = int(round(mp.WIN[0] * FS))


@pytest.fixture(scope="module")
def pulse():
    """Pulse model with an Rx01-like measured shape: -16 dB roll-off toward
    40 kHz and ~2 rad of quadratic phase."""
    u = (np.arange(200) + 0.5) / 200
    w = np.interp(u, [0.05, 0.25, 0.5, 0.75, 0.95], [1.51, 1.71, 0.89, 0.47, 0.24]) \
        * np.exp(2j * (u - 0.5) ** 2 * 4)
    mf = detect.MatchedFilter(P, FS, int(0.25 * FS), shapes=w[:, None])
    half = int(0.015 * FS)
    tx = np.arange(-half * mp.UPS, half * mp.UPS + 1) / mp.UPS
    return mp.Pulse(mf.xamb(1.0, 1.0, half, 0)(tx))


def synth(pulse, R, zs, D, zr=10.0, amps=(1.0, -0.8, 0.5, -0.35, -0.35, 0.25), snr_db=30.0,
          t0=12e-6, seed=0):
    """y(t) = sum a_p X(t - tau_p) on the extraction grid, plus white noise
    (snr_db: direct-path peak power over noise power per sample)."""
    rng = np.random.default_rng(seed)
    n = int(round(mp.WIN[1] * FS)) - N0 + 1
    tau = (mp.rel_delays(R, zs, D, zr) + t0) * FS - N0
    a = np.asarray(amps, complex)
    y = mp.model_columns(pulse, np.arange(n).astype(float), tau) @ a
    sig = np.abs(y).max() ** 2 / 10 ** (snr_db / 10)
    noise = np.sqrt(sig / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    return y + noise, sig


def test_grid_energy_matches_direct_least_squares(pulse):
    y, _ = synth(pulse, 400.0, 7.0, 24.0)
    rel = mp.rel_delays(400.0, 7.0, 24.0, 10.0)[None] * FS
    t0 = np.array([-N0 + 12e-6 * FS])
    e = mp.grid_energy(y, pulse, rel, t0, ridge=0.0)[0, 0]
    G = mp.model_columns(pulse, np.arange(len(y)).astype(float), rel[0] + t0[0])
    a = np.linalg.lstsq(G, y, rcond=None)[0]
    assert abs(e - np.sum(np.abs(G @ a) ** 2)) < 0.02 * e


@pytest.mark.parametrize("R,zs,D,snr", [(380.0, 7.0, 24.0, 30.0), (700.0, 9.0, 28.0, 40.0)])
def test_full_image_model_recovers_geometry(pulse, R, zs, D, snr):
    """With all six image paths specular and visible, the coherent fit
    recovers R, zs and D (zr known). At 700 m it needs ~40 dB: at 30 dB 3 of
    8 noise draws land on the near-degenerate (zs, D, R) family instead (the
    double-bounce paths are what breaks the degeneracy, and only weakly)."""
    y, sig = synth(pulse, R, zs, D, snr_db=snr, seed=int(R))
    Rg = np.exp(np.arange(np.log(R * 0.6), np.log(R * 1.6), 0.002))
    zg = np.arange(zs - 3, zs + 3.01, 0.25)
    Dg = np.arange(D - 4, D + 4.01, 0.5)
    e, Rb = mp.ping_profile(y, pulse, FS, N0, 10.0, Rg, zg, Dg)
    iz, iD = np.unravel_index(np.argmax(e), e.shape)
    assert abs(zg[iz] - zs) <= 0.5
    assert abs(Dg[iD] - D) <= 1.0
    assert abs(Rb[iz, iD] / R - 1) < 0.03


def test_far_field_degeneracy_with_two_paths(pulse):
    """Direct + surface + bottom only: (R, zs, D) and the equivalent
    geometry for another pinger depth fit equally well (the real-data case)."""
    R, zs, D = 400.0, 7.0, 24.0
    amps = (1.0, -0.8, 0.5, 0, 0, 0)
    y, _ = synth(pulse, R, zs, D, amps=amps, snr_db=40.0)
    R2, D2 = mp.equivalent_geometry(R, zs, D, 10.0, 9.0)
    assert abs(R2 / R - 9.0 / 7.0) < 1e-12                 # 2 m of depth -> 29 % of range
    t0 = -N0 + 12e-6 * FS + np.arange(-2, 2.01, 0.25)
    rel = mp.rel_delays(np.array([R, R2]), np.array([zs, 9.0]), np.array([D, D2]), 10.0)[:, :3] * FS
    e = mp.grid_energy(y, pulse, rel, t0, amax=np.array([1, 2, 2.0])).max(1)
    assert abs(e[1] / e[0] - 1) < 2e-3


def test_ridge_tracker_follows_late_arrival():
    """A late arrival tau = K / R(k) over clutter is tracked without geometry."""
    rng = np.random.default_rng(1)
    K, n = 120, 900
    t = (np.arange(n) + N0) / FS
    Rk = 400 + 60 * np.sin(np.arange(K) / 25)
    tau = 0.33 / Rk
    As = 0.3 * np.exp(-t / 0.4e-3)[None, :] * rng.uniform(0.5, 1.5, (K, n))
    As += 0.5 * np.exp(-0.5 * ((t[None, :] - tau[:, None]) / 25e-6) ** 2)
    est, con = mp.track_ridge(As, t, np.ones(K, bool))
    assert np.median(np.abs(est / tau - 1)) < 0.01
    assert np.median(con) > 3
