"""Synthetic checks of processing/usbl.py: a 4-channel LFM with a known array
geometry, a wall echo and noise -> delays, bearing, self-calibration, wall."""
import numpy as np
import usbl

FS = 96000
C = 1520.0
POS = np.array([[0.0, 0.0], [0.19, 0.016], [0.313, 0.0], [0.505, 0.0]])   # like the real array
WALL_H0, WALL_PHI = 1.2, np.radians(120.0)


def _fdelay(x, d):
    """Fractional delay d [s] of a real signal via FFT phase."""
    n = len(x) * 2
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / FS)
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d), n)[: len(x)]


def _ping(az, rng, snr_db=30.0, wall=True, t0=0.003):
    """Compressed snippets (ch, n) of one ping from azimuth az (array frame)."""
    rep = usbl.lfm_replica(FS)
    n = int(0.12 * FS)
    p = np.zeros(n)
    p[: len(rep)] = rep.real
    u = np.array([np.cos(az), np.sin(az)])
    lag = usbl.wall_lags(POS, np.array([az]), WALL_H0, WALL_PHI, C)[0]
    x = np.zeros((n, 4))
    for i in range(4):
        x[:, i] = _fdelay(p, t0 - POS[i] @ u / C)
        if wall and np.isfinite(lag[i]):
            x[:, i] += 0.4 * _fdelay(p, t0 - POS[i] @ u / C + lag[i])
    sig = np.sqrt(np.mean(p[: len(rep)] ** 2))
    x += rng.normal(0, sig * 10 ** (-snr_db / 20) * np.sqrt(len(rep)) / 30, x.shape)
    return usbl.compress(x, rep)[: int(0.009 * FS)].T


def test_pair_delays_and_bearing():
    rng = np.random.default_rng(0)
    for az in np.radians([45.0, 90.0, 140.0]):
        y = _ping(az, rng)
        tau, q, tc, _ = usbl.pair_delays(y, FS, search=(int(0.0015 * FS), int(0.0045 * FS)))
        true = usbl.plane_delays(POS, np.array([az]), C)[0]
        assert np.abs(tau - true).max() < 1.5e-6, (np.degrees(az), (tau - true) * 1e6)
        a, rms, azm, rmsm, _ = usbl.bearing_ls(tau, POS, C)
        assert abs(np.degrees(a[0] - az)) < 0.3
        assert rmsm[0] > 3 * rms[0]          # mirror side rejected by the off-line element


def test_self_calibration_recovers_spacing():
    rng = np.random.default_rng(1)
    az = np.radians(np.r_[np.linspace(38, 155, 300)])
    tau = usbl.plane_delays(POS, az, C) + rng.normal(0, 0.5e-6, (len(az), 6))
    # cycle slips on 10 % of the pings, as on the real data
    tc = tau + rng.normal(0, 4e-6, tau.shape)
    slip = rng.random(len(az)) < 0.1
    tau_s = tau.copy()
    tau_s[slip, 2] += 1 / 35e3
    tau_s[slip, 4] += 1 / 35e3
    tau_s[slip, 5] += 1 / 35e3
    G = usbl.calibrate(tau_s, tc, C)
    pos = G["pos"]
    assert np.abs(pos - POS).max() < 0.01, pos
    assert G["ok"].mean() > 0.95
    assert np.abs(G["tau"][G["ok"]] - tau[G["ok"]]).max() < 3e-6


def test_wall_fit():
    rng = np.random.default_rng(2)
    az = np.radians(np.linspace(40, 155, 40))
    Y = np.array([_ping(a, rng) for a in az])
    te = np.array([usbl.channel_toa(y, FS, search=(int(0.0015 * FS), int(0.0045 * FS)))[0] for y in Y])
    lags = np.arange(-0.5e-3, 3e-3, 1 / (FS * 8))
    E = usbl.echo_stack(Y, te, FS, lags)
    W = usbl.fit_wall(E, lags, POS, az, C, h_grid=np.arange(0.3, 2.0, 0.01), phi_grid=np.radians(np.arange(0, 360, 1.0)))
    assert abs(W["h0"] - WALL_H0) < 0.03
    assert abs(np.degrees(W["phi"] - WALL_PHI)) < 2
