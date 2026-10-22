"""Synthetic checks of processing/noise_bearing.py: a broadband plane wave on
the measured USBL geometry, with a wall echo, snapping-shrimp-like clicks,
independent channel noise and a zero-lag common-mode component -> windowed
GCC delays -> bearing within 1 deg."""
import numpy as np
import usbl
import noise_bearing as nb

FS = 96000
C = 1520.0
POS = np.array([[0.0, 0.0], [0.189, 0.0165], [0.313, 0.0], [0.505, 0.0]])   # like the calibrated array
WALL_H0, WALL_PHI = 1.25, np.radians(119.0)
WIN, PAD = 1.0, 0.05


def _fdelay(X, f, d):
    """Real signal with spectrum X delayed by d [s] (FFT phase)."""
    return np.fft.irfft(X * np.exp(-2j * np.pi * f * d), 2 * (len(X) - 1))


def _block(az, rng, snr_db=-6.0, wall=True, clicks=True, common_db=None):
    """Raw block (n, 4): window + PAD each side. Source = white noise from
    azimuth az (array frame); per-channel noise snr_db below/above it."""
    n = int(round((WIN + 2 * PAD) * FS)) // 2 * 2
    f = np.fft.rfftfreq(n, 1 / FS)
    S = np.fft.rfft(rng.normal(size=n))
    u = np.array([np.cos(az), np.sin(az)])
    lag = usbl.wall_lags(POS, np.array([az]), WALL_H0, WALL_PHI, C)[0]
    x = np.zeros((n, 4))
    for i in range(4):
        d = 0.002 - POS[i] @ u / C
        x[:, i] = _fdelay(S, f, d)
        if wall and np.isfinite(lag[i]):
            x[:, i] += 0.25 * _fdelay(S, f, d + lag[i])
    x += rng.normal(0, 10 ** (-snr_db / 20), x.shape)
    if common_db is not None:
        x += rng.normal(0, 10 ** (common_db / 20), n)[:, None]
    if clicks:
        uc = np.array([np.cos(az + 1.2), np.sin(az + 1.2)])
        burst = np.exp(-np.arange(20) / 4.0) * np.sin(2 * np.pi * 9e3 * np.arange(20) / FS)
        for t0 in rng.uniform(0.01, WIN + 2 * PAD - 0.01, 30):
            for i in range(4):
                k = int(round((t0 - POS[i] @ uc / C) * FS))
                x[k:k + 20, i] += 150 * burst
    return x


def _bearings(azs, rng, remove_crosstalk=False, **kw):
    out = [nb.block_spectra(_block(a, rng, **kw), FS, WIN, pad=PAD) for a in azs]
    S, P, frac, _ = (np.array(v) for v in zip(*out))
    f, band = nb.band_bins(FS, (2e3, 28e3))
    return nb.delays_and_bearings(S, P, f[band], POS, C, remove_crosstalk=remove_crosstalk), frac


def test_plane_wave_wall_clicks():
    rng = np.random.default_rng(0)
    azs = np.radians([40.0, 70.0, 90.0, 115.0, 150.0])
    B, frac = _bearings(azs, rng)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (B["az"] - azs)))))
    assert err.max() < 1.0, err
    assert (frac > 0).all()                                   # clicks were blanked
    tau_true = usbl.plane_delays(POS, azs, C)
    assert np.abs(B["tau"] - tau_true).max() < 5e-6           # direct-path delays, not the wall cross terms
    assert (B["rms_mirror"] > 2 * B["rms"]).all()             # side resolved by the ch1 offset


def test_common_mode_removed():
    rng = np.random.default_rng(1)
    azs = np.radians(np.linspace(45.0, 150.0, 8))
    kw = dict(snr_db=-12.0, common_db=-4.0, clicks=False)
    B, _ = _bearings(azs, rng, remove_crosstalk=True, **kw)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (B["az"] - azs)))))
    assert err.max() < 1.0, err


def test_blank_clicks_keeps_continuous_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(FS, 4))
    x[int(0.3 * FS):int(0.4 * FS)] *= 30                      # a 100 ms ping-like burst: kept (mostly)
    x[int(0.7 * FS):int(0.7 * FS) + 10, 2] += 200             # a click on one channel: blanked on all
    y, frac = nb.blank_clicks(x, FS)
    assert np.all(y[int(0.7 * FS):int(0.7 * FS) + 10] == 0)
    assert np.count_nonzero(y[int(0.3 * FS):int(0.4 * FS), 0]) > 0.7 * 0.1 * FS
    assert frac < 0.05


def test_usbl_time():
    t = np.array([52.14, 1052.14])
    tu = nb.usbl_time(t, 102.821832, -8.74094e-5, 52.14)
    assert np.allclose(tu - t, [102.821832, 102.821832 - 0.0874094])
