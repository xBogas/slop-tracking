"""Direct-path pick, ping-to-ping scatter and Doppler budget of multipath_paths / multipath_predict."""
import numpy as np

import config
import detect
import multipath_paths as mpaths
import multipath_predict as mpp


def test_direct_is_earliest_above_level_not_strongest():
    # row 0: weak direct (-4 dB) then a stronger reflection; a -20 dB precursor is below the level
    # row 1: nothing in the window -> no direct path
    cmp = dict(row=np.array([0, 0, 0, 1]), tau=np.array([-0.05e-3, 0.0, 0.2e-3, 0.5e-3]),
               amp=np.array([0.1, 10 ** (-4 / 20), 1.0, 1.0]) + 0j)
    direct, rel = mpaths.pick_direct(cmp, centre=np.zeros(2), half=0.1e-3, thr_db=-9.0)
    assert direct.tolist() == [1, -1]
    assert np.allclose(rel, [-20, -4, 0, 0])


def test_scatter_recovers_white_noise_sigma():
    rng = np.random.default_rng(0)
    n, sig = 400, 7e-6
    pred = np.linspace(200e-6, 300e-6, n)                        # predicted drift is removed
    m = np.zeros(n, mpaths.MEAS_DTYPE)
    m["row"] = m["ping"] = np.arange(n)
    m["dtau"] = pred + rng.normal(0, sig, n)
    assert abs(mpaths.scatter(m, pred) / sig - 1) < 0.15
    assert np.isnan(mpaths.scatter(m[::2], pred))                 # no consecutive pings


def test_doppler_budget():
    cfg = config.load()
    c = 1520.0
    launch = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, -3.0, 4.0, -6.0, 8.0]])
    pred = dict(v_r=np.array([2.0, 2.0]), launch=launch)
    b = mpp.doppler_budget(pred, cfg, c)
    assert np.allclose(b["common"], detect.coupling(cfg.pinger) * 2.0 / c)
    assert np.allclose(b["residual"][0], 0.0)                    # same angle as D: no residual
    cos = np.cos(np.radians(launch[1]))
    assert np.allclose(b["residual"][1], detect.coupling(cfg.pinger) * 2.0 * (cos[1:] - cos[0]) / c)
