"""Association tests on simulated detection lists."""
import numpy as np

import associate

C = 1520.0
# receiver geometry like the site (ENU m, z up): Rx902, Rx01, USBL
RX = np.array([[-338.5, -365.0, -10.0], [-109.8, -702.8, -10.0], [0.0, 0.0, -2.0]])


def track(n=700):
    """Straight leg to the south-west, then a zig-zag back, 3 m/s, in the sea
    south-west of the shore receiver."""
    t = np.arange(n, dtype=float)
    hdg = np.where(t < 220, np.radians(-120), np.radians(30) + np.radians(50) * np.sign(np.sin(2 * np.pi * (t - 220) / 80)))
    return np.cumsum(np.where(t < 220, 3.0, 1.2)[:, None] * np.c_[np.cos(hdg), np.sin(hdg)], axis=0) \
        + np.array([-50.0, -120.0])


def simulate(b_true=103.05, jitter=1e-4, noise=2e-5, miss=0.05, seed=1):
    rng = np.random.default_rng(seed)
    p = track()
    n = len(p)
    xyz = np.c_[p, np.full(n, -10.0)]
    e = 37.25 + 1.0000 * np.arange(n) + rng.normal(0, jitter, n)
    R = np.linalg.norm(xyz[:, None, :] - RX[None], axis=2)
    toa = e[:, None] + R / C + rng.normal(0, noise, (n, 3))
    D, truth = {}, {}
    for j, name in enumerate(associate.NAMES):
        keep = rng.random(n) > miss
        t = toa[keep, j] + (b_true if name == "USBL" else 0.0)
        ids = np.flatnonzero(keep)
        fa = rng.uniform(t[0], t[-1], 4)             # a few false alarms
        tt = np.r_[t, fa]
        idd = np.r_[ids, -np.ones(4, int)]
        o = np.argsort(tt)
        tt, idd = tt[o], idd[o]
        k, ok = associate.index_pings(tt, 1.0)
        D[name] = dict(t=tt[ok], k=k[ok])
        truth[name] = idd[ok]
    tdoa = np.max(np.abs(toa[:, 2] + b_true - 102.7 - toa[:, :2].T), axis=0)
    return D, truth, tdoa


def test_association_resolves_lags_and_offset():
    b_true = 103.05                                  # coarse guess 102.7 is 0.35 s off
    D, truth, tdoa = simulate(b_true)
    assert tdoa.max() > 0.6                          # apparent cross-receiver delays up to > 0.6 s
    out = associate.solve(D, RX, -10.0, [420.0, 900.0], 102.7, 1.0, verbose=False)
    assert abs(out["usbl_offset"] - b_true) < 1e-3
    assert out["usbl_offset_margin"] > 3
    # every row holds TOAs of one and the same true ping on all receivers (no false alarms)
    idx = out["det_index"]
    ids = np.full(idx.shape, -2)
    for j, name in enumerate(associate.NAMES):
        m = idx[:, j] >= 0
        ids[m, j] = truth[name][idx[m, j]]
    assert np.all(ids != -1)
    for r in ids:
        assert len(set(r[r >= 0].tolist())) == 1
    assert (idx >= 0).sum() >= 0.9 * 3 * 700
    assert abs(out["Tp"] - 1.0) < 1e-5
