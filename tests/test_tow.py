"""Inverting the towed-pinger model: tow point and boat centre from the
pinger track of `towed_body_track` (the motion-model demo truth)."""
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from boat_model import towed_body_track, tow_point_from_body, boat_from_towed_body, ct_track

TRUTH = Path(__file__).resolve().parent.parent / "motion_model" / "demo_truth.csv"


def _truth():
    d = np.genfromtxt(TRUTH, delimiter=",", names=True)
    t = d["t"]
    P = np.column_stack([d["pinger_x"], d["pinger_y"]])
    C = np.column_stack([d["x"], d["y"]])
    head = np.column_stack([np.cos(d["psi"]), np.sin(d["psi"])])
    return t, P, C, C - 3.0 * head, np.abs(d["r"])     # stern = L/2 = 3 m behind centre


def test_recover_boat_from_pinger_track():
    t, P, C, tow, r = _truth()
    V = np.gradient(P, t, axis=0)
    Q, Cb = boat_from_towed_body(P, V, t, L=5.0, stern_offset=3.0)
    eq, ec = np.hypot(*(Q - tow).T), np.hypot(*(Cb - C).T)
    # straight: yaw rate < 0.5 deg/s over the last 5 s, and past start-up
    dt = t[1] - t[0]
    n = int(5 / dt)
    straight = (maximum_filter1d(r, 2 * n + 1, origin=n) < np.deg2rad(0.5)) & (t > 15)
    turn = ~straight & (t > 15)
    print(f"\nstraight legs ({straight.mean():.0%} of samples): tow point max {eq[straight].max():.3f} m, "
          f"boat centre max {ec[straight].max():.3f} m")
    print(f"turns: tow point median/95%/max {np.median(eq[turn]):.2f}/{np.percentile(eq[turn], 95):.2f}/"
          f"{eq[turn].max():.2f} m, boat centre {np.median(ec[turn]):.2f}/"
          f"{np.percentile(ec[turn], 95):.2f}/{ec[turn].max():.2f} m")
    assert straight.sum() > 500
    assert eq[straight].max() < 0.5 and ec[straight].max() < 0.5
    assert ec[turn].max() < 2.0          # turns: stern swings out, course != heading


def test_recover_on_1hz_tracker_rate():
    """At the ping rate (1 Hz, what the tracker delivers) straight legs still < 0.5 m."""
    t, P, C, tow, r = _truth()
    s = slice(None, None, 20)
    t1, P1, C1 = t[s], P[s], C[s]
    V1 = np.gradient(P1, t1, axis=0)
    _, Cb = boat_from_towed_body(P1, V1, t1)
    ec = np.hypot(*(Cb - C1).T)
    straight = (maximum_filter1d(r, 201, origin=100)[s] < np.deg2rad(0.5)) & (t1 > 15)
    print(f"\n1 Hz: boat centre straight max {ec[straight].max():.2f} m, all max {ec[t1 > 15].max():.2f} m")
    assert ec[straight].max() < 0.5


def test_fresh_simulation_and_stop_hold():
    """Independent run (no current, a stop at the end): line slack -> the
    last direction is held, no NaN."""
    t = np.arange(3001) * 0.05
    U = np.minimum(t / 5, 1.0) * 9.0 * np.exp(-np.maximum(t - 90, 0) / 5)   # coast to a stop
    w = np.where((t >= 40) & (t < 60), U / 34.0, 0.0)
    S = ct_track([0.0, 0.0, 0.3], U, w, 0.05)
    P = towed_body_track(S, tow_len_h=5.0, L=6.0)
    tow = S[:, :2] - 3.0 * np.column_stack([np.cos(S[:, 2]), np.sin(S[:, 2])])
    V = np.gradient(P, t, axis=0)
    Q = tow_point_from_body(P, V, L=5.0)
    assert np.isfinite(Q).all()
    leg = (t > 20) & (t < 40)
    assert np.hypot(*(Q - tow)[leg].T).max() < 0.05
    V0 = np.zeros((4, 2)); V0[1] = [0.0, 2.0]
    Q0 = tow_point_from_body(np.zeros((4, 2)), V0, L=5.0)
    assert np.allclose(Q0, [[0, 5]] * 4)                   # back-filled and held
