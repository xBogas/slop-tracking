"""
Planar motion models for the tracked emitter (pinger or boat).

1. Towed pinger - the pinger dragged on a line behind the stern (tractrix),
   and its inverse (pinger track -> tow point -> boat centre).

2. ct_* functions - a kinematic coordinated-turn (CT) model with no inputs.
   This is what the tracker uses, because throttle and rudder are never
   observed in our problem.

Frames: world is East-North (x east, y north), heading psi is measured
from +x (east), counter-clockwise positive (maths convention).
"""

import numpy as np

# ----------------------------------------------------------------------------
# Towed pinger: a body dragged on a line behind the stern (tractrix)
# ----------------------------------------------------------------------------


def towed_body_track(S, tow_len_h=5.0, stern_offset=None, L=6.0):
    """
    Horizontal track of a body towed behind the boat. S [N, >=3] holds the
    boat's x, y and heading psi in its first three columns.

    The body is pulled along the line towards the tow point (stern) and
    stays at horizontal distance tow_len_h once the line is taut; when the
    boat slows the line goes slack and the body stops (no drift modelled).
    This is the pure-pursuit / tractrix approximation, fine for a light
    buoy + pinger at ~10 m depth with a short line.

    Returns P [N, 2] pinger horizontal positions.
    """
    if stern_offset is None:
        stern_offset = L / 2
    x, y, psi = S[:, 0], S[:, 1], S[:, 2]
    tow = np.column_stack(
        [x - stern_offset * np.cos(psi), y - stern_offset * np.sin(psi)]
    )
    P = np.zeros_like(tow)
    P[0] = tow[0] - tow_len_h * np.array([np.cos(psi[0]), np.sin(psi[0])])
    for k in range(1, len(tow)):
        d = tow[k] - P[k - 1]
        dist = np.hypot(*d)
        P[k] = P[k - 1] + d * (1 - tow_len_h / dist) if dist > tow_len_h else P[k - 1]
    return P


def _unit_hold(V, v_min):
    """Unit vectors of V (N, 2); where |V| < v_min, hold the last valid
    direction (the first valid one before it). NaN if none is valid."""
    V = np.asarray(V, dtype=float)
    sp = np.hypot(V[:, 0], V[:, 1])
    ok = sp >= v_min
    u = np.full_like(V, np.nan)
    u[ok] = V[ok] / sp[ok, None]
    if ok.any():
        idx = np.where(ok, np.arange(len(V)), 0)
        idx = np.maximum.accumulate(idx)            # last valid index so far
        idx[: np.argmax(ok)] = np.argmax(ok)        # back-fill the start
        u = u[idx]
    return u


def tow_point_from_body(P, V, L=5.0, v_min=0.3):
    """
    Invert `towed_body_track`: tow point (stern) from the towed body track.

    In the pure-pursuit / tractrix model a taut line always lies along the
    towed body's velocity (the body moves straight towards the tow point),
    so the tow point is  Q = P + L V/|V|. P, V are (N, 2) body positions and
    velocities over ground (e.g. the phase 1 tracker's pinger estimate).
    Where |V| < v_min (stopped, or slack line) the last direction is held;
    the line is then not taut, so Q is only known to within L there.
    """
    return np.asarray(P, dtype=float) + L * _unit_hold(V, v_min)


def boat_centre_from_tow(Q, VQ, stern_offset=3.0, v_min=0.3):
    """Boat centre = tow point + stern_offset along the tow point's course.
    VQ is the tow point's ground velocity (e.g. np.gradient(Q, t, axis=0)).
    Uses course over ground for heading, so crab angle (current, sideslip in
    turns) gives an error of about stern_offset * crab angle."""
    return np.asarray(Q, dtype=float) + stern_offset * _unit_hold(VQ, v_min)


def boat_from_towed_body(P, V, t, L=5.0, stern_offset=3.0, v_min=0.3):
    """Convenience: towed body track (P, V at times t) -> (Q, C) = tow
    point and boat centre tracks."""
    Q = tow_point_from_body(P, V, L, v_min)
    VQ = np.gradient(Q, np.asarray(t, dtype=float), axis=0)
    return Q, boat_centre_from_tow(Q, VQ, stern_offset, v_min)


# ----------------------------------------------------------------------------
# Kinematic coordinated-turn model for tracking
# ----------------------------------------------------------------------------
#
# State z = [x, y, chi, U, omega]
#   chi   course over ground [rad] (absorbs heading + sideslip + current)
#   U     speed over ground [m/s]
#   omega turn rate [rad/s]
# Driven by white noise on dU/dt (sigma_a) and d omega/dt (sigma_w).
# Note: chi/U/omega describe the TRACKED EMITTER (pinger in phase 1, boat
# noise centre in phase 2), not necessarily the boat's bow.


def ct_predict(z, dt):
    """Exact discrete CT transition. z is (5,) or (..., 5) (vectorized).

    Written with sinc-type terms, so it is smooth through omega = 0 (no
    branch): with a = omega dt, S = sin(a)/a and C = (1 - cos a)/a,
      x+ = x + U dt (S cos chi - C sin chi)
      y+ = y + U dt (S sin chi + C cos chi)
    This also keeps finite-difference derivatives w.r.t. omega correct at
    omega ~ 0, where a branch on |omega| would give zero."""
    z = np.asarray(z, dtype=float)
    x, y, chi, U, w = np.moveaxis(z, -1, 0)
    a = w * dt
    S = np.sinc(a / np.pi)
    C = 0.5 * a * np.sinc(a / (2 * np.pi)) ** 2
    c, s = np.cos(chi), np.sin(chi)
    x2 = x + U * dt * (S * c - C * s)
    y2 = y + U * dt * (S * s + C * c)
    return np.stack([x2, y2, chi + a, U, w], axis=-1)


def ct_jacobian(z, dt, eps=1e-6):
    """Numerical Jacobian of ct_predict (good enough for an EKF)."""
    F = np.zeros((5, 5))
    f0 = ct_predict(z, dt)
    for i in range(5):
        dz = np.zeros(5)
        dz[i] = eps
        F[:, i] = (ct_predict(z + dz, dt) - f0) / eps
    return F


def ct_process_noise(z, dt, sigma_a=0.5, sigma_w=np.deg2rad(3.0)):
    """
    Q for CT with piecewise-constant random along-track acceleration
    (std sigma_a [m/s^2]) and turn acceleration (std sigma_w [rad/s^2])
    over each step; position and course noise follow by integration.
    z is (5,) or (..., 5); returns (5, 5) or (..., 5, 5).
    """
    chi = np.asarray(z, dtype=float)[..., 2]
    c, s = np.cos(chi), np.sin(chi)
    # Noise enters as accel along track (a) and turn-rate accel (alpha)
    G = np.zeros(chi.shape + (5, 2))
    G[..., 0, 0] = 0.5 * dt**2 * c
    G[..., 1, 0] = 0.5 * dt**2 * s
    G[..., 2, 1] = 0.5 * dt**2
    G[..., 3, 0] = dt
    G[..., 4, 1] = dt
    Qc = np.diag([sigma_a**2, sigma_w**2])
    return G @ Qc @ np.swapaxes(G, -1, -2)


def ct_track(z0, U, omega, dt):
    """Integrate the CT model through given speed and turn-rate histories
    U, omega [N] (sampled every dt) from z0 = [x, y, chi, ...]; Z [N, 5]."""
    Z = np.zeros((len(U), 5))
    Z[0] = [z0[0], z0[1], z0[2], U[0], omega[0]]
    for k in range(len(U) - 1):
        Z[k + 1] = ct_predict(Z[k], dt)
        Z[k + 1, 3:] = U[k + 1], omega[k + 1]
    return Z
