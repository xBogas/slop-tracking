"""
Demo: straight-line run, then free manoeuvres, with a towed pinger.
A kinematic synthetic truth: scripted speed and path curvature (with lags),
integrated with the CT model, plus a constant water current.
Writes demo_trajectory.png and demo_truth.csv next to this file.
Run: uv run python motion_model/demo.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boat_model import towed_body_track, ct_track, ct_predict

HERE = Path(__file__).parent
DT, T_END = 0.05, 300.0
CURRENT = np.array([0.2, -0.1])   # water current [m/s]
TAU_U, TAU_W = 2.5, 1.0           # speed and turn-rate lags [s]
L, TOW = 6.0, 5.0                 # boat length, horizontal tow line [m]


def schedule(t):
    """Scripted speed through water [m/s] and curvature 1/radius [1/m]
    (+ port)."""
    if t < 70:   return 10.0, 0.0             # accelerate, straight run
    if t < 90:   return 8.6, 1 / 19           # turn to port
    if t < 110:  return 12.6, 0.0             # fast straight
    if t < 170:  # zig-zag
        return 10.4, np.sign(np.sin(2 * np.pi * (t - 110) / 12)) / 36
    if t < 190:  return 5.2, 0.0              # slow down
    if t < 240:  return 6.1, -1 / 11.6        # tight starboard circle
    return 8.9, np.sin(2 * np.pi * t / 30) / 57   # wander


def simulate():
    t = np.arange(int(round(T_END / DT)) + 1) * DT
    U, w = np.zeros(len(t)), np.zeros(len(t))
    for k in range(len(t) - 1):
        Uc, kappa = schedule(t[k])
        U[k + 1] = U[k] + DT / TAU_U * (Uc - U[k])
        w[k + 1] = w[k] + DT / TAU_W * (U[k] * kappa - w[k])
    Z = ct_track([0.0, 0.0, 0.0], U, w, DT)       # through the water
    xy = Z[:, :2] + CURRENT * t[:, None]
    return t, np.column_stack([xy, Z[:, 2], U, w])


def main():
    t, S = simulate()
    x, y, psi, U, r = S.T
    P = towed_body_track(S, tow_len_h=TOW, L=L)
    vg = np.column_stack([U * np.cos(psi), U * np.sin(psi)]) + CURRENT

    sep = np.hypot(*(S[:, :2] - P).T)
    print(f"boat-pinger horizontal separation: {sep.min():.1f}..{sep.max():.1f} m")

    # How well does a CT model (no inputs) predict the pinger 1 s ahead?
    Vp = np.gradient(P, t, axis=0)
    chi = np.unwrap(np.arctan2(Vp[:, 1], Vp[:, 0]))
    Zp = np.column_stack([P, chi, np.hypot(*Vp.T), np.gradient(chi, t)])
    k = int(1.0 / DT)
    err = np.hypot(*(ct_predict(Zp[200:-k], 1.0)[:, :2] - Zp[200 + k:, :2]).T)
    print(f"CT 1 s prediction error on pinger: median {np.median(err):.2f} m, "
          f"95% {np.percentile(err, 95):.2f} m")

    # Truth CSV for later tracker tests
    hdr = "t,x,y,psi,u,r,vx_ground,vy_ground,pinger_x,pinger_y"
    np.savetxt(HERE / "demo_truth.csv", np.column_stack([t, S, vg, P]),
               delimiter=",", header=hdr, comments="", fmt="%.4f")

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), width_ratios=[2, 1, 1])
    a = ax[0]
    a.plot(x, y, label="boat")
    a.plot(*P.T, "--", lw=1, label="towed pinger")
    for tm in range(0, 301, 30):
        a.annotate(f"{tm}s", S[int(tm / DT), :2], fontsize=7)
    a.plot(x[0], y[0], "ko")
    a.set_aspect("equal"); a.set_xlabel("east [m]"); a.set_ylabel("north [m]")
    a.set_title("Track"); a.legend(); a.grid(alpha=.3)

    a = ax[1]
    a.plot(t, U, label="speed through water")
    a.plot(t, np.hypot(*vg.T), label="speed over ground")
    a2 = a.twinx()
    a2.plot(t, np.rad2deg(r), "C2", lw=1, label="turn rate [deg/s]")
    a.set_title("Speed [m/s] and turn rate"); a.set_xlabel("t [s]"); a.grid(alpha=.3)
    a.legend(loc="upper left"); a2.legend(loc="upper right")

    a = ax[2]
    a.plot(t[200:-k], err)
    a.set_title("CT model 1 s-ahead error on pinger [m]")
    a.grid(alpha=.3); a.set_xlabel("t [s]")

    fig.tight_layout()
    fig.savefig(HERE / "demo_trajectory.png", dpi=110)


if __name__ == "__main__":
    main()
