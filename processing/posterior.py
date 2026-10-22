"""
Phase 1 posterior for the phase 2 steps: the USBL clock, sound speed, array
orientation and receiver positions from out/phase1/state.npz (run_phase1.py),
with the depths from the config. Read from the file each time it changes,
never copied into the code.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

STATE = config.ROOT / "out" / "phase1" / "state.npz"
NAMES = ("Rx902", "Rx01", "USBL")


def phase1(path=STATE, cfg=None):
    """dict: b (usbl_time - buoy_time at t_ref [s]), d_S (drift), t_ref, c,
    alpha (array axis ENU azimuth [rad]), pos {name: (e, n, z)} with z = -depth."""
    cfg = cfg or config.load()
    S = np.load(path)
    g = dict(zip([str(n) for n in S["names"]], S["globals"]))
    pos = {n: np.r_[S["buoys"][i], -cfg.receiver(n).depth] for i, n in enumerate(NAMES[:2])}
    u = cfg.receiver("USBL")
    pos["USBL"] = np.r_[u.en, -u.depth]
    return dict(b=float(S["usbl_offset"]) + g["b_S"], d_S=g["d_S"], t_ref=float(S["t0"]),
                c=g["c"], alpha=g["alpha"], pos=pos)
