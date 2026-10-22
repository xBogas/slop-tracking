"""Range-dependent eigenray tests against the closed-form image method."""
import numpy as np

import bathy
import multipath_range as mp


def test_flat_bottom_matches_image_method():
    R, zs, D, zr = 400.0, 6.8, 13.0, 10.0
    rays = bathy.eigenrays(zs, zr, [0.0, R], [D, D], max_bounces=3)
    L = {bathy.family(r): r["L"] for r in rays}
    ref = dict(zip(("D", "S", "B", "SB", "BS", "SBS"), mp.path_delays(R, zs, D, zr) * mp.C_SOUND))
    for k, v in ref.items():
        assert abs(L[k] - v) < 1e-3, (k, L[k], v)


def test_sloping_bottom_single_bounce():
    # planar bottom z = z0 + s x: the bottom path is the straight line to the
    # mirror image of the source in that plane
    R, zs, zr, z0, s = 300.0, 4.0, 10.0, 8.0, 0.02
    rays = bathy.eigenrays(zs, zr, [0.0, R], [z0, z0 + s * R], max_bounces=1)
    B = [r for r in rays if bathy.family(r) == "B"][0]
    nrm = np.array([-s, 1.0]) / np.hypot(s, 1.0)
    p = np.array([0.0, zs])
    img = p + 2 * ((np.array([0.0, z0]) - p) @ nrm) * nrm
    assert abs(B["L"] - np.hypot(R - img[0], zr - img[1])) < 1e-3


def test_arrival_angles_flat_bottom():
    # flat bottom: each path arrives at its image-source elevation (positive down)
    R, zs, D, zr = 300.0, 3.0, 13.0, 10.0
    rays = {bathy.family(r): r for r in bathy.eigenrays(zs, zr, [0.0, R], [D, D])}
    for f, zi in (("D", zs), ("S", -zs), ("B", 2 * D - zs)):
        assert abs(rays[f]["arrival"] - np.degrees(np.arctan2(zr - zi, R))) < 0.01, f


def test_water_depth_includes_configured_tide():
    import config
    cfg = config.load()
    B = bathy.Bathymetry(cfg)
    e, n = np.array([-338.5, -109.8]), np.array([-365.0, -702.8])        # the buoys
    lat, lon = cfg.to_llh(e, n)
    assert B.tide == cfg.site["tide"]
    assert np.allclose(B.depth(e, n), B.depth_ll(lat, lon) + cfg.site["tide"])


def test_moored_hydrophone_depth_follows_seabed_and_tide():
    # depth below the surface = charted depth at the mooring + tide - height above the seabed
    import config
    cfg = config.load()
    B = bathy.Bathymetry(cfg)
    for r in cfg.receivers:
        if r.height_above_seabed is not None:
            water = float(B.depth_ll(r.lat, r.lon)) + cfg.site["tide"]
            assert abs(r.depth - (water - r.height_above_seabed)) < 1e-9
