"""
Run configuration and local geometry.

Loads a TOML run file (see forfun.toml) into plain dataclasses and converts
receiver lat/lon into a local East-North frame (metres) around the site
origin. The conversion is the exact WGS84 tangent plane (ECEF -> ENU), which
is sub-millimetre over a few km.

Run: python3 config.py [forfun.toml]   -> prints the receiver geometry
"""
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# WGS84
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def _ecef(lat, lon, h=0.0):
    lat, lon = np.radians(lat), np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - _E2) + h) * np.sin(lat)])


def _enu_rot(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0.0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def enu_from_llh(lat, lon, lat0, lon0):
    """Local (east, north) [m] of lat/lon around (lat0, lon0)."""
    d = _ecef(lat, lon) - _ecef(lat0, lon0)[:, None] if np.ndim(lat) else _ecef(lat, lon) - _ecef(lat0, lon0)
    return (_enu_rot(lat0, lon0) @ d)[:2]


def llh_from_enu(e, n, lat0, lon0):
    """Inverse of enu_from_llh (at sea level), by two fixed-point steps."""
    e, n = np.asarray(e, float), np.asarray(n, float)
    m_lat = 111132.954 - 559.822 * np.cos(2 * np.radians(lat0))
    m_lon = 111412.84 * np.cos(np.radians(lat0))
    lat, lon = lat0 + n / m_lat, lon0 + e / m_lon
    for _ in range(3):
        de, dn = enu_from_llh(np.atleast_1d(lat), np.atleast_1d(lon), lat0, lon0)
        lat = lat + (n - dn.reshape(n.shape)) / m_lat
        lon = lon + (e - de.reshape(e.shape)) / m_lon
    return lat, lon


@dataclass
class Receiver:
    name: str
    kind: str                     # "buoy" or "usbl"
    file: Path
    lat: float
    lon: float
    pos_sigma: float
    depth: float = None           # [m] below the surface; derived when height_above_seabed is given
    height_above_seabed: float = None   # [m]: depth = water depth at (lat, lon) incl. tide - this
    en: np.ndarray = None         # local (east, north) [m]
    clock_offset_guess: float = 0.0   # receiver_time = buoy_time + offset
    element_spacing: list = None  # nominal gaps [m]; usbl.py self-calibrates the geometry
    wall_distance: list = None
    gps_offset: float = 0.0       # [m] GPS antenna -> receiver; `en` includes it
    gps_offset_bearing: float = 0.0   # [deg true]
    axis_bearing: float = None    # [deg magnetic] array axis, first -> last channel
    axis_bearing_sigma: float = None
    axis_enu: float = None        # the same as ENU azimuth [rad] (atan2(n, e)), declination applied


@dataclass
class Pinger:
    f0: float
    f1: float
    duration: float
    interval: float
    depth: float
    tow_length: float


@dataclass
class Run:
    name: str
    start: float
    end: float
    pinger: bool
    lat: float = None             # boat start position (organisers' log), if given
    lon: float = None
    en: np.ndarray = None         # local (east, north) [m] of (lat, lon)


@dataclass
class RunConfig:
    origin: tuple
    buoy_t0_local: str
    receivers: list
    pinger: Pinger
    runs: list
    priors: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)     # phase 2 settings ([noise])
    site: dict = field(default_factory=dict)      # [site] as read (date, tide, ...)

    def receiver(self, name):
        return next(r for r in self.receivers if r.name == name)

    def to_llh(self, e, n):
        return llh_from_enu(e, n, *self.origin)


def _depths_from_seabed(recs, raw, path):
    """Hydrophone depth for receivers moored at a height above the seabed: the EMODnet
    water depth (bathy.Bathymetry, charted + [site] tide) at the receiver's lat/lon,
    minus the height."""
    hs = [r for r in recs if r.height_above_seabed is not None]
    if not hs:
        return
    import bathy                                  # lazy: bathy imports config
    s = raw["site"]
    B = bathy.Bathymetry(RunConfig(origin=(s["origin_lat"], s["origin_lon"]), buoy_t0_local=s["buoy_t0_local"],
                                   receivers=[], pinger=None, runs=[], site=s))
    for r in hs:
        water = float(B.depth_ll(r.lat, r.lon)) + B.tide
        if r.depth is not None:
            raise ValueError(f"{path}: {r.name} gives both depth and height_above_seabed")
        r.depth = water - r.height_above_seabed
        if r.depth <= 0:
            raise ValueError(f"{path}: {r.name} {r.height_above_seabed} m above a {water:.1f} m seabed")


def load(path=ROOT / "processing" / "forfun.toml"):
    raw = tomllib.loads(Path(path).read_text())
    s = raw["site"]
    origin = (s["origin_lat"], s["origin_lon"])
    recs = []
    for r in raw["receiver"]:
        rx = Receiver(**{**r, "file": ROOT / r["file"]})
        b = np.radians(rx.gps_offset_bearing)
        rx.en = enu_from_llh(rx.lat, rx.lon, *origin) + rx.gps_offset * np.array([np.sin(b), np.cos(b)])
        if rx.axis_bearing is not None:
            rx.axis_enu = np.radians(90.0 - rx.axis_bearing - s.get("declination", 0.0))
        recs.append(rx)
    _depths_from_seabed(recs, raw, path)
    runs = [Run(**r) for r in raw["run"]]
    for r in runs:
        if r.lat is not None:
            r.en = enu_from_llh(r.lat, r.lon, *origin)
    return RunConfig(origin=origin, buoy_t0_local=s["buoy_t0_local"], receivers=recs,
                     pinger=Pinger(**raw["pinger"]), runs=runs,
                     priors=raw.get("priors", {}), noise=raw.get("noise", {}), site=s)


if __name__ == "__main__":
    cfg = load(*sys.argv[1:])
    for r in cfg.receivers:
        print(f"{r.name:6s} {r.kind:5s} E {r.en[0]:8.1f}  N {r.en[1]:8.1f}  depth {r.depth:4.2f}")
    b = [r.en for r in cfg.receivers if r.kind == "buoy"]
    print(f"buoy baseline {np.hypot(*(b[0] - b[1])):.1f} m")
