"""
Rebuild what is out of date, in order: the processing steps and the notebooks.

Each stage lists the config sections it reads, its input files and its
outputs; its code is the script (or notebook) plus every repo module it
imports. A stage is stale when it has never run, an output is missing, or
any of those changed since its last run (config compared after parsing, so
comments do not count; see stamp.py). Running a stage changes its outputs,
which makes the stages that read them stale in turn, so one pass in order
brings everything up to date. Stamps are kept in out/.stamps/<stage>.json.

Run: uv run python processing/pipeline.py              # rebuild stale stages
     uv run python processing/pipeline.py --status     # what is stale, and why
     uv run python processing/pipeline.py --dry-run    # what would run
     uv run python processing/pipeline.py nb_multipath phase2   # only these stages (if stale)
     uv run python processing/pipeline.py --force usbl  # rerun usbl, then whatever goes stale

--dry-run is a pass of --status: a stage after a stale one shows "ok" if
its inputs have not changed yet, although it will usually run.

Caches inside a stage (multipath arrivals and profiles, USBL pings, the
notebook's family grid) carry their own keys, so running a script by hand
never reuses a stale one either.
"""
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import stamp

ROOT = config.ROOT
OUT = ROOT / "out"
STAMPS = OUT / ".stamps"
ALL = ("site", "receiver", "pinger", "run", "priors", "noise")
# detect.py and the chirp notebook read only these (a 3.5 min rerun otherwise)
DETECT = ("receiver:name,kind,file,clock_offset_guess", "pinger:f0,f1,duration,interval",
          "run:start,end,pinger")


@dataclass
class Stage:
    name: str
    script: str                       # .py (run with args) or .ipynb (executed in place)
    inputs: list
    outputs: list
    args: list = field(default_factory=list)
    sections: tuple = ALL

    @property
    def path(self):
        return ROOT / self.script

    def parts(self):
        return stamp.parts(self.sections, files=[ROOT / f for f in self.inputs], code=self.path,
                           args=self.args)

    def command(self):
        if self.path.suffix == ".ipynb":
            return [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=3600", str(self.path)]
        return [sys.executable, str(self.path), *self.args]


def stages(cfg=None):
    cfg = cfg or config.load()
    wav = {r.name: str(r.file.relative_to(ROOT)) for r in cfg.receivers}
    buoys = [n for n in wav if n != "USBL"]
    det = [f"out/detections_{n}.npz" for n in wav]
    p1 = ["out/phase1/state.npz", "out/phase1/track.csv"]
    DTM = "data/bathymetry/emodnet_mean_vodice.tif"
    return [
        Stage("detect", "processing/detect.py", list(wav.values()), det, sections=DETECT),
        Stage("associate", "processing/associate.py", det, ["out/toa_phase1.npz"]),
        Stage("phase1_toa_only", "processing/run_phase1.py", ["out/toa_phase1.npz"],
              ["out/phase1_toa_only/track.csv", "out/phase1_toa_only/state.npz"], ["--no-bearing", "--no-validate"]),
        Stage("usbl", "processing/usbl.py",
              ["out/toa_phase1.npz", "out/detections_USBL.npz", wav["USBL"], "out/phase1_toa_only/track.csv"],
              ["out/usbl/bearings_phase1.npz"]),
        Stage("phase1", "processing/run_phase1.py", ["out/toa_phase1.npz", "out/usbl/bearings_phase1.npz"],
              p1 + ["out/phase1/validation.txt"]),
        Stage("phase1_fixes", "processing/run_phase1.py", ["out/toa_phase1.npz", "out/usbl/bearings_phase1.npz"],
              ["out/phase1_fixes/state.npz", "out/phase1_fixes/track.csv"], ["--fixes"]),
        Stage("multipath", "processing/multipath_range.py",
              ["out/toa_phase1.npz", *[f"out/detections_{n}.npz" for n in buoys], *[wav[n] for n in buoys], *p1],
              ["out/multipath/mp_range_phase1.npz"]),
        Stage("multipath_paths", "processing/multipath_paths.py",
              ["out/toa_phase1.npz", *[f"out/detections_{n}.npz" for n in buoys], *[wav[n] for n in buoys], *p1, DTM],
              ["out/multipath/paths_phase1.npz", "out/multipath/paths_phase1.csv"]),
        Stage("export", "processing/export.py",
              ["out/toa_phase1.npz", *p1, "out/usbl/bearings_phase1.npz", "out/multipath/paths_phase1.npz"],
              [*[f"out/export/first_arrival_{n}.csv" for n in wav], "out/export/usbl_bearing_range.csv",
               *[f"out/export/multipath_{n}.csv" for n in buoys], "out/export/README.md"]),
        Stage("noise_tdoa", "processing/noise_tdoa.py", [*wav.values(), "out/phase1/state.npz"],
              ["out/phase2/tdoa_noise.npz"]),
        Stage("noise_tdoa_pinger", "processing/noise_tdoa.py", [*wav.values(), *p1],
              ["out/phase2/tdoa_pinger_check.txt", "out/phase2/tdoa_pinger_check.npz"], ["--pinger"]),
        Stage("noise_bearing", "processing/noise_bearing.py",
              [wav["USBL"], "out/usbl/bearings_phase1.npz", "out/phase1/state.npz"],
              ["out/phase2/bearings_noise.npz"]),
        Stage("phase2", "processing/run_phase2.py",
              [*p1, "out/phase2/tdoa_noise.npz", "out/phase2/bearings_noise.npz", "out/phase2/tdoa_pinger_check.txt"],
              ["out/phase2/track.csv", "out/phase2/validation.txt"]),
        Stage("nb_chirp", "notebooks/chirp_timeline.ipynb", list(wav.values()),
              ["notebooks/chirp_timeline.ipynb"], sections=DETECT),
        Stage("nb_multipath", "notebooks/multipath_reflections.ipynb",
              ["out/toa_phase1.npz", *p1, "out/multipath/mp_range_phase1.npz", "out/phase2/track.csv",
               DTM, *[wav[n] for n in buoys]],
              ["notebooks/multipath_reflections.ipynb"]),
        Stage("nb_multipath_paths", "notebooks/multipath_paths.ipynb",
              ["out/multipath/paths_phase1.npz", "out/toa_phase1.npz", *p1, DTM,
               *[f"out/detections_{n}.npz" for n in buoys], *[wav[n] for n in buoys]],
              ["notebooks/multipath_paths.ipynb"]),
    ]


def _stamp_path(s):
    return STAMPS / f"{s.name}.json"


def why_stale(s):
    """Reasons the stage must run ([] if up to date)."""
    f = _stamp_path(s)
    if not f.exists():
        return ["never run"]
    missing = [o for o in s.outputs if not (ROOT / o).exists()]
    if missing:
        return [f"output {o} missing" for o in missing]
    return stamp.diff(json.loads(f.read_text()), s.parts())


def run_stage(s):
    p = s.parts()                                      # the inputs it runs on
    tic = time.time()
    print(f"== {s.name}: {' '.join(Path(a).name if i < 2 else a for i, a in enumerate(s.command()))}", flush=True)
    r = subprocess.run(s.command(), cwd=s.path.parent if s.path.suffix == ".ipynb" else ROOT)
    if r.returncode:
        raise SystemExit(f"{s.name} failed (exit {r.returncode}); later stages not run")
    STAMPS.mkdir(parents=True, exist_ok=True)
    _stamp_path(s).write_text(json.dumps(p, indent=1, sort_keys=True))
    print(f"== {s.name} done in {time.time() - tic:.0f} s", flush=True)


def main(argv):
    flags = {a for a in argv if a.startswith("--")}
    names = [a for a in argv if not a.startswith("--")]
    all_stages = stages()
    known = {s.name for s in all_stages}
    bad = [n for n in names if n not in known]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; stages: {', '.join(known)}")
    sel = [s for s in all_stages if s.name in names] if names and "--force" not in flags else all_stages
    forced = set(names) if "--force" in flags else set()
    if "--status" in flags or "--dry-run" in flags:
        for s in sel:
            r = ["forced"] if s.name in forced else why_stale(s)
            print(f"{s.name:18s} {'stale' if r else 'ok'}" + "".join(f"\n{'':20s}{x}" for x in r))
        return
    for s in sel:
        r = ["forced"] if s.name in forced else why_stale(s)
        if r:
            print(f"{s.name}: " + "; ".join(r))
            run_stage(s)

if __name__ == "__main__":
    main(sys.argv[1:])
