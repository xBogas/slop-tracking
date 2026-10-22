"""Input keys (stamp.py) and stage staleness (pipeline.py)."""
import numpy as np

import pipeline
import stamp

TOML = """
[pinger]
f0 = 30000.0   # comment
depth = 3.0

[[run]]
name = "run1"
start = 0.0
lat = 43.7
"""


def _cfg(tmp_path, text=TOML):
    f = tmp_path / "cfg.toml"
    f.write_text(text)
    return f


def test_key_follows_config_not_comments(tmp_path):
    f = _cfg(tmp_path)
    k0 = stamp.key(["pinger"], cfg_path=f)
    f.write_text(TOML.replace("# comment", "# other comment"))
    assert stamp.key(["pinger"], cfg_path=f) == k0
    f.write_text(TOML.replace("depth = 3.0", "depth = 4.0"))
    assert stamp.key(["pinger"], cfg_path=f) != k0


def test_field_selection(tmp_path):
    f = _cfg(tmp_path)
    spec = ["pinger:f0", "run:start"]
    k0 = stamp.key(spec, cfg_path=f)
    f.write_text(TOML.replace("depth = 3.0", "depth = 4.0").replace("lat = 43.7", "lat = 43.8"))
    assert stamp.key(spec, cfg_path=f) == k0          # unselected fields changed
    f.write_text(TOML.replace("start = 0.0", "start = 1.0"))
    assert stamp.key(spec, cfg_path=f) != k0


def test_key_follows_files_and_extra(tmp_path):
    f = _cfg(tmp_path)
    d = tmp_path / "in.npz"
    np.savez(d, a=np.arange(3))
    k0 = stamp.key(files=[d], cfg_path=f, x=np.arange(3))
    assert stamp.key(files=[d], cfg_path=f, x=np.arange(4)) != k0
    np.savez(d, a=np.arange(4))
    assert stamp.key(files=[d], cfg_path=f, x=np.arange(3)) != k0


def test_cache_roundtrip(tmp_path):
    c = tmp_path / "c.npz"
    assert stamp.load(c, "k1") is None
    stamp.save(c, "k1", a=np.arange(3))
    assert np.array_equal(stamp.load(c, "k1")["a"], np.arange(3))
    assert stamp.load(c, "k2") is None                # stale key = miss
    np.savez(c, a=np.arange(3))                       # old cache without a key = miss
    assert stamp.load(c, "k1") is None


def test_local_imports_are_recursive():
    files = {p.name for p in stamp.local_imports(stamp.ROOT / "processing" / "run_phase1.py")}
    assert {"run_phase1.py", "config.py", "fg_tracker.py", "boat_model.py", "validate.py"} <= files


def test_stage_staleness(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "STAMPS", tmp_path / "stamps")
    monkeypatch.setattr(stamp, "CONFIG", _cfg(tmp_path))
    inp, out = tmp_path / "in.txt", tmp_path / "out.txt"
    inp.write_text("1")
    s = pipeline.Stage("toy", "processing/config.py", [inp], [out], sections=("pinger",))
    assert pipeline.why_stale(s) == ["never run"]
    (tmp_path / "stamps").mkdir()
    pipeline._stamp_path(s).write_text(__import__("json").dumps(s.parts()))
    assert pipeline.why_stale(s) == [f"output {out} missing"]
    out.write_text("x")
    assert pipeline.why_stale(s) == []
    inp.write_text("2")
    assert pipeline.why_stale(s) == [f"files {inp} changed"]
    inp.write_text("1")
    stamp.CONFIG.write_text(TOML.replace("depth = 3.0", "depth = 5.0"))
    assert pipeline.why_stale(s) == ["config pinger changed"]


def test_stage_table_inputs_are_produced_upstream():
    """Every out/ input of a stage is an output of an earlier stage (no cycles, nothing missing)."""
    made = set()
    for s in pipeline.stages():
        for f in s.inputs:
            if str(f).startswith("out/"):
                assert f in made, (s.name, f)
        made |= set(s.outputs)
