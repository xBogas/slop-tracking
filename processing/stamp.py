"""
Input keys for derived files, so that a change to forfun.toml, an input file
or the code that made a file is noticed.

`parts(...)` describes what a result depends on: the named config sections
(parsed, so comments and layout do not count), input files (content hash; size
and mtime for the multi-GB recordings), the local Python modules a script
imports (recursively), and any extra parameters (numbers, arrays). `key`
hashes that to a short string.

`load(path, key)` / `save(path, key, **arrays)` are a self-checking npz
cache: the key is stored in the file, and a cache whose key differs is
treated as missing.

Config sections: "pinger", "priors", ..., or "pinger:f0,f1" to keep only
some fields (per entry for the [[receiver]] / [[run]] arrays).
"""
import ast
import functools
import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

ROOT = config.ROOT
CONFIG = ROOT / "processing" / "forfun.toml"
CODE_DIRS = [ROOT / d for d in ("processing", "estimation", "motion_model")]
BIG = 256 * 2 ** 20                 # files above this are keyed by size + mtime, not content


def file_digest(path):
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    if st.st_size > BIG:
        return f"size {st.st_size} mtime {st.st_mtime_ns}"
    return _content_digest(p, st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _content_digest(p, size, mtime):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(2 ** 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_sections(sections, path=None):
    """{section spec: parsed value} from the TOML file (default CONFIG)."""
    raw = tomllib.loads(Path(path or CONFIG).read_text())
    out = {}
    for spec in sections:
        name, _, fields = spec.partition(":")
        v = raw.get(name)
        if fields:
            keep = fields.split(",")
            pick = lambda r: {k: r.get(k) for k in keep}
            v = [pick(r) for r in v] if isinstance(v, list) else pick(v or {})
        out[spec] = v
    return out


def local_imports(path):
    """Source files of the repo modules imported by a .py file or notebook, recursively."""
    seen, todo = set(), [Path(path)]
    while todo:
        p = todo.pop()
        if p in seen or not p.exists():
            continue
        seen.add(p)
        for name in _imported_names(p):
            for d in CODE_DIRS:
                f = d / f"{name.split('.')[0]}.py"
                if f.exists():                     # first match, like sys.path order
                    todo.append(f)
                    break
    return sorted(seen)


def _imported_names(p):
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text())
        src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    else:
        src = p.read_text()
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
    return names


def code_sources(path):
    """Code that a script or notebook runs: itself plus its local imports (notebook
    outputs excluded, so re-executing a notebook does not change its own key)."""
    out = {}
    for f in local_imports(path):
        if f.suffix == ".ipynb":
            nb = json.loads(f.read_text())
            src = "\n".join("".join(c["source"]) for c in nb["cells"])
            out[_rel(f)] = hashlib.sha1(src.encode()).hexdigest()
        else:
            out[_rel(f)] = file_digest(f)
    return out


def _rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        v = np.ascontiguousarray(v)
        return f"array {v.dtype} {v.shape} {hashlib.sha1(v.tobytes()).hexdigest()}"
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, Path):
        return _rel(v)
    return repr(v)


def parts(sections=(), files=(), code=None, cfg_path=None, **extra):
    """Everything a result depends on, as a JSON-able dict (see the module doc)."""
    return dict(config=config_sections(sections, cfg_path),
                files={_rel(f): file_digest(f) for f in files},
                code=code_sources(code) if code else {},
                extra=json.loads(json.dumps(extra, sort_keys=True, default=_jsonable)))


def key(*args, **kw):
    p = parts(*args, **kw)
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]


def diff(old, new):
    """What changed between two `parts` dicts, as short strings."""
    out = []
    for grp in ("config", "files", "code", "extra"):
        a, b = old.get(grp, {}), new.get(grp, {})
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                what = "missing" if b.get(k) is None else "new" if k not in a else "changed"
                out.append(f"{grp} {k} {what}")
    return out


def load(path, k):
    """Arrays of an npz cache written by `save` with the same key, else None."""
    path = Path(path)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    if "_key" not in d.files or str(d["_key"]) != k:
        return None
    return {n: d[n] for n in d.files if n != "_key"}


def save(path, k, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, _key=np.array(k), **arrays)
