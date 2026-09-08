"""File and runtime guards for one bounded native-operator candidate generation."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(x) for key, x in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(x) for x in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def activate():
    for name, digest in read(ROOT / "source-sha256.json").items():
        if sha(ROOT / "source" / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    for name, entry in read(ROOT / "inputs-sha256.json").items():
        if sha(ROOT / "inputs" / name) != entry["sha256"]:
            raise ValueError("Frozen input changed: " + name)
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(ROOT / "source/src"))


def environment(inherited=None):
    profile = read(ROOT / "profile.json")
    env = {
        key: value
        for key, value in (os.environ if inherited is None else inherited).items()
        if not key.startswith(("SPACEPDHCG_", "QOCO_"))
    }
    env.update(profile["environment"])
    env.update({key: value["path"] for key, value in profile["native_libraries"].items()})
    env.update(SPACEPDHCG_GTOC12_DATA=profile["data"], LD_LIBRARY_PATH=profile["LD_LIBRARY_PATH"])
    return profile, env


def runtime_check(actual=None):
    actual = os.environ if actual is None else actual
    profile, expected = environment({})
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Exact frozen native environment required")
    if any(key.startswith(("SPACEPDHCG_", "QOCO_")) and key not in expected for key in actual):
        raise ValueError("Unreviewed native switch")
    for item in profile["native_libraries"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Changed native library")
    return profile


def ready_check():
    ready = read(ROOT / "ready-manifest.json")
    for name, digest in ready["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Prepared kit changed: " + name)
    return ready
