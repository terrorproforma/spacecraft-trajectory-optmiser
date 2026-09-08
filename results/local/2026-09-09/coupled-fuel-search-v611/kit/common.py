"""Frozen inputs, objective guards and finite experiment configuration."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRICES = (0.05, 0.25, 1.0)
MESH = (30.0, 10.0, 3.0, 1.0)
MAX_MOVES = 12
MAX_SEARCHES = 12
MAX_REFINEMENTS = 4
MAX_PROBES = 3
SHIPS = (20, 7, 10, 11)
RAW_FLOOR = 23 * math.log(23 / 2) / 0.004


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {key: clean(v) for key, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def activate():
    for name, expected in read(ROOT / "source-sha256.json").items():
        if sha(ROOT / "source" / name) != expected:
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
        k: v
        for k, v in (os.environ if inherited is None else inherited).items()
        if not k.startswith(("SPACEPDHCG_", "QOCO_"))
    }
    env.update(profile["environment"])
    env.update({k: v["path"] for k, v in profile["native_libraries"].items()})
    env.update(SPACEPDHCG_GTOC12_DATA=profile["data"], LD_LIBRARY_PATH=profile["LD_LIBRARY_PATH"])
    return profile, env


def runtime_check(actual=None):
    actual = os.environ if actual is None else actual
    profile, expected = environment({})
    if any(actual.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact frozen native environment required")
    if any(k.startswith(("SPACEPDHCG_", "QOCO_")) and k not in expected for k in actual):
        raise ValueError("Unreviewed native switch")
    for entry in profile["native_libraries"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Changed native library")
    return profile


def ready_check():
    result = read(ROOT / "ready-manifest.json")
    for name, expected in result["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Prepared kit changed: " + name)
    return result


def eligible(raw, weighted, baseline, original):
    return bool(
        math.isfinite(raw)
        and math.isfinite(weighted)
        and weighted > original["weighted_kg"] + 1e-8
        and baseline["raw_kg"] - original["raw_kg"] + raw >= RAW_FLOOR - 1e-8
    )


def fleet_accept(checked, baseline, expected):
    return bool(
        checked.get("ok")
        and checked.get("official", {}).get("ok")
        and checked.get("independent", {}).get("ok")
        and checked.get("score_kg") is not None
        and math.isfinite(checked["score_kg"])
        and checked.get("total_mass_kg") is not None
        and math.isfinite(checked["total_mass_kg"])
        and abs(checked["score_kg"] - expected["weighted_kg"]) <= 1e-7
        and abs(checked["total_mass_kg"] - expected["raw_kg"]) <= 1e-7
        and checked["score_kg"] > baseline["weighted_kg"] + 1e-8
        and checked["total_mass_kg"] >= RAW_FLOOR - 1e-8
    )


def best_combination(qualified, baseline):
    """At most eight subsets of already both-checked independent improvements."""
    if len(qualified) > 3:
        raise ValueError("At most three qualified candidates")
    best, best_score = [], baseline["weighted_kg"]
    for mask in range(1, 1 << len(qualified)):
        subset = [item for i, item in enumerate(qualified) if mask & (1 << i)]
        if len({item["ship"] for item in subset}) != len(subset):
            continue
        if any(
            not item["checked"].get("ok")
            or not item["checked"]["official"].get("ok")
            or not item["checked"]["independent"].get("ok")
            for item in subset
        ):
            continue
        raw = baseline["raw_kg"] + sum(item["raw_gain"] for item in subset)
        score = baseline["weighted_kg"] + sum(item["weighted_gain"] for item in subset)
        if (
            math.isfinite(raw)
            and math.isfinite(score)
            and raw >= RAW_FLOOR - 1e-8
            and score > best_score + 1e-8
        ):
            best, best_score = subset, score
    return best, best_score
