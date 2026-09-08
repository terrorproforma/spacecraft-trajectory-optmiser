"""Pinned inputs and execution environment for the finite v607b rescue."""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_FLOOR = 23 * math.log(23 / 2) / 0.004


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=float) + "\n")
    pending.replace(path)


def activate():
    for name, expected in read(ROOT / "source-sha256.json").items():
        if sha(ROOT / "source" / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    for name, expected in read(ROOT / "inputs-sha256.json").items():
        if sha(ROOT / "inputs" / name) != expected["sha256"]:
            raise ValueError(f"frozen input changed: {name}")
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(ROOT / "source/src"))


def environment(inherited=None):
    profiles = read(ROOT / "profiles.json")
    profile = profiles["profiles"]["local"]
    env = {
        key: value
        for key, value in (os.environ if inherited is None else inherited).items()
        if not key.startswith(("SPACEPDHCG_", "QOCO_"))
    }
    env.update(profiles["common_environment"])
    env.update({key: value["path"] for key, value in profile["native_libraries"].items()})
    env.update(SPACEPDHCG_GTOC12_DATA=profile["data"], LD_LIBRARY_PATH=profile["LD_LIBRARY_PATH"])
    return profile, env


def validate_runtime(actual=None):
    actual = os.environ if actual is None else actual
    profile, expected = environment({})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"reviewed native environment changed: {key}")
    if any(key.startswith(("SPACEPDHCG_", "QOCO_")) and key not in expected for key in actual):
        raise ValueError("unreviewed native flag present")
    for key, value in profile["native_libraries"].items():
        if not Path(actual[key]).is_file() or sha(actual[key]) != value["sha256"]:
            raise ValueError(f"native library hash mismatch: {key}")
    return profile


def validate_ready():
    ready = read(ROOT / "ready-manifest.json")
    for name, expected in ready["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"reviewed kit changed: {name}")
    return ready


def qualifies(checked, expected_raw, expected_score):
    return bool(
        checked.get("ok")
        and checked.get("independent", {}).get("ok")
        and checked.get("official", {}).get("ok")
        and checked.get("total_mass_kg") is not None
        and checked.get("score_kg") is not None
        and math.isfinite(checked["total_mass_kg"])
        and math.isfinite(checked["score_kg"])
        and checked["total_mass_kg"] >= RAW_FLOOR - 1e-8
        and abs(checked["total_mass_kg"] - expected_raw) <= 1e-7
        and abs(checked["score_kg"] - expected_score) <= 1e-7
    )


def better(checked, best, dry_mass, control_passed):
    if not (
        control_passed
        and checked.get("prescribed_cargo_verified")
        and checked.get("ok")
        and checked.get("independent", {}).get("ok")
        and checked.get("official", {}).get("ok")
        and math.isfinite(dry_mass)
        and dry_mass >= 500 - 1e-9
        and checked.get("score_kg") is not None
        and math.isfinite(checked["score_kg"])
        and checked.get("total_mass_kg") is not None
        and math.isfinite(checked["total_mass_kg"])
        and checked["total_mass_kg"] >= RAW_FLOOR - 1e-8
    ):
        return False
    score = checked["score_kg"]
    return bool(
        score > best["score_kg"] + 1e-8
        or (
            abs(score - best["score_kg"]) <= 1e-8
            and dry_mass > best.get("ship7_final_dry_mass_kg", math.inf) + 1e-6
        )
    )
