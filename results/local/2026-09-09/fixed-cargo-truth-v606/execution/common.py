"""Immutable inputs, source activation and strict fleet objective for v606."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCUMBENT_SHA = "33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8"
DATA_SHA = "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
BONUS_SHA = "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
FLEET_RAW_FLOOR = 23 * math.log(23 / 2.0) / 0.004


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


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


def activate_source():
    for name, expected in read(ROOT / "source-sha256.json").items():
        if sha(ROOT / "source" / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(ROOT / "source/src"))


def load_inputs():
    from spacepdhcg.gtoc12.solution import Solution

    manifest = read(ROOT / "inputs/sha256.json")
    for name, expected in manifest.items():
        if sha(ROOT / "inputs" / name) != expected:
            raise ValueError(f"retained input changed: {name}")
    if sha(ROOT / "inputs/Result.txt") != INCUMBENT_SHA:
        raise ValueError("exact retained v595 fleet is required")
    audited = read(ROOT / "inputs/fresh-verification.json")
    if not audited["independent"]["ok"] or not audited["official"]["ok"]:
        raise ValueError("both retained checker certificates are required")
    if audited["solution_sha256"] != INCUMBENT_SHA or audited["bonus_sha256"] != BONUS_SHA:
        raise ValueError("retained audit provenance mismatch")
    solution = Solution.read(ROOT / "inputs/Result.txt")
    summaries = {i: read(ROOT / f"inputs/ship-{i:02d}.json") for i in range(1, 24)}
    if solution.ship_count != 23:
        raise ValueError("retained fleet must contain 23 ships")
    for ship in solution.ships:
        summary = summaries[ship.ship_id]
        if not summary["certified"]:
            raise ValueError("uncertified original ship")
        expected = Counter(
            (int(body), float(epoch))
            for phase in ("deploy_epochs", "collect_epochs")
            for body, epoch in summary["plan"][phase].items()
        )
        if (
            Counter((event.event_id, event.before.epoch) for event in ship.asteroid_visits())
            != expected
        ):
            raise ValueError("original summary is inconsistent with flown Result")
        if any(
            abs(float(mass) - audited["scored_masses"][str(body)]) > 1e-7
            for body, mass in summary["collected_mass_kg"].items()
        ):
            raise ValueError("original cargo differs from independent scored masses")
    return summaries, audited, solution


def weighted(cargo, weights):
    if any(
        body not in weights or not math.isfinite(weights[body]) or weights[body] <= 0
        for body in cargo
    ):
        raise ValueError("complete finite positive fixed-bonus weights are mandatory")
    return sum(float(mass) * weights[body] for body, mass in cargo.items())


def checker_pass(checked):
    return bool(
        checked.get("ok")
        and checked.get("independent", {}).get("ok")
        and checked.get("official", {}).get("ok")
        and checked.get("score_kg") is not None
        and checked.get("total_mass_kg") is not None
        and math.isfinite(checked["score_kg"])
        and math.isfinite(checked["total_mass_kg"])
        and checked["total_mass_kg"] >= FLEET_RAW_FLOOR - 1e-8
    )


def can_promote(checked, baseline_score, control_passed):
    return bool(
        control_passed and checker_pass(checked) and checked["score_kg"] > baseline_score + 1e-8
    )


def profile_environment(name, inherited=None):
    profiles = read(ROOT / "profiles.json")
    profile = profiles["profiles"][name]
    environment = {
        key: value
        for key, value in (os.environ if inherited is None else inherited).items()
        if not key.startswith(("SPACEPDHCG_", "QOCO_REPLAY_"))
    }
    environment.update(profiles["common_environment"])
    environment.update({key: row["path"] for key, row in profile["native_libraries"].items()})
    environment["SPACEPDHCG_GTOC12_DATA"] = profile["data"]
    environment["LD_LIBRARY_PATH"] = profile["LD_LIBRARY_PATH"]
    return profile, environment


def validate_profile(name, environment=None):
    actual = os.environ if environment is None else environment
    profile, expected = profile_environment(name, {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"validated runtime profile changed: {key}")
    unexpected = [
        key
        for key in actual
        if key.startswith(("SPACEPDHCG_", "QOCO_REPLAY_")) and key not in expected
    ]
    if unexpected:
        raise ValueError(f"unreviewed native flags: {unexpected}")
    runtime = {}
    for key, row in profile["native_libraries"].items():
        path = Path(actual[key])
        if not path.is_file() or sha(path) != row["sha256"]:
            raise ValueError(f"unvalidated native library: {key}")
        runtime[key] = {"path": str(path.resolve()), "sha256": sha(path)}
    return profile, runtime


def validate_ready():
    publication = read(ROOT / "ready-manifest.json")
    for name, expected in publication["files"].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"reviewed preparation changed: {name}")
    return publication
