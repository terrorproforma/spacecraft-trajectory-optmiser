"""Frozen host and immutable input helpers for the one positive ship-8 candidate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[3]
MAPPING = ROOT / "build/performance/seeded-candidates-v626/output/mapping.json"
PRIOR = ROOT / "build/performance/refinement-queue-v623"
SEED = ROOT / "build/performance/native-seed-v624"
CONTROL = ROOT / "build/performance/seeded-route-v625"
_PROFILE = json.loads((KIT / "profile.json").read_text())
CORE_MANIFEST = ROOT / _PROFILE["native_manifest"]["path"]
DATA = "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data"
RESULT_SHA = "d368babcf0656ab55b9a21bd8ccde4eef22629a37f9bc8f8a1877eb15a3e7efa"
REQUEST_SHA = "f51f89dfebf9dd71eb9ea3061f0d54e2e1bad93c2a996413917a5852a53c8834"
CORE = _PROFILE["library"]["path"]
CORE_SHA = _PROFILE["library"]["sha256"]
MANIFEST_SHA = _PROFILE["native_manifest"]["sha256"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(value):
    if dataclasses.is_dataclass(value):
        return safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(safe(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def current(path, value):
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def activate():
    overlay = read(KIT / "host-overlay.json")
    assert sha(PRIOR / "source-sha256.json") == overlay["base_source_manifest_sha256"]
    for name, expected in read(PRIOR / "source-sha256.json").items():
        assert sha(PRIOR / "source" / name) == expected, name
    for name, expected in overlay["files"].items():
        assert sha(KIT / "overlay" / name) == expected, name
    sys.meta_path = [
        f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"
    ]
    sys.path.insert(0, str(PRIOR / "source/src"))
    import spacepdhcg.gtoc12 as package

    package.__path__.insert(0, str(KIT / "overlay/spacepdhcg/gtoc12"))


def catalogue_and_bonus():
    from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue

    os.environ["SPACEPDHCG_GTOC12_DATA"] = DATA
    catalogue, bonus = load_catalogue(), load_bonus_table()
    assert (
        catalogue.source_sha256
        == "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675"
    )
    assert bonus.source_sha256 == "e8a3795e599556ed5b66713ab1fa176de93ef37f93cb2a4a87d561539b1caa21"
    return catalogue, bonus


def ship_byte_parts(payload, ship_id=8):
    lines = payload.splitlines(keepends=True)
    found = [
        i
        for i, line in enumerate(lines)
        if line.split() and line.split()[0] == str(ship_id).encode()
    ]
    assert found and found == list(range(found[0], found[-1] + 1))
    return (
        b"".join(lines[: found[0]]),
        b"".join(lines[found[0] : found[-1] + 1]),
        b"".join(lines[found[-1] + 1 :]),
    )


def splice_ship(payload, replacement):
    prefix, old, suffix = ship_byte_parts(payload)
    assert replacement and not replacement.endswith(b"\n")
    assert all(line.split()[0] == b"8" for line in replacement.splitlines() if line.split())
    separator = b"\n" if suffix else b""
    assert not suffix or old.endswith(b"\n")
    fleet = prefix + replacement + separator + suffix
    assert ship_byte_parts(fleet)[0] == prefix and ship_byte_parts(fleet)[2] == suffix
    return fleet
