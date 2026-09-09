"""Portable saved-byte and scalar residual audit; never executes archived workers."""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import posixpath
import tempfile
import zipfile

import audit_saved

BASE = "build/performance/interval-defect-replay-v633"


def compare_findings(calculated, stored):
    """Only the identified reporting-only expm1 value has a two-ULP allowance."""
    calculated = json.loads(json.dumps(calculated))
    assert set(calculated) == set(stored)
    key = "impulsive_rocket_equivalent_propellant_scale_kg"
    actual, expected = calculated[key], stored[key]
    assert isinstance(actual, float) and isinstance(expected, float)
    assert math.isfinite(actual) and math.isfinite(expected)
    limit = 2 * max(math.ulp(actual), math.ulp(expected))
    assert abs(actual - expected) <= limit
    assert {k: v for k, v in calculated.items() if k != key} == {
        k: v for k, v in stored.items() if k != key
    }
    return {"field": key, "calculated": actual, "stored": expected,
            "absolute_difference_kg": abs(actual - expected), "maximum_ulps": 2,
            "absolute_bound_kg": limit, "physics_and_byte_checks_unchanged": True}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def verify(directory, index_sha=None, roundtrip=False):
    if not __debug__:
        raise RuntimeError("assertions required; python -O is unsupported")
    directory = Path(directory)
    index_bytes = (directory / "index.json").read_bytes()
    if index_sha is not None:
        assert sha(index_bytes) == index_sha
    index = json.loads(index_bytes)
    for name, item in index["package_files"].items():
        data = (directory / name).read_bytes()
        assert len(data) == item["bytes"] and sha(data) == item["sha256"], name
    dependency = directory / index["dependency"]["directory"]
    dep_index = (dependency / "index.json").read_bytes()
    assert sha(dep_index) == index["dependency"]["index_sha256"]
    dep_map = json.loads(dep_index)
    assert sha((dependency / "evidence.zip").read_bytes()) == dep_map["archive_sha256"]
    with zipfile.ZipFile(directory / "evidence.zip") as own, zipfile.ZipFile(dependency / "evidence.zip") as old:
        assert set(own.namelist()) == set(index["members"])
        assert len(own.namelist()) == len(set(own.namelist()))
        for name, item in index["members"].items():
            data = own.read(name)
            assert len(data) == item["bytes"] and sha(data) == item["sha256"], name
        for name, expected in index["dependency_members"].items():
            assert dep_map["files"][name]["sha256"] == expected
            assert sha(old.read(name)) == expected, name

        def get(name):
            key = posixpath.normpath(str(PurePosixPath(BASE) / name))
            return own.read(key) if key in index["members"] else old.read(key)

        result = audit_saved.audit(get)
        stored = json.loads(get("saved-audit.json"))
        portability = compare_findings(result, stored)
        if roundtrip:
            with tempfile.TemporaryDirectory(prefix="v633-evidence-") as temp:
                target = Path(temp)
                for name in own.namelist():
                    path = PurePosixPath(name)
                    assert not path.is_absolute() and ".." not in path.parts
                    destination = target.joinpath(*path.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(own.read(name))
                    assert sha(destination.read_bytes()) == index["members"][name]["sha256"]
    return {"passed": True, "archive_members": len(index["members"]),
            "linked_dependency_members": len(index["dependency_members"]),
            "residual_components_checked": result["residual_components_checked"],
            "roundtrip": roundtrip, "GPU_calls": 0, "optimizer_calls": 0,
            "orbital_propagations": 0, "index_sha256": sha(index_bytes),
            "reporting_only_libm_comparison": portability}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-sha256")
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent, args.index_sha256, args.roundtrip), indent=2))
