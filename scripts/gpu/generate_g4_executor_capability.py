#!/usr/bin/env python3
"""Generate a content-addressed G4 executor capability record."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED_AXES = {
    "family",
    "intervals",
    "policy",
    "quality_tier",
    "conditioning",
    "scaling_mode",
    "warm_start_mode",
    "family_classes",
    "evaluation_seed",
    "repeat",
    "solver_order",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    executable = arguments.executable.resolve()
    policy_path = repository / "benchmarks/g4_policy.json"
    lock = (repository / "benchmarks/g4_policy.sha256").read_text().split()
    if len(lock) != 2 or lock[0] != sha256_path(policy_path):
        raise SystemExit("G4 policy lock mismatch")
    policy = json.loads(policy_path.read_text())
    split = policy["tuning_evaluation_split"]
    if set(split["tuning_seeds"]) & set(split["evaluation_seeds"]):
        raise SystemExit("G4 tuning and evaluation seed sets overlap")
    if (
        len(set(split["evaluation_seeds"]))
        != policy["matrix"]["randomised_instances_per_coordinate"]
    ):
        raise SystemExit("G4 evaluation seed cardinality mismatch")
    matrix_sha256 = hashlib.sha256(canonical_bytes(policy["matrix"]).rstrip(b"\n")).hexdigest()
    source_commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("capability generation requires a clean source commit")
    emitted = subprocess.run(
        [executable, "--g4-capabilities"],
        check=True,
        capture_output=True,
        text=True,
    )
    capability = json.loads(emitted.stdout)
    if set(capability.get("axes", {})) != EXPECTED_AXES:
        raise SystemExit("executor did not audit every frozen axis")
    if any(
        value.get("status") not in {"applied", "execution_only"}
        for value in capability["axes"].values()
    ):
        raise SystemExit("executor reports an unapplied roadmap-required axis")
    capability.update(
        {
            "source_commit": source_commit,
            "executable_sha256": sha256_path(executable),
            "policy_sha256": lock[0],
            "matrix_sha256": matrix_sha256,
        }
    )
    capability["capability_sha256"] = hashlib.sha256(
        canonical_bytes(capability).rstrip(b"\n")
    ).hexdigest()
    encoded = canonical_bytes(capability)
    output = arguments.output.resolve()
    if arguments.check:
        if not output.is_file() or output.read_bytes() != encoded:
            raise SystemExit("G4 executor capability record drift")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    print(capability["capability_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
