"""Verify the prepared source, tests and late correctness prerequisite, without GPU."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert not (KIT / "ready.json").exists()
    assert not (KIT / "output").exists()
    assert not (KIT / "launch-marker.json").exists()
    source = read(KIT / "source-sha256.json")
    for name, expected in source.items():
        assert sha(KIT / "source" / name) == expected, name
    for name, expected in read(KIT / "input-sha256.json").items():
        assert sha(KIT / "inputs" / name) == expected, name
    extracted = {}
    with tarfile.open(KIT / "inputs/compact-g-source.tar.gz", "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = member.name.removeprefix("./").removeprefix("source/")
            assert name in source and name not in extracted, name
            extracted[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    native = read(KIT / "inputs/compact-g-report.json")
    assert extracted == native["owned_sources"]
    full_archive = KIT / "source-full.tar.gz"
    if not full_archive.exists():
        with tarfile.open(full_archive, "x:gz") as archive:
            for name in sorted(source):
                archive.add(KIT / "source" / name, arcname=name, recursive=False)
    with tarfile.open(full_archive, "r:gz") as archive:
        complete_source = {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive
            if member.isfile()
        }
    assert complete_source == source
    profile = read(KIT / "profile.json")
    assert sha(Path(profile["python"])) == profile["python_sha256"]
    assert sha(Path(profile["library"]["path"])) == profile["library"]["sha256"]
    assert sha(Path(profile["native_root"]) / "report.json") == profile["source_report_sha256"]
    prior_tests = KIT / "validation/attempt-04/report.json"
    tests = [x for x in read(prior_tests) if x["name"] == "pytest"]
    assert len(tests) == 1 and tests[0]["exit_code"] == 0
    assert "18 passed" in (KIT / "validation/attempt-04/pytest.log").read_text()
    lint = read(KIT / "validation/final-lint.json")
    assert len(lint) == 2 and all(row["exit_code"] == 0 for row in lint)
    prerequisite = ROOT / "build/performance/completion-model-gpu-v622d/report.json"
    expected = "643caf4e5550373ac3b06761197ae4ee1f7801569e614498786ba089b34a54ff"
    assert sha(prerequisite) == expected
    from launch import validate_prerequisite

    validate_prerequisite(read(prerequisite), profile)
    write(
        KIT / "freeze-audit.json",
        {
            "preparation_only": True,
            "GPU_calls": 0,
            "source_files": len(source),
            "original_owned_overlay_archive_files": len(extracted),
            "original_owned_overlay_archive_roundtrip": True,
            "complete_source_archive_roundtrip": True,
            "complete_source_archive_sha256": sha(full_archive),
            "CPU_tests": {"passed": 18, "report": str(prior_tests.relative_to(KIT))},
            "prior_ready_before_raw_retention_fix_sha256": sha(
                KIT / "validation/pre-raw-retention/ready.json"
            ),
            "matching_GPU_correctness": {
                "path": str(prerequisite.relative_to(ROOT)),
                "sha256": expected,
                "tests_passed": 7,
                "scope": "Separate already-completed correctness run, not this benchmark",
            },
            "future_evaluate_calls": 80,
            "future_kernel_launches_from_frozen_source": 160,
            "future_candidate_evaluations": 27040,
        },
    )
    files = {
        path.relative_to(KIT).as_posix(): sha(path)
        for path in sorted(KIT.rglob("*"))
        if path.is_file() and not any(x in path.parts for x in ("__pycache__", ".ruff_cache"))
    }
    ready = {
        "status": "prepared_not_executed",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum((KIT / name).stat().st_size for name in files),
        "benchmark_GPU_calls_so_far": 0,
        "future_evaluate_calls": 80,
        "candidate_evaluations_each_arm": 13520,
        "historical_unique_controls": 20,
        "source_report_sha256": profile["source_report_sha256"],
        "library_sha256": profile["library"]["sha256"],
        "required_prerequisite_sha256": expected,
    }
    write(KIT / "ready.json", ready)
    for name, expected in files.items():
        assert sha(KIT / name) == expected
    print(json.dumps({k: value for k, value in ready.items() if k != "files"}))
    print("ready_sha256=" + sha(KIT / "ready.json"))


if __name__ == "__main__":
    main()
