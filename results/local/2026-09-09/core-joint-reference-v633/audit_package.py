"""Portable stdlib byte/status audit; never executes archived numerical source."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

KIT = "build/performance/core-joint-reference-v633/"
TINY = "build/performance/core-joint-prox-v632/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    path = PurePosixPath(name)
    if path.is_absolute() or "\\" in name or ":" in name or ".." in path.parts:
        raise ValueError("unsafe archive name")
    if any(p in (".git", "__pycache__", ".pytest_cache") for p in path.parts):
        raise ValueError("unintended Git/cache entry")
    if path.suffix.lower() in (".exe", ".dll", ".so", ".o", ".a", ".pyc", ".pyd", ".pyo", ".pem"):
        raise ValueError("binary/credential file is outside publication scope")


def verify(package):
    index_path = package/"index.json"
    index = json.loads(index_path.read_text())
    archive_path = package/"evidence.zip"
    assert archive_path.stat().st_size == index["archive"]["bytes"]
    assert sha(archive_path.read_bytes()) == index["archive"]["sha256"]
    for name, item in index["auxiliary_files"].items():
        data = (package/name).read_bytes()
        assert len(data) == item["bytes"] and sha(data) == item["sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        assert len(entries) == len({e.filename for e in entries}) == index["file_count"]
        assert {e.filename for e in entries} == set(index["files"])
        data = {}
        for entry in entries:
            safe(entry.filename)
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            payload = archive.read(entry)
            pin = index["files"][entry.filename]
            assert len(payload) == entry.file_size == pin["bytes"] and sha(payload) == pin["sha256"]
            assert not payload.startswith((b"\x7fELF", b"MZ"))
            assert not any(line.startswith(b"-----BEGIN ") and line.endswith(b"PRIVATE KEY-----")
                           for line in payload.splitlines())
            data[entry.filename] = payload
    assert sum(len(v) for v in data.values()) == index["expanded_bytes"]
    read = lambda path: json.loads(data[path])
    ready = read(KIT+"ready.json")
    assert sha(data[KIT+"ready.json"]) == "c24911306be6845714b0c307dab1b9fe3dbb8cbe0d35a50263f9d08f25ba1267"
    for name, item in ready["files"].items():
        value = data[KIT+name]
        assert len(value) == item["bytes"] and sha(value) == item["sha256"]
    tiny = read(TINY+"index.json")
    assert sha(data[TINY+"index.json"]) == "27bbd087aedd788f920b434484c56b2a7feec9ce648a45ba038dcfe02bffb854"
    assert tiny["file_count"] == 8 and tiny["bytes"] == 56515
    for name, item in tiny["files"].items():
        value = data[TINY+name]
        assert len(value) == item["bytes"] and sha(value) == item["sha256"]
    launch, work = read(KIT+"run-a/report.json"), read(KIT+"run-a/worker/report.json")
    assert sha(data[KIT+"run-a/report.json"]) == "f81fdc2153aaaf864cfe8b4dc544c39deaf680d85937c38a8dbea09e48bacaad"
    assert sha(data[KIT+"run-a/worker/report.json"]) == "b9429ec1405fc0c4c2adc09a6d1585b451428fc8beec4b248f32a4e755b9864d"
    assert launch["complete"] and launch["passed"] and launch["worker_terminal"]
    assert launch["processes_started"] == 1 and launch["final_returncode"] == 0
    assert work["complete"] and work["actual_inner_calls"] == 77 and work["actual_directions"] == 66
    assert work["actual_outer_updates"] == 73 and work["original_numeric_passes"] == 0
    assert sha(data[KIT+"runtime.json"]) == work["runtime_sha256"] == launch["runtime_sha256"]
    assert work["GPU_calls"] == work["native_solver_calls"] == work["known_points_loaded"] == 0
    audits = read(KIT+"audit-a/findings.json")
    assert audits["complete"] and len(audits["rows"]) == 6 and len(audits["finals"]) == 2
    assert audits["original_numeric_passes"] == audits["accepted_outer_results"] == 0
    for case in work["cases"]:
        assert case["reference_status"] == "numerical_failure" and not case["qualified_numeric"]
        trace_path = KIT+"run-a/worker/"+case["capture"]+"/inner-work.jsonl"
        assert sha(data[trace_path]) == case["inner_trace_sha256"]
        trace = [json.loads(line) for line in data[trace_path].splitlines()]
        assert len(trace) == case["completed_inner_records"]
        for key, count in case["total_counts"].items():
            assert count == case["setup_counts"][key]+sum(r["counts"][key] for r in trace)
        assert case["total_counts"]["full_factor_attempts"] == case["total_counts"]["full_factor_successes"] == 1
        assert case["total_counts"]["active_factor_attempts"] == case["total_counts"]["active_factor_successes"] == 1
        assert case["total_counts"]["active_backtrack_exhaustions"] == case["total_counts"]["numerical_failures"] == 1
        assert case["total_counts"]["inner_limits"] == 0
        for cp in case["checkpoints"]:
            payload = data[KIT+"run-a/worker/"+cp["path"]]
            assert sha(payload) == cp["sha256"]
            point = json.loads(payload)
            audit = next(row for row in audits["rows"] if row["point_sha256"] == cp["sha256"])
            assert audit["long_double"] == cp["original_audit"]
            assert not audit["Decimal65"]["passes"] and not audit["long_double"]["passes_common_kkt_gate"]
            assert not audit["Decimal65"]["qualified"] and not audit["long_double"]["qualified"]
            assert point["termination"] == point["termination_code"] == 0
            for key in ("x", "y", "z", "s"):
                values = point[key]
                assert sha(struct.pack("<"+"d"*len(values), *values)) == audit["vector_FP64_sha256"][key]
    diagnosis = read(KIT+"saved-diagnosis/findings.json")
    assert diagnosis["complete"] and not diagnosis["counterfactual_trial_pass_proved"]
    assert all(r["failed_base_FP64_metrics_reproduced_exactly"] for r in diagnosis["rows"])
    assert diagnosis["factorizations"] == diagnosis["linear_solves"] == diagnosis["new_directions"] == diagnosis["GPU_calls"] == 0
    assert sha(data[KIT+"saved-diagnosis/diagnose_stdlib.py"]) == diagnosis["source_sha256"]
    for name, digest in work["source_sha256"].items():
        assert sha(data[KIT+name]) == digest
    return dict(passed=True, index_sha256=sha(index_path.read_bytes()), files=len(data),
                expanded_bytes=index["expanded_bytes"], original_numeric_passes=0,
                saved_vector_records=6, v632_tiny_files=8, archived_code_executed=False,
                new_factors=0, new_proximal_calls=0, new_outer_updates=0, GPU_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    if args.out:
        if args.out.exists():
            raise ValueError("fresh verification output required")
        args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
