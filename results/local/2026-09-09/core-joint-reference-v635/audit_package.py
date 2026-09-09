"""Portable stdlib byte/status audit; never executes archived numerical source."""

if not __debug__:
    raise RuntimeError("Assertions must remain enabled; Python -O is not supported")

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

KIT = "build/performance/core-joint-reference-v635/"
TINY = "build/performance/core-joint-merit-v634/"
READY = "463e3fcf130a62db6c3266859984b7139fcf7842381ef8f8ea8b04497c548cbc"
RUN = "7a5f60b87acb4c3fd0ad845ffc23ebcb0cff48c399b14e6def79b9bd30a269f8"
WORK = "a46b6ba955e62dfff51e1df87a67ea2d8f36b4ea86a58a972727e2abbfc14068"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name, payload):
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.as_posix() == name and "\\" not in name and ":" not in name
    assert not any(p in (".", "..", ".git", "__pycache__", ".pytest_cache", ".venv") for p in path.parts)
    assert path.suffix.lower() not in (".exe", ".dll", ".so", ".o", ".a", ".pyc", ".pyd", ".pyo", ".pem")
    assert not payload.startswith((b"\x7fELF", b"MZ"))
    assert not any(line.startswith(b"-----BEGIN ") and line.endswith(b"PRIVATE KEY-----") for line in payload.splitlines())


def verify(package, expected_index=None):
    raw_index = (package/"index.json").read_bytes()
    if expected_index is not None:
        assert sha(raw_index) == expected_index
    index = json.loads(raw_index)
    archive_path = package/"evidence.zip"
    assert archive_path.stat().st_size == index["archive"]["bytes"]
    assert sha(archive_path.read_bytes()) == index["archive"]["sha256"]
    for name, entry in index["auxiliary_files"].items():
        payload = (package/name).read_bytes()
        safe(name, payload)
        assert len(payload) == entry["bytes"] and sha(payload) == entry["sha256"]
    assert (package/".gitattributes").read_bytes().strip() == b"* -text -whitespace"
    data = {}
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        assert len(entries) == len({e.filename for e in entries}) == index["file_count"]
        assert {e.filename for e in entries} == set(index["files"])
        for entry in entries:
            assert not entry.is_dir() and not stat.S_ISLNK(entry.external_attr >> 16)
            payload = archive.read(entry)
            safe(entry.filename, payload)
            pin = index["files"][entry.filename]
            assert len(payload) == entry.file_size == pin["bytes"] and sha(payload) == pin["sha256"]
            data[entry.filename] = payload
    assert sum(map(len, data.values())) == index["expanded_bytes"]
    read = lambda name: json.loads(data[name])
    assert sha(data[KIT+"ready.json"]) == READY
    ready = read(KIT+"ready.json")
    for name, entry in ready["files"].items():
        payload = data[KIT+name]
        assert len(payload) == entry["bytes"] and sha(payload) == entry["sha256"]
    tiny = read(KIT+"tiny-a/report.json")
    assert tiny["complete"] and tiny["passed"] and len(tiny["tests"]) == 3
    assert tiny["captured_inputs_parsed"] == tiny["captured_factorizations"] == tiny["captured_updates"] == tiny["GPU_calls"] == 0
    for name, entry in tiny["tests"].items():
        assert entry["passed"] and sha(data[KIT+"tiny-a/"+name+".json"]) == entry["sha256"]
    previous_tiny = read(TINY+"index.json")
    assert sha(data[TINY+"index.json"]) == "add2dd078b1105a48c6a2c5b6b9e094a86b966dfdbb78a09d9a5b8137e7e9ef9"
    for name, entry in previous_tiny["files"].items():
        payload = data[TINY+name]
        assert len(payload) == entry["bytes"] and sha(payload) == entry["sha256"]
    launch, work = read(KIT+"run-a/report.json"), read(KIT+"run-a/worker/report.json")
    assert sha(data[KIT+"run-a/report.json"]) == RUN and sha(data[KIT+"run-a/worker/report.json"]) == WORK
    assert launch["complete"] and launch["passed"] and launch["worker_terminal"]
    assert launch["processes_started"] == 1 and launch["worker_returncode"] == launch["final_returncode"] == 0
    assert launch["worker_report_sha256"] == launch["terminal_worker_report_sha256"] == WORK
    assert work["complete"] and work["actual_outer_updates"] == 20000 and work["actual_inner_calls"] == 20002
    assert work["actual_directions"] == 15469 and work["original_numeric_passes"] == 0
    assert work["GPU_calls"] == work["native_solver_calls"] == work["known_points_loaded"] == 0
    assert sha(data[KIT+"runtime.json"]) == work["runtime_sha256"] == launch["runtime_sha256"]
    for name, digest in work["source_sha256"].items():
        assert sha(data[KIT+name]) == digest
    audit = read(KIT+"audit-a/findings.json")
    assert audit["complete"] and audit["worker_report_sha256"] == WORK
    assert len(audit["rows"]) == 8 and len(audit["finals"]) == 2
    assert audit["original_numeric_passes"] == audit["accepted_outer_results"] == 0
    assert audit["solver_calls"] == audit["factorizations"] == audit["projection_calls"] == audit["GPU_calls"] == 0
    assert sha(data[KIT+"audit_saved.py"]) == audit["source_sha256"]
    for name, digest in audit["auditor_sha256"].items():
        assert sha(data[KIT+"audit-inputs/"+name]) == digest
    scalar_KKT, scalar_Armijo = 0, 0
    for case in work["cases"]:
        assert case["iterations"] == case["attempted_outer_updates"] == 10000
        assert case["reference_status"] == "outer_iteration_limit" and not case["qualified_numeric"]
        assert case["total_counts"]["full_factor_attempts"] == case["total_counts"]["full_factor_successes"] == 1
        assert case["total_counts"]["active_factor_attempts"] == case["total_counts"]["active_factor_successes"] == 1
        assert all(case["total_counts"][key] == 0 for key in ("active_factor_rejections", "gradient_fallbacks", "Newton_halvings", "numerical_failures", "inner_limits"))
        trace_path = KIT+"run-a/worker/"+case["capture"]+"/inner-work.jsonl"
        assert sha(data[trace_path]) == case["inner_trace_sha256"]
        totals = dict.fromkeys(case["total_counts"], 0)
        records = 0
        for line in data[trace_path].splitlines():
            record = json.loads(line)
            records += 1
            assert record["status"] == "inner_residual_pass" and record["counts"]["calls"] == 1
            for key, value in record["counts"].items():
                totals[key] += value
            assert len(record["trial_scalars"]) == record["counts"]["trial_attempts"]
            for trial in record["trial_scalars"]:
                if trial["acceptance"] == "trial_KKT_stop":
                    assert trial["trial_residual_pass"] and not trial["merit_attempted"]
                    scalar_KKT += 1
                else:
                    assert trial["acceptance"] == "Armijo" and trial["merit_computed"]
                    assert trial["merit"]["mode"] == "active_exact"
                    assert trial["merit"]["upper"] <= trial["nominal_Armijo_rhs"]
                    scalar_Armijo += 1
        assert records == case["completed_inner_records"] == 10001
        assert all(case["total_counts"][key] == case["setup_counts"][key]+value for key, value in totals.items())
        for checkpoint in case["checkpoints"]:
            payload = data[KIT+"run-a/worker/"+checkpoint["path"]]
            assert sha(payload) == checkpoint["sha256"]
            point = json.loads(payload)
            row = next(row for row in audit["rows"] if row["point_sha256"] == checkpoint["sha256"])
            assert point["termination"] == point["termination_code"] == 0
            assert row["long_double"] == checkpoint["original_audit"]
            assert not row["Decimal65"]["passes"] and not row["long_double"]["passes_common_kkt_gate"]
            assert not row["Decimal65"]["qualified"] and not row["long_double"]["qualified"]
            for key in ("x", "y", "z", "s"):
                assert all(math.isfinite(value) for value in point[key])
                assert sha(struct.pack("<"+"d"*len(point[key]), *point[key])) == row["vector_FP64_sha256"][key]
    assert scalar_KKT == 15460 and scalar_Armijo == 9
    analysis = read(KIT+"saved-analysis/findings.json")
    assert analysis["complete"] and analysis["audit_sha256"] == sha(data[KIT+"audit-a/findings.json"])
    assert analysis["solver_calls"] == analysis["factorization_calls"] == analysis["new_proximal_calls"] == analysis["new_directions"] == analysis["GPU_calls"] == 0
    assert sha(data[KIT+"analyze_saved.py"]) == analysis["source_sha256"]
    assert all(row["exact_zero_virtual_pairs"] == 525 and row["pair_sums_exact"] and row["fixed_primal_gate_eligibility"] and not row["dual_correction_feasibility_established"] for row in analysis["rows"])
    return dict(passed=True, index_sha256=sha(raw_index), files=len(data), expanded_bytes=index["expanded_bytes"],
                saved_vector_records=8, original_numeric_passes=0, outer_updates=20000,
                inner_calls=20002, directions=15469, trial_KKT_exits=15460, merit_trials=9,
                archived_code_executed=False, new_solver_or_factor_calls=0, GPU_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--index-sha256")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.package, args.index_sha256)
    if args.out:
        if args.out.exists():
            raise ValueError("fresh verification output required")
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, allow_nan=False))
