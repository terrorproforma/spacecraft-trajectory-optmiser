"""Portable byte/status audit; never imports or executes archived numerical code."""

if not __debug__:
    raise RuntimeError("Assertions must remain enabled; Python -O is not supported")

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

KIT = "build/performance/core-joint-halpern-v636/"
READY = "c3448eea56181220ab6b5585c33e0ea7673cd77b3291fd86cb895ceec9ecb533"
RUN = "168027c5d81ec57c7edc11744b32131537c329bcb26cfa0a2c9bdf763d5fd5df"
WORK = "3770901465533302e15878fbf281278cce53e95848c46dc2e350028bcd1d042b"
AUDIT = "c9237302b17887ec4c546ca976905ba4562ea518821f46cfd19bfeb8f145340c"
ANALYSIS = "3c948f67eda330947fd5b242117dc9ca1b4b623f1160ec9fb7bfff98c56b1f5a"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bits(values):
    return struct.pack("<"+"d"*len(values), *values)


def safe(name, payload):
    path = PurePosixPath(name)
    assert not path.is_absolute() and path.as_posix() == name and "\\" not in name and ":" not in name
    assert not any(p in (".", "..", ".git", "__pycache__", ".pytest_cache", ".venv") for p in path.parts)
    assert path.suffix.lower() not in (".exe", ".dll", ".so", ".o", ".a", ".pyc", ".pyd", ".pyo", ".pem")
    assert not payload.startswith((b"\x7fELF", b"MZ"))
    assert not any(line.startswith(b"-----BEGIN ") and line.endswith(b"PRIVATE KEY-----") for line in payload.splitlines())


def verify(package, expected_index=None):
    raw_index = (package/"index.json").read_bytes()
    if expected_index:
        assert sha(raw_index) == expected_index
    index = json.loads(raw_index)
    archive_path = package/"evidence.zip"
    assert archive_path.stat().st_size == index["archive"]["bytes"]
    assert sha(archive_path.read_bytes()) == index["archive"]["sha256"]
    for name, pin in index["auxiliary_files"].items():
        payload = (package/name).read_bytes()
        safe(name, payload)
        assert len(payload) == pin["bytes"] and sha(payload) == pin["sha256"]
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
    for name, digest in (("ready.json",READY), ("run-a/report.json",RUN),
                         ("run-a/worker/report.json",WORK), ("audit-a/findings.json",AUDIT),
                         ("saved-analysis/findings.json",ANALYSIS)):
        assert sha(data[KIT+name]) == digest
    ready = read(KIT+"ready.json")
    for name, pin in ready["files"].items():
        payload = data[KIT+name]
        assert len(payload) == pin["bytes"] and sha(payload) == pin["sha256"]
    tiny = read(KIT+"tiny-a/report.json")
    assert tiny["complete"] and tiny["passed"] and set(tiny["tests"]) == {"Halpern"}
    assert tiny["captured_inputs_parsed"] == tiny["captured_factorizations"] == tiny["captured_updates"] == tiny["GPU_calls"] == 0
    assert sha(data[KIT+"tiny-a/Halpern.json"]) == tiny["tests"]["Halpern"]["sha256"]
    old_tiny = "build/performance/core-joint-reference-v635/tiny-a/"
    assert sha(data[old_tiny+"report.json"]) == "4fdee7392649bc48d1d705cf74a4080fa3553eced571acae5108bc33bf656e3a"
    for name, pin in read(old_tiny+"report.json")["tests"].items():
        assert pin["passed"] and sha(data[old_tiny+name+".json"]) == pin["sha256"]
    launch, work = read(KIT+"run-a/report.json"), read(KIT+"run-a/worker/report.json")
    assert launch["complete"] and launch["passed"] and launch["worker_terminal"]
    assert launch["processes_started"] == 1 and launch["worker_returncode"] == launch["final_returncode"] == 0
    assert launch["worker_report_sha256"] == launch["terminal_worker_report_sha256"] == WORK
    assert work["complete"] and work["actual_outer_updates"] == 20000 and work["actual_inner_calls"] == 20002
    assert work["actual_directions"] == 20010 and work["original_numeric_passes"] == 0
    assert work["GPU_calls"] == work["native_solver_calls"] == work["known_points_loaded"] == 0
    assert not work["reference_costs_used_by_algorithm"] and work["weight_updates"] == 0
    assert sha(data[KIT+"runtime.json"]) == work["runtime_sha256"] == launch["runtime_sha256"]
    for name, digest in work["source_sha256"].items():
        assert sha(data[KIT+name]) == digest
    audit, analysis = read(KIT+"audit-a/findings.json"), read(KIT+"saved-analysis/findings.json")
    assert audit["complete"] and audit["worker_report_sha256"] == analysis["worker_report_sha256"] == WORK
    assert len(audit["rows"]) == 8 and len(audit["finals"]) == 2
    assert audit["original_numeric_passes"] == audit["accepted_outer_results"] == 0
    assert audit["solver_calls"] == audit["factorizations"] == audit["projection_calls"] == audit["GPU_calls"] == 0
    assert sha(data[KIT+"audit_saved.py"]) == audit["source_sha256"]
    for name, digest in audit["auditor_sha256"].items():
        assert sha(data[KIT+"audit-inputs/"+name]) == digest
    assert analysis["complete"] and analysis["audit_sha256"] == AUDIT
    assert analysis["solver_calls"] == analysis["factorizations"] == analysis["proximal_calls"] == analysis["projections"] == analysis["GPU_calls"] == 0
    assert sha(data[KIT+"analyze_saved.py"]) == analysis["source_sha256"]
    reference = read(KIT+"reporting-objectives.json")
    support_path = "build/performance/core-dual-correction-v636/saved-support-a.json"
    support = read(support_path)
    assert sha(data[support_path]) == "1ea38dc836b1935a877ffdc43e653901799eb45a70b09d770f30407861d60ec9"
    total_acceptance = Counter()
    for case, final, derived, prior in zip(work["cases"],audit["finals"],analysis["rows"],support["rows"]):
        capture = case["capture"]
        assert capture == final["capture"] == derived["capture"] == prior["capture"]
        assert case["iterations"] == case["attempted_outer_updates"] == 10000
        assert case["reference_status"] == "outer_iteration_limit" and not case["qualified_numeric"]
        assert all(case["total_counts"][key] == 0 for key in ("active_factor_rejections","gradient_fallbacks","Newton_halvings","numerical_failures","inner_limits"))
        assert case["total_counts"]["full_factor_attempts"] == case["total_counts"]["full_factor_successes"] == 1
        assert case["total_counts"]["active_factor_attempts"] == case["total_counts"]["active_factor_successes"] == 1
        trace = KIT+"run-a/worker/"+capture+"/inner-work.jsonl"
        assert sha(data[trace]) == case["inner_trace_sha256"]
        sums, acceptance = Counter(), Counter()
        restart_iterations, metric_iterations = [], []
        records = 0
        for line in data[trace].splitlines():
            record = json.loads(line)
            records += 1
            assert record["status"] == "inner_residual_pass" and record["counts"]["calls"] == 1
            sums.update(record["counts"])
            assert len(record["trial_scalars"]) == record["counts"]["trial_attempts"]
            for trial in record["trial_scalars"]:
                acceptance[trial["acceptance"]] += 1
                if trial["acceptance"] == "trial_KKT_stop":
                    assert trial["trial_residual_pass"] and not trial["merit_attempted"]
                else:
                    assert trial["acceptance"] == "Armijo" and trial["merit_computed"]
                    assert trial["merit"]["mode"] == "active_exact" and trial["merit"]["upper"] <= trial["nominal_Armijo_rhs"]
            if not record["initial"]:
                scalar = record["outer"]
                assert scalar["completed_map"] and scalar["public_point"] == "proximal_output"
                if scalar["restarted"]:
                    restart_iterations.append(scalar["total"])
                if scalar["metric_attempted"]:
                    metric_iterations.append(scalar["total"])
        assert records == case["completed_map_records"] == case["completed_inner_records"] == 10001
        assert all(case["total_counts"][k] == case["setup_counts"][k]+sums[k] for k in sums)
        assert restart_iterations == derived["restart_iterations"] == [200,400,800,1400,2200,3600,5800,9200]
        assert len(metric_iterations) == case["outer_counts"]["metric_evaluations"] == 58
        assert dict(acceptance) == derived["acceptance"]
        total_acceptance.update(acceptance)
        p_verified = 0
        for checkpoint in case["checkpoints"]:
            payload = data[KIT+"run-a/worker/"+checkpoint["path"]]
            assert sha(payload) == checkpoint["sha256"]
            point = json.loads(payload)
            row = next(r for r in audit["rows"] if r["point_sha256"] == checkpoint["sha256"])
            assert point["termination"] == point["termination_code"] == 0
            assert row["long_double"] == checkpoint["original_audit"]
            assert not row["Decimal65"]["passes"] and not row["long_double"]["passes_common_kkt_gate"]
            assert not row["Decimal65"]["qualified"] and not row["long_double"]["qualified"]
            for key in ("x","y","z","s"):
                assert all(math.isfinite(v) for v in point[key])
                assert sha(bits(point[key])) == row["vector_FP64_sha256"][key]
            if point["outer"]["vectors"] is not None:
                vectors = point["outer"]["vectors"]
                assert bits(point["u_reduced"]) == bits(vectors["proximal_u"])
                pair_rows = {r for _,_,rp,rm,_ in point["l1_map"] for r in (rp,rm)}
                assert bits([v for j,v in enumerate(point["z"]) if j not in pair_rows]) == bits(vectors["proximal_z"])
                p_verified += 1
        assert p_verified == derived["retained_P_checkpoints_bit_verified"] == 3
        assert derived["cost_is_infeasible_not_qualified_progress"]
        assert final["gate_ratios"]["primal_cone_violation"] > 1 and final["gate_ratios"]["block_complementarity_normalized"] > 1
        assert derived["prior_v635_objective"] == reference[capture]["v635_objective"] == prior["original_objective"]
        assert derived["qualified_same_input_QOCO_objective"] == reference[capture]["qualified_QOCO_objective"]
        qoco = prior["same_input_saved_reference"]
        assert qoco["saved_both_original_audits_pass"] and qoco["native_status"] in (1,2)
        assert qoco["snapshot_sha256"] == case["snapshot_sha256"]
        assert qoco["original_objective"] == qoco["recomputed_objective"] == reference[capture]["qualified_QOCO_objective"]
        assert reference[capture]["saved_support_sha256"] == sha(data[support_path])
        assert reference[capture]["snapshot_sha256"] == case["snapshot_sha256"] and not reference[capture]["algorithm_may_read"]
    assert total_acceptance == {"trial_KKT_stop":20002,"Armijo":8}
    for name, dependency in index["published_predecessor_dependencies"].items():
        old_index = data[dependency["package_path"]+"/index.json"]
        assert sha(old_index) == dependency["index_sha256"]
        prior = json.loads(old_index)
        assert (prior["archive"]["sha256"] if "archive" in prior else prior["archive_sha256"]) == dependency["archive_sha256"]
        if name == "core-primal-decision-v636":
            prior_map = json.loads(old_index)["files"]
            assert prior_map[support_path]["sha256"] == sha(data[support_path])
    return dict(passed=True,index_sha256=sha(raw_index),files=len(data),expanded_bytes=index["expanded_bytes"],
                saved_vector_records=8,original_numeric_passes=0,outer_updates=20000,inner_calls=20002,directions=20010,
                restarts=16,fixed_point_metrics=116,retained_P_points_bit_verified=6,
                archived_code_executed=False,new_solver_or_factor_calls=0,GPU_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--index-sha256")
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    result = verify(args.package,args.index_sha256)
    if args.out:
        if args.out.exists():
            raise ValueError("fresh output required")
        args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,allow_nan=False))
