"""Portable stdlib saved-byte audit. Never compiles, assembles, loads native code or connects."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import zipfile

CAPTURE_INDEX = "0689e0d08b41397144b8328bd56981451d76fb46d96fbdcafbae295c5e63e199"
STATUS_SOURCE = "f00dd0dc96c70aaff10e633743fe30a02dcad7e241ed4498101410c714fdb493"
INCUMBENT = "1f420928bbef8da91e3a6f8b74d3bf00039c78d4257215c103a484bc55bb22da"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(data):
    return {"bytes": len(data), "sha256": digest(data)}


def canonical(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def safe(name, data):
    parts = PurePosixPath(name).parts
    require(name and not name.startswith("/") and "\\" not in name and ":" not in name,
            "unsafe member name")
    require(all(p not in (".", "..", "", ".git", "__pycache__") for p in parts)
            and "/".join(parts) == name, "unsafe path component")
    require(Path(name).suffix.lower() not in (".exe", ".dll", ".so", ".o", ".obj", ".a", ".pyc", ".pyd"),
            "compiled member excluded")
    require(not data.startswith((b"\x7fELF", b"MZ")), "executable bytes excluded")
    require(not re.search(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", data), "private key excluded")


def bits(values):
    return b"".join(struct.pack("<d", float(v)) for v in values)


def capture_summary(files):
    prefix = "capture/"
    get = lambda name: json.loads(files[prefix + name])
    original_index = get("sha256.json")
    require(digest(files[prefix + "sha256.json"]) == CAPTURE_INDEX, "original capture index")
    require(original_index["file_count"] == 25 and original_index["bytes"] == 173817, "original kit totals")
    for name, pin in original_index["files"].items():
        require(identity(files[prefix + name]) == pin, "original capture bytes: " + name)
    ready = get("ready.json")
    for name, pin in ready["files"].items():
        require(identity(files[prefix + name]) == pin, "ready bytes: " + name)
    manifest = get("source-manifest.json")
    require(len(manifest["headers"]) == 4, "four exact headers")
    for name, pin in manifest["headers"].items():
        require(identity(files[prefix + name]) == {k: pin[k] for k in ("bytes", "sha256")}, "header bytes")
    report = get("execution-a/report.json")
    cap = get("execution-a/capture/capture-report.json")
    validation = get("execution-a/capture-validation.json")
    ctx = get("execution-a/capture/context.json")
    native = get("execution-a/capture/native.json")
    require(report["complete"] and report["passed"] and report["status"] == "captured_not_solved", "capture status")
    require(report["assembly_calls"] == report["assembly_attempts"] == 1 and report["retries"] == 0, "one assembly")
    require(report["optimizer_calls"] == report["gpu_calls"] == 0 and report["children_started"] == 2, "execution scope")
    require(report["cleanup"]["verified_empty"] and not report["cleanup"]["survivors"], "cleanup")
    require([p["stage"] for p in report["processes"]] == ["compile", "capture"], "two process roles")
    require(all(p["exit_code"] == 0 and p["cleanup"]["verified_empty"] and not p["cleanup"]["survivors"]
                for p in report["processes"]), "process exits and cleanup")
    for name, pin in report["log_files"].items():
        require(identity(files[prefix + "execution-a/" + name]) == pin, "raw log bytes")
    for field, name in (("ready_sha256", "ready.json"), ("runner_sha256", "capture.py"),
                        ("source_manifest_sha256", "source-manifest.json"),
                        ("validation_sha256", "execution-a/capture-validation.json")):
        require(report[field] == digest(files[prefix + name]), "execution binding: " + field)
    require(cap["source_sha256"] == digest(files[prefix + "export_hcw.cpp"])
            and cap["header_map_sha256"] == digest(files[prefix + "source-manifest.json"]), "compiled source binding")
    require(cap["complete"] and cap["passed"] and cap["assembly_calls"] == 1
            and cap["reference_rollout_intervals"] == 20 and cap["optimizer_calls"] == cap["gpu_calls"] == 0, "export scope")
    for name, pin in validation["files"].items():
        require(identity(files[prefix + "execution-a/capture/" + name]) == pin, "saved validation binding")
    require(validation["passed"] and validation["native_Q_A_c_bit_preserved"]
            and validation["cone_permutation_and_sign_verified"], "saved validation status")

    # Parse only retained serialization; no transcription, dynamics, objective or solver evaluation.
    tokens = iter(files[prefix + "execution-a/capture/snapshot.txt"].decode().split())
    require(next(tokens) == "SPACEPDHCG_QOCO_QP_V1", "snapshot format")
    require([int(next(tokens)) for _ in range(9)] == [186, 132, 80, 186, 1212, 60, 0, 20, 0], "snapshot dimensions")
    require([int(next(tokens)) for _ in range(4)] == [200, 0, 5, 0], "integer settings")
    require([float(next(tokens)) for _ in range(9)] == [1e-6, 1e-13, 1e-8, 1e-13, 1e-11, 1e-8, 1e-8, 1e-5, 1e-5], "numeric settings")
    def vector(cast, size):
        require(int(next(tokens)) == size, "vector length")
        return [cast(next(tokens)) for _ in range(size)]
    mats = []
    for nnz in (186, 1212, 60):
        mats.append({"offsets": vector(int, 187), "indices": vector(int, nnz)})
    require(vector(int, 20) == [4] * 20, "SOC sizes")
    values = vector(float, 1856)
    require(all(math.isfinite(v) for v in values), "finite coefficient bytes")
    require(vector(float, 0) == vector(float, 0) == [] and float(next(tokens)) == 0
            and next(tokens, None) is None, "no shifts or objective offset")
    at = 0
    for m, name, size in zip(mats, ("Q", "A", "F"), (186, 1212, 60)):
        require(m["offsets"] == native[name]["offsets"], "column map")
        rows = native[name]["indices"]
        expected_rows = rows if name != "F" else [4 * (r // 4) + (0 if r % 4 == 3 else r % 4 + 1) for r in rows]
        require(m["indices"] == expected_rows, "row map")
        expected_values = native[name]["values"] if name != "F" else [-v for v in native[name]["values"]]
        require(bits(values[at:at+size]) == bits(expected_values), "native matrix bits/sign")
        at += size
    require(bits(values[at:at+186]) == bits(native["c"]), "original c bits")
    require(bits(values[at+186:at+318]) == bits(native["scalar_lower"]) == bits(native["scalar_upper"]), "equality bounds")
    perm = [4*k+j for k in range(20) for j in (3, 0, 1, 2)]
    require(ctx["qoco_row_to_native_affine"] == perm, "cone permutation")
    require(bits(values[at+318:]) == bits([native["affine_offset"][r] for r in perm]), "cone offset")
    require(bits(native["Q"]["values"]) == bits([2e-4 if j % 6 < 3 else 2e-2 for j in range(126)] + [2.] * 60), "nonzero Q recipe")
    require(native["variable_lower"] == ["-Infinity"] * 186 and native["variable_upper"] == ["Infinity"] * 186, "original free bounds")
    require(ctx["canonical_objective_constant"] == native["canonical_objective_constant"] == 0
            and not ctx["physical_constant_used_by_solver_or_original_audit"], "constant convention")
    require(ctx["final_vectors"] is None and not ctx["reference_used_as_solver_seed"], "no solver result")
    require(ctx["cold_x"] == [0]*186 and ctx["cold_y"] == [0]*132 and ctx["cold_z"] == [0]*80
            and bits(ctx["cold_s"]) == bits(values[at+318:]), "declared cold vectors")
    require(ctx["initial_state"] == [.1, -.05, .02, 1e-4, -2e-4, 5e-5]
            and ctx["target_state"] == [0]*6 and ctx["times_seconds"] == list(range(0, 201, 10)), "physical recipe")
    return {"status": "host_coefficient_capture_only", "variables": 186, "equalities": 132, "soc4": 20,
            "positive_diagonal_P_values": 186, "canonical_objective_constant": 0,
            "tracking_reference_constant_metadata_only": ctx["physical_tracking_constant_fp64_metadata_only"],
            "assembly_calls": 1, "reference_rollout_intervals": 20, "optimizer_calls": 0, "gpu_calls": 0,
            "final_solver_vectors_available": False, "source_headers_verified": 4,
            "compile_wall_seconds": report["processes"][0]["wall_seconds"],
            "capture_process_wall_seconds": report["processes"][1]["wall_seconds"],
            "snapshot_sha256": digest(files[prefix + "execution-a/capture/snapshot.txt"]),
            "original_kit_index_sha256": CAPTURE_INDEX,
            "excluded_binary": original_index["excluded_binary"],
            "scope": "Saved hashes, settings and bitwise native/QOCO serialization; no new assembly, model or solver evaluation."}


def lambda_summary(raw_bytes):
    require(digest(raw_bytes) == STATUS_SOURCE, "retrieved raw source pin")
    raw = json.loads(raw_bytes)
    require(raw["complete"] and raw["passed"] and raw["source_uploads"] == raw["launches"] == 0, "retrieval scope")
    queries = raw["queries"]
    require(len(queries) == 3 and all(q["exit_code"] == 0 and not q["stderr"] for q in queries), "read-only query exits")
    gpu = [v.strip() for v in queries[0]["stdout"].strip().split(",")]
    require(gpu == ["GPU-6543ef0f-153d-d517-489b-8850b5a18c8b", "NVIDIA H100 80GB HBM3", "580.105.08", "0 %", "0 MiB"], "idle observation")
    require(queries[1]["stdout"] == "", "compute process observation")
    reports = json.loads(queries[2]["stdout"])
    expected = {"851": (True, 3, 6, 3, 3, 2, 3), "853": (False, 0, 0, 2, 0, None, None),
                "854": (True, 30, 4, 2, 2, 16, 17), "855": (True, 3, 48, 3, 3, 16, 17)}
    require(set(reports) == set(expected), "remote report inventory")
    rows = {}
    for version, (success, solves, hits, requests, attempts, cert_legs, legs) in expected.items():
        r = reports[version]
        require(r["complete"] and r["success"] is success and r["incumbent_sha256"] == INCUMBENT, "process/incumbent status")
        require((r["native_solves"], r["cache_hits"], r["fixed_requests"], len(r["attempts"])) == (solves, hits, requests, attempts), "reported call/request accounting")
        require(r["fleet_checks"] == 0, "no fleet checker claim")
        records = []
        for a in r["attempts"]:
            require(a["certified"] is False and a["certified_legs"] == cert_legs and a["legs"] == legs, "partial leg/route distinction")
            require(len(a["failures"]) == 1, "one retained failure per attempted route")
            f = a["failures"][0]
            require(f["certified"] is False and f["certificate"] is None and f["certification_backend"] is None, "failed flight not certified")
            s = f["solution"]
            record = {k: a[k] for k in ("rank", "request_sha256", "certified", "legs", "certified_legs", "seconds")}
            record["failed_flight"] = {k: f[k] for k in ("leg", "from", "to", "departure", "arrival", "initial_mass_kg", "minimum_final_mass_kg", "status", "diagnostic", "certificate")}
            record["failed_flight"]["solution"] = {k: s[k] for k in ("status", "iterations", "accepted_iterations", "propellant_kg", "max_defect", "solve_seconds")}
            records.append(record)
        rows[version] = {"process_complete": r["complete"], "process_success": r["success"],
                         "complete_routes_certified": 0, "native_solves": solves, "cache_hits": hits,
                         "fleet_checks": r["fleet_checks"], "fixed_requests": requests,
                         "attempt_count": attempts, "campaign_seconds": r["seconds"], "attempts": records,
                         "protocol": r["protocol"], "error": r.get("error"),
                         "timing_policy": r.get("timing_policy"), "return_policy": r.get("return_policy"),
                         "core_sha256": r["core_sha256"], "qoco_sha256": r["qoco_sha256"],
                         "source_manifest_sha256": r["source_manifest_sha256"],
                         "settings_canonical_sha256": digest(canonical(r["settings"])),
                         "parsed_report_canonical_sha256": digest(canonical(r))}
    require("prescribed cargo violates mining production" in rows["853"]["error"], "prior preparation failure retained")
    require([a["failed_flight"]["arrival"] for a in rows["855"]["attempts"]] == [69578, 69638, 69698], "three return horizons")
    return {"scope": "Read-only interpretation of already retrieved reports; no propagation, certification or solver rerun.",
            "raw_source": identity(raw_bytes), "remote_reports_stdout_sha256": digest(queries[2]["stdout"].encode()),
            "derived_hash_convention": "Parsed report/settings hashes use UTF-8 sorted indented JSON plus LF; not remote file-byte hashes.",
            "retrieval_complete": raw["complete"], "retrieval_passed": raw["passed"],
            "retrieval_seconds": raw["seconds"], "retrieval_source_uploads": raw["source_uploads"], "retrieval_launches": raw["launches"],
            "ephemeral_identity_removed": raw["ephemeral_identity_removed"],
            "idle_observation_UTC": raw["observed_at_UTC"], "gpu_observation": gpu,
            "compute_process_query_stdout": queries[1]["stdout"], "reports": rows,
            "reported_native_solves_total": sum(r["native_solves"] for r in reports.values()),
            "reported_cache_hits_total": sum(r["cache_hits"] for r in reports.values()),
            "attempted_routes_total": sum(len(r["attempts"]) for r in reports.values()),
            "fleet_checks_total": 0, "complete_routes_certified": 0, "fleet_gain_evidenced": False,
            "incumbent_sha256": INCUMBENT,
            "limitations": ["success is process completion, not trajectory certification",
                            "partial certification is reported metadata, not independently re-propagated here",
                            "cached prefix legs are not fresh native solves",
                            "native infeasible status is not an infeasibility proof",
                            "idle observation applies only to the stated UTC; no speed comparison"]}


def verify(directory, index_sha256=None):
    if not __debug__:
        raise RuntimeError("Python -O is forbidden")
    directory = Path(directory)
    index_bytes = (directory / "index.json").read_bytes()
    if index_sha256:
        require(digest(index_bytes) == index_sha256, "publication index pin")
    index = json.loads(index_bytes)
    for name, pin in index["top_files"].items():
        data = (directory / name).read_bytes()
        safe(name, data)
        require(identity(data) == pin, "top file bytes")
    archive = directory / "evidence.zip"
    require(digest(archive.read_bytes()) == index["archive_sha256"], "ZIP identity")
    files = {}
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        require(len(infos) == len({x.filename for x in infos}) == index["file_count"] <= 64, "ZIP names/count")
        require({x.filename for x in infos} == set(index["files"]), "exact member map")
        require(sum(x.file_size for x in infos) == index["bytes"] <= 5_000_000, "expanded bytes")
        for info in infos:
            mode = info.external_attr >> 16
            require(not info.is_dir() and not info.flag_bits & 1 and stat.S_IFMT(mode) in (0, stat.S_IFREG), "regular unencrypted members only")
            data = z.read(info)
            safe(info.filename, data)
            require(identity(data) == index["files"][info.filename], "member hash: " + info.filename)
            files[info.filename] = data
    if index["package"] == "nonzero-q-capture-v637":
        summary = capture_summary(files)
    elif index["package"] == "lambda-run-status-v637":
        summary = lambda_summary(files["lambda-status-v637.json"])
    else:
        raise ValueError("unknown package")
    require(files["summary.json"] == canonical(summary), "summary exactly recomputed from saved evidence")
    return {"passed": True, "package": index["package"], "index_sha256": digest(index_bytes),
            "archive_sha256": index["archive_sha256"], "file_count": len(files), "bytes": index["bytes"],
            "audit_scope": "stdlib saved-byte/provenance/status checks only", "new_gpu_solver_model_calls": 0}


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Python -O is forbidden")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--index-sha256")
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.index_sha256), indent=2, sort_keys=True))
