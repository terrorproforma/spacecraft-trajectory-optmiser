"""Portable saved-byte/source/report bindings. Never executes project/numerical code."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import tarfile
import zipfile

BP = "build/performance/"
OLD = BP + "seeded-candidates-v626/campaign"
NEW = BP + "seeded-candidate-boundary-merit-v627/campaign"
DQ = BP + "gpu-dual-polish-v626"
RESULT = "931defc362c9299d28d089aa46130c05f92bf9ee38ea03d870b4dc2a83915946"
PINS = {
    BP + "boundary-merit-core-v627a/manifest.json": "197076c91583c32cd39f79ae57d9e020d4011b8171d9b6105b555ff175b0f789",
    BP + "boundary-merit-core-v627a/source.tar.gz": "5f0324d1ab1985db06b7907dcbfa0bf4dfab32b67e8c302572c41cb85c7f2405",
    BP + "boundary-merit-tests-v627a/report.json": "df1a4e932f1e38d964968d0fdf9695daa8e7a28cf0836c5946000fd0af544632",
    BP + "boundary-candidate-review-v627/findings.json": "c96fc9544af283979839f56e67a968c76fd3392d49061400c2ae8ceb8a420f4a",
    BP + "boundary-candidate-review-v627/index.json": "d8f29835244cb50147fb9853e4e04ea40282e9373b8faec76e74fdb7f4b03614",
    BP + "mesh-refinement-review-v626/findings.json": "9bef8ecaa4465360cb6fe2e261343f10f7c37728c5c9236405c01166f1174742",
    BP + "scvx-boundary-review-v626/index.json": "45754e6d99e6ee6019f148ae9180dd9095e5ccf18754774c60b9261b429c68ec",
    DQ + "/sha256.json": "86312946bab71d80950c1ca33fb56ad199f6e6048843c3fd201996ae34508f95",
    DQ + "/real-a/report.json": "6f7f3156745878ecca01598522b0d324959b587462070c5f0078fb549fc8843b",
    BP + "dual-qr-real-review-v626/index.json": "89d42503863871753ce324e881d0943bbcb0244971d678512156f8b38ff6a3f3",
    BP + "dual-qr-real-review-v626/findings.json": "07cc34c9f7d361cbd4fa3772304a6cb74939d2673731fede78bb6e3b063f4e73",
    NEW + "/output/report.json": "3b5ffb1b56f2e53446a4fb18bb10489b311b3d85d298927a4432ab029fd52015",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_name(name):
    path = PurePosixPath(name.rstrip("/"))
    assert name and "\\" not in name and not path.is_absolute()
    assert all(part not in ("..", ".git", "__pycache__", ".pytest_cache", ".ruff_cache") and ":" not in part for part in path.parts), name
    assert path.suffix.lower() not in (".pem", ".key", ".p12", ".pfx", ".pyc", ".so", ".exe", ".dll", ".o", ".obj"), name


def inspect(raw, name, stats, depth=0):
    safe_name(name)
    assert depth < 6 and len(raw) <= 64 * 1024 * 1024, name
    assert not raw.startswith((b"\x7fELF", b"MZ")), name
    assert not re.search(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----", raw), name
    stats["payloads"] += 1
    stats["expanded_bytes"] += len(raw)
    assert stats["expanded_bytes"] <= 384 * 1024 * 1024
    if raw.startswith(b"PK\x03\x04"):
        stats["containers"] += 1
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            seen = set()
            for item in archive.infolist():
                safe_name(item.filename)
                assert item.filename not in seen and not stat.S_ISLNK(item.external_attr >> 16)
                seen.add(item.filename)
                assert not item.flag_bits & 1 and item.file_size <= 64 * 1024 * 1024
                if not item.is_dir():
                    data = archive.read(item)
                    assert len(data) == item.file_size
                    inspect(data, item.filename, stats, depth + 1)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        stats["containers"] += 1
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
            seen = set()
            for item in archive:
                safe_name(item.name)
                assert item.name not in seen and (item.isfile() or item.isdir())
                seen.add(item.name)
                assert item.size <= 64 * 1024 * 1024
                if item.isfile():
                    data = archive.extractfile(item).read()
                    assert len(data) == item.size
                    inspect(data, item.name, stats, depth + 1)


def main():
    if not __debug__:
        raise RuntimeError("Run this verifier without Python -O")
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-index")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    raw_index = (package / "index.json").read_bytes()
    if args.expected_index:
        assert sha(raw_index) == args.expected_index
    index = json.loads(raw_index)
    assert index["schema"] == "ENDPOINT_MERIT_AND_GPU_DUAL_V627" and index["complete"]
    for name, item in index["auxiliary_files"].items():
        raw = (package / name).read_bytes()
        assert sha(raw) == item["sha256"] and len(raw) == item["bytes"]
    raw_archive = (package / "evidence.zip").read_bytes()
    assert len(raw_archive) == index["archive_bytes"] and sha(raw_archive) == index["archive_sha256"]
    payload = {}
    stats = {"payloads": 0, "expanded_bytes": 0, "containers": 0}
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) == index["file_count"]
        assert set(names) == set(index["files"])
        for item in archive.infolist():
            name = item.filename
            assert "v628" not in name and "visualizer" not in name and "visualiser" not in name
            assert not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16)
            raw = archive.read(item)
            expected = index["files"][name]
            assert len(raw) == item.file_size == expected["bytes"] and sha(raw) == expected["sha256"], name
            inspect(raw, name, stats)
            payload[name] = raw
    assert sum(len(raw) for raw in payload.values()) == index["uncompressed_bytes"]
    for name, digest in PINS.items():
        assert sha(payload[name]) == digest, name

    def read(name):
        return json.loads(payload[name])

    def bound(name, digest):
        if name in payload:
            assert sha(payload[name]) == digest, name
        else:
            assert name in index["omitted_files"] and index["omitted_files"][name]["sha256"] == digest, name

    for prefix in (BP + "boundary-candidate-review-v627", BP + "dual-qr-real-review-v626", BP + "scvx-boundary-review-v626"):
        saved = read(prefix + "/index.json")
        for name, item in saved["files"].items():
            bound(prefix + "/" + name, item["sha256"])
            assert len(payload[prefix + "/" + name]) == item["bytes"]
    for prefix in (OLD, NEW):
        ready = read(prefix + "/ready.json")
        for name, digest in ready["files"].items():
            bound(prefix + "/" + name, digest)
        for name, digest in ready["external_files"].items():
            resolved = posixpath.normpath(prefix + "/" + name)
            assert not resolved.startswith("../")
            bound(resolved, digest)
        launch = read(prefix + "/output/launch-report.json")
        assert launch["complete"] and launch["passed"] and launch["exit_code"] == 0
        assert not launch["compute_processes_after_cleanup"]
        bound(prefix + "/ready.json", launch["ready_sha256"])
    old, new = read(OLD + "/output/report.json"), read(NEW + "/output/report.json")
    assert old["complete"] and old["status"] == "route_or_wait_failed_no_fleet_checks"
    assert (old["native_solves_started"], old["native_iterations"], old["flight_certificate_calls"], old["full_fleet_CPU_checks"], old["full_fleet_official_checks"]) == (8, 25, 7, 0, 0)
    assert new["complete"] and new["status"] == "certified_positive_candidate"
    assert (new["native_solves_started"], new["native_solves_returned"], new["native_iterations"], new["flight_certificate_calls"], new["wait_certificate_calls"]) == (17, 17, 34, 17, 3)
    assert new["incumbent_promotions"] == 0 and new["search_or_Lambert_calls"] == 0
    for field in ("request_sha256", "source_result_sha256", "selected_case"):
        assert new[field] == old[field]
    bound(NEW + "/output/fleet/Result.txt", RESULT)
    bound(NEW + "/output/fleet/official/Result.txt", RESULT)
    checker = read(NEW + "/output/fleet/checker-binding.json")
    assert checker["result_sha256"] == new["fleet_result_sha256"] == RESULT
    for key in ("independent", "official"):
        assert checker[key]["ok"] and checker[key]["result_sha256"] == RESULT
    assert checker["official"]["return_code"] == 0 and checker["independent"]["violations"] == []
    assert checker["independent"]["total_mass_kg"] == new["acceptance"]["raw_kg"]
    assert checker["independent"]["weighted_score_fixed_bonus_kg"] == new["acceptance"]["weighted_kg"] == 12843.92777477852
    assert new["acceptance"]["weighted_delta_from_retained_kg"] == 0.37207919333195605
    assert new["acceptance"]["both_checkers_passed"] and not new["acceptance"]["incumbent_promoted"]
    audit = read(BP + "boundary-candidate-review-v627/findings.json")
    assert audit["passed"] and audit["six_native_tests_and_sanitizers_pass"]
    bound(NEW + "/output/report.json", audit["report_sha256"])
    for key, value in audit["counts"].items():
        assert new[key] == value
    for name, digest in audit["output_hashes"].items():
        bound(NEW + "/output/" + name, digest)
    units = read(BP + "boundary-merit-tests-v627a/report.json")
    assert units["complete"] and units["passed"] and len(units["stages"]) == 6
    for stage in units["stages"]:
        assert stage["exit_code"] == stage["terminal_exit"] == 0
        bound(BP + "boundary-merit-tests-v627a/" + stage["name"] + ".log", stage["log_sha256"])
    manifest = read(BP + "boundary-merit-core-v627a/manifest.json")
    assert manifest["complete"] and manifest["gpu_calls"] == 0
    with tarfile.open(fileobj=io.BytesIO(payload[BP + "boundary-merit-core-v627a/source.tar.gz"]), mode="r:gz") as archive:
        actual = {item.name: sha(archive.extractfile(item).read()) for item in archive if item.isfile()}
    assert actual == manifest["source_sha256"] and len(actual) == 388
    mesh = read(BP + "mesh-refinement-review-v626/findings.json")
    assert mesh["passed"] and mesh["saved_cpu_tests_passed"] == 82 and mesh["non_docstring_ast_unchanged"]
    for name, digest in mesh["live_source_sha256"].items():
        bound(name, digest)
    qr = read(DQ + "/real-a/report.json")
    assert qr["complete"] and qr["passed"] and qr["evaluate_calls_observed"] == 1
    assert qr["result_flags"] == 64 and qr["numeric_pass"] == 0 and qr["original_termination"] == 2
    assert qr["fixed_x_s_retained_z_bits"] and not qr["compute_after"]
    assert (qr["QR_calls"], qr["apply_Q_calls"], qr["triangular_calls"]) == (1, 1, 2)
    bound(DQ + "/real-a/replay.log", qr["raw_log_sha256"])
    independent = read(BP + "dual-qr-real-review-v626/findings.json")
    assert independent["passed"] and independent["after_decimal65"]["passes"]
    assert not independent["diagnostic_numeric_pass"] and independent["diagnostic_flags"] == 64
    assert independent["original_native_termination"] == 2 and not independent["native_qualified"]
    assert independent["fixed_x_s_retained_z_bits"] and independent["quality_rejection_preserved"]
    bound(DQ + "/real-a/report.json", independent["report_sha256"])
    bound(DQ + "/real-a/records.json", independent["records_sha256"])
    own = read(DQ + "/sha256.json")
    for name, item in own["files"].items():
        bound(DQ + "/" + name, item["sha256"])
    tiny = read(DQ + "/tiny-a/report.json")
    assert tiny["complete"] and tiny["passed"] and tiny["evaluate_calls_observed"] == 10
    assert (tiny["QR_calls"], tiny["apply_Q_calls"], tiny["triangular_calls"]) == (7, 6, 11)
    omitted = index["omitted_files"]
    official = NEW + "/output/fleet/official/GTOC12_Verify"
    assert omitted[official]["bytes"] == 187200 and omitted[official]["sha256"] == "d4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6"
    expected_binaries = {official} | {DQ + "/build-b/binaries/" + name for name in ("libgpu_dual_qr.so", "dual_qr_test", "dual_qr_replay")}
    assert {name for name, item in omitted.items() if item["reason"].startswith("Compiled binary")} == expected_binaries
    for name, item in omitted.items():
        if name not in expected_binaries:
            assert any(part in ("__pycache__", ".pytest_cache", ".ruff_cache", ".git") for part in PurePosixPath(name).parts)
            assert item["reason"] == "Cache or Git metadata excluded."
    result = {"passed": True, "index_sha256": sha(raw_index), "archive_sha256": sha(raw_archive),
              "files": len(payload), "bytes": index["uncompressed_bytes"], "nested_inspection": stats,
              "historical_v626_status": old["status"], "historical_v627_status": new["status"],
              "historical_v627_Result_sha256": RESULT, "both_saved_checkers_bound": True,
              "historical_weighted_kg": new["acceptance"]["weighted_kg"],
              "GPU_dual_original_numeric_gates_pass": True, "GPU_dual_diagnostic_rejection_preserved": True,
              "native_termination": 2, "omitted_binary_bytes_reverified": False,
              "omitted_files": omitted, "project_code_executions": 0, "numerical_reruns": 0,
              "GPU_calls": 0, "checker_runs": 0, "propagations": 0}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        assert not args.output.exists(), "fresh verification output required"
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
