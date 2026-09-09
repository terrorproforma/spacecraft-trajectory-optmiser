"""Portable, stdlib-only byte/source/status audit. Never runs archived code."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile

NEW = "build/performance/gpu-dual-polish-v629"
OLD = "build/performance/gpu-dual-polish-v626"
PROFILE = "build/performance/dual-profile-v629"
REVIEW = "build/performance/dual-qr-real-review-v629"
PINS = {
    NEW + "/sha256.json": "ecaa60396775d3d3c258cb772faa76c777b52f84751a2882db06b6122f5be9f3",
    OLD + "/sha256.json": "86312946bab71d80950c1ca33fb56ad199f6e6048843c3fd201996ae34508f95",
    "build/performance/dual-isolation-review-v629/index.json": "6cf8cc2b2a043a08385536c5a3584fd9ae419819c31331cf269de7180d270c36",
    REVIEW + "/index.json": "595a39b8bf53dd83f14d13d11334e8bfb280963a3214a4a9a323d8757bc88248",
    "build/performance/dual-qr-real-review-v626/index.json": "89d42503863871753ce324e881d0943bbcb0244971d678512156f8b38ff6a3f3",
}
MAX_MEMBER = 64 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
BAD_SUFFIX = re.compile(r"\.(?:exe|dll|so(?:\.\d+)*|a|o|obj|pdb|cubin|fatbin|pyc|pyo)$", re.I)
PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def safe_name(name):
    p = PurePosixPath(name)
    require(name and "\\" not in name and "\x00" not in name, "unsafe name")
    require(not p.is_absolute() and str(p) == name, "noncanonical name: " + name)
    require(all(x not in ("", ".", "..", ".git", "__pycache__", ".pytest_cache", ".ruff_cache") for x in p.parts), "unsafe path: " + name)
    require(":" not in name and not BAD_SUFFIX.search(name), "forbidden member: " + name)
    require(p.name not in ("id_rsa", "id_ed25519", "id_ecdsa", ".env"), "credential filename")


def file_info(data):
    return {"bytes": len(data), "sha256": digest(data)}


def safe_bytes(name, data):
    safe_name(name)
    require(len(data) <= MAX_MEMBER, "oversized member: " + name)
    require(not data.startswith((b"\x7fELF", b"MZ")), "compiled executable: " + name)
    require(not PRIVATE_KEY.search(data), "private key data: " + name)


def unpack_tar(data):
    result = {}
    names = set()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive:
            name = member.name.rstrip("/")
            safe_name(name)
            require(name not in names, "duplicate tar path: " + name)
            names.add(name)
            require(member.isdir() or member.isfile(), "tar link/special member: " + name)
            if member.isdir():
                continue
            require(0 <= member.size <= MAX_MEMBER, "oversized tar member")
            stream = archive.extractfile(member)
            require(stream is not None, "unreadable tar member")
            payload = stream.read(MAX_MEMBER + 1)
            require(len(payload) == member.size, "truncated tar member")
            result[name] = payload
    return result


def unpack_zip(data):
    result = {}
    names = set()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        require(len(archive.infolist()) <= 5000, "too many ZIP members")
        for item in archive.infolist():
            name = item.filename.rstrip("/")
            safe_name(name)
            require(name not in names, "duplicate ZIP path: " + name)
            names.add(name)
            mode = item.external_attr >> 16
            kind = stat.S_IFMT(mode)
            require(kind in (0, stat.S_IFREG, stat.S_IFDIR), "ZIP link/special member")
            require(not item.flag_bits & 1, "encrypted ZIP member")
            if item.is_dir():
                continue
            require(0 <= item.file_size <= MAX_MEMBER, "oversized ZIP member")
            result[name] = archive.read(item)
    require(sum(map(len, result.values())) <= MAX_TOTAL, "oversized ZIP expansion")
    return result


def scan_members(files, depth=0, counters=None):
    if counters is None:
        counters = {"nested_archives": 0, "nested_members": 0, "expanded_scan_bytes": 0}
    require(depth <= 4, "archive nesting limit")
    for name, payload in files.items():
        safe_bytes(name, payload)
        counters["expanded_scan_bytes"] += len(payload)
        require(counters["expanded_scan_bytes"] <= MAX_TOTAL, "recursive byte limit")
        nested = None
        if name.endswith((".tar.gz", ".tgz")):
            nested = unpack_tar(payload)
        elif name.endswith((".zip", ".npz")) or payload.startswith(b"PK\x03\x04"):
            nested = unpack_zip(payload)
        if nested is not None:
            counters["nested_archives"] += 1
            counters["nested_members"] += len(nested)
            scan_members(nested, depth + 1, counters)
    return counters


def raw_records(data):
    records = {}
    for line in data.decode("utf-8").splitlines():
        if line.startswith("GPU_DUAL_QR_"):
            tag, body = line.split(" ", 1)
            records.setdefault(tag, []).append(strict_json(body))
    return records


def audit(package, expected_index=None):
    package = Path(package).resolve()
    raw_index = (package / "index.json").read_bytes()
    if expected_index:
        require(digest(raw_index) == expected_index, "index pin mismatch")
    index = strict_json(raw_index)
    require(index["schema"] == "gpu-dual-isolation-package-v629/1", "wrong package schema")
    for name, info in index["top_files"].items():
        safe_name(name)
        require(len(PurePosixPath(name).parts) == 1, "unexpected top-level nesting")
        data = (package / name).read_bytes()
        require(file_info(data) == info, "top-file mismatch: " + name)
        safe_bytes(name, data)
    require(index["top_files"]["audit_package.py"]["sha256"] == digest(Path(__file__).read_bytes()), "auditor identity mismatch")
    archive = (package / "evidence.zip").read_bytes()
    require(digest(archive) == index["archive_sha256"] and len(archive) == index["archive_bytes"], "archive mismatch")
    files = unpack_zip(archive)
    require(set(files) == set(index["files"]), "ZIP/member map mismatch")
    for name, info in index["files"].items():
        require(file_info(files[name]) == info, "member mismatch: " + name)
    require(len(files) == index["file_count"], "file count mismatch")
    require(sum(map(len, files.values())) == index["total_bytes"], "expanded bytes mismatch")
    safety = scan_members(files)

    def get(path):
        require(path in files, "missing dependency: " + path)
        return files[path]

    def read(path):
        return strict_json(get(path))

    def sha(path, expected):
        require(digest(get(path)) == expected, "binding mismatch: " + path)

    for path, pin in PINS.items():
        sha(path, pin)
        sealed = read(path)
        prefix = str(PurePosixPath(path).parent) + "/"
        require(sealed["file_count"] == len(sealed["files"]), "sealed count mismatch")
        for name, info in sealed["files"].items():
            require(file_info(get(prefix + name)) == info, "sealed member mismatch: " + name)
        omitted = sealed.get("omitted_binaries", {})
        if isinstance(omitted, dict):
            require(not any(prefix + name in files for name in omitted), "omitted binary included")

    source_counts = {}
    for folder in (OLD + "/build-a", OLD + "/build-b", NEW + "/build-a"):
        manifest = read(folder + "/manifest.json")
        sha(folder + "/source.tar.gz", manifest["source_archive_sha256"])
        sources = unpack_tar(get(folder + "/source.tar.gz"))
        actual = {name: file_info(data) for name, data in sorted(sources.items())}
        require(actual == manifest["source_files"], "compiled source member mismatch")
        tree_sha = digest(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode())
        require(tree_sha == manifest["source_tree_sha256"], "source tree mismatch")
        source_counts[folder] = len(sources)
    current = read(NEW + "/build-a/manifest.json")
    old = read(OLD + "/build-b/manifest.json")
    require(current["passed"] and current["complete"] and current["GPU_evaluate_calls"] == 0, "build status")
    for name, info in current["source_files"].items():
        require(file_info(get(NEW + "/" + name)) == info, "frozen source/root mismatch")

    tiny = read(NEW + "/tiny-a/report.json")
    tiny_rows = read(NEW + "/tiny-a/records.json")
    require(tiny["complete"] and tiny["passed"] and tiny["retries"] == 0, "tiny status")
    require(tiny["evaluate_calls_observed"] == len(tiny_rows) == 13, "tiny count")
    require([row["case"] for row in tiny_rows] == tiny["expected_cases"], "tiny case order")
    sha(NEW + "/tiny-a/test.log", tiny["raw_log_sha256"])
    require(raw_records(get(NEW + "/tiny-a/test.log"))["GPU_DUAL_QR_RESULT"] == tiny_rows, "tiny raw records")
    for key, report_key, count in (("qr_calls", "QR_calls", 9), ("apply_q_calls", "apply_Q_calls", 7), ("triangular_calls", "triangular_calls", 13)):
        require(sum(row[key] for row in tiny_rows) == tiny[report_key] == count, "tiny operation count")
    require(all(row["source_sha256"] == current["source_tree_sha256"] and row["original_termination"] == 2 for row in tiny_rows), "tiny source/status")
    require(tiny_rows[-3]["numeric_pass"] == 1 and tiny_rows[-2]["flags"] == 8 and tiny_rows[-1]["flags"] == 128, "isolation controls")
    require(all(tiny_rows[-1][k] == 0 for k in ("qr_calls", "apply_q_calls", "triangular_calls")), "all-isolated no-op work")

    real = read(NEW + "/real-a/report.json")
    real_rows = read(NEW + "/real-a/records.json")
    require(set(real_rows) == {"GPU_DUAL_QR_RESULT", "GPU_DUAL_QR_META", "GPU_DUAL_QR_COMPLETE"} and all(isinstance(v, list) and len(v) == 1 for v in real_rows.values()), "real record cardinality")
    result = real_rows["GPU_DUAL_QR_RESULT"][0]
    meta = real_rows["GPU_DUAL_QR_META"][0]
    require(real["complete"] and real["passed"] and real["retries"] == 0, "real status")
    require(real["evaluate_calls_observed"] == 1 and real["CPU_numerical_factorizations"] == real["CPU_optimizations"] == 0, "real work scope")
    require(result["flags"] == 0 and result["numeric_pass"] == 1 and result["original_termination"] == meta["original_termination"] == 2, "real diagnostic/native distinction")
    require((result["active_columns"], result["isolated_columns"]) == (538, 4), "real structure")
    require((result["qr_calls"], result["apply_q_calls"], result["triangular_calls"]) == (1, 1, 2), "real operation count")
    sha(NEW + "/real-a/replay.log", real["raw_log_sha256"])
    require(real_rows == raw_records(get(NEW + "/real-a/replay.log")), "real raw records")
    for sub, report in (("tiny-a", tiny), ("real-a", real)):
        sha(NEW + "/build-a/manifest.json", report["build_manifest_sha256"])
        sha(NEW + "/" + sub + "/build-manifest.json", report["build_manifest_sha256"])
        require(report["source_tree_sha256"] == current["source_tree_sha256"], "report tree")
        runner = "run_tiny.py" if sub == "tiny-a" else "run_real.py"
        sha(NEW + "/" + sub + "/" + runner, report["runner_sha256"])
    require(meta["source_sha256"] == result["source_sha256"] == current["source_tree_sha256"], "real compiled tree")

    findings = read(REVIEW + "/findings.json")
    require(findings["passed"] and findings["all_original_quality_checks_passed"], "original audit pass")
    require(findings["diagnostic_numeric_pass"] and findings["diagnostic_flags"] == 0, "audit diagnostic status")
    require(not findings["native_qualified"] and findings["original_native_termination"] == 2, "audit native status")
    require(findings["long_double_audits"][-1]["passes_common_kkt_gate"] and not findings["long_double_audits"][-1]["qualified"], "original gate/status distinction")
    require(findings["actual_pair_sum_exact_lambda_count"] == 525, "exact pair count")
    require(findings["fixed_x_s_retained_z_bits"] and findings["isolated_y_bits_preserved"], "fixed-vector assertions")
    for field, name in (("report_sha256", "report.json"), ("records_sha256", "records.json"), ("raw_log_sha256", "replay.log")):
        sha(NEW + "/real-a/" + name, findings[field])
    for name, pin in findings["input_sha256"].items():
        sha(NEW + "/inputs/" + name, pin)
    for kind, name in (("decimal", "decimal_auditor.py"), ("long_double", "long_double_auditor.py")):
        sha("build/performance/dual-polish-v625/prepared-a/inputs/" + name, findings["auditor_sha256"][kind])

    profile = read(PROFILE + "/profile-a/report.json")
    require(profile["complete"] and profile["passed"] and profile["records_passed"] and profile["diagnostic_rejection_preserved"], "profile status")
    require(profile["evaluations"] == profile["evaluation_limit"] == profile["profile_limit"] == 1 and profile["unprofiled_replays"] == 0, "profile scope")
    require(profile["exit_code"] == profile["child_exit_after_cleanup"] == 0 and profile["cleanup"]["verified_empty"] and not profile["cleanup"]["survivors"], "profile cleanup")
    sha(PROFILE + "/profile.py", profile["runner_sha256"])
    sha(PROFILE + "/process_guard.py", profile["process_guard_sha256"])
    sha(OLD + "/build-b/manifest.json", profile["manifest_sha256"])
    for name, pin in profile["inputs"].items():
        sha(OLD + "/inputs/" + name, pin)
        sha(NEW + "/inputs/" + name, pin)
    profile_rows = read(PROFILE + "/profile-a/records.json")
    require(set(profile_rows) == {"GPU_DUAL_QR_RESULT", "GPU_DUAL_QR_META", "GPU_DUAL_QR_COMPLETE"} and all(isinstance(v, list) and len(v) == 1 for v in profile_rows.values()), "profile record cardinality")
    require(profile_rows == raw_records(get(PROFILE + "/profile-a/profile.log")), "profile raw records")
    pr = profile_rows["GPU_DUAL_QR_RESULT"][0]
    require(pr["flags"] == 64 and pr["numeric_pass"] == 0 and pr["original_termination"] == 2, "old rejection changed")
    require(profile_rows["GPU_DUAL_QR_META"][0]["source_sha256"] == old["source_tree_sha256"], "profile old source")
    require(set(profile["artifacts"]) == {"trace.nsys-rep", "trace.sqlite"}, "trace artifact names")
    for name, info in profile["artifacts"].items():
        require(file_info(get(PROFILE + "/profile-a/" + name)) == info and info["bytes"] > 0, "trace artifact binding")
    analysis = read(PROFILE + "/profile-a/analysis.json")
    sha(PROFILE + "/profile-a/trace.sqlite", analysis["sqlite_sha256"])
    require(not analysis["kernel_activity_available"] and not analysis["GPU_memory_activity_available"], "profile capability scope")
    require("not supported" in json.dumps(analysis["diagnostics"]), "missing driver limitation")

    timing = read("build/performance/dual-isolation-review-v629/qoco-timing-findings.json")
    require(timing["passed"] and not timing["matched_qualified_complete_latency_comparison_exists"], "historical timing scope")
    for name, pin in timing["input_sha256"].items():
        sha(name, pin)
    return {
        "passed": True, "index_sha256": digest(raw_index),
        "archive_sha256": digest(archive), "archive_bytes": len(archive),
        "file_count": len(files), "total_bytes": sum(map(len, files.values())),
        "source_archive_members": source_counts, "safety": safety,
        "tiny_evaluations": 13, "real_evaluations": 1,
        "diagnostic_numeric_pass": True, "original_native_termination": 2,
        "native_qualified": False, "profile_old_rejection_preserved": True,
        "profile_kernel_activity_available": False,
        "executed_numerical_audits": 0, "GPU_calls": 0,
        "optimizer_calls": 0, "propagation_calls": 0,
        "scope": "Saved bytes, source maps, exact raw/report bindings and recorded status only; archived code was not executed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-index")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.package, args.expected_index)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
