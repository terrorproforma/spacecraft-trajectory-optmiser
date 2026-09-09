"""Portable stdlib byte/readback-binding audit. No numerical or native execution."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tempfile
import zipfile

PREFIX = "build/performance/qoco-exact-comparison-v629/"
REPORT = "a0fdfc00da6b41d6a68e91ab9978818a3da5b3e901fc489020ce232e85eed520"
AUDIT = "eda8bf6fd6c64246edd69f66e5143f37296bdf5ad962565135fd2a738fa881ca"
SNAPSHOT = "4d3b8c1641722c0e44fa5e9518e5bf0311dbc8ad3847f45eaf9cc2af2f6eede5"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(raw, pin):
    assert len(raw) == pin["bytes"] and sha(raw) == pin["sha256"]


def safe(name, raw):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and ".." not in path.parts and ":" not in name and "\\" not in name
    assert not {".git", "__pycache__", ".pytest_cache", ".venv"}.intersection(path.parts)
    assert path.suffix.lower() not in {".exe", ".dll", ".so", ".o", ".obj", ".pyc", ".pem", ".key"}
    assert not raw.startswith((b"\x7fELF", b"MZ"))
    for line in raw.splitlines():
        assert not line.strip().startswith((b"-----BEGIN PRIVATE KEY-----", b"-----BEGIN OPENSSH PRIVATE KEY-----"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-index", required=True)
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    index_raw = (args.package / "index.json").read_bytes()
    assert sha(index_raw) == args.expected_index
    index = json.loads(index_raw)
    for name, pin in index["top_files"].items():
        check((args.package / name).read_bytes(), pin)
    assert (args.package / ".gitattributes").read_text().strip() == "* -text -whitespace"
    raw_zip = (args.package / "evidence.zip").read_bytes()
    assert sha(raw_zip) == index["archive_sha256"] and len(raw_zip) == index["archive_bytes"]
    blobs = {}
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as archive:
        assert len(archive.infolist()) == len(set(archive.namelist())) == index["file_count"]
        assert set(archive.namelist()) == set(index["files"])
        for entry in archive.infolist():
            assert not entry.is_dir() and ((entry.external_attr >> 16) & 0o170000) != 0o120000
            raw = archive.read(entry)
            safe(entry.filename, raw)
            check(raw, index["files"][entry.filename])
            blobs[entry.filename] = raw
    assert sum(map(len, blobs.values())) == index["total_bytes"]
    get = lambda name: json.loads(blobs[PREFIX + name])
    nested_sources = {}
    for attempt in ("a", "b"):
        manifest = get("build-" + attempt + "/manifest.json")
        archive_raw = blobs[PREFIX + "build-" + attempt + "/source.tar.gz"]
        assert sha(archive_raw) == manifest["source_archive_sha256"]
        entries = {}
        with tarfile.open(fileobj=io.BytesIO(archive_raw)) as archive:
            for entry in archive:
                assert entry.isfile() and entry.name not in entries
                raw = archive.extractfile(entry).read()
                safe(entry.name, raw)
                check(raw, manifest["source_files"][entry.name])
                entries[entry.name] = raw
        assert set(entries) == set(manifest["source_files"]) and len(entries) == 19
        tree = sha(json.dumps(manifest["source_files"], sort_keys=True, separators=(",", ":")).encode())
        assert tree == manifest["source_tree_sha256"] and manifest["GPU_solve_calls"] == 0
        for stage in manifest["stages"]:
            assert sha(blobs[PREFIX + "build-" + attempt + "/" + stage["name"] + ".log"]) == stage["log_sha256"]
            assert stage["exit_code"] == stage["expected_exit"]
        assert manifest["complete"] and manifest["passed"] == (attempt == "b")
        nested_sources[attempt] = entries
    a, b = nested_sources["a"], nested_sources["b"]
    assert {name for name in a if a[name] != b[name]} == {"replay.cu"}
    assert a["replay.cu"].replace(b'<<s.verbose', b'<<static_cast<int>(s.verbose)') == b["replay.cu"]
    assert b["replay.cu"] == blobs[PREFIX + "replay.cu"]
    assert sha(blobs[PREFIX + "audit-inputs/snapshot.txt"]) == SNAPSHOT
    report, records, audit, build = [get(name) for name in ("run-a/report.json", "run-a/records.json", "audit-findings.json", "build-b/manifest.json")]
    assert sha(blobs[PREFIX + "run-a/report.json"]) == REPORT and sha(blobs[PREFIX + "audit-findings.json"]) == AUDIT
    for name, pin in audit["input_pins"].items():
        assert sha(blobs[PREFIX + name]) == pin
    parsed = {}
    for line in blobs[PREFIX + "run-a/replay.log"].decode().splitlines():
        if line.startswith("QOCO_COMPARE_"):
            tag, text = line.split(" ", 1)
            parsed.setdefault(tag, []).append(json.loads(text))
    assert parsed == records and all(len(value) == 1 for value in records.values())
    meta, point, timing = [records[key][0] for key in ("QOCO_COMPARE_META", "QOCO_COMPARE_RESULT", "QOCO_COMPARE_COMPLETE")]
    assert point["status"] == point["solution_status"] == report["native_status"] == audit["native_status"] == 2
    assert audit["native_status_name"] == "QOCO_SOLVED_INACCURATE"
    assert audit["common_numeric_gate_pass"] and audit["qoco_backend_qualified"]
    assert audit["original_decimal65"]["passes"] and audit["original_long_double"]["qualified"]
    assert point["iterations"] == 30 and point["ir_iterations"] == 177
    assert point["solve_calls"] == report["solve_calls"] == timing["solve_calls"] == 1
    assert point["numeric_updates"] == report["numeric_updates"] == timing["numeric_updates"] == 2
    assert report["complete"] and report["passed"] and report["child_terminal"] and report["child"]["exit_code"] == 0
    assert report["process_cleanup"]["verified_empty"] and report["final_process_cleanup"]["verified_empty"]
    assert not report["compute_before"] and not report["compute_after"] and not report["final_compute_inventory"]
    assert meta["settings"] == report["original_settings"] == build["original_settings"] and len(meta["settings"]) == 13
    assert meta["source_sha256"] == report["source_tree_sha256"] == build["source_tree_sha256"]
    assert timing == report["host_timing"] and timing["host_elapsed_phases"] == audit["host_elapsed_phases_seconds"]
    for name, pin in audit["vector_hashes"].items():
        assert len(point[name]) == pin["elements"]
        assert sha(struct.pack("<" + "d" * len(point[name]), *point[name])) == pin["sha256_fp64_little_endian"]
    inspection = get("source-inspection.json")
    assert sha(blobs[PREFIX + "source-inspection.json"]) == build["source_inspection_sha256"]
    for name, pin in inspection["source_files"].items():
        check(blobs[PREFIX + "source-observations/qoco/" + name], pin)
    check(blobs[PREFIX + "source-observations/v683-build-report.json"], inspection["historical_build_report"])
    for name, pin in inspection["compile_headers"].items():
        check(b["qoco/" + name], pin)
    assert index["external_runtime_binaries"] == [build[key] for key in ("artifact", "qoco_library", "cudss_runtime")]
    assert blobs[PREFIX + "run-a/run_once.py"] == blobs[PREFIX + "run_once.py"]
    assert blobs[PREFIX + "run-a/process_guard.py"] == blobs[PREFIX + "process_guard.py"]
    roundtrip_count = 0
    if args.roundtrip:
        with tempfile.TemporaryDirectory(prefix="qoco-static-v629-") as tmp:
            for name, raw in blobs.items():
                path = Path(tmp).joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            for name, pin in index["files"].items():
                check(Path(tmp).joinpath(*PurePosixPath(name).parts).read_bytes(), pin)
                roundtrip_count += 1
    findings = {"passed": True, "index_sha256": args.expected_index, "archive_sha256": index["archive_sha256"],
                "files": len(blobs), "expanded_bytes": sum(map(len, blobs.values())), "nested_source_files": 38,
                "roundtrip_files": roundtrip_count, "native_status": 2, "native_status_name": "QOCO_SOLVED_INACCURATE",
                "saved_common_gate_pass": True, "recorded_solve_calls": 1,
                "external_binary_bytes_reverified": False, "numerical_audits_repeated": 0,
                "archived_code_executed": 0, "native_loads": 0, "GPU_calls": 0}
    if args.output:
        with args.output.open("x") as output:
            json.dump(findings, output, indent=2, sort_keys=True)
            output.write("\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
