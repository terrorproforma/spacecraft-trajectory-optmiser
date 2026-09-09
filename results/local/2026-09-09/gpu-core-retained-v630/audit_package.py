"""Portable stdlib archive/readback audit. Executes no archived numerical code."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import tarfile
import tempfile
import zipfile

PREFIX = "build/performance/gpu-core-retained-v630/"
REPORT = "460e79d92b9b1940af925d408a17a26168b26f42cb6f6411d0f14dc3ee450169"
AUDIT = "ad5db3aa98540bd0cd1fa24710f590c5e34f42ac8c6a04313cc94f2bb8534a90"
BUILD = "44baa8f8356b52166fb3940d221e4c2ad34daae60c4d032c064e77735544491b"


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
    archive_raw = (args.package / "evidence.zip").read_bytes()
    assert sha(archive_raw) == index["archive_sha256"] and len(archive_raw) == index["archive_bytes"]
    blobs = {}
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        assert len(archive.infolist()) == len(set(archive.namelist())) == index["file_count"]
        assert set(archive.namelist()) == set(index["files"])
        for item in archive.infolist():
            assert not item.is_dir() and ((item.external_attr >> 16) & 0o170000) != 0o120000
            raw = archive.read(item)
            safe(item.filename, raw)
            check(raw, index["files"][item.filename])
            blobs[item.filename] = raw
    assert sum(map(len, blobs.values())) == index["total_bytes"]
    get = lambda name: json.loads(blobs[PREFIX + name])
    build, report, audit = [get(name) for name in ("build-a/manifest.json", "run-b/report.json", "audit-findings.json")]
    assert sha(blobs[PREFIX + "build-a/manifest.json"]) == BUILD
    assert sha(blobs[PREFIX + "run-b/report.json"]) == REPORT
    assert sha(blobs[PREFIX + "audit-findings.json"]) == AUDIT
    assert build["complete"] and build["passed"] and build["GPU_solve_calls"] == build["dual_evaluate_calls"] == 0
    assert sha(blobs[PREFIX + "build-a/source.tar.gz"]) == build["source_archive_sha256"]
    source_files = {}
    with tarfile.open(fileobj=io.BytesIO(blobs[PREFIX + "build-a/source.tar.gz"])) as archive:
        for item in archive:
            assert item.isfile() and item.name not in source_files
            raw = archive.extractfile(item).read()
            safe(item.name, raw)
            check(raw, build["source_files"][item.name])
            source_files[item.name] = raw
    assert set(source_files) == set(build["source_files"]) and len(source_files) == 29
    tree = sha(json.dumps(build["source_files"], sort_keys=True, separators=(",", ":")).encode())
    assert tree == build["source_tree_sha256"]
    for name, raw in source_files.items():
        if not name.startswith("qoco/"):
            assert raw == blobs[PREFIX + name]
    assert len(build["stages"]) == 21
    for stage in build["stages"]:
        assert sha(blobs[PREFIX + "build-a/" + stage["name"] + ".log"]) == stage["log_sha256"]
        assert stage["exit_code"] == stage["expected_exit"]
    assert get("validation/report.json")["CPU_tests"] == 22
    old = blobs[PREFIX + "preparation-a/run_batch.py"]
    current = blobs[PREFIX + "run_batch.py"]
    assert old.replace(b'output = HERE / "run-a"', b'output = HERE / "run-b"') == current
    assert blobs[PREFIX + "run-a/run_batch.py"] == old
    assert blobs[PREFIX + "run-b/run_batch.py"] == current
    assert get("run-a/report.json")["processes"] == []
    assert "BlockingIOError" in get("run-a/report.json")["error"]
    assert report["complete"] and report["passed"] and report["primary_solves"] == 8
    assert report["PDHCG_updates"] == 40000 and report["QOCO_iterations"] == 115 and report["QOCO_ir_iterations"] == 783
    assert report["correction_evaluations"] == 0
    assert report["final_process_cleanup"]["verified_empty"] and not report["final_compute_inventory"]
    assert not report["compute_before"] and report["runner_sha256"] == sha(current)
    assert report["libraries"] == build["libraries"]
    assert index["external_runtime_binaries"] == [*build["artifacts"].values(), *build["libraries"].values()]
    for name, pin in audit["input_pins"].items():
        assert sha(blobs[PREFIX + name]) == pin
    for saved in ("localization.json", "qoco-comparison.json"):
        for name, pin in get(saved)["pins"].items():
            # Windows analysis records use native separators in provenance labels.
            # ZIP membership itself is strictly POSIX and already safety-checked.
            assert sha(blobs[PREFIX + name.replace("\\", "/")]) == pin
    order = [("early", "persistent_chain"), ("early", "qoco_retained"),
             ("near_converged", "qoco_retained"), ("near_converged", "persistent_chain")]
    qualified = {"persistent_chain": 0, "qoco_retained": 0}
    work = {"PDHCG_updates": 0, "QOCO_iterations": 0, "QOCO_ir_iterations": 0}
    for i, (entry, outcome, expected) in enumerate(zip(report["processes"], audit["outcomes"], order)):
        capture, backend = expected
        assert (entry["capture"], entry["backend"]) == (outcome["capture"], outcome["backend"]) == expected
        assert entry["child"]["exit_code"] == 0 and entry["child_terminal"] and entry["process_cleanup"]["verified_empty"]
        assert not entry["compute_before"] and not entry["compute_after"]
        folder = f"run-b/{i}-{capture}-{backend}/"
        raw_log = blobs[PREFIX + folder + "replay.log"]
        assert sha(raw_log) == entry["raw_log_sha256"]
        assert sha(blobs[PREFIX + folder + "records.json"]) == entry["records_sha256"]
        rows = get(folder + "records.json")
        parsed = {}
        for line in raw_log.decode().splitlines():
            if line.startswith(("CORE_RETAINED_", "QOCO_RETAINED_", "GPU_DUAL_QR_")):
                tag, text = line.split(" ", 1)
                parsed.setdefault(tag, []).append(json.loads(text))
        assert parsed == rows and "GPU_DUAL_QR_RESULT" not in rows
        assert sha(blobs[PREFIX + "inputs/" + capture + ".txt"]) == build["inputs"][capture + ".txt"] == entry["snapshot_sha256"]
        native = rows["CORE_RETAINED_PDHCG" if backend == "persistent_chain" else "QOCO_RETAINED_RESULT"]
        trials = rows["CORE_RETAINED_TRIAL" if backend == "persistent_chain" else "QOCO_RETAINED_TRIAL"]
        assert [r["trial"] for r in native] == [0, 1] == [r["trial"] for r in trials]
        for point, trial, saved in zip(native, trials, outcome["trials"]):
            result = saved["native"]
            for key in ("x", "y", "z", "s"):
                digest = sha(struct.pack("<" + "d" * len(point[key]), *point[key]))
                assert digest == result["FP64_vector_sha256"][key]
            assert result["original_decimal65"]["passes"] == result["original_long_double"]["passes_common_kkt_gate"]
            qualified[backend] += result["original_long_double"]["qualified"]
            if backend == "persistent_chain":
                assert result["native_status"] == point["termination"] == 2
                assert point["cold_zero_bits_checked"] and point["scaling_refreshed"] == 1
                assert point["iterations"] == 10000 and not point["deadline_requested"]
                assert trial["reason"] == "correction_skipped_primal_gate" and not trial["composite_qualified"]
                assert trial["correction_calls_total"] == trial["dual_creates_total"] == 0
                assert result["original_decimal65"]["primal"] > 1e-9
                work["PDHCG_updates"] += point["iterations"]
            else:
                assert result["native_status"] == point["status"] == point["solution_status"]
                assert trial["cold_start_action"] == 0 and trial["numeric_updates_this_trial"] == 2
                work["QOCO_iterations"] += point["iterations"]
                work["QOCO_ir_iterations"] += point["ir_iterations"]
    assert qualified == {"persistent_chain": 0, "qoco_retained": 1}
    assert all(report[key] == value for key, value in work.items())
    assert audit["work"]["primary_solves"] == 8 and audit["work"]["correction_evaluations"] == 0
    assert blobs["docs/GPU_CORE_RETAINED_COLD_STARTS.md"] == blobs[PREFIX + "REPORT.md"]
    roundtrip = 0
    if args.roundtrip:
        with tempfile.TemporaryDirectory(prefix="retained-static-v630-") as tmp:
            for name, raw in blobs.items():
                path = Path(tmp).joinpath(*PurePosixPath(name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            for name, pin in index["files"].items():
                check(Path(tmp).joinpath(*PurePosixPath(name).parts).read_bytes(), pin)
                roundtrip += 1
    findings = {"passed": True, "index_sha256": args.expected_index, "archive_sha256": index["archive_sha256"],
                "files": len(blobs), "expanded_bytes": sum(map(len, blobs.values())), "nested_source_files": 29,
                "roundtrip_files": roundtrip, "qualified": qualified, "recorded_primary_solves": 8,
                "recorded_corrections": 0, "external_binary_bytes_reverified": False,
                "numerical_audits_repeated": 0, "archived_code_executed": 0, "native_loads": 0, "GPU_calls": 0}
    if args.output:
        with args.output.open("x") as target:
            json.dump(findings, target, indent=2, sort_keys=True)
            target.write("\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
