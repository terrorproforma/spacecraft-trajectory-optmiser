"""Standard-library static byte/readback verifier; never executes archived code."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile

KIT = "build/performance/core-primal-reference-v631/"
PINS = {
    "ready.json": "38e02779892d7e8c2a0b85121fda0d46a3a0660259e9519648d956b39f454186",
    "prepare-a/findings.json": "44e825703b250a8f1ad02ca69b73ca1e74f53b4e97787244614445c57d937080",
    "validation/tiny-a.json": "a1465b7246d945525205fbe4fab68c47a571f2be11c3a7aefead9bc893e94a1d",
    "run-a/report.json": "b3cd0507d6fcaf9b54265cabb8526a65fa00a3afb41a74ae39a796c977e90a6c",
    "run-a/worker/report.json": "a7aff487abd60b6afea5dddbddd020e3044647dd5c17c1db2ed1c640bc499e69",
    "audit-a/findings.json": "67f158a159c8a3156ceb34883a5009e38b7ec21c9f35ea1e2266a5ceee82f86f",
    "reference.py": "21a9da338e3e698ea5cc5a85477e3c5db80dfde2275a89a1a098401dcbdf494f",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    path = PurePosixPath(name)
    assert name and str(path) == name and not path.is_absolute()
    assert "\\" not in name and ":" not in name and ".." not in path.parts
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-index")
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    raw_index = (package / "index.json").read_bytes()
    if args.expected_index:
        assert digest(raw_index) == args.expected_index
    index = json.loads(raw_index)
    assert index["schema"] == "core_primal_reference_v631_static_package"
    for name, metadata in index["auxiliary_files"].items():
        safe(name)
        data = (package / name).read_bytes()
        assert len(data) == metadata["bytes"] and digest(data) == metadata["sha256"]
    archive_path = package / "evidence.zip"
    archive = archive_path.read_bytes()
    assert len(archive) == index["archive"]["bytes"]
    assert digest(archive) == index["archive"]["sha256"]
    payload = {}
    with zipfile.ZipFile(archive_path) as z:
        assert len(z.infolist()) == len(index["files"])
        assert set(z.namelist()) == set(index["files"])
        for member in z.infolist():
            path = safe(member.filename)
            assert not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16)
            assert not any(p in (".git", "__pycache__", ".pytest_cache") for p in path.parts)
            assert path.suffix.lower() not in (".so", ".dll", ".exe", ".o", ".a", ".pyc", ".pem", ".key")
            data = z.read(member)
            assert not data.startswith((b"\x7fELF", b"MZ"))
            assert not any(line.startswith(b"-----BEGIN ") and b"PRIVATE KEY-----" in line
                           for line in data.splitlines())
            metadata = index["files"][member.filename]
            assert len(data) == metadata["bytes"] == member.file_size
            assert digest(data) == metadata["sha256"]
            payload[member.filename] = data
    assert len(payload) == index["member_count"]
    assert sum(map(len, payload.values())) == index["expanded_bytes"]
    for name, expected in PINS.items():
        assert digest(payload[KIT + name]) == expected, name

    def read(name):
        return json.loads(payload[KIT + name])

    ready, prep, tiny = read("ready.json"), read("prepare-a/findings.json"), read("validation/tiny-a.json")
    for name, metadata in ready["files"].items():
        data = payload[KIT + name]
        assert digest(data) == metadata["sha256"] and len(data) == metadata["bytes"]
    assert tiny["passed"] and tiny["captured_inputs_loaded"] == tiny["GPU_calls"] == 0
    assert prep["complete"] and prep["captured_iterations"] == prep["captured_primal_projection_calls"] == 0
    launch, worker, audited = read("run-a/report.json"), read("run-a/worker/report.json"), read("audit-a/findings.json")
    assert launch["complete"] and launch["passed"] and launch["processes_started"] == 1
    assert launch["worker_terminal"] and launch["final_returncode"] == launch["worker_returncode"] == 0
    assert launch["preflight"]["ready_sha256"] == digest(payload[KIT + "ready.json"])
    assert launch["worker_report_sha256"] == digest(payload[KIT + "run-a/worker/report.json"])
    assert worker["complete"] and worker["cold_runs_started"] == 2
    assert worker["actual_updates"] == 20000 and worker["actual_projection_calls"] == 20002
    assert worker["GPU_calls"] == worker["native_solver_calls"] == 0
    assert worker["original_numeric_passes"] == audited["original_numeric_passes"] == 0
    assert audited["complete"] and len(audited["rows"]) == 8 and len(audited["finals"]) == 2
    assert audited["solver_calls"] == audited["projection_calls"] == audited["factorizations"] == 0
    assert audited["worker_report_sha256"] == digest(payload[KIT + "run-a/worker/report.json"])
    for source, pin in worker["source_sha256"].items():
        assert digest(payload[KIT + source]) == pin
    for source, pin in worker["auditor_sha256"].items():
        assert digest(payload[KIT + "audit-inputs/" + source]) == pin
    checkpoints = 0
    for case in worker["cases"]:
        assert case["iterations"] == case["attempted_updates"] == 10000
        assert case["projection_calls"] == 10001 and case["reference_status"] == "iteration_limit"
        assert not case["qualified_numeric"]
        input_data = payload[KIT + "inputs/" + case["capture"] + ".txt"]
        assert digest(input_data) == case["snapshot_sha256"]
        lines = input_data.decode().splitlines()
        assert len(lines) == 15 and lines[0] == "SPACEPDHCG_QOCO_QP_V1"
        n, p, m, nq, *_ = map(int, lines[1].split())
        assert (n, p, m, nq) == (1886, 542, 3324, 453)
        assert all(float(value) == 0.0 for value in lines[11].split()[1:1 + nq])
        for cp in case["checkpoints"]:
            data = payload[KIT + "run-a/worker/" + cp["path"]]
            assert digest(data) == cp["sha256"]
            point = json.loads(data)
            assert not point["native_solver_called"]
            assert point["termination"] == point["termination_code"] == 0
            row = next(row for row in audited["rows"]
                       if row["capture"] == case["capture"] and row["iterations"] == cp["iterations"])
            assert row["point_sha256"] == cp["sha256"]
            assert row["original_export_identities_pass"]
            assert row["Decimal65"]["passes"] == row["long_double"]["passes_common_kkt_gate"]
            assert not row["Decimal65"]["qualified"] and not row["long_double"]["qualified"]
            for key, length in zip(("x", "y", "z", "s"), (1886, 542, 3324, 3324)):
                assert len(point[key]) == length
                vector = struct.pack("<" + "d" * length, *point[key])
                assert digest(vector) == row["vector_FP64_sha256"][key]
            checkpoints += 1
    assert checkpoints == 8
    extraction = None
    if args.extract:
        root = args.extract.resolve()
        root.mkdir(parents=True, exist_ok=False)
        for name, data in payload.items():
            path = root.joinpath(*safe(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            assert path.read_bytes() == data
        extraction = str(root)
    result = {
        "passed": True, "index_sha256": digest(raw_index),
        "archive_sha256": index["archive"]["sha256"],
        "members_verified": len(payload), "expanded_bytes": index["expanded_bytes"],
        "full_vector_checkpoints_bound": checkpoints, "CPU_updates_recorded": 20000,
        "projections_recorded": 20002, "numeric_qualifications_recorded": 0,
        "archived_code_executed": 0, "numeric_audits_rerun": 0, "GPU_calls": 0,
        "fresh_roundtrip_directory": extraction,
    }
    if args.output:
        assert not args.output.exists()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
