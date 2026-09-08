"""Freeze final additive integration source and every control/test artifact."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
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


def function(path, name):
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", "") == name)
    return ast.dump(node, include_attributes=False)


def main():
    assert all(x["returncode"] == 0 for x in read(KIT / "validation/attempt-05/report.json"))
    assert "76 passed" in (KIT / "validation/attempt-05/pytest.log").read_text()
    assert read(KIT / "output/report.json")["fresh_scalar_finish_cpu_calls"] == 34
    assert read(KIT / "readback-audit.json")["passed"]
    old = KIT / "inputs/fixed_refine-v606.py"
    current = ROOT / "src/spacepdhcg/gtoc12/fixed_refinement.py"
    assert function(old, "refine_fixed") == function(current, "refine_fixed")
    original_queue = KIT / "source/src/spacepdhcg/gtoc12/refinement_admission.py"
    current_queue = ROOT / "src/spacepdhcg/gtoc12/refinement_admission.py"
    a = ast.parse(original_queue.read_text())
    b = ast.parse(current_queue.read_text())
    for tree in (a, b):
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "RefinementAdmissionQueue":
                node.body = [n for n in node.body if getattr(n, "name", "") != "retained"]
    assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    base = read(KIT / "source-sha256.json")
    overlay = KIT / "implementation"
    overlay.mkdir(exist_ok=False)
    files = sorted((ROOT / "src").rglob("*.py"))
    files += [
        ROOT / "tests" / name
        for name in (
            "test_gtoc12_refinement_admission.py",
            "test_gtoc12_fixed_refinement.py",
            "test_gtoc12_completion_capture.py",
        )
    ]
    retained = {}
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        digest = sha(path)
        if base.get(name) == digest:
            continue
        destination = overlay / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(path) == sha(destination) == digest
        retained[name] = digest
    write(KIT / "implementation-sha256.json", retained)
    write(
        KIT / "final-source-audit.json",
        {
            "passed": True,
            "original_refine_fixed_function_AST_unchanged": True,
            "new_direct_input_validation": "Immutable prescription validation before runner creation",
            "new_leg_telemetry": "Preserve actual record.certification_backend",
            "historical_policy_difference": "Only additive retained() getter; no policy/model rerun",
            "frozen_control_source_index_sha256": sha(KIT / "source-sha256.json"),
            "final_integration_overlay_index_sha256": sha(KIT / "implementation-sha256.json"),
            "overlay_files": len(retained),
            "reconstruction": "Copy source/, then implementation/ over it; use explicit reconstructed src import path.",
            "validation": "76 CPU tests, no native load/solve; no new core validation claimed here",
        },
    )
    index = {}
    for path in sorted(KIT.rglob("*")):
        if not path.is_file():
            continue
        assert "__pycache__" not in path.parts and path.suffix not in {".so", ".dll", ".exe"}
        index[path.relative_to(KIT).as_posix()] = {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    ready = {
        "scope": "CPU-only v622 admission controls and additive integration; no GPU launch authorized by this file",
        "files": index,
        "file_count": len(index),
        "total_bytes": sum(x["bytes"] for x in index.values()),
        "control_report_sha256": sha(KIT / "output/report.json"),
        "readback_audit_sha256": sha(KIT / "readback-audit.json"),
        "owned_production_sha256": {
            name: sha(ROOT / name)
            for name in (
                "src/spacepdhcg/gtoc12/refinement_admission.py",
                "src/spacepdhcg/gtoc12/fixed_refinement.py",
                "tests/test_gtoc12_refinement_admission.py",
                "tests/test_gtoc12_fixed_refinement.py",
                "tests/test_gtoc12_completion_capture.py",
            )
        },
    }
    write(KIT / "ready.json", ready)
    for name, item in index.items():
        assert sha(KIT / name) == item["sha256"]
    print(
        json.dumps(
            {
                "ready_sha256": sha(KIT / "ready.json"),
                "files": len(index),
                "bytes": ready["total_bytes"],
                "owned": ready["owned_production_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
