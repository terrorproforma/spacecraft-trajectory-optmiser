"""Index the reviewed local preparation; refuse absent/failed CPU evidence."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


validation = json.loads((root / "validation.json").read_text())
audit = json.loads((root / "cpu-audit.json").read_text())
assert validation["passed"] and audit["passed"] and audit["gpu_calls"] == 0
assert not audit["native_library_load_attempts"]
assert not (root / "launch.json").exists(), "preparation has already launched"
for name, expected in validation["files"].items():
    assert digest(root / name) == expected, name
source = json.loads((root / "source-sha256.json").read_text())
inputs = json.loads((root / "inputs/sha256.json").read_text())
for name, expected in source.items():
    assert digest(root / "source" / name) == expected, name
for name, expected in inputs.items():
    assert digest(root / "inputs" / name) == expected, name
suite = ET.parse(root / "pytest.xml").getroot().find("testsuite")
assert suite is not None
assert suite.attrib["failures"] == suite.attrib["errors"] == suite.attrib["skipped"] == "0"
names = set(validation["files"])
names.update("source/" + name for name in source)
names.update("inputs/" + name for name in inputs)
names.update(
    {
        "README.md",
        "finalize.py",
        "source-sha256.json",
        "inputs/sha256.json",
        "preparation.json",
        "plan/plan.json",
        "plan/report.json",
        "cpu-audit.json",
        "validation.json",
        "pytest.xml",
        "pytest.log",
        "cpu-audit.log",
        "ruff-check.log",
        "ruff-format.log",
        "launch-recipe.log",
    }
)
record = {
    "status": "ready_for_root_review_not_GPU_executed",
    "source_commit": audit["source_commit"],
    "source_files": len(source),
    "input_files": len(inputs),
    "tests_passed": int(suite.attrib["tests"]),
    "tests_skipped": 0,
    "cases": audit["cases"],
    "epoch_seeds": audit["epoch_seeds"],
    "maximum_joint_candidates_including_polish": audit["maximum_joint_candidates_including_polish"],
    "gpu_calls": 0,
    "files": {name: digest(root / name) for name in sorted(names)},
}
(root / "ready-manifest.json").write_text(json.dumps(record, indent=2) + "\n")
for name, expected in record["files"].items():
    assert digest(root / name) == expected, name
print(json.dumps({k: v for k, v in record.items() if k != "files"}))
print(
    json.dumps(
        {
            "ready_manifest_sha256": digest(root / "ready-manifest.json"),
            "indexed_files": len(record["files"]),
            "driver_sha256": digest(root / "run.py"),
        }
    )
)
