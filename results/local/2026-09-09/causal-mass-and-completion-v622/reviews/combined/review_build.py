"""Read-only CPU audit of the final combined CMake source/build evidence."""
from pathlib import Path
import hashlib
import io
import json
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BUILD = ROOT / "build/performance/combined-core-v622a"
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (BUILD / "manifest.json").read_bytes()
assert sha(raw) == "9bff6c54af1edca6323502908d440305b4edc8447550572a5d93b50a421e3730"
m = json.loads(raw)
assert m["complete"] and m["gpu_calls"] == 0 and m["reused_compiled_objects"] == []
assert len(m["stages"]) == 17
assert all(x["returncode"] == x["expected"] for x in m["stages"])
assert sha((BUILD / "source.tar.gz").read_bytes()) == m["source_archive_sha256"]
with tarfile.open(BUILD / "source.tar.gz") as archive:
    assert set(archive.getnames()) == set(m["source_sha256"])
    for path, digest in m["source_sha256"].items():
        assert sha(archive.extractfile(path).read()) == digest, path
mass_path = ROOT / "build/performance/mass-core-v622a/manifest.json"
compact_path = ROOT / "build/performance/completion-model-v622g/report.json"
assert sha(mass_path.read_bytes()) == m["mass_component_manifest_sha256"] == "ca970d7acd9893ba71d7e58f85393cb2a26c003ca931e2071e2718e4292ad026"
assert sha(compact_path.read_bytes()) == m["compact_component_report_sha256"] == "56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a"
mass, compact = json.loads(mass_path.read_text()), json.loads(compact_path.read_text())
assert len(m["owned_sha256"]) == 14
for path, digest in m["owned_sha256"].items():
    reference = mass["owned_sha256"].get(path, compact["owned_sources"].get(path))
    assert reference == digest == m["source_sha256"][path] == sha((ROOT / path).read_bytes()), path
base_bytes = subprocess.check_output(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", m["base_commit"], "cpp", "third_party"], cwd=ROOT)
assert sha(base_bytes) == m["base_archive_sha256"]
with tarfile.open(fileobj=io.BytesIO(base_bytes), mode="r:") as archive:
    base = {x.name: sha(archive.extractfile(x).read()) for x in archive if x.isfile()}
assert base == m["base_source_sha256"]
assert all(m["source_sha256"][name] == value for name, value in base.items() if name not in m["owned_paths"])
assert set(m["source_sha256"]) - set(base) == set(m["owned_paths"]) - set(base)
assert sha("".join(name + ":" + value + "\n" for name, value in m["source_sha256"].items()).encode()) == m["source_tree_sha256"]
patterns = ["cooperative_l1_kernelILb1", "cooperative_l1_kernelILb0", "cooperative_solve_kernelILb1", "cooperative_solve_kernelILb0", "12solve_kernelILb1", "12solve_kernelILb0", "cooperative_halpern_kernel", "cooperative_mass_kernel", "cooperative_mass_initialise_kernel"]


def resources(path):
    found = {}
    for name, row in re.findall(r"Function ([^\n]+):\n\s+(REG:[^\n]+)", path.read_text()):
        keys = [key for key in patterns if key in name]
        if keys:
            assert len(keys) == 1 and keys[0] not in found
            found[keys[0]] = {k: int(v) for k, v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", row)}
    assert len(found) == len(patterns)
    return found


current_resources = resources(BUILD / "resources.log")
assert current_resources == resources(ROOT / "build/performance/mass-core-v622a/resources.log")
parsers = {}
for case in ("conditioning", "difficult"):
    for mode in ("unit", "mass", "mass-seed"):
        path = BUILD / f"validate-{case}-{mode}.log"
        records = [json.loads(line.split(" ", 1)[1], parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s))) for line in path.read_text().splitlines() if line.startswith("PERSISTENT_")]
        assert records and records[0]["source_tree_sha256"] == m["source_tree_sha256"]
        if mode != "unit":
            assert len(records[0]["mass_map"]) == (211 if case == "conditioning" else 234)
        parsers[f"{case}-{mode}"] = {"records": len(records), "sha256": sha(path.read_bytes())}
commands = json.loads((BUILD / "compile-commands.json").read_text())
assert sha((BUILD / "compile-commands.json").read_bytes()) == m["compile_commands_sha256"]
selected_commands = [x for x in commands if Path(x["file"]).name in ("gtoc12_completion.cu", "persistent_pdhcg.cu")]
assert len(selected_commands) == 2
for entry in selected_commands:
    command = entry["command"]
    assert "sm_120" in command and "-rdc=true" in command
    if Path(entry["file"]).name == "gtoc12_completion.cu":
        assert "--fmad=false" in command
    else:
        isolated_commands = json.loads((ROOT / "build/performance/mass-core-v622a/compile-commands.json").read_text())
        prior = next(x["command"] for x in isolated_commands if Path(x["file"]).name == "persistent_pdhcg.cu")
        assert command.replace("spacepdhcg-combined-v622a", "spacepdhcg-mass-v622a") == prior
symbols = (BUILD / "symbols.log").read_text()
for symbol in ("spacepdhcg_cuda_workspace_set_mass_options", "spacepdhcg_cuda_workspace_mass_diagnostics", "spacepdhcg_gtoc12_completion_evaluate_host", "spacepdhcg_gtoc12_completion_evaluate_compact_host"):
    assert symbol in symbols
findings = {
    "scope": "CPU-only source/archive/build/log audit; no native loading or GPU execution by reviewer",
    "manifest_sha256": sha(raw), "source_tree_sha256": m["source_tree_sha256"],
    "source_archive_sha256": m["source_archive_sha256"], "source_files": len(m["source_sha256"]),
    "verified_git_base": m["base_commit"], "exact_reviewed_overlays": len(m["owned_sha256"]),
    "component_pins_verified": True, "working_tree_foreign_changes_excluded": True,
    "core": m["library"], "mass_test": m["test"], "completion_test": m["completion_test"], "replay": m["replay"],
    "resources_equal_isolated_mass_build": current_resources,
    "native_compilation_flags": selected_commands,
    "hidden_replay_checks": parsers,
    "stage_logs_sha256": {x["name"]: sha((BUILD / (x["name"] + ".log")).read_bytes()) for x in m["stages"]},
    "decision": "Source/build checks pass for bounded integration smoke; no cold solve or performance qualification inferred",
}
(OUT / "build-findings.json").write_text(json.dumps(findings, indent=2) + "\n")
print(json.dumps({"passed": True, "files": len(m["source_sha256"]), "manifest_sha256": sha(raw), "findings_sha256": sha((OUT / "build-findings.json").read_bytes())}))
