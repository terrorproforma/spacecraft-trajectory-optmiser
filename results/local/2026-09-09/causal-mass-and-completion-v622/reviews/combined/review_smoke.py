"""CPU-only source/hash review of the finite integration supervisor; never run it."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
path = ROOT / "build/performance/run_combined_smoke_v622.py"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = path.read_text()
tree = ast.parse(source)
constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)}
kit = ROOT / "build/performance/completion-pack-benchmark-v622"
assert sha(kit / "source-sha256.json") == constants["HOST_FULL_SHA"]
assert sha(kit / "source-full.tar.gz") == constants["HOST_ARCHIVE_SHA"]
assert sha(ROOT / "build/performance/combined-core-v622a/manifest.json") == constants["MANIFEST_SHA"]
assert sha(ROOT / "build/performance/completion-model-v622g/report.json") == constants["HOST_SHA"]
index = json.loads((kit / "source-sha256.json").read_text())
with tarfile.open(kit / "source-full.tar.gz") as archive:
    assert {m.name for m in archive if m.isfile()} == set(index)
    for name, digest in index.items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
        assert sha(kit / "source" / name) == digest
assert len(index) == 341
for text in ("pass_fds=(lock.fileno(),)", "child.wait(timeout=90)", "start_new_session=True", "fcntl.LOCK_EX | fcntl.LOCK_NB", "os.killpg(child.pid, signal.SIGTERM)", "os.killpg(child.pid, signal.SIGKILL)", "assert seen == set(host_sources)", "SPACEPDHCG_COMPLETION_TEST_LIBRARY=manifest['library']['path']", "assert UUID in report['gpu_inventory'] and not report['compute_before']"):
    assert text in source, text
findings = {
    "scope": "Source/AST/hash review only; no native imports, supervisor preflight or GPU calls executed by reviewer",
    "runner_sha256": sha(path),
    "manifest_sha256": constants["MANIFEST_SHA"], "host_report_sha256": constants["HOST_SHA"],
    "host_full_source_sha256": constants["HOST_FULL_SHA"], "host_full_archive_sha256": constants["HOST_ARCHIVE_SHA"],
    "host_files_verified": len(index), "full_host_dependency_binding_fixed": True,
    "maximum_processes": 3, "per_process_timeout_seconds": 90,
    "mass_solve_APIs": 7, "mass_iteration_caps": 9, "mass_expected_updates": 5,
    "legacy_nonempty_completion_evaluations": 2, "legacy_completion_candidates": 262,
    "compact_valid_calls": 18, "compact_candidate_evaluations": 2358, "compact_malformed_calls": 6,
    "scope_limit": "Same existing tests on combined linkage. No cold 100k repeat, numerical change, performance trial or fleet promotion.",
    "decision": "GO for one bounded supervised integration smoke with this exact runner and pins; actual outcomes remain to be audited",
}
(OUT / "smoke-preparation-findings.json").write_text(json.dumps(findings, indent=2) + "\n")
print(json.dumps({"passed": True, "runner_sha256": findings["runner_sha256"], "findings_sha256": sha(OUT / "smoke-preparation-findings.json")}))
