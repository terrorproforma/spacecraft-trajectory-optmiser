"""Seal completed v621 evidence after saved-output CPU rechecks; no GPU/Git work."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "results/local/2026-09-09/gpu-route-completion-v621"
SCRATCH = ROOT / "build/performance/completion-package-v621"
assert not (DEST / "sha256.json").exists()
success = SCRATCH / "portable-recheck-c/cpu-recheck"
report = json.loads((success / "report.json").read_text())
assert report["passed"] and len(report["commands"]) == 5
assert all(c["returncode"] == 0 for c in report["commands"])
assert report["GPU_calls"] == report["native_library_loads"] == 0
shutil.copytree(success, DEST / "reproduce/portable-pass")
shutil.copy2(Path(__file__), DEST / "reproduce/seal.py")
(DEST / "reproduce/PORTABILITY.md").write_text(
    "The final CPU-only portability run passes five audits: saved native controls, "
    "saved production adapter vectors, coefficient-only diagonal metric, exact mass "
    "structure, and every saved benchmark vector/timing. No native library is loaded.\n\n"
    "Attempt a preserved an audit dependency whose pinned Windows-authored JSON was "
    "regenerated with Linux LF before the dependent hash check. Its contents were identical. "
    "Attempt b passed all five mathematical audits but the wrapper's final equality check "
    "compared Windows and POSIX input-path labels literally. The final wrapper audits the "
    "untouched dependency first and compares regenerated JSON values with only input-hash "
    "path labels normalized. Numerical fields and hash values are unchanged. Original "
    "evidence and both failed wrapper attempts remain byte-preserved.\n",
    encoding="utf-8")
status = {
    "state": "complete", "sealed": True, "pending": [],
    "scope": "Local RTX 5090 route-completion proxy costing; default off; no fleet promotion.",
    "native_validation": {"passed": True, "kernels": 3, "candidate_evaluations": 304},
    "adapter_validation": {"passed": True, "evaluation_calls": 8, "candidate_evaluations": 1048, "lambert_requests": 0},
    "benchmark": {"passed": True, "GPU_calls": 60, "CPU_batches": 60, "candidate_evaluations_per_backend": 14200,
                  "unique_historical_controls": 20, "default_promoted": False},
    "CPU_portability_audits": {"passed": True, "audits": 5, "GPU_calls": 0, "native_library_loads": 0},
    "source_storage": {"snapshots": 5, "unique_objects": 344, "executable_binaries_included": False},
}
(DEST / "STATUS.json").write_text(json.dumps(status, indent=2)+"\n")
index = {}
for path in sorted(DEST.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(DEST)
    assert not any(p in {"__pycache__", ".pytest_cache", ".ruff_cache"} for p in relative.parts)
    assert path.suffix.lower() not in {".pyc", ".so", ".dll", ".exe", ".o", ".a", ".pem", ".key"}
    data = path.read_bytes()
    index[relative.as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
(DEST / "sha256.json").write_text(json.dumps(index, indent=2)+"\n")
sys.path.insert(0, str(DEST / "reproduce"))
from portable import verify
verification = verify(DEST)
print(json.dumps({**verification, "indexed_bytes": sum(x["bytes"] for x in index.values()),
                  "index_sha256": hashlib.sha256((DEST / "sha256.json").read_bytes()).hexdigest(),
                  "package": str(DEST)}, indent=2))
