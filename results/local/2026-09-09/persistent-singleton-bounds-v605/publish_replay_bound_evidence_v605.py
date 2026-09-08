"""Reconcile native source archives and independently re-audit the bound experiment."""

import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(root):
    records = json.loads((root / "sha256.json").read_text(encoding="utf-8-sig"))
    for name, record in records.items():
        assert sha(root / name) == record["sha256"], name
        assert (root / name).stat().st_size == record["bytes"], name
    return len(records)


def archive_hashes(path):
    result = {}
    with tarfile.open(path) as archive:
        for item in archive.getmembers():
            name = item.name.removeprefix("./")
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            if item.isdir():
                continue
            assert item.isfile() and name not in result
            result[name] = hashlib.sha256(archive.extractfile(item).read()).hexdigest()
    return result


workspace = Path(__file__).resolve().parents[2]
baseline = workspace / "results/local/2026-09-09/persistent-snapshot-v603"
analytic = workspace / "build/performance/persistent-replay-v604b"
real = workspace / "build/performance/persistent-bound-ablation-v605"
output = workspace / "results/local/2026-09-09/persistent-singleton-bounds-v605"
assert not output.exists(), "do not overwrite published evidence"
analytic_count = index(analytic)
source = archive_hashes(analytic / "source.tar.gz")
old_source = archive_hashes(baseline / "analytic/source.tar.gz")
manifest = json.loads((analytic / "manifest.json").read_text())
assert manifest["complete"]
assert manifest["immutable_core_sha256"] == (
    "d4b0bea9672bb0cea612b7d4e8db5d448f1b79b9831be5710723276edd128633"
)
for name, expected in manifest["source_sha256"].items():
    assert source.get(name) == expected, name
for name in set(source) - set(manifest["source_sha256"]):
    assert name.startswith("third_party/") and source[name] == old_source[name], name
for name in manifest["owned_paths"]:
    assert sha(workspace / name) == source[name], name
for name in set(source) - set(manifest["owned_paths"]):
    assert source[name] == old_source[name], name

report = json.loads((real / "report.json").read_text())
assert report["complete"] and len(report["cases"]) == 4
assert report["adapter_manifest_sha256"] == sha(analytic / "manifest.json")
assert report["executable_sha256"] == manifest["executable_sha256"]
assert report["source_pin"] == manifest["frozen_commit"]
assert report["independent_auditor_sha256"] == sha(real / "audit_persistent_snapshot.py")
spec = importlib.util.spec_from_file_location("audit", real / "audit_persistent_snapshot.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audited = []
for case in report["cases"]:
    assert case["returncode"] == 0 and not case["timed_out"]
    assert case["runtime_identity_matches"] and case["independent_native_audit_agrees"]
    log = real / (case["name"] + ".log")
    snapshot = real / "inputs" / (case["label"] + ".txt")
    assert sha(log) == case["log_sha256"] and sha(snapshot) == case["input_sha256"]
    results = audit.audit_log(
        audit.load_snapshot(snapshot), log, sha(snapshot), backend="persistent",
        coordinates="original", record_prefix="PERSISTENT_REPLAY",
    )
    assert results == case["audits"]
    assert len(results) == 1 and case["qualified"] == sum(r["qualified"] for r in results)
    audited.append({"name": case["name"], "qualified": case["qualified"],
                    "log_sha256": sha(log), "audit_recomputed": True})

output.mkdir()
shutil.copytree(analytic, output / "analytic")
(output / "real").mkdir()
shutil.copyfile(real / "report.json", output / "real/report.json")
with tarfile.open(output / "real/raw.tar.gz", "w:gz") as archive:
    for path in sorted(real.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            archive.add(path, arcname="run/" + path.relative_to(real).as_posix())
raw = archive_hashes(output / "real/raw.tar.gz")
for name, expected in raw.items():
    assert name.startswith("run/") and sha(real / name.removeprefix("run/")) == expected
shutil.copyfile(__file__, output / Path(__file__).name)
record = {
    "complete": True,
    "analytic_index_files": analytic_count,
    "source_members": len(source),
    "cpp_sources_verified": len(manifest["source_sha256"]),
    "owned_source_files_matching_checkout": len(manifest["owned_paths"]),
    "non_owned_sources_unchanged_from_v603": len(set(source) - set(manifest["owned_paths"])),
    "raw_members": len(raw),
    "independent_cases_recomputed": audited,
}
(output / "publication-audit.json").write_text(json.dumps(record, indent=2) + "\n")
(output / ".gitattributes").write_bytes(b"* -text whitespace=cr-at-eol\n*.log -whitespace\n")
print(json.dumps(record))
