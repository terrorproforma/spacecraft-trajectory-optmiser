"""Package exact known-point and fixed-cargo outcomes without binaries or reruns."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

workspace = Path(__file__).resolve().parents[2]
scratch = workspace / "build/performance"
base = workspace / "results/local/2026-09-09"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def archive_files(target, rows):
    assert not target.exists()
    expected = {}
    with tarfile.open(target, "w:gz") as archive:
        for name, path in sorted(rows):
            assert name not in expected and path.is_file() and not path.is_symlink()
            assert not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
            expected[name] = sha(path)
            archive.add(path, arcname=name, recursive=False)
    with tarfile.open(target) as archive:
        actual = {member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                  for member in archive.getmembers() if member.isfile()}
    assert actual == expected
    return len(expected)


def regular_files(root):
    return [(path.relative_to(root).as_posix(), path) for path in root.rglob("*")
            if path.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".ruff_cache") for part in path.parts)]


def verify_index(root):
    rows = read(root / "sha256.json")
    for name, value in rows.items():
        assert sha(root / name) == value["sha256"], name
        assert (root / name).stat().st_size == value["bytes"], name
    return len(rows)


known = base / "persistent-known-point-v606"
known.mkdir(exist_ok=True)
analytic = scratch / "persistent-known-point-v606c"
real = scratch / "known-point-replay-v606"
assert verify_index(analytic) == 74
manifest = read(analytic / "manifest.json")
with tarfile.open(analytic / "source.tar.gz") as tar:
    for name in manifest["owned_paths"]:
        member = next(m for m in tar.getmembers() if m.name.removeprefix("./") == name)
        assert sha(workspace / name) == hashlib.sha256(tar.extractfile(member).read()).hexdigest(), name
real_report = read(real / "run/report.json")
audit = read(real / "audit/findings.json")
assert real_report["complete"] and audit["complete"]
assert audit["report_sha256"] == sha(real / "run/report.json")
assert len(real_report["cases"]) == 8 and all(c["runtime_identity_matches"] and c["independent_native_audit_agrees"] for c in real_report["cases"])
assert sha(analytic / "manifest.json") == real_report["manifest_sha256"]
assert (real / "analysis/REPORT.md").is_file(), "wait for completed numerical assessment"
shutil.copytree(analytic, known / "analytic")
for directory in ("audit", "analysis"):
    shutil.copytree(real / directory, known / directory, ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(scratch / "objective-balance-v606", known / "objective-balance", ignore=shutil.ignore_patterns("__pycache__"))
copy_file(real / "run/report.json", known / "real/report.json")
known_members = archive_files(known / "real/raw.tar.gz", [("known-point-replay-v606/" + name, path) for name, path in regular_files(real)])
copy_file(scratch / "encode_known_points_v606.py", known / "encode_known_points_v606.py")
copy_file(Path(__file__), known / Path(__file__).name)

truth = base / "fixed-cargo-truth-v606"
truth.mkdir(exist_ok=True)
kit = scratch / "truth-set-v606"
ready = read(kit / "ready-manifest.json")
assert sha(kit / "ready-manifest.json") == "7e87c5360efaee14ddaec94602ac4132b610073d782118d6e4cb122312c4d2cc"
for name, expected in ready["files"].items():
    assert sha(kit / name) == expected, name
report = read(kit / "output/report.json")
assert report["complete"] and report["profile"] == "local" and report["status"] == "incumbent_retained"
assert report["whole_route_refinements_started"] == 4 and report["native_solves_completed"] == report["native_solves_started"] == 66
assert len(report["promotions"]) == 0
assert read(kit / "launch.json")["returncode"] == 0
truth_audit = read(scratch / "truth-set-audit-v606/report.json")
assert truth_audit["complete"] and truth_audit["report_sha256"] == sha(kit / "output/report.json")
source_rows = [("execution/" + name, kit / name) for name in ready["files"]]
source_rows += [("execution/ready-manifest.json", kit / "ready-manifest.json")]
truth_source_members = archive_files(truth / "source.tar.gz", source_rows)
truth_raw_members = archive_files(truth / "raw.tar.gz", [("output/" + name, path) for name, path in regular_files(kit / "output")])
for name in ("README.md", "common.py", "fixed_refine.py", "run.py", "launch.py", "profiles.json", "preparation.json", "ready-manifest.json", "cpu-audit.json", "validation/report.json", "launch.json", "run.log", "plan/plan.json"):
    copy_file(kit / name, truth / "execution" / name)
copy_file(kit / "output/report.json", truth / "report.json")
copy_file(kit / "output/native-solves.jsonl", truth / "native-solves.jsonl")
shutil.copytree(scratch / "truth-set-audit-v606", truth / "audit", ignore=shutil.ignore_patterns("__pycache__"))
copy_file(Path(__file__), truth / Path(__file__).name)

record = {"complete": True, "analytic_index_files": 74, "known_raw_members": known_members,
          "truth_source_members": truth_source_members, "truth_raw_members": truth_raw_members,
          "known_report_sha256": sha(real / "run/report.json"), "truth_report_sha256": sha(kit / "output/report.json"),
          "owned_diagnostic_sources_match_frozen_build": True, "new_gpu_calls": 0,
          "lambda_execution": False, "lambda_payload_transfer": False, "fleet_gain_kg": 0}
for output in (known, truth):
    (output / "publication-audit.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / ".gitattributes").write_bytes(b"* -text whitespace=cr-at-eol\n*.log -whitespace\n")
print(json.dumps(record))
