"""Lossless portable negative evidence; no native binaries or numerical reruns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIT = ROOT / "build/performance/fleet-budgeted-prefix-v630"
DEST = ROOT / "results/local/2026-09-09/fleet-budgeted-prefix-v630"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(source, target):
    target = DEST / target
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def archive(path, rows):
    with tarfile.open(DEST / path, "x:gz", compresslevel=9) as output:
        for source, name in rows:
            output.add(source, arcname=name, recursive=False)


def main():
    if not __debug__:
        raise RuntimeError("assertions required")
    ready = json.loads((KIT / "ready.json").read_bytes())
    for name, expected in ready["files"].items():
        assert sha(KIT / name) == expected, name
    prep = [(KIT / name, name) for name in ready["files"] if not name.startswith("host/")]
    prep.append((KIT / "ready.json", "ready.json"))
    archive("preparation.tar.gz", prep)
    output_files = [(p, str(p.relative_to(KIT / "output"))) for p in sorted((KIT / "output").rglob("*")) if p.is_file()]
    archive("outputs.tar.gz", output_files)
    for name in ("ready.json", "protocol.json", "profile.json"):
        copy(KIT / name, name)
    for name in ("report.json", "launch-report.json", "production-admission.json"):
        copy(KIT / "output" / name, name)
    native = ROOT / "build/performance/gpu-route-ephemeris-v630"
    copy(native / "manifest.json", "native-manifest.json")
    copy(native / "source.tar.gz", "native-source.tar.gz")
    copy(native / "gpu-tests-a/report.json", "prerequisites/native-gpu-tests.json")
    copy(ROOT / "build/performance/route-ephemeris-audit-v630/execution-a/worker/report.json", "prerequisites/archived-boundary-parity.json")
    for name in ("bundles.py", "refinement_admission.py"):
        copy(KIT / "host/spacepdhcg/gtoc12" / name, "policy/" + name)
    for name in ("test_gtoc12_fleet_admission.py", "test_gtoc12_bundles.py"):
        copy(ROOT / "tests" / name, "policy/" + name)
    for name in ("report.json", "README.md", "test-output.txt"):
        copy(ROOT / "build/performance/fleet-budget-admission-v630" / name, "policy-validation/" + name)
    for name in ("audit.py", "report.json", "package.py"):
        copy(HERE / name, "audit/" + name)
    # The validated wait helper is a ready-bound external. Preserve it for replay.
    copy(ROOT / "build/performance/seeded-route-v625/runtime.py", "support/validated-wait-runtime.py")
    for name in ("REPORT.md", "findings.json", "positive-requests.json", "current-ships.json", "attempt-signatures.json", "index.json"):
        copy(ROOT / "build/performance/frontier-opportunities-v630" / name, "opportunities/" + name)
    data = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}
    with (DEST / "sha256.json").open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"files": len(data), "bytes": sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()),
                      "index_sha256": sha(DEST / "sha256.json"), "raw_outputs": len(output_files)}))


if __name__ == "__main__":
    main()
