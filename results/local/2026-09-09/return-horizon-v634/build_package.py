"""Package terminal saved evidence only, excluding executables and external data."""

import hashlib
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KIT = ROOT / "build/performance/return-horizon-v634"
OUT = ROOT / "results/local/2026-09-09/return-horizon-v634"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    require(not OUT.exists(), "Fresh publication directory required")
    ready_raw = (KIT / "ready.json").read_bytes()
    require(sha(ready_raw) == "ac6379bdaf8bdc28f4296509ab9aa3bddbac2903c53c9edd15759529879d5edf", "Ready identity")
    ready = json.loads(ready_raw)
    report = json.loads((KIT / "output/report.json").read_bytes())
    launch = json.loads((KIT / "output/launch-report.json").read_bytes())
    require(report["complete"] and launch["complete"] and launch["passed"], "Terminal execution/cleanup required")
    require(launch["owned_descendant_cleanup"]["verified_empty"], "Cleanup must be complete")
    payloads = {}
    for path in sorted(KIT.rglob("*")):
        if path.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".ruff_cache") for part in path.parts):
            payloads[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name, digest in ready["files"].items():
        require(sha((KIT / name).read_bytes()) == digest, f"Sealed input changed: {name}")
    runtime = {}
    for name, digest in ready["external_files"].items():
        if name.startswith("/"):
            runtime[name] = {"sha256": digest, "scope": "Exact runtime/data reference; checked by saved launch preflight and omitted from this source/results package"}
            continue
        path = (KIT / name).resolve()
        require(path.is_relative_to(ROOT), "Unexpected external project path")
        raw = path.read_bytes()
        require(sha(raw) == digest, f"External source drift: {name}")
        require(path.suffix not in (".so", ".dll", ".exe", ".pyc"), "No executables in package")
        payloads[path.relative_to(ROOT).as_posix()] = raw
    for name in ("build/performance/mass-merit-return-v632/output/report.json",
                 "build/performance/mass-merit-return-v632/output/candidate_mass_return/legs/00/native-raw.json"):
        payloads[name] = (ROOT / name).read_bytes()
    payloads["package/runtime-only-references.json"] = encode(runtime)
    files = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in sorted(payloads.items())}
    OUT.mkdir()
    with zipfile.ZipFile(OUT / "evidence.zip", "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(payloads.items()):
            archive.writestr(name, raw)
        archive.writestr("FILES.json", encode(files))
    for name in ("audit_saved.py", "verify_package.py", "saved-audit.json", "build_package.py", "README.md", "NEXT.md"):
        (OUT / name).write_bytes((HERE / name).read_bytes())
    (OUT / ".gitattributes").write_bytes(b"* -text -whitespace\n")
    for name in ("report.json", "launch-report.json"):
        (OUT / name).write_bytes((KIT / "output" / name).read_bytes())
    (OUT / "package-map.json").write_bytes(encode({"archive_original_paths": True, "archive_files": len(files),
        "prepared_files": len(ready["files"]), "raw_output_files": len(list((KIT / "output").rglob("*"))) - len([p for p in (KIT / "output").rglob("*") if p.is_dir()]),
        "archived_external_project_files": len(ready["external_files"]) - len(runtime),
        "runtime_only_references": runtime, "ready_sha256": sha(ready_raw),
        "no_new_solver_or_verifier_executions": True}))
    index = {path.name: {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())} for path in sorted(OUT.iterdir()) if path.is_file()}
    (OUT / "index.json").write_bytes(encode({"files": index}))
    print(json.dumps({"directory": str(OUT), "files": len(index), "archive_files": len(files),
          "index_sha256": sha((OUT / "index.json").read_bytes()), "evidence_sha256": sha((OUT / "evidence.zip").read_bytes())}, indent=2))


if __name__ == "__main__":
    main()
