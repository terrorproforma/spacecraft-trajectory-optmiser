"""Verify/extract v619 evidence, or explicitly rerun its CPU-only archived checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

PACKAGE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def read(path):
    return json.loads(Path(path).read_text())


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def payloads():
    index = read(PACKAGE / "sha256.json")
    for name, expected in index["files"].items():
        assert digest((PACKAGE / name).read_bytes()) == expected["sha256"], name
    assert len(index["files"]) == index["file_count"]
    reconstruction = read(PACKAGE / "reconstruction.json")
    blobs = {}
    for archive, members in reconstruction["archives"].items():
        with tarfile.open(PACKAGE / archive, "r:gz") as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and ".." not in path.parts
                content = stream.extractfile(member).read()
                assert digest(content) == members[member.name]
                key = (archive, member.name)
                assert key not in blobs
                blobs[key] = content
        assert set(name for file, name in blobs if file == archive) == set(members)
    recovered = {}
    for name, location in reconstruction["kit"].items():
        recovered[name] = (blobs[tuple(location["archive"])] if "archive" in location
                           else (PACKAGE / location["file"]).read_bytes())
    original = read(PACKAGE / "kit/evidence-manifest.json")
    assert set(recovered) == set(original["files"])
    for name, content in recovered.items():
        expected = original["files"][name]
        assert digest(content) == expected["sha256"] and len(content) == expected["bytes"], name
    roots = {}
    for name, location in reconstruction["root_audit"].items():
        roots[name] = (blobs[tuple(location["archive"])] if "archive" in location
                       else (PACKAGE / location["file"]).read_bytes())
        assert digest(roots[name]) == reconstruction["root_audit_sha256"][name]
    return recovered, roots


def materialize(work, task, recovered, roots):
    work.mkdir(parents=True, exist_ok=False)
    skip = ({"output/baseline.json", "output/candidate-final.json", "output/audit.json"}
            if task == "bridges" else {"output/audit.json"} if task == "audit" else set())
    for name, payload in recovered.items():
        dest = work / "kit" / name
        if name in skip:
            dest = work / "reference" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    for name, payload in roots.items():
        dest = work / "root-audit" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    (work / "kit/evidence-manifest.json").write_bytes((PACKAGE / "kit/evidence-manifest.json").read_bytes())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def operation(work, name, data):
    kit = work / "kit"
    sys.path.insert(0, str(kit))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if name in ("baseline", "candidate-final"):
        filename = "replay-initial.py" if name == "baseline" else "replay.py"
        module = load_module("packaged_replay", kit / filename)
        original_read = module.read

        def relocated_read(path):
            value = original_read(path)
            if Path(path) == kit / "inputs/v616-profile.json":
                value["data"] = str(data)
            return value

        module.read = relocated_read
        sys.argv = [str(kit / filename), name]
        module.main()
    elif name == "audit":
        module = load_module("packaged_audit", kit / "audit.py")
        original_read = module.read

        def relocated_read(path):
            value = original_read(path)
            if Path(path) == kit / "inputs/incumbent-inventory.json":
                for row in value["routes"]:
                    row["path"] = str(work / "root-audit/inputs" / f"ship-{row['ship']:02}.json")
            return value

        module.read = relocated_read
        module.main()
    elif name == "tests":
        module = load_module("packaged_validation", kit / "validate_final.py")
        original_read = module.read

        def relocated_read(path):
            value = original_read(path)
            if Path(path) == kit / "inputs/v616-profile.json":
                value["data"] = str(data)
            return value

        module.read = relocated_read
        raise SystemExit(module.tests())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("verify", "extract", "audit", "bridges", "tests"))
    parser.add_argument("--work", type=Path)
    parser.add_argument("--data", type=Path, help="Pinned GTOC12 data directory; no automatic downloads")
    parser.add_argument("--_operation", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._operation:
        operation(args.work.resolve(), args._operation, args.data.resolve() if args.data else None)
        return
    recovered, roots = payloads()
    if args.task == "verify":
        print(json.dumps({"verified_kit_files": len(recovered), "verified_root_files": len(roots),
                          "GPU_calls": 0, "numerical_reruns": 0}))
        return
    if args.work is None:
        parser.error("--work must name a new directory")
    if args.task in ("bridges", "tests") and (args.data is None or not args.data.is_dir()):
        parser.error("--data must point to the existing pinned GTOC12 data directory")
    work = args.work.resolve()
    materialize(work, args.task, recovered, roots)
    if args.task == "extract":
        for name, payload in recovered.items():
            assert (work / "kit" / name).read_bytes() == payload
        print(json.dumps({"roundtrip_kit_files": len(recovered), "root_files": len(roots),
                          "GPU_calls": 0, "numerical_reruns": 0}))
        return
    operations = ["baseline", "candidate-final", "audit"] if args.task == "bridges" else [args.task]
    for name in operations:
        command = [sys.executable, "-B", str(__file__), args.task, "--work", str(work), "--_operation", name]
        if args.data is not None:
            command += ["--data", str(args.data.resolve())]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=240,
                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        (work / f"{name}.log").write_text(completed.stdout + completed.stderr)
        print(completed.stdout, end="")
        if completed.returncode:
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
