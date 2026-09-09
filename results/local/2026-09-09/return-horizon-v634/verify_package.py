"""Portable archive/hash and saved-output audit; never execute the campaign."""

import argparse
import hashlib
import json
import posixpath
import zipfile
from pathlib import Path

from audit_saved import KIT, READY, audit, require, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    index_raw = (root / "index.json").read_bytes()
    require(sha(index_raw) == args.index_sha256, "Package index identity")
    index = json.loads(index_raw)
    for name, pin in index["files"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), "Package path escapes root")
        data = path.read_bytes()
        require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], f"Package file: {name}")
    with zipfile.ZipFile(root / "evidence.zip") as archive:
        names = archive.namelist()
        files = json.loads(archive.read("FILES.json"))
        require(len(names) == len(set(names)) and set(names) == set(files) | {"FILES.json"}, "Archive inventory")
        for name, pin in files.items():
            raw = archive.read(name)
            require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], f"Archive member: {name}")
        ready = json.loads(archive.read(f"{KIT}/ready.json"))
        require(sha(archive.read(f"{KIT}/ready.json")) == READY, "Ready manifest")
        runtime = json.loads(archive.read("package/runtime-only-references.json"))
        for name, digest in ready["external_files"].items():
            if name.startswith("/"):
                require(runtime[name]["sha256"] == digest, "Runtime-only reference")
            else:
                normalized = posixpath.normpath(f"{KIT}/{name}")
                require(sha(archive.read(normalized)) == digest, f"External evidence: {name}")
        findings = audit(archive.read)
        require(findings == json.loads((root / "saved-audit.json").read_bytes()), "Saved audit reproduction")
    print(json.dumps({"passed": True, "archive_members": len(files),
          "prepared_files_verified": len(ready["files"]), "external_source_files_verified": len(ready["external_files"]) - len(runtime),
          "runtime_or_official_data_references_not_reexecuted": len(runtime),
          "native_returns": findings["native_returns"], "native_updates": findings["native_updates"],
          "certificates": 0, "full_fleet_checks": 0, "new_numerical_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
