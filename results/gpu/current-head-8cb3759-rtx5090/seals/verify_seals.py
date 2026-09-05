#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archives = json.loads((ROOT / "seals/archives.json").read_text(encoding="utf-8"))
assert [record["gate"] for record in archives["archives"]] == ["G2", "G3"]
for record in archives["archives"]:
    path = ROOT.parents[2] / Path(record["path"])
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    sidecar = path.with_name(path.name + ".sha256")
    assert sidecar.read_text(encoding="utf-8").split()[0] == record["sha256"]
    with tarfile.open(path, "r:gz") as archive:
        member = archive.extractfile("evidence-index.json")
        assert member is not None
        index = json.load(member)
        assert index["repository"]["commit"] == archives["source_commit"]
        assert index["repository"]["dirty"] is False
        assert index["artifacts"]
        names = set(archive.getnames())
        assert "summary.json" in names and "status.txt" in names and "commands.txt" in names

index = json.loads((ROOT / "evidence-index.json").read_text(encoding="utf-8"))
excluded = {
    (ROOT / "evidence-index.json").resolve(),
    (ROOT / "evidence-index.json.sha256").resolve(),
}
indexed_paths = set()
for record in index["artifacts"]:
    path = ROOT / record["path"]
    indexed_paths.add(path.resolve())
    assert path.stat().st_size == record["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
expected = {path.resolve() for path in ROOT.rglob("*") if path.is_file()} - excluded
assert indexed_paths == expected
assert index["repository"]["commit"] == archives["source_commit"]
assert index["repository"]["dirty"] is False
assert (
    hashlib.sha256((ROOT / "evidence-index.json").read_bytes()).hexdigest()
    == (ROOT / "evidence-index.json.sha256").read_text(encoding="utf-8").split()[0]
)
print(
    json.dumps(
        {
            "status": "PASS",
            "archives": len(archives["archives"]),
            "root_artifacts": len(index["artifacts"]),
            "source_commit": archives["source_commit"],
            "root_index_sha256": hashlib.sha256(
                (ROOT / "evidence-index.json").read_bytes()
            ).hexdigest(),
            "local_only": True,
            "immutable_uri": None,
        },
        sort_keys=True,
    )
)
