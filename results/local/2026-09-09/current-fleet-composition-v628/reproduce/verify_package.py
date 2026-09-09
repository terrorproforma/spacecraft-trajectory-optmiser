"""Read-only package byte/section/binding audit; no numerical project imports."""

import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("Package verification requires Python assertions; do not use -O.")
    package = Path(__file__).resolve().parent.parent
    index = json.loads((package / "index.json").read_text())
    actual = {
        str(path.relative_to(package)).replace("\\", "/")
        for path in package.rglob("*") if path.is_file() and path.name != "index.json"
    }
    # A nested preserved index is an ordinary indexed evidence file.
    actual.update(str(path.relative_to(package)).replace("\\", "/")
                  for path in package.rglob("index.json") if path.parent != package)
    assert actual == set(index["files"])
    for name, expected in index["files"].items():
        data = (package / name).read_bytes()
        assert len(data) == expected["bytes"] and digest(data) == expected["sha256"], name
    archives = json.loads((package / "archive-audit.json").read_text())
    for item in archives["files"]:
        data = gzip.decompress((package / item["path"]).read_bytes())
        assert len(data) == item["uncompressed_bytes"]
        assert digest(data) == item["uncompressed_sha256"]
    result = (package / "Result.txt").read_bytes()
    binding = json.loads((package / "checker-binding.json").read_text())
    report = json.loads((package / "report.json").read_text())
    assert digest(result) == binding["result_sha256"] == report["result_sha256"]
    assert report["qualified"] and binding["independent"]["ok"] and binding["official"]["ok"]
    sections = {}
    for line in result.splitlines(keepends=True):
        row = line.split()
        if row:
            sections.setdefault(str(int(row[0])), bytearray()).extend(line)
    proof = json.loads((package / "lineage/retained-sections.json").read_text())
    for ship, expected in proof["sections"].items():
        assert digest(sections[ship]) == expected["composed_sha256"]
        assert len(sections[ship]) == expected["bytes"]
        assert expected["composed_sha256"] == expected["selected_source_sha256"]
    assert len(sections) == 23
    print(json.dumps({"status": "passed", "indexed_files": len(index["files"]),
                      "bytes": sum(x["bytes"] for x in index["files"].values()),
                      "gzip_roundtrips": len(archives["files"]), "ship_sections": 23,
                      "Result_sha256": digest(result), "GPU_calls": 0,
                      "propagations_or_checker_reruns": 0}))


if __name__ == "__main__":
    main()
