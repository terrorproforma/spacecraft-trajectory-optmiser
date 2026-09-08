"""Freeze published Python and exact retained fleet inputs; never invoke CUDA."""

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
REVISION = "3091c716714c8bdec364d54c5e7357f2b5d85730"
INCUMBENT = "33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (
        subprocess.check_output(["git", "rev-parse", REVISION], cwd=REPO, text=True).strip()
        == REVISION
    )
    source = ROOT / "source"
    source.mkdir(exist_ok=False)
    paths = ["src", "benchmarks/gtoc12", "pyproject.toml", "results/gtoc12/hop_inflation_fit.json"]
    data = subprocess.check_output(["git", "archive", REVISION, *paths], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            target = (source / member.name).resolve()
            assert target.is_relative_to(source) and not member.issym() and not member.islnk()
        archive.extractall(source)
    inputs = ROOT / "inputs"
    original_inputs = REPO / "build/performance/next-score-v597/source/inputs"
    original_manifest = json.loads((original_inputs / "sha256.json").read_text())
    for name, expected in original_manifest.items():
        assert digest(original_inputs / name) == expected, name
    assert digest(original_inputs / "Result.txt") == INCUMBENT
    shutil.copytree(original_inputs, inputs)
    shutil.copy2(
        REPO / "results/local/2026-09-09/orphan-recovery-v595/audit/fresh-verification.json",
        inputs / "fresh-verification.json",
    )
    original_manifest["fresh-verification.json"] = digest(inputs / "fresh-verification.json")
    (inputs / "sha256.json").write_text(json.dumps(original_manifest, indent=2) + "\n")
    source_hashes = {
        p.relative_to(source).as_posix(): digest(p)
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    (ROOT / "source-sha256.json").write_text(json.dumps(source_hashes, indent=2) + "\n")
    record = {
        "source_commit": REVISION,
        "source_files": len(source_hashes),
        "source_paths": paths,
        "incumbent_sha256": INCUMBENT,
        "input_manifest_sha256": digest(inputs / "sha256.json"),
        "preparation_gpu_calls": 0,
        "production_sources_modified": False,
    }
    (ROOT / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
