"""Append terminal v621 evidence losslessly; no GPU, Git or network operations."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / "build/performance/completion-package-v621"
DEST = ROOT / "results/local/2026-09-09/gpu-route-completion-v621"
OBJECTS = SCRATCH / "source-objects"
sha = lambda data: hashlib.sha256(data).hexdigest()
origins = json.loads((SCRATCH / "copy-origins.json").read_text())
snapshots = json.loads((DEST / "python/source-snapshots.json").read_text())


def copy(source, destination):
    data = source.read_bytes()
    if destination.exists():
        assert destination.read_bytes() == data, destination
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    assert source.suffix.lower() not in {".pyc", ".so", ".dll", ".exe", ".o", ".a", ".pem", ".key"}
    origins[destination.relative_to(DEST).as_posix()] = {
        "source": source.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha(data)}


def tree(source, destination, exclude=()):
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", "pytest-tmp"} for part in relative.parts):
            continue
        if any(relative == Path(x) or Path(x) in relative.parents for x in exclude):
            continue
        copy(path, destination / relative)


def archive(path, members):
    assert not path.exists(), path
    with path.open("wb") as out, gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))


benchmark = ROOT / "build/performance/completion-benchmark-v621"
assert (benchmark / "RESULTS.md").is_file() and (benchmark / "summary.json").is_file()
assert json.loads((benchmark / "output/launch-report.json").read_text())["passed"]
raw = {p.relative_to(benchmark).as_posix(): p.read_bytes()
       for p in sorted((benchmark / "output").rglob("*")) if p.is_file() and p.parent != benchmark / "output"}
tree(benchmark, DEST / "runs/benchmark", exclude=("source", *raw))
archive(DEST / "runs/benchmark/readbacks.tar.gz", raw)
(DEST / "runs/benchmark/readbacks.json").write_text(json.dumps({name: {"bytes": len(data), "sha256": sha(data)} for name, data in raw.items()}, indent=2)+"\n")
mapping = json.loads((benchmark / "source-sha256.json").read_text())
rows = {}
for name, digest in mapping.items():
    data = (benchmark / "source" / name).read_bytes()
    assert sha(data) == digest
    blob = OBJECTS / digest
    if not blob.exists():
        blob.write_bytes(data)
    else:
        assert blob.read_bytes() == data
    rows[name] = {"bytes": len(data), "sha256": digest}
assert rows == snapshots["cpu-attempt-b"]["files"]
snapshots["benchmark"] = {"source_root": "build/performance/completion-benchmark-v621/source",
                          "original_manifest": "runs/benchmark/source-sha256.json", "files": rows}
tree(ROOT / "build/performance/mass-diagonal-metric-v621", DEST / "reviews/mass-metric")
copy(ROOT / "build/performance/l1-scaling-v615b/findings.json", DEST / "reviews/old-l1-scaling/findings.json")
copy(ROOT / "build/performance/gpu-state-v621.json", DEST / "health/gpu-state-v621.json")
for name in ("GPU_ROUTE_COMPLETION.md", "SOTA_EXECUTION_PLAN_2026-09-09.md"):
    copy(ROOT / "docs" / name, DEST / "documentation" / name)
for name in ("prepare_completion_tranche_v621.py", "commit_completion_tranche_v621.py"):
    source = ROOT / "build/performance" / name
    if source.exists():
        copy(source, DEST / "reproduce/publication" / name)
for source in (ROOT / "build/performance/prepare_completion_package_v621.py", Path(__file__), SCRATCH / "portable.py"):
    copy(source, DEST / "reproduce" / source.name)
(DEST / "python/source-snapshots.json").write_text(json.dumps(snapshots, indent=2)+"\n")
archive(DEST / "python/source-objects.tar.gz", {"objects/"+p.name: p.read_bytes() for p in sorted(OBJECTS.iterdir())})
(SCRATCH / "copy-origins.json").write_text(json.dumps(origins, indent=2)+"\n")
(DEST / "reproduce/copy-origins.json").write_text(json.dumps(origins, indent=2)+"\n")
print(json.dumps({"appended": True, "sealed": False, "source_objects": len(list(OBJECTS.iterdir())),
                  "benchmark_raw_members": len(raw), "benchmark_raw_bytes": sum(map(len, raw.values()))}))
