"""Verify and materialize the v621 evidence; never load a native/GPU library."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def safe(name):
    path = PurePosixPath(name)
    assert not path.is_absolute() and ".." not in path.parts and "\\" not in name
    return path


def archive_members(path):
    rows = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            safe(member.name)
            assert member.isfile() and member.name not in rows, member.name
            assert Path(member.name).suffix.lower() not in {".so", ".dll", ".exe", ".o", ".a", ".pyc", ".pem", ".key"}
            data = archive.extractfile(member).read()
            assert not data.startswith((b"\x7fELF", b"MZ"))
            rows[member.name] = data
    return rows


def verify(package):
    index_path = package / "sha256.json"
    count = 0
    if index_path.exists():
        index = read(index_path)
        disk = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
        assert disk == set(index) | {"sha256.json"}, disk ^ (set(index) | {"sha256.json"})
        for name, pin in index.items():
            safe(name)
            data = (package / name).read_bytes()
            assert pin == {"bytes": len(data), "sha256": sha(data)}, name
        count = len(index)
    objects = archive_members(package / "python/source-objects.tar.gz")
    for name, data in objects.items():
        assert name == "objects/" + sha(data), name
    snapshots = read(package / "python/source-snapshots.json")
    for snapshot in snapshots.values():
        for name, pin in snapshot["files"].items():
            safe(name)
            data = objects["objects/" + pin["sha256"]]
            assert len(data) == pin["bytes"]
    archives = 1
    for path in sorted(package.rglob("*.tar.gz")):
        if path.name == "source-objects.tar.gz":
            continue
        rows = archive_members(path)
        archives += 1
        member_index = path.with_name(path.name.removesuffix(".tar.gz") + ".json")
        if member_index.exists():
            expected = read(member_index)
            assert set(rows) == set(expected), path
            for name, data in rows.items():
                assert expected[name] == {"bytes": len(data), "sha256": sha(data)}, name
        elif path.name == "source.tar.gz":
            manifest = read(path.with_name("manifest.json"))
            expected = manifest["source_sha256"]
            assert set(rows) == set(expected), path
            assert all(sha(data) == expected[name] for name, data in rows.items())
    return {"passed": True, "indexed_files": count, "archives": archives,
            "source_snapshots": len(snapshots), "unique_source_objects": len(objects),
            "GPU_calls": 0, "native_library_loads": 0}


LAYOUT = {
    "build/standalone": "completion-native-core-v621",
    "build/fullcore": "completion-fullcore-v621",
    "controls": "completion-native-controls-v621",
    "python/cpu-attempt-a": "completion-adapter-cpu-v621a",
    "python/cpu-attempt-b": "completion-adapter-cpu-v621b",
    "runs/adapter": "completion-adapter-gpu-v621",
    "runs/benchmark": "completion-benchmark-v621",
    "reviews/orchestration": "completion-integration-review-v621",
    "reviews/mass-design": "mass-structure-review-v621",
    "reviews/mass-metric": "mass-diagonal-metric-v621",
    "reviews/old-l1-scaling": "l1-scaling-v615b",
    "reviews/benchmark": "completion-benchmark-review-v621",
}


def materialize(package, out):
    assert not out.exists(), "Use a new output directory; existing results are never overwritten."
    out.mkdir(parents=True)
    performance = out / "build/performance"
    for relative, original in LAYOUT.items():
        source = package / relative
        if source.exists():
            shutil.copytree(source, performance / original)
    objects = archive_members(package / "python/source-objects.tar.gz")
    snapshots = read(package / "python/source-snapshots.json")
    target = {"controls": "completion-native-controls-v621/source",
              "cpu-attempt-a": "completion-adapter-cpu-v621a/source",
              "cpu-attempt-b": "completion-adapter-cpu-v621b/source",
              "benchmark": "completion-benchmark-v621/source"}
    for name, snapshot in snapshots.items():
        if name not in target:
            continue
        for relative, pin in snapshot["files"].items():
            path = performance / target[name] / safe(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(objects["objects/" + pin["sha256"]])
    for name in ("completion-adapter-gpu-v621", "completion-benchmark-v621"):
        kit = performance / name
        archive = kit / "readbacks.tar.gz"
        if archive.exists():
            for relative, data in archive_members(archive).items():
                path = kit / safe(relative)
                path.parent.mkdir(parents=True, exist_ok=True)
                assert not path.exists(), path
                path.write_bytes(data)
    mass = performance / "mass-structure-review-v621"
    inputs = performance / "known-point-replay-v606/inputs"
    shutil.copytree(mass / "inputs", inputs)
    prior = performance / "l1-weight-v618b"
    prior.mkdir()
    shutil.copy2(mass / "prior-manifest.json", prior / "manifest.json")
    # Only the two source members consumed by this audit, explicitly a subset.
    with tarfile.open(prior / "source.tar.gz", "w:gz") as archive:
        import io
        for name, pin in snapshots["mass-source-subset"]["files"].items():
            data = objects["objects/" + pin["sha256"]]
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            archive.addfile(info, io.BytesIO(data))
    (prior / "SUBSET.txt").write_text("Two accessed C++ members only; not the original full v618b source archive.\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--materialize", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    if args.materialize:
        result["materialized_root"] = str(materialize(args.package, args.materialize))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
