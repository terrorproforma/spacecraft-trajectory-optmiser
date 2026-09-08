"""Archive all exact v616 inputs/source/output and independent negative-result evidence."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "build/performance/depth-diverse-generation-v616"
AUDIT = REPO / "build/performance/depth-generation-audit-v616"
DEST = REPO / "results/local/2026-09-09/depth-diverse-generation-v616"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def archive(name, members):
    target = DEST / (name + ".tar.gz")
    with (
        target.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as stream:
            for relative in sorted(members):
                payload = (KIT / name / relative).read_bytes()
                member = tarfile.TarInfo(relative)
                member.size, member.mode, member.mtime = len(payload), 0o644, 0
                stream.addfile(member, io.BytesIO(payload))
    recovered = {}
    with tarfile.open(target, "r:gz") as stream:
        for member in stream.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute()
            assert ".." not in Path(member.name).parts
            digest = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            assert digest == members[member.name]
            recovered[member.name] = digest
    assert recovered == members
    return {"sha256": sha(target), "files": len(members), "members": recovered}


def main():
    if DEST.exists():
        raise FileExistsError("New publication directory required")
    ready = read(KIT / "ready-manifest.json")
    original = {name: sha(KIT / name) for name in ready["files"]}
    assert original == ready["files"]
    report = read(AUDIT / "report-v2.json")
    raw_before = {name: sha(KIT / "output" / name) for name in report["raw_output_hashes"]}
    assert raw_before == report["raw_output_hashes"]
    DEST.mkdir(parents=True)
    archived = {}
    for name in ("source", "inputs"):
        members = {
            key[len(name) + 1 :]: digest
            for key, digest in original.items()
            if key.startswith(name + "/")
        }
        archived[name] = archive(name, members)
    assert archived["source"]["files"] == 190 and archived["inputs"]["files"] == 6
    reconstructed = {}
    for name, digest in original.items():
        if name.startswith(("source/", "inputs/")):
            continue
        copy(KIT / name, DEST / "kit" / name)
        reconstructed[name] = digest
    copy(KIT / "ready-manifest.json", DEST / "kit/ready-manifest.json")
    for name in ("launch.json", "run.log"):
        copy(KIT / name, DEST / "execution" / name)
    for name in raw_before:
        copy(KIT / "output" / name, DEST / "output" / name)
    for name in (
        "audit.py",
        "diagnose.py",
        "finish_control.py",
        "report-v2.json",
        "diagnosis-v2.json",
        "finish-control.json",
    ):
        copy(AUDIT / name, DEST / "audit" / name)
    for name in ("report.json", "diagnosis.json"):
        copy(AUDIT / name, DEST / "audit/superseded" / name)
    copy(AUDIT / "README-publication.md", DEST / "README.md")
    copy(Path(__file__), DEST / "audit/package.py")
    (DEST / ".gitattributes").write_text(
        "# Preserve exact evidence bytes and indexed hashes.\n* -text\n"
    )
    for name, value in archived.items():
        reconstructed.update(
            {name + "/" + relative: digest for relative, digest in value["members"].items()}
        )
    assert reconstructed == original
    write(
        DEST / "archive-audit.json",
        {
            "archives": archived,
            "all_prepared_hashes_match": True,
            "reconstructed_prepared_files": len(reconstructed),
            "raw_output_files": len(raw_before),
            "all_raw_output_hashes_match": True,
            "ready_sha256": sha(KIT / "ready-manifest.json"),
            "authoritative_output_audit_sha256": sha(AUDIT / "report-v2.json"),
            "authoritative_diagnosis_sha256": sha(AUDIT / "diagnosis-v2.json"),
            "scalar_finish_controls_sha256": sha(AUDIT / "finish-control.json"),
        },
    )
    assert {name: sha(KIT / name) for name in original} == original
    assert {name: sha(KIT / "output" / name) for name in raw_before} == raw_before
    files = {
        str(path.relative_to(DEST)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(DEST.rglob("*"))
        if path.is_file()
    }
    write(
        DEST / "sha256.json",
        {
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(item["bytes"] for item in files.values()),
        },
    )
    for name, value in files.items():
        assert sha(DEST / name) == value["sha256"]
    print(
        json.dumps(
            {
                "destination": str(DEST),
                "indexed_files": len(files),
                "indexed_bytes": sum(item["bytes"] for item in files.values()),
                "index_sha256": sha(DEST / "sha256.json"),
                "archive_audit_sha256": sha(DEST / "archive-audit.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
