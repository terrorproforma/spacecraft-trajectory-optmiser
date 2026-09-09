"""Package exact saved v629 evidence; no numerical execution or source mutation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
CAMPAIGN = ROOT / "build/performance/mass-budgeted-frontier-v629"
VIEW = ROOT / "build/performance/frontier-visualizer-v629"
DEST = ROOT / "results/local/2026-09-09/mass-budgeted-frontier-v629"
DATASET = ROOT / "results/lambda/2026-09-06/visualiser/data/gtoc12-frontier-v629"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def files_under(path):
    return sorted(p for p in path.rglob("*") if p.is_file())


def main():
    if not __debug__:
        raise RuntimeError("Assertions required")
    DEST.mkdir()
    ready = read(CAMPAIGN / "ready.json")
    for name, expected in ready["files"].items():
        assert sha(CAMPAIGN / name) == expected
    assert sha(CAMPAIGN / "ready.json") == "6a63b9506655dd1590c7c470a5a5314315ba565475658a82ea3adbd28cd9bf6b"
    assert sha(KIT / "inputs/Result.txt") == "765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da"
    assert read(KIT / "output/report.json")["qualified"]
    copies = {
        "Result.txt": KIT / "inputs/Result.txt", "RESULTS.md": CAMPAIGN / "RESULTS.md",
        "checker-binding.json": KIT / "output/checker-binding.json",
        "report.json": KIT / "output/report.json",
        "independent.json": KIT / "output/independent.json",
        "independent-full.json": KIT / "output/independent-full.json",
        "official.json": KIT / "output/official.json", "official-full.json": KIT / "output/official-full.json",
        "official.stdout": KIT / "output/official.stdout", "official.stderr": KIT / "output/official.stderr",
        "composition-launch-report.json": KIT / "output/launch-report.json",
        "campaign-report.json": CAMPAIGN / "output/report.json",
        "campaign-launch-report.json": CAMPAIGN / "output/launch-report.json",
        "campaign-ready.json": CAMPAIGN / "ready.json",
        "saved-node-provenance.json": VIEW / "export/saved-node-provenance.json",
        "viewer-export-audit.json": VIEW / "export-audit.json",
        "viewer-import-audit.json": VIEW / "import-audit.json",
        "viewer-http-audit.json": VIEW / "http-audit.json",
        "package.py": Path(__file__),
        "verify_package.py": KIT / "verify_package.py",
    }
    for name, source in copies.items():
        shutil.copyfile(source, DEST / name)
    (DEST / ".gitattributes").write_text("* -text -whitespace\n")
    assigned, omitted, archives = set(), [], {}

    def archive(name, paths):
        rows = {}
        with tarfile.open(DEST / name, "w:gz", compresslevel=6) as target:
            for path in sorted(set(paths)):
                relative = path.relative_to(ROOT).as_posix()
                if relative in assigned:
                    continue
                if path.name == "GTOC12_Verify" or path.name == "GTOC12_Asteroids_Data.txt" or "__pycache__" in path.parts or path.suffix == ".pyc":
                    omitted.append({"path": relative, "sha256": sha(path), "bytes": path.stat().st_size, "reason": "copied checker executable or duplicate public catalogue; caches omitted"})
                    continue
                assert path.is_file() and not path.is_symlink()
                data_sha = sha(path)
                rows[relative] = {"sha256": data_sha, "bytes": path.stat().st_size}
                target.add(path, arcname=relative, recursive=False)
                assigned.add(relative)
        archives[name] = {"sha256": sha(DEST / name), "bytes": (DEST / name).stat().st_size, "files": rows}

    archive("campaign.tar.gz", files_under(CAMPAIGN))
    archive("composition.tar.gz", files_under(KIT))
    archive("viewer.tar.gz", files_under(VIEW) + files_under(DATASET))
    prerequisite = []
    prior = ROOT / "build/performance/seeded-candidate-boundary-merit-v627/campaign"
    prerequisite.append(prior / "ready.json")
    prerequisite.extend(prior / name for name in read(prior / "ready.json")["files"])
    base = ROOT / "build/performance/refinement-queue-v623"
    prerequisite.append(base / "source-sha256.json")
    prerequisite.extend(base / "source" / name for name in read(base / "source-sha256.json"))
    for name in ready["external_files"]:
        path = (CAMPAIGN / name).resolve()
        if path.name in ("local.tar.gz", "local-files.json"):
            continue  # Existing committed v799 pool; hash pointer retained below.
        prerequisite.append(path)
    prerequisite.extend(ROOT / name for name in read(KIT / "checker-plan.json")["source_sha256"])
    archive("frozen-prerequisites.tar.gz", prerequisite)
    write(DEST / "archive-manifest.json", archives)
    write(DEST / "omitted-artifacts.json", omitted)
    write(DEST / "external-runtime.json", {
        "profile": read(CAMPAIGN / "profile.json"),
        "external_saved_pool": {name: digest for name, digest in ready["external_files"].items() if name.endswith(("local.tar.gz", "local-files.json"))},
        "catalogue_sha256": read(KIT / "output/report.json")["catalogue_sha256"],
        "bonus_sha256": read(KIT / "output/report.json")["bonus_sha256"],
        "no_runtime_binaries_in_package": True,
    })
    (DEST / "README.md").write_text((KIT / "package-README.md").read_text())
    index = {p.relative_to(DEST).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files_under(DEST)}
    write(DEST / "index.json", {"files": index, "count": len(index), "bytes": sum(x["bytes"] for x in index.values())})
    print(json.dumps({"directory": str(DEST), "index_sha256": sha(DEST / "index.json"), "files": len(index), "bytes": sum(x["bytes"] for x in index.values()), "archive_members": sum(len(x["files"]) for x in archives.values()), "omitted": len(omitted)}))


if __name__ == "__main__":
    main()
