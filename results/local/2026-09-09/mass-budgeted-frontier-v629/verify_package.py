"""Package-only byte, archive round-trip, fleet-section and checker-binding audit."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile

KIT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def sections(data):
    result = {}
    for line in data.splitlines(keepends=True):
        if line.split():
            ship = int(line.split()[0])
            result[ship] = result.get(ship, b"") + line
    return result


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; package checks require assertions")
    index = read(KIT / "index.json")
    for name, row in index["files"].items():
        data = (KIT / name).read_bytes()
        assert digest(data) == row["sha256"] and len(data) == row["bytes"], name
    payloads, count = {}, 0
    archives = read(KIT / "archive-manifest.json")
    needed = {
        "build/performance/current-fleet-union-v629/inputs/Result.txt",
        "build/performance/frontier-union-v629/inputs/Result.txt",
        "build/performance/mass-budgeted-frontier-v629/output/v794-ship08-rank0000/ship.txt",
        "build/performance/mass-budgeted-frontier-v629/output/v794-ship19-rank0006/ship.txt",
        "build/performance/mass-budgeted-frontier-v629/ready.json",
    }
    all_members = {}
    for name, manifest in archives.items():
        assert digest((KIT / name).read_bytes()) == manifest["sha256"]
        observed = set()
        with tarfile.open(KIT / name, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and ".." not in path.parts
                assert member.name not in observed and member.name not in all_members
                data = archive.extractfile(member).read()
                row = manifest["files"][member.name]
                assert len(data) == row["bytes"] and digest(data) == row["sha256"], member.name
                all_members[member.name] = row["sha256"]
                if member.name in needed:
                    payloads[member.name] = data
                observed.add(member.name)
                count += 1
        assert observed == set(manifest["files"])
    assert set(payloads) == needed
    ready = json.loads(payloads["build/performance/mass-budgeted-frontier-v629/ready.json"])
    for name, expected in ready["files"].items():
        assert all_members["build/performance/mass-budgeted-frontier-v629/"+name] == expected
    result = (KIT / "Result.txt").read_bytes()
    result_sha = digest(result)
    assert result == payloads["build/performance/frontier-union-v629/inputs/Result.txt"]
    assert result_sha == "765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da"
    binding = read(KIT / "checker-binding.json")
    assert binding["result_sha256"] == result_sha
    assert all(binding[key]["ok"] and binding[key]["result_sha256"] == result_sha for key in ("independent", "official"))
    assert binding["independent"]["ships"] == 23 and binding["independent"]["mined_asteroids"] == 199
    assert binding["independent"]["weighted_score_fixed_bonus_kg"] == 13023.704900978004
    old = sections(payloads["build/performance/current-fleet-union-v629/inputs/Result.txt"])
    new = sections(result)
    assert len(new) == 23 and all(old[i] == new[i] for i in old if i not in (8, 19))
    for ship, case in ((8, "v794-ship08-rank0000"), (19, "v794-ship19-rank0006")):
        assert new[ship].rstrip(b"\n") == payloads[f"build/performance/mass-budgeted-frontier-v629/output/{case}/ship.txt"]
    run = read(KIT / "campaign-report.json")
    assert len(run["cases"]) == 4 and all(c["certified"] for c in run["cases"])
    assert run["native_solves_started"] == run["native_solves_returned"] == 72
    assert run["native_iterations"] == 286 and run["wait_certificate_calls"] == 6
    print(json.dumps({"passed": True, "indexed_files": len(index["files"]), "archive_members_round_trip": count, "original_ready_files_preserved": len(ready["files"]), "other21_sections_exact": True, "result_sha256": result_sha, "GPU_or_numerical_calls": 0}))


if __name__ == "__main__":
    main()
