"""Portable stdlib-only hash/archive/accounting audit; never run mission numerics."""

import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unpack(path, expected):
    with tarfile.open(path) as source:
        members = [m for m in source if m.isfile()]
        assert len(members) == len({m.name for m in members}) == len(expected)
        values = {}
        for member in members:
            assert not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            payload = source.extractfile(member).read()
            assert expected[member.name] == {"sha256": digest(payload), "bytes": len(payload)}
            values[member.name] = payload
        return values


def main():
    if not __debug__:
        raise RuntimeError("This verifier requires enabled assertions; do not use python -O")
    root = Path(__file__).resolve().parent
    read = lambda name: json.loads((root / name).read_bytes())
    index = read("sha256.json")
    for name, expected in index.items():
        assert digest((root / name).read_bytes()) == expected, name
    kit = unpack(root / "frozen-kit.tar.gz", read("frozen-kit-files.json"))
    external = unpack(root / "external-evidence.tar.gz", read("external-evidence-files.json"))
    raw = unpack(root / "raw-output.tar.gz", read("raw-output-files.json"))
    units = unpack(root / "native-validation.tar.gz", read("native-validation-files.json"))
    ready = read("ready.json")
    assert digest(kit["ready.json"]) == "f053a57f9996244c815924d6b41e9cefa8f1fc7c47a652d50a4cd1d3d643fefe"
    assert kit["ready.json"] == (root / "ready.json").read_bytes()
    for name, expected in ready["files"].items():
        assert digest(kit[name]) == expected
    external_map = read("external-map.json")
    for name, expected in ready["external_files"].items():
        assert digest(external[external_map[name]]) == expected
    native = read("native-manifest.json")
    archive = external["build/performance/gpu-scaled-zoh-v631/source.tar.gz"]
    assert digest(archive) == native["source_archive_sha256"]
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        members = [m for m in source if m.isfile()]
        source_hashes = {m.name: digest(source.extractfile(m).read()) for m in members}
        assert len(source_hashes) == len(members)
        assert source_hashes == native["sources"]
    host = json.loads(kit["host-index.json"])
    assert all(digest(kit["host/" + name]) == source_hashes["src/" + name] == value for name, value in host.items())
    report, launch = json.loads(raw["report.json"]), json.loads(raw["launch-report.json"])
    assert report == read("report.json") and launch == read("launch-report.json")
    assert launch["passed"] and launch["exit_code"] == 0 and launch["owned_descendant_cleanup"]["verified_empty"]
    assert not launch["compute_processes_after_cleanup"]
    assert report["complete"] and report["status"] == "candidate_return_failed"
    assert (report["native_solves_started"], report["native_iterations"], report["flight_certificate_calls"]) == (2, 24, 1)
    assert report["full_fleet_CPU_checks"] == report["full_fleet_official_checks"] == 0
    assert not report["incumbent_promoted"]
    assert [x["certified"] for x in report["cases"]] == [True, False]
    assert json.loads(units["report.json"])["passed"]
    assert json.loads(kit["validation-harness-final/report.json"])["passed"]
    print(json.dumps({"passed": True, "indexed_files": len(index), "prepared_files": len(ready["files"]),
                      "external_files": len(external_map), "raw_output_files": len(raw),
                      "native_source_files": len(source_hashes), "host_files": len(host),
                      "scope": "Static package/hash/accounting validation; no physics rerun"}, indent=2))


if __name__ == "__main__":
    main()
