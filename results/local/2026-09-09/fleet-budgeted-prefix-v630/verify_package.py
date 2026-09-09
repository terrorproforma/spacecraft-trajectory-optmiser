"""Portable source/ready/readback archive check; no project imports or native code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def members(path):
    with tarfile.open(path, "r:gz") as archive:
        entries = [x for x in archive if x.isfile()]
        assert len({x.name for x in entries}) == len(entries)
        assert all(not x.name.startswith("/") and ".." not in Path(x.name).parts for x in entries)
        return {x.name: archive.extractfile(x).read() for x in entries}


def main():
    if not __debug__:
        raise RuntimeError("Run without -O; this verifier requires assertions")
    index = json.loads((ROOT / "sha256.json").read_bytes())
    for name, expected in index.items():
        assert digest((ROOT / name).read_bytes()) == expected, name
    actual_names = {str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.rglob("*") if p.is_file()}
    assert actual_names == set(index) | {"sha256.json"}
    ready_bytes = (ROOT / "ready.json").read_bytes()
    assert digest(ready_bytes) == "a145167e71d52a709bd80bf3af87458fd824c13c14484e702eb326f27ae60643"
    ready = json.loads(ready_bytes)
    preparation = members(ROOT / "preparation.tar.gz")
    native = members(ROOT / "native-source.tar.gz")
    manifest = json.loads((ROOT / "native-manifest.json").read_bytes())
    assert digest((ROOT / "native-source.tar.gz").read_bytes()) == manifest["source_archive_sha256"]
    assert {k: digest(v) for k, v in native.items()} == manifest["sources"]
    overlay = json.loads(preparation["policy-overlays.json"])
    for name, expected in ready["files"].items():
        if name.startswith("host/"):
            local = name.removeprefix("host/")
            data = (ROOT / "policy" / Path(local).name).read_bytes() if local in overlay else native["src/" + local]
        else:
            data = preparation[name]
        assert digest(data) == expected, name
    assert preparation["ready.json"] == ready_bytes
    outputs = members(ROOT / "outputs.tar.gz")
    audit = json.loads((ROOT / "audit/report.json").read_bytes())
    assert {name: digest(data) for name, data in outputs.items()} == audit["outputs"]
    assert outputs["report.json"] == (ROOT / "report.json").read_bytes()
    assert outputs["launch-report.json"] == (ROOT / "launch-report.json").read_bytes()
    report, launch = json.loads(outputs["report.json"]), json.loads(outputs["launch-report.json"])
    assert report["complete"] and report["status"] == "no_new_certified_routes_incumbent_retained"
    assert launch["passed"] and launch["exit_code"] == 0 and launch["owned_descendant_cleanup"]["verified_empty"]
    assert not launch["compute_processes_after_cleanup"]
    assert report["native_solves_started"] == report["native_solves_returned"] == 24
    assert report["native_iterations"] == 157
    assert report["flight_certificate_calls"] == 21 and report["wait_certificate_calls"] == 1
    assert report["full_fleet_CPU_checks"] == report["full_fleet_official_checks"] == report["incumbent_promotions"] == 0
    assert not any(Path(name).name in ("Result.txt", "ship.txt") for name in outputs)
    print(json.dumps({"passed": True, "package_files": len(index), "prepared_hashes_reconstructed": len(ready["files"]),
                      "native_source_files": len(native), "raw_output_files": len(outputs),
                      "unchanged_Result_sha256": report["source_Result_sha256"]}))


if __name__ == "__main__":
    main()
