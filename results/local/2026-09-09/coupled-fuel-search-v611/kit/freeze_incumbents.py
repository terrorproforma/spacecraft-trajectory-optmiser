"""Freeze feasible incumbent seeds and verify exact archived fleet control; CPU only."""

import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import common

SHIPS = (20, 7, 10, 11)


def ship_bytes(path, ship):
    return b"\n".join(
        line
        for line in Path(path).read_bytes().splitlines()
        if line.split() and int(line.split()[0]) == ship
    )


def main():
    root = common.ROOT
    repo = root.parents[2]
    inputs = common.read(root / "inputs-sha256.json")
    if "archived-control-fleet.txt" in inputs:
        raise FileExistsError("Incumbent freezing already completed")
    for ship in SHIPS:
        source = root.parent / "incident-window-substitution-v604/inputs" / f"ship-{ship:02d}.json"
        name = f"ship-{ship:02d}.json"
        shutil.copyfile(source, root / "inputs" / name)
        inputs[name] = {"origin": str(source), "sha256": common.sha(source)}
    published = repo / "results/lambda/2026-09-09/gpu-device-search-v707"
    manifest = common.read(published / "local-raw.manifest.json")
    selected = {
        "campaign/candidate0/best/Result.txt": "archived-control-fleet.txt",
        "campaign/candidate0/report.json": "archived-control-campaign.json",
    }
    extraction = {}
    with tarfile.open(published / "local-raw.tar.gz", "r:gz") as tar:
        for member, destination in selected.items():
            data = tar.extractfile(member).read()
            expected = manifest[member]
            if (
                len(data) != expected["bytes"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]
            ):
                raise AssertionError("Archived local control member hash mismatch")
            (root / "inputs" / destination).write_bytes(data)
            inputs[destination] = {
                "origin": str(published / "local-raw.tar.gz") + "::" + member,
                "sha256": expected["sha256"],
            }
            extraction[member] = expected
    common.write(root / "inputs-sha256.json", inputs)
    evidence = common.read(root / "inputs/v707-campaign.json")
    campaign = next(item for item in evidence["runs"] if item["name"] == "candidate0")
    same = {
        str(ship): ship_bytes(root / "inputs/Result.txt", ship)
        == ship_bytes(root / "inputs/archived-control-fleet.txt", ship)
        for ship in SHIPS
    }
    if (
        not all(same.values())
        or not campaign["best"]["official"]["ok"]
        or not campaign["best"]["independent"]["ok"]
    ):
        raise AssertionError("Chosen incumbent control is not byte-exact and both-checked")
    profile = common.read(root / "profile.json")
    if (
        evidence["core_sha256"]
        != profile["native_libraries"]["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]["sha256"]
        or evidence["qoco_sha256"]
        != profile["native_libraries"]["SPACEPDHCG_QOCO_LIBRARY"]["sha256"]
    ):
        raise AssertionError("Archived local control library mismatch")
    symbols = subprocess.run(
        [
            "nm",
            "-D",
            "--defined-only",
            profile["native_libraries"]["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]["path"],
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    exported = [
        line
        for line in symbols.splitlines()
        if line.endswith(" spacepdhcg_gtoc12_joint_search_host")
    ]
    if len(exported) != 1:
        raise AssertionError("Actual local library lacks graph entry point")
    original = common.read(root / "source-sha256.json")
    validated = common.read(root / "inputs/v707-source-manifest.json")
    comparisons = {
        name: {"frozen_sha256": checksum, "v707_sha256": validated.get("repo/" + name)}
        for name, checksum in original.items()
        if validated.get("repo/" + name) != checksum
    }
    critical = (
        "gpu_joint.py",
        "jointopt.py",
        "gpu_lambert.py",
        "low_thrust.py",
        "pipeline.py",
        "gpu_scvx.py",
        "gpu_execution.py",
        "official.py",
        "verifier.py",
        "solution.py",
    )
    critical_differences = [
        name
        for name in comparisons
        if name.startswith("src/spacepdhcg/gtoc12/") and Path(name).name in critical
    ]
    common.write(
        root / "incumbent-controls.json",
        {
            "ships": SHIPS,
            "exact_ship_bytes_equal": same,
            "archived_control_both_fleet_checks_pass": True,
            "archived_fullfleet_sha256": common.sha(root / "inputs/archived-control-fleet.txt"),
            "archived_local_campaign_core_and_qoco_match": True,
            "native_search_export": exported,
            "nm_output_sha256": hashlib.sha256(symbols.encode()).hexdigest(),
            "archive_members": extraction,
            "source_differences_from_v707": comparisons,
            "critical_source_differences": critical_differences,
            "fresh_control_refinement_required": False,
            "runtime_control_gate": (
                "Fresh both-checker verification of retained fleet and repeat exact selected-ship "
                "byte comparison; no new control full route"
            ),
            "GPU_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "ships": SHIPS,
                "byte_equal": same,
                "critical_differences": critical_differences,
                "native_search_export": exported,
                "GPU_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
