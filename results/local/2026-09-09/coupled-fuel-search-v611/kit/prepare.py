"""Freeze published f8b2ac7a and ship-7 lineage; CPU/file operations only."""

# Keep provenance paths and exact runtime strings together.
# ruff: noqa: E501

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
PIN = "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    if (ROOT / "source").exists() or (ROOT / "inputs").exists():
        raise FileExistsError("Fresh preparation required")
    data = subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            PIN,
            "src",
            "benchmarks/gtoc12",
            "pyproject.toml",
            "results/gtoc12/hop_inflation_fit.json",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    source = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Unsafe Git archive member")
            destination = ROOT / "source" / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(tar.extractfile(member).read())
            source[member.name] = sha(destination)
    if len(source) != 190:
        raise AssertionError("Expected 190 frozen source/benchmark files")
    write(ROOT / "source-sha256.json", source)
    inputs = ROOT / "inputs"
    inputs.mkdir()
    paths = {
        "Result.txt": "build/performance/return-window-rescue-v607b/inputs/Result.txt",
        "control-route.json": "build/performance/incident-window-substitution-v604/inputs/ship-07.json",
        "probe-prescription.json": "build/performance/truth-set-v606/inputs/ship_07_replace_15206_with_3150/prescription.json",
        "v606-report.json": "build/performance/truth-set-v606/output/report.json",
        "v607b-audit.json": "build/performance/return-rescue-audit-v607b/final-report.json",
        "v707-published-source.json": "results/lambda/2026-09-09/gpu-device-search-v707/published-source.json",
        "v707-source-manifest.json": "results/lambda/2026-09-09/gpu-device-search-v707/local/final/source-manifest.json",
        "v707-campaign.json": "results/lambda/2026-09-09/gpu-device-search-v707/local/campaign.json",
        "v707-final-report.json": "results/lambda/2026-09-09/gpu-device-search-v707/local/final/report.json",
        "v707-followup-report.json": "results/lambda/2026-09-09/gpu-device-search-v707/local/followup/report.json",
    }
    for i in range(17):
        paths[f"probe-leg-{i:02d}.json"] = (
            "build/performance/truth-set-v606/output/ship_07_replace_15206_with_3150/"
            f"leg-{i:02d}.json"
        )
    # Locate the recorded paths from the v607b exact input index, rather than guessing.
    previous = json.loads(
        (REPO / "build/performance/return-window-rescue-v607b/inputs-sha256.json").read_text()
    )
    paths["probe-prescription.json"] = str(
        REPO / "build/performance/return-window-rescue-v607b/inputs/probe/prescription.json"
    )
    for i in range(17):
        paths[f"probe-leg-{i:02d}.json"] = str(
            REPO / "build/performance/return-window-rescue-v607b/inputs/probe" / f"leg-{i:02d}.json"
        )
    provenance = {}
    for name, origin in paths.items():
        origin = REPO / origin
        shutil.copyfile(origin, inputs / name)
        provenance[name] = {"origin": str(origin), "sha256": sha(inputs / name)}
    shutil.copyfile(
        REPO / "build/performance/truth-set-v606/fixed_refine.py", ROOT / "fixed_refine.py"
    )
    write(ROOT / "inputs-sha256.json", provenance)
    published = json.loads((inputs / "v707-published-source.json").read_text())["files"]
    comparisons = {}
    for name in ("src/spacepdhcg/gtoc12/gpu_joint.py", "src/spacepdhcg/gtoc12/jointopt.py"):
        comparisons[name] = {
            "frozen_sha256": source[name],
            "validated_v707_sha256": published[name],
            "exact_match": source[name] == published[name],
        }
        if not comparisons[name]["exact_match"]:
            raise AssertionError("Native search wrapper must match validated v707 exactly")
    profile = {
        "python": "/home/angus/worktrees/spacepdhcg-literature-venv/bin/python",
        "data": "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data",
        "lock": "/home/angus/.spacepdhcg-gpu.lock",
        "native_libraries": {
            "SPACEPDHCG_GTOC12_CUDA_LIBRARY": {
                "path": "/home/angus/spacepdhcg-joint-search-v702/final/libspacepdhcg_cuda.so",
                "sha256": "65f1335e3af820d22451e0d4f587f2d26aeccf364a0ed3bb9597bfd009bac5aa",
            },
            "SPACEPDHCG_QOCO_LIBRARY": {
                "path": "/home/angus/spacepdhcg-retry-conditioning-v683/final/libqoco.so",
                "sha256": "5b1b1a047f9d5041f821b64490562a3d598f51b47fedb08d89cdae211728940a",
            },
        },
        "LD_LIBRARY_PATH": "/home/angus/spacepdhcg-joint-search-v702/final:/home/angus/spacepdhcg-retry-conditioning-v683/final:/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64",
        "environment": {
            "SPACEPDHCG_TEST_GTOC12_JOINT_BATCH": "1",
            "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION": "1",
            "SPACEPDHCG_TEST_GTOC12_JOINT_RESIDENT_GEOMETRY": "1",
            "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_MESH": "1",
            "SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SEARCH": "1",
            "SPACEPDHCG_TEST_GTOC12_CONDITIONING_RETRY": "1",
            "SPACEPDHCG_TEST_GTOC12_PARALLEL_DIRECTIONS": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "prior_validation": {
            "GPU_tests": "111 passed without skips on RTX5090 and H100 in v707",
            "local_active_loop_memcheck": "CUDA error999 unresolved; not claimed memory-clean",
            "local_racecheck_and_synccheck": "passed moving-search regression",
            "fullfleet": "v707 local baseline/candidate each36 converged legs and both fleet checkers",
        },
    }
    for item in profile["native_libraries"].values():
        if sha(Path(item["path"])) != item["sha256"]:
            raise AssertionError("Native runtime hash mismatch")
    write(ROOT / "profile.json", profile)
    write(
        ROOT / "preparation.json",
        {
            "source_commit": PIN,
            "source_files": len(source),
            "input_files": len(provenance),
            "source_compatibility": comparisons,
            "source_manifest_sha256": sha(ROOT / "source-sha256.json"),
            "fixed_refine_sha256": sha(ROOT / "fixed_refine.py"),
            "GPU_calls": 0,
            "prior_prefix_input_manifest_entries": len(previous),
        },
    )
    print(
        json.dumps(
            {"source_files": len(source), "input_files": len(provenance), "GPU_calls": 0}, indent=2
        )
    )


if __name__ == "__main__":
    main()
