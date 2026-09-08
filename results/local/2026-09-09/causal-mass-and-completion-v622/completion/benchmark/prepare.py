"""Freeze a finite ordinary-versus-compact component benchmark without GPU work."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
G = Path("/home/angus/spacepdhcg-completion-model-v622g")
OLD = ROOT / "build/performance/completion-benchmark-v621"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert sha(G / "report.json") == (
        "56a65032e639a6eb9af08052e9f3df87f5f1c84cfe9cf2ece6f3f63eafb2a12a"
    )
    report = json.loads((G / "report.json").read_text())
    assert sha(G / "source.tar.gz") == report["source_archive_sha256"]
    assert sha(Path(report["library"]["path"])) == report["library"]["sha256"]
    for name, expected in report["owned_sources"].items():
        assert sha(G / "source" / name) == expected
    (KIT / "inputs").mkdir(exist_ok=False)
    # Preserve the exact archive; use the source directory for unambiguous paths.
    shutil.copyfile(G / "source.tar.gz", KIT / "inputs/compact-g-source.tar.gz")
    shutil.copyfile(G / "report.json", KIT / "inputs/compact-g-report.json")
    with tarfile.open(G / "source.tar.gz", "r:gz") as archive:
        archive_names = [x.name for x in archive.getmembers() if x.isfile()]
    assert archive_names
    shutil.copytree(G / "source", KIT / "source")
    source_map = {
        p.relative_to(KIT / "source").as_posix(): sha(p)
        for p in sorted((KIT / "source").rglob("*"))
        if p.is_file()
    }
    for name in (
        "fixtures.json",
        "features.json",
        "ship-01.json",
        "ship-04.json",
        "ship-07.json",
        "ship-10.json",
        "ship-23.json",
        "v616-report.json",
        "hop_inflation_fit.json",
        "gpu-profile.json",
    ):
        shutil.copyfile(OLD / "inputs" / name, KIT / "inputs" / name)
    shutil.copyfile(OLD / "control_comparator.py", KIT / "control_comparator.py")
    old_groups = json.loads((OLD / "plan.json").read_text())["groups"]
    groups = [x for x in old_groups if x["size"] in (24, 48, 256, 1024)]
    assert len(groups) == 8 and sum(x["size"] for x in groups) * 5 == 13520
    plan = {
        "preparation_only": True,
        "scope": "Production GpuCompletion.run packing/native/reconstruction component",
        "historical_unique_controls": 20,
        "groups": groups,
        "backends": ["ordinary", "compact"],
        "warm_order": [
            "ordinary",
            "compact",
            "compact",
            "ordinary",
            "compact",
            "ordinary",
            "ordinary",
            "compact",
        ],
        "first_order": "ordinary/compact on even groups, compact/ordinary on odd groups",
        "calls_per_backend_per_group": 5,
        "native_evaluate_calls": 80,
        "candidate_evaluations_per_backend": 13520,
        "total_candidate_evaluations": 27040,
        "worker_seconds": 180,
        "cleanup_term_seconds": 10,
        "cleanup_kill_seconds": 30,
        "models_refitted": 0,
        "fresh_Lambert_solves": 0,
        "fresh_SCvx_refinements": 0,
        "fresh_CPU_finish_calls": 0,
        "catalogue_sha256": "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675",
        "data_path": "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data",
        "geometry": "Both arms use the official catalogue. No injected saved geometry cache.",
        "timer": (
            "Entire production run; includes per-call model/source hashing, packing, "
            "native call and reconstruction"
        ),
        "outside_timer": (
            "Catalogue load, request construction, common workspace create/close, "
            "capture and validation"
        ),
        "first_call": (
            "Reported separately for each fresh arm workspace; "
            "compact model configure remains inside run"
        ),
        "throughput_unit": (
            "repeated archived proxy completions/second, not mission solutions/second"
        ),
        "launch_gate": (
            "A matching complete passing compact-g GPU correctness report is required by exact hash"
        ),
    }
    write(KIT / "plan.json", plan)
    write(KIT / "source-sha256.json", source_map)
    write(KIT / "input-sha256.json", {p.name: sha(p) for p in sorted((KIT / "inputs").iterdir())})
    write(
        KIT / "profile.json",
        {
            "python": "/home/angus/worktrees/spacepdhcg-literature-venv/bin/python",
            "python_sha256": sha(
                Path("/home/angus/worktrees/spacepdhcg-literature-venv/bin/python")
            ),
            "source_report_sha256": sha(G / "report.json"),
            "native_root": str(G),
            "library": report["library"],
            "gpu_uuid": "GPU-4df2f6b5-e866-14a0-eeac-332cb2b757d4",
            "lock": "/home/angus/.spacepdhcg-gpu.lock",
            "nvidia_smi": "/usr/lib/wsl/lib/nvidia-smi",
        },
    )
    print(
        json.dumps(
            {
                "groups": len(groups),
                "future_calls": 80,
                "GPU_calls_now": 0,
                "source_files": len(source_map),
            }
        )
    )


if __name__ == "__main__":
    main()
