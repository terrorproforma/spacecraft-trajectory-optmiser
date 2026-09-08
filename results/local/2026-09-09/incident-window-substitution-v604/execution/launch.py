"""Print the pinned local recipe; --execute explicitly launches the reviewed job."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--execute", action="store_true")
parser.add_argument("--output", type=Path, default=root / "output")
parser.add_argument("--proxy-only", action="store_true")
args = parser.parse_args()
python = "/home/angus/worktrees/spacepdhcg-literature-venv/bin/python"
environment = {
    k: v for k, v in os.environ.items() if not k.startswith(("SPACEPDHCG_", "QOCO_REPLAY_"))
}
environment.update(
    SPACEPDHCG_GTOC12_CUDA_LIBRARY="/home/angus/build-spacepdhcg-joint-v596/build/cuda/libspacepdhcg_cuda.so",
    SPACEPDHCG_QOCO_LIBRARY="/home/angus/build-qoco-scaled-pool-v540/final/libqoco.so",
    SPACEPDHCG_GTOC12_DATA="/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data",
    SPACEPDHCG_TEST_GTOC12_JOINT_BATCH="1",
    SPACEPDHCG_TEST_GTOC12_JOINT_DEVICE_SELECTION="1",
    LD_LIBRARY_PATH="/home/angus/build-qoco-scaled-pool-v540/final:/home/angus/cudss-isolated-0.8.0.10/nvidia/cu12/lib:/usr/local/cuda-12.8/lib64",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    PYTHONHASHSEED="0",
    PYTHONDONTWRITEBYTECODE="1",
)
command = [
    python,
    str(root / "run.py"),
    "--execute",
    "--output",
    str(args.output.resolve()),
    "--proxy-seconds",
    "180",
    "--wall-seconds",
    "1800",
    "--max-refinements",
    "4",
]
if args.proxy_only:
    command.append("--proxy-only")
record = {
    "command": command,
    "cwd": str(root / "source"),
    "environment": {
        k: v
        for k, v in environment.items()
        if k.startswith("SPACEPDHCG_")
        or k
        in (
            "LD_LIBRARY_PATH",
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONHASHSEED",
        )
    },
    "driver_sha256": hashlib.sha256((root / "run.py").read_bytes()).hexdigest(),
    "launched": False,
}
if args.execute:
    publication = json.loads((root / "ready-manifest.json").read_text())
    for name, expected in publication["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"reviewed preparation changed: {name}")
    if args.output.exists():
        raise FileExistsError("choose a fresh output directory")
    with (root / "launch.json").open("x") as marker:
        with (root / "run.log").open("x") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=root / "source",
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record.update(launched=True, pid=process.pid)
        marker.write(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
