"""Foreground local supervisor; print-only by default, exclusive single launch."""

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import support


def main(args):
    profile, env = support.environment()
    command = [
        profile["python"],
        "-B",
        str(support.ROOT / "run.py"),
        "--execute",
        "--output",
        str(args.output.resolve()),
        "--lock",
        profile["lock"],
        "--wall-seconds",
        str(args.wall_seconds),
    ]
    record = {
        "profile": "local",
        "command": command,
        "supervisor_pid": os.getpid(),
        "supervisor_sha256": support.sha(__file__),
        "driver_sha256": support.sha(support.ROOT / "run.py"),
        "environment": {
            key: value
            for key, value in env.items()
            if key.startswith("SPACEPDHCG_")
            or key
            in (
                "LD_LIBRARY_PATH",
                "PYTHONHASHSEED",
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "maximum_native_returns": 5,
        "maximum_candidate_returns": 4,
        "whole_route_reruns": 0,
        "launched": False,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if not args.execute:
        print(json.dumps(record, indent=2))
        return 0
    import fcntl

    support.validate_ready()
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    with (support.ROOT / "launch.json").open("x") as marker:
        marker.write(json.dumps(record, indent=2) + "\n")
    try:
        support.validate_runtime(env)
        with Path(profile["lock"]).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            record["compute_process_observation"] = processes.stdout.strip()
            if processes.stdout.strip():
                raise RuntimeError("compute process present; no child launched")
            record["gpu_inventory"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.used",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
            command += ["--lock-fd", str(lock.fileno())]
            with (support.ROOT / "run.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    cwd=support.ROOT / "source",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                    start_new_session=True,
                )
                record.update(launched=True, pid=child.pid, command=command)
                support.write(support.ROOT / "launch.json", record)
                print(
                    json.dumps(
                        {
                            "pid": child.pid,
                            "supervisor_pid": os.getpid(),
                            "output": str(args.output.resolve()),
                        }
                    ),
                    flush=True,
                )
                code = child.wait()
            record.update(
                returncode=code, finished_utc=datetime.datetime.now(datetime.UTC).isoformat()
            )
            support.write(support.ROOT / "launch.json", record)
            return code
    except Exception as error:
        record.update(
            error=repr(error),
            status="supervisor_failed" if record["launched"] else "refused_before_GPU_launch",
        )
        support.write(support.ROOT / "launch.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=support.ROOT / "output")
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("wall budget must be positive and at most 1800 seconds")
    raise SystemExit(main(args))
