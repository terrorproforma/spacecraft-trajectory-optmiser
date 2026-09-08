"""Foreground supervisor; print-only by default, one reviewed host/profile per truth set."""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import common


def recipe(profile_name, output, wall_seconds):
    profile, environment = common.profile_environment(profile_name)
    command = [
        profile["python"],
        str(common.ROOT / "run.py"),
        "--execute",
        "--profile",
        profile_name,
        "--output",
        str(output.resolve()),
        "--wall-seconds",
        str(wall_seconds),
        "--lock",
        profile["lock"],
    ]
    return profile, environment, command


def main(args):
    profile, environment, command = recipe(args.profile, args.output, args.wall_seconds)
    record = {
        "profile": args.profile,
        "command": command,
        "budget": "at most four routes on one host; no second-profile run",
        "environment": {
            key: value
            for key, value in environment.items()
            if key.startswith("SPACEPDHCG_")
            or key
            in (
                "LD_LIBRARY_PATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
        "supervisor_sha256": common.sha(__file__),
        "supervisor_pid": os.getpid(),
        "driver_sha256": common.sha(common.ROOT / "run.py"),
        "launched": False,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if not args.execute:
        import json

        print(json.dumps(record, indent=2))
        return 0
    import fcntl

    common.validate_ready()
    if args.output.exists():
        raise FileExistsError(
            "output must be fresh; completed or partial runs cannot be overwritten"
        )
    # A single marker for both profiles prevents another local launch from this kit.
    # The coordinator must preserve this same four-route budget when transferring a kit.
    with (common.ROOT / "launch.json").open("x") as marker:
        import json

        marker.write(json.dumps(record, indent=2) + "\n")
    try:
        common.validate_profile(args.profile, environment)
        with Path(profile["lock"]).open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            )
            record["compute_process_observation"] = check.stdout.strip()
            if check.stdout.strip():
                raise RuntimeError("compute process present; preserved no-GPU launch attempt")
            record["gpu_inventory"] = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.used",
                    "--format=csv,noheader",
                ],
                text=True,
                capture_output=True,
                timeout=15,
                check=True,
            ).stdout.strip()
            command += ["--lock-fd", str(lock.fileno())]
            with (common.ROOT / "run.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=common.ROOT / "source",
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                    start_new_session=True,
                )
                record.update(launched=True, pid=child.pid, command=command)
                common.write(common.ROOT / "launch.json", record)
                print(
                    json.dumps(
                        {
                            "pid": child.pid,
                            "supervisor_pid": os.getpid(),
                            "profile": args.profile,
                            "output": str(args.output.resolve()),
                        }
                    ),
                    flush=True,
                )
                returncode = child.wait()
            record.update(
                returncode=returncode,
                finished_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            )
            common.write(common.ROOT / "launch.json", record)
            return returncode
    except Exception as error:
        record.update(
            error=repr(error),
            status="supervisor_failed" if record["launched"] else "refused_before_GPU_launch",
        )
        common.write(common.ROOT / "launch.json", record)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", choices=("local", "h100"), required=True)
    parser.add_argument("--output", type=Path, default=common.ROOT / "output")
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 3600:
        parser.error("bounded wall budget of 0..3600 seconds required")
    sys.exit(main(args))
