"""Foreground local supervisor; print-only by default, exclusive single launch."""

import argparse
import datetime
import json
import os
import signal
import subprocess
from pathlib import Path

import common


def wait_owned_child(child, record, *, timeout, grace=10.0):
    """Bound only the new-session child that this supervisor created and owns."""
    record["outer_timeout_seconds"] = timeout
    record["termination_grace_seconds"] = grace
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        record.update(status="supervisor_timeout", timed_out=True)

        # start_new_session=True makes the owned child the process-group leader.
        # Check immediately before signalling so another group is never targeted.
        def stop(sig):
            try:
                group = os.getpgid(child.pid)
            except ProcessLookupError:
                return
            if group != child.pid or child.pid <= 1:
                # Unexpected group changes must not result in a signal to its group.
                record["unexpected_child_process_group"] = group
                child.send_signal(sig)
            else:
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass

        stop(signal.SIGTERM)
        try:
            actual_code = child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            record["forced_kill"] = True
            stop(signal.SIGKILL)
            actual_code = child.wait()
        record["child_returncode"] = actual_code
        return 124


def retain_timeout_state(output, record):
    """Keep the raw partial driver report intact and mark its counts as incomplete."""
    state = {
        "status": "supervisor_timeout",
        "execution_ended": True,
        "counts_complete": False,
        "reason": "Owned child exceeded wall budget plus 30-second cleanup allowance",
        "count_scope": "Last saved counts may omit an interrupted native call",
        "child_returncode": record.get("child_returncode"),
        "automatic_retry": False,
    }
    report_path = output / "report.json"
    if report_path.is_file():
        report = common.read(report_path)
        state["preserved_report_sha256"] = common.sha(report_path)
        state["preserved_report"] = {
            key: report.get(key)
            for key in (
                "status",
                "complete",
                "native_legs_started",
                "native_legs_finished",
                "active_case",
                "best",
                "searches_completed",
                "full_refinements_started",
            )
        }
    record["timeout_evidence"] = state
    if output.is_dir():
        common.write(output / "supervisor-timeout.json", state)


def main(args):
    profile, env = common.environment()
    command = [
        profile["python"],
        "-B",
        str(common.ROOT / "run.py"),
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
        "supervisor_sha256": common.sha(__file__),
        "driver_sha256": common.sha(common.ROOT / "run.py"),
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
        "maximum_native_legs": 68,
        "maximum_full_refinements": 4,
        "maximum_device_searches": 12,
        "launched": False,
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if not args.execute:
        print(json.dumps(record, indent=2))
        return 0
    import fcntl

    common.ready_check()
    if args.output.exists():
        raise FileExistsError("output directory must be new")
    with (common.ROOT / "launch.json").open("x") as marker:
        marker.write(json.dumps(record, indent=2) + "\n")
    try:
        common.runtime_check(env)
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
            with (common.ROOT / "run.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    env=env,
                    cwd=common.ROOT / "source",
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
                            "output": str(args.output.resolve()),
                        }
                    ),
                    flush=True,
                )
                code = wait_owned_child(child, record, timeout=args.wall_seconds + 30)
                if record.get("timed_out"):
                    retain_timeout_state(args.output, record)
            record.update(
                returncode=code, finished_utc=datetime.datetime.now(datetime.UTC).isoformat()
            )
            common.write(common.ROOT / "launch.json", record)
            return code
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
    parser.add_argument("--output", type=Path, default=common.ROOT / "output")
    parser.add_argument("--wall-seconds", type=float, default=1800)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 1800:
        parser.error("wall budget must be positive and at most 1800 seconds")
    raise SystemExit(main(args))
