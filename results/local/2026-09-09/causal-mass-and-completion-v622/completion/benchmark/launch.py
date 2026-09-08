"""Single-use foreground supervisor; never bypass a busy GPU or missing prerequisite."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_prerequisite(report, profile):
    if (
        report.get("complete") is not True
        or report.get("exit_code") != 0
        or report.get("expected_readbacks_present") is not True
        or report.get("source_report_sha256") != profile["source_report_sha256"]
        or report.get("library_sha256") != profile["library"]["sha256"]
        or report.get("junit") != {"tests": 7, "failures": 0, "errors": 0, "skipped": 0}
    ):
        raise ValueError("matching compact-g GPU correctness evidence has not passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--prerequisite-sha256", required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Evidence assertions must be enabled")
    write(
        KIT / "launch-marker.json",
        {"pid": os.getpid(), "created": time.time(), "launcher_sha256": sha(__file__)},
    )
    output = KIT / "output"
    output.mkdir(exist_ok=False)
    report = {"complete": False, "passed": False, "supervisor_pid": os.getpid(), "failures": []}
    child = lock = None

    def reap():
        if child is None or child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        ready, profile = read(KIT / "ready.json"), read(KIT / "profile.json")
        for name, expected in ready["files"].items():
            assert sha(KIT / name) == expected, name
        assert sha(args.prerequisite) == args.prerequisite_sha256
        prerequisite = read(args.prerequisite)
        validate_prerequisite(prerequisite, profile)
        # Preserve the exact late-arriving prerequisite without changing frozen code.
        (output / "prerequisite.json").write_bytes(args.prerequisite.read_bytes())
        assert sha(profile["python"]) == profile["python_sha256"]
        assert sha(profile["library"]["path"]) == profile["library"]["sha256"]
        assert sha(Path(profile["native_root"]) / "report.json") == profile["source_report_sha256"]
        native = read(KIT / "inputs/compact-g-report.json")
        for name, expected in native["owned_sources"].items():
            assert sha(Path(profile["native_root"]) / "source" / name) == expected
        lock = open(profile["lock"], "a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        def query(flag):
            return subprocess.check_output(
                [profile["nvidia_smi"], flag, "--format=csv,noheader"], text=True, timeout=15
            ).strip()

        inventory = query("--query-gpu=uuid,name,driver_version")
        processes = query("--query-compute-apps=pid,process_name")
        report.update(gpu_inventory=inventory, compute_processes=processes)
        assert profile["gpu_uuid"] in inventory and "RTX 5090" in inventory
        assert not processes or processes == "No running processes found"
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SPACEPDHCG_", "QOCO_", "PDHCG_"))
        }
        env.update(
            CUDA_VISIBLE_DEVICES=profile["gpu_uuid"],
            PYTHONOPTIMIZE="0",
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            SPACEPDHCG_PACK_BENCHMARK_SUPERVISED=str(os.getpid()),
            SPACEPDHCG_GTOC12_DATA=read(KIT / "plan.json")["data_path"],
        )
        command = [profile["python"], "-B", str(KIT / "run.py")]
        report.update(
            ready_sha256=sha(KIT / "ready.json"),
            prerequisite_sha256=args.prerequisite_sha256,
            command=command,
            core_sha256=profile["library"]["sha256"],
        )
        write(output / "preflight.json", report)
        with (output / "worker.log").open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=KIT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
            report["child_pid"] = child.pid
            write(output / "child-launch.json", {"pid": child.pid, "deadline_seconds": 180})
            print(json.dumps({"supervisor_pid": os.getpid(), "child_pid": child.pid}), flush=True)
            try:
                report["exit_code"] = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
                reap()
                raise
        assert report["exit_code"] == 0
        measured = read(output / "benchmark-report.json")
        assert measured["passed"] and measured["native_calls"] == 80
        assert measured["candidates"] == {"ordinary": 13520, "compact": 13520}
        report["passed"] = True
    except BaseException as error:
        report["failures"].append({"error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            reap()
        except BaseException as error:
            report["failures"].append({"reaping_error": repr(error)})
        report.update(complete=True, ended=time.time())
        write(output / "launch-report.json", report)
        if lock is not None:
            lock.close()  # Any unreaped child still owns the inherited descriptor.


if __name__ == "__main__":
    main()
