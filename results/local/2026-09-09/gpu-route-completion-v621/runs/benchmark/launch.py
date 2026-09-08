"""One foreground, single-use, locked benchmark process; no retry."""

from __future__ import annotations

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


def main():
    if not __debug__:
        raise RuntimeError("Evidence assertions must be enabled")
    write(
        KIT / "launch-marker.json",
        {"supervisor_pid": os.getpid(), "time": time.time(), "launcher_sha256": sha(__file__)},
    )
    output = KIT / "output"
    output.mkdir(exist_ok=False)
    report = {"complete": False, "passed": False, "supervisor_pid": os.getpid(), "failures": []}
    child = None
    lock = None

    def interrupt(signum, frame):
        raise InterruptedError(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    def reap():
        if child is None or child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)

    try:
        ready = read(KIT / "ready.json")
        for name, digest in ready["files"].items():
            assert sha(KIT / name) == digest, name
        pins = read(KIT / "inputs/gpu-profile.json")
        prerequisite = read(KIT / "adapter-gpu-prerequisite.json")
        assert prerequisite["passed"] and not prerequisite["failures"]
        assert prerequisite["pins"]["library"] == pins["library"]
        assert sha(pins["python"]) == prerequisite["pins"]["python_sha256"]
        assert sha(pins["native_manifest_path"]) == pins["native_manifest_sha256"]
        native = read(pins["native_manifest_path"])
        assert native["library"] == pins["library"]
        assert sha(pins["library"]["path"]) == pins["library"]["sha256"]
        for name, digest in native["source_sha256"].items():
            assert sha(Path(pins["native_source_root"]) / name) == digest
        for name, digest in read(KIT / "source-sha256.json").items():
            assert sha(KIT / "source" / name) == digest
        for name, digest in read(KIT / "input-sha256.json").items():
            assert sha(KIT / "inputs" / name) == digest
        report["ready_sha256"] = sha(KIT / "ready.json")
        report["fullcore_sha256"] = pins["library"]["sha256"]
        lock = open(pins["gpu_lock"], "a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = subprocess.check_output(
            [pins["nvidia_smi"], "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=15,
        )
        report["gpu_inventory"] = inventory
        assert pins["gpu_uuid"] in inventory and "RTX 5090" in inventory
        processes = subprocess.check_output(
            [pins["nvidia_smi"], "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
            timeout=15,
        )
        report["compute_processes_before"] = processes
        assert not processes.strip() or processes.strip() == "No running processes found"
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": pins["gpu_uuid"],
            "PYTHONOPTIMIZE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "SPACEPDHCG_GTOC12_CUDA_LIBRARY": pins["library"]["path"],
            "SPACEPDHCG_TEST_GTOC12_COMPLETION_BATCH": "1",
            "SPACEPDHCG_COMPLETION_BENCHMARK_SUPERVISED": str(os.getpid()),
            "LD_LIBRARY_PATH": pins["ld_library_path"],
        }
        command = [pins["python"], "-B", str(KIT / "run.py")]
        report["command"] = command
        write(output / "preflight.json", report)
        with (output / "worker.log").open("x") as stream:
            child = subprocess.Popen(
                command,
                cwd=KIT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
            report["child_pid"] = child.pid
            write(
                output / "child-launch.json",
                {"pid": child.pid, "command": command, "deadline_seconds": 180},
            )
            print(json.dumps({"supervisor_pid": os.getpid(), "child_pid": child.pid}), flush=True)
            try:
                report["exit_code"] = child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                report["timed_out"] = True
                reap()
                report["exit_code"] = child.returncode
                raise
        assert report["exit_code"] == 0
        outcome = read(output / "benchmark-report.json")
        assert outcome["passed"] and outcome["GPU_evaluate_calls"] == 60
        report["passed"] = True
    except BaseException as error:
        report["failures"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        try:
            reap()
        except BaseException as error:
            report["failures"].append({"cleanup_error": repr(error)})
        report.update(complete=True, end=time.time())
        write(output / "launch-report.json", report)
        if lock is not None:
            lock.close()  # An unreaped child retains its inherited descriptor/lock.


if __name__ == "__main__":
    main()
