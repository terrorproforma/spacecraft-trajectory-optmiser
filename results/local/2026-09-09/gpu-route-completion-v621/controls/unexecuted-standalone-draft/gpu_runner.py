"""Single foreground supervisor: one native test and one 42-case C ABI batch."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

KIT = Path(__file__).resolve().parent
OUTPUT = KIT / "gpu-output"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    # Consume a single execution marker before any health probe or GPU operation.
    with (KIT / "gpu-launch-marker.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "time": time.time(), "runner_sha256": sha(__file__)}, stream)
        stream.write("\n")
    OUTPUT.mkdir(exist_ok=False)
    report = {"complete": False, "passed": False, "supervisor_pid": os.getpid(),
              "processes": [], "failures": [], "launch_time": time.time(),
              "expected_maximum_kernel_launches": 3, "expected_case_evaluations": 304,
              "new_mission_searches": 0, "new_refinements": 0}
    owned = None
    lock = None

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def reap(child):
        if child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=30)

    def launch(name, command, environment):
        nonlocal owned
        item = {"name": name, "command": command, "start": time.time(), "deadline_seconds": 180}
        report["processes"].append(item)
        with (OUTPUT / f"{name}.log").open("x") as stream:
            owned = subprocess.Popen(command, cwd=KIT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            item["pid"] = owned.pid
            write(OUTPUT / f"{name}-launch.json", item)
            print(json.dumps({"stage": name, "supervisor_pid": os.getpid(), "child_pid": owned.pid}), flush=True)
            try:
                item["exit_code"] = owned.wait(timeout=180)
            except subprocess.TimeoutExpired:
                item["timed_out"] = True
                reap(owned)
                item["exit_code"] = owned.returncode
                raise
            finally:
                item["end"] = time.time()
        assert item["exit_code"] == 0, f"{name} failed; no retry or next stage"
        owned = None

    try:
        pins = read(KIT / "gpu-profile.json")
        ready = read(KIT / "gpu-ready.json")
        for name, digest in ready["files"].items():
            assert sha(KIT / name) == digest, name
        assert sha(pins["native_manifest_path"]) == pins["native_manifest_sha256"]
        native = read(pins["native_manifest_path"])
        assert native["library"] == pins["library"] and native["test"] == pins["test"]
        for item in (pins["library"], pins["test"]):
            assert sha(item["path"]) == item["sha256"]
        for name, digest in native["source_sha256"].items():
            assert sha(Path(pins["native_source_root"]) / name) == digest
        for name, digest in read(KIT / "source-sha256.json").items():
            assert sha(KIT / "source" / name) == digest
        for name, digest in read(KIT / "input-provenance.json")["inputs"].items():
            assert sha(KIT / "inputs" / name) == digest
        report["pins"] = pins
        report["ready_sha256"] = sha(KIT / "gpu-ready.json")
        lock = open(pins["gpu_lock"], "a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        report["lock_acquired"] = time.time()
        smi = pins["nvidia_smi"]
        inventory = subprocess.check_output([smi, "--query-gpu=name,uuid,driver_version,memory.used", "--format=csv,noheader"], text=True, timeout=15)
        report["gpu_inventory"] = inventory
        assert pins["gpu_uuid"] in inventory and "RTX 5090" in inventory
        processes = subprocess.check_output([smi, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True, timeout=15)
        report["compute_processes_before"] = processes
        assert not processes.strip() or processes.strip() == "No running processes found", "GPU already in use; execution marker retained"
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": pins["gpu_uuid"],
                       "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": "1", "SPACEPDHCG_COMPLETION_SUPERVISED": str(os.getpid()),
                       "LD_LIBRARY_PATH": pins["ld_library_path"]}
        write(OUTPUT / "preflight.json", report)
        launch("native-test", [pins["test"]["path"]], environment)
        lines = (OUTPUT / "native-test.log").read_text().splitlines()
        records = [json.loads(line[len("COMPLETION_TEST "):]) for line in lines if line.startswith("COMPLETION_TEST ")]
        assert records == [{"phase": "native", "kernel_launches": 2, "candidates": 262, "passed": True}]
        report["native_records"] = records
        launch("fixture-batch", [pins["python"], "-B", str(KIT / "gpu_batch.py"), str(OUTPUT)], environment)
        comparison = read(OUTPUT / "batch-comparison.json")
        assert comparison["passed"] and comparison["api_evaluate_calls"] == 1
        report.update(passed=True, native_kernel_launches=2, fixture_kernel_launches=1,
                      actual_case_evaluations=304, fixtures=42)
    except BaseException as error:
        report["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if owned is not None:
            try:
                reap(owned)
            except BaseException as error:
                report["failures"].append({"cleanup_error": repr(error), "owned_pid": owned.pid})
        report.update(complete=True, end=time.time())
        write(OUTPUT / "report.json", report)
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        print(json.dumps({"complete": True, "passed": report["passed"], "output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
