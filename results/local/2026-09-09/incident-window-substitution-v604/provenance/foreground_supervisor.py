"""Launch the approved v604 job once and retain a foreground WSL parent."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED_READY = "bb33c0656f5f60354c263b26eb4b9fc9ec4b9140874016f4175215febf39038c"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, record):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(path)


def observe(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return {
            "command": command,
            "exitcode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": repr(error)}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()
if not args.execute:
    print(
        json.dumps(
            {
                "launched": False,
                "source_sha256": digest(Path(__file__)),
                "ready_manifest_sha256": digest(ROOT / "ready-manifest.json"),
            }
        )
    )
    raise SystemExit(0)

if digest(ROOT / "ready-manifest.json") != EXPECTED_READY:
    raise ValueError("reviewed ready manifest changed")
if any((ROOT / name).exists() for name in ("launch.json", "output", "supervisor.json")):
    raise FileExistsError("an existing launch/output must be observed, never duplicated")
attempt = 1
while (ROOT / f"supervisor-attempt-{attempt:02d}.json").exists():
    attempt += 1
attempt_path = ROOT / f"supervisor-attempt-{attempt:02d}.json"
record = {
    "supervisor_pid": os.getpid(),
    "supervisor_sha256": digest(Path(__file__)),
    "ready_manifest_sha256": EXPECTED_READY,
    "attempt": attempt,
    "started_unix": time.time(),
    "launched": False,
}
with attempt_path.open("x") as stream:
    stream.write(json.dumps(record, indent=2) + "\n")

lock_path = Path.home() / ".spacepdhcg-gpu.lock"
with lock_path.open("a+") as lock:
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        record.update(status="GPU_lock_busy_no_child_launched", lock_available=False)
        record["lock_holders"] = observe(["fuser", "-v", str(lock_path)])
        save(attempt_path, record)
        print(json.dumps(record), flush=True)
        raise SystemExit(75) from None
    record["lock_available"] = True
    record["GPU_processes"] = observe(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]
    )
    observed = record["GPU_processes"]
    if observed.get("exitcode") != 0 or observed.get("stdout", "").strip():
        record["status"] = "GPU_process_observation_needs_review_no_child_launched"
        save(attempt_path, record)
        print(json.dumps(record), flush=True)
        raise SystemExit(76)
    # The reviewed child takes this same nonblocking lock before all GPU work.
    # A racing acquisition can only produce a retained failed no-GPU attempt.
    fcntl.flock(lock, fcntl.LOCK_UN)
save(attempt_path, record)

sys.argv = [str(ROOT / "launch.py"), "--execute"]
namespace = runpy.run_path(str(ROOT / "launch.py"), run_name="__main__")
child = namespace["process"]
record.update(
    launched=True,
    child_pid=child.pid,
    status="running",
    output=str(ROOT / "output"),
    launch_sha256=digest(ROOT / "launch.json"),
)
save(attempt_path, record)
save(ROOT / "supervisor.json", record)
print(
    json.dumps(
        {
            "child_pid": child.pid,
            "supervisor_pid": os.getpid(),
            "output": record["output"],
            "supervisor_sha256": record["supervisor_sha256"],
        }
    ),
    flush=True,
)
previous = None
while child.poll() is None:
    report_path = ROOT / "output/report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
            progress = {k: report.get(k) for k in ("status", "stage", "complete", "seconds")}
            progress.update(
                cases=len(report.get("screened_cases", [])),
                retimings=len(report.get("retimings", [])),
                refinements=len(report.get("refinements", [])),
            )
            encoded = json.dumps(progress, sort_keys=True)
            if encoded != previous:
                print(encoded, flush=True)
                previous = encoded
        except (OSError, json.JSONDecodeError):
            pass
    time.sleep(2)
record.update(status="complete", returncode=child.returncode, finished_unix=time.time())
save(attempt_path, record)
save(ROOT / "supervisor.json", record)
print(json.dumps({"child_pid": child.pid, "returncode": child.returncode}), flush=True)
raise SystemExit(child.returncode)
