"""Keep the reviewed v599 child attached to a WSL command until it exits."""
import json
from pathlib import Path
import runpy
import sys
import time

root = Path(__file__).resolve().parent / "asteroid-substitution-v599"
sys.argv = [str(root / "launch.py"), "--execute"]
namespace = runpy.run_path(str(root / "launch.py"), run_name="__main__")
child = namespace["process"]
print(json.dumps({"child_pid": child.pid, "output": str(root / "output")}), flush=True)
last_report = None
while child.poll() is None:
    path = root / "output" / "report.json"
    if path.exists():
        try:
            report = json.loads(path.read_text())
            compact = {key: report.get(key) for key in ("status", "complete", "stage", "seconds")}
            encoded = json.dumps(compact, sort_keys=True)
            if encoded != last_report:
                print(encoded, flush=True)
                last_report = encoded
        except (OSError, json.JSONDecodeError):
            pass
    time.sleep(5)
print(json.dumps({"child_pid": child.pid, "returncode": child.returncode}), flush=True)
sys.exit(child.returncode)
