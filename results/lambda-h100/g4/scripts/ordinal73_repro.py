"""Run the exact ordinal-73 coordinate (P1-E N=100 adaptive, censoring twin: 600 s / 1M cap) once
on the H100 through the real --g4-session executor and record every attempt's wall against its
deadline.

The executor requires the full nine-attempt manifest, so the *group* deadline is set to
2 x 600 s + 60 s: warm-up/0 and warm-up/1 get the full 600 s attempt deadline, measured/0 gets
the ~60 s remainder (the executor clamps the attempt deadline to the group deadline), and the
remaining six attempts are recorded as ``unrun`` ("group deadline prevented launch"). That
exercises the recovery-phase cancel at 600 s twice and a short cancel once in ~21 minutes
instead of ~90.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ["G4_ROOT"])
sys.path.insert(0, str(ROOT / "tests"))
import test_g4_pdhcg_deadline_gpu as deadline_test  # noqa: E402

EXECUTOR = os.environ["SPACEPDHCG_G4_EXECUTOR"]
OUT = Path(sys.argv[1])
DEADLINE = float(os.environ.get("REPRO_ATTEMPT_DEADLINE", "600"))
GROUP_DEADLINE = float(os.environ.get("REPRO_GROUP_DEADLINE", str(2 * DEADLINE + 60)))
CAP = int(os.environ.get("REPRO_INNER_CAP", "1000000"))

manifest = deadline_test.manifest_for("adaptive", 100, f"ordinal73-h100-{DEADLINE}")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "execution-group.json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
)
environment = dict(os.environ)
environment.update(
    {
        "SPACEPDHCG_G4_GROUP_ID": manifest["group_id"],
        "SPACEPDHCG_G4_POLICY_RESET": "independent-with-persistent-workspace",
        "SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS": str(DEADLINE),
        "SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS": str(GROUP_DEADLINE),
        "SPACEPDHCG_G4_POLICY_AMENDMENT": "single-gpu-v1.2",
        "SPACEPDHCG_G4_CENSORING_STRATUM": "censoring_sensitivity",
        "SPACEPDHCG_G4_INNER_ITERATION_CAP": str(CAP),
        "SPACEPDHCG_G4_DETERMINISTIC_REPLAY": "1",
    }
)
policy_sha256 = (ROOT / "benchmarks/g4_policy.sha256").read_text().split()[0]
command = [EXECUTOR, "--g4-session", str(OUT / "execution-group.json"), policy_sha256, "b" * 64, "c" * 64]
(OUT / "command.txt").write_text(" ".join(command) + "\n")
started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
started = time.monotonic()
completed = subprocess.run(
    command, check=False, capture_output=True, text=True, timeout=GROUP_DEADLINE + 900, env=environment
)
wall = time.monotonic() - started
(OUT / "stdout.jsonl").write_text(completed.stdout)
(OUT / "stderr.log").write_text(completed.stderr)
records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
attempts = [r for r in records if r.get("case") == "g4_attempt"]
rows = []
for r in attempts:
    trace = r.get("trace") or {}
    elapsed = r["timing"]["elapsed_seconds"]
    launched = r.get("launched") is True
    rows.append(
        {
            "attempt": f"{r['repeat_kind']}/{r['repeat']}",
            "launched": launched,
            "disposition": r["disposition"],
            "elapsed_seconds": elapsed,
            "over_deadline_seconds": (elapsed - DEADLINE) if launched and r["disposition"] == "timeout" else None,
            "inner_iterations": trace.get("inner_iterations"),
            "canonical_residual": trace.get("canonical_residual"),
            "reason": r.get("reason"),
        }
    )
summary = {
    "coordinate": manifest["coordinate"],
    "attempt_deadline_seconds": DEADLINE,
    "group_deadline_seconds": GROUP_DEADLINE,
    "inner_iteration_cap": CAP,
    "executor": EXECUTOR,
    "returncode": completed.returncode,
    "session_wall_seconds": wall,
    "started_utc": started_utc,
    "attempts": rows,
    "deadline_exercised": any(row["disposition"] == "timeout" for row in rows),
    "verdict": (
        "PASS"
        if completed.returncode == 0
        and all(
            row["elapsed_seconds"] <= DEADLINE + deadline_test.ATTEMPT_GRACE_SECONDS
            for row in rows
            if row["launched"]
        )
        else "FAIL"
    ),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
raise SystemExit(0 if summary["verdict"] == "PASS" else 1)
