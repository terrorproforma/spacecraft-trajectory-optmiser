"""Regression: the G4 per-attempt deadline must bound every launched PDHCG attempt.

Background (integration/single-gpu-v1, campaign g4-claim-core-4db5047, ordinal 73): an adaptive
P1-E N=100 censoring twin (600 s deadline, 1,000,000 inner cap) ran its first five attempts to a
prompt 600 s cancellation, then the sixth attempt ran for more than 47 minutes and the group hit
the 5760 s safety boundary. The inner solve had left the PDHG loop for the recovery kernel, whose
projected-gradient loop read the mapped cancellation flag per thread (a barrier-divergence
hazard) and whose feasibility-polish / KKT-reconstruction phases never polled it at all.

This test drives the real ``--g4-session`` executor on the exact ordinal-73 coordinate and on its
fixed-tight / hybrid-pdhcg-ipm and N=2000 variants with short attempt deadlines, and asserts
that every launched attempt's measured wall stays within ``deadline + grace`` and that the
session as a whole stays within nine deadlines plus startup and grace. It needs the CUDA build
and the QOCO library, so it skips unless ``SPACEPDHCG_G4_EXECUTOR`` points at
``device_scvx_integration_test`` and ``SPACEPDHCG_QOCO_LIBRARY`` is set. The full matrix takes
about 25 minutes on an RTX 5090; ``SPACEPDHCG_G4_DEADLINE_TEST_QUICK=1`` runs the N=100 5 s
cases only.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = os.environ.get("SPACEPDHCG_G4_EXECUTOR", "")
QUICK = os.environ.get("SPACEPDHCG_G4_DEADLINE_TEST_QUICK", "") == "1"

pytestmark = pytest.mark.skipif(
    not EXECUTOR or not Path(EXECUTOR).is_file() or not os.environ.get("SPACEPDHCG_QOCO_LIBRARY"),
    reason="needs SPACEPDHCG_G4_EXECUTOR (CUDA executor) and SPACEPDHCG_QOCO_LIBRARY",
)

# Seconds an attempt may run past its deadline: the deadline thread's cancel must be honoured
# within one kernel loop body (well under a second), plus host-side finalisation (checkpoint
# restore, diagnostics, NVML sampler join).
ATTEMPT_GRACE_SECONDS = 2.0
# Per-session slack on top of nine attempts: CUDA startup, workspace creation and the
# inter-attempt reset boundaries.
SESSION_OVERHEAD_SECONDS = 45.0

# The exact ordinal-73 coordinate of campaign g4-claim-core-4db5047 (P1-E, N=100, adaptive,
# tight, conditioning 4.0, seed 173, censoring_sensitivity twin).
ORDINAL_73 = {
    "censoring_stratum": "censoring_sensitivity",
    "conditioning": 4.0,
    "family": "P1-E-low-thrust",
    "intervals": 100,
    "policy": "adaptive",
    "quality_tier": "tight",
    "quality_tolerance": 1e-06,
    "scaling_mode": "refresh_if_needed",
    "seed": 173,
    "solver_order": 3,
    "transfer_class": "combined",
    "trust_class": 1.0,
    "warm_mode": "primal",
}


def _group_id(tag: str) -> str:
    digest = tag.encode().hex()
    return "g4-group-v1-" + (digest * 8)[:64]


def _instance_id(tag: str) -> str:
    digest = tag.encode().hex()
    return "g4-instance-v2-" + (digest * 8)[:64]


def manifest_for(policy: str, intervals: int, tag: str) -> dict:
    coordinate = {**ORDINAL_73, "policy": policy, "intervals": intervals}
    group_id = _group_id(tag)
    instance = _instance_id(tag)
    attempts = []
    for kind, repeat in (("warmup", 0), ("warmup", 1), *(("measured", r) for r in range(7))):
        attempts.append(
            {
                **coordinate,
                "group_id": group_id,
                "instance": instance,
                "repeat_kind": kind,
                "repeat": repeat,
                "statistics_eligible": kind == "measured",
            }
        )
    return {
        "attempts": attempts,
        "coordinate": coordinate,
        "group_id": group_id,
        "physical_instance_id": instance,
        "process_contract": {
            "persistent_session": True,
            "persistent_workspace": True,
            "policy_reset_between_attempts": True,
            "processes": 1,
        },
        "record_kind": "execution_group",
        "schema_version": "1.0.0",
    }


def run_session(policy: str, intervals: int, deadline: float, cap: int) -> dict:
    tag = f"{policy}-{intervals}-{deadline}"
    manifest = manifest_for(policy, intervals, tag)
    group_deadline = 9 * deadline + 60
    with tempfile.TemporaryDirectory(prefix="spacepdhcg-g4-deadline-") as directory:
        path = Path(directory) / "execution-group.json"
        path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
        environment = dict(os.environ)
        environment.update(
            {
                "SPACEPDHCG_G4_GROUP_ID": manifest["group_id"],
                "SPACEPDHCG_G4_POLICY_RESET": "independent-with-persistent-workspace",
                "SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS": str(deadline),
                "SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS": str(group_deadline),
                "SPACEPDHCG_G4_POLICY_AMENDMENT": "single-gpu-v1.2",
                "SPACEPDHCG_G4_CENSORING_STRATUM": "censoring_sensitivity",
                "SPACEPDHCG_G4_INNER_ITERATION_CAP": str(cap),
                "SPACEPDHCG_G4_DETERMINISTIC_REPLAY": "1",
            }
        )
        # The executor refuses any policy hash other than its compiled frozen policy; the
        # matrix and capability hashes are echoed into the records only.
        policy_sha256 = (ROOT / "benchmarks/g4_policy.sha256").read_text().split()[0]
        started = time.monotonic()
        completed = subprocess.run(
            [EXECUTOR, "--g4-session", str(path), policy_sha256, "b" * 64, "c" * 64],
            check=False,
            capture_output=True,
            text=True,
            timeout=group_deadline + 300,
            env=environment,
        )
        wall = time.monotonic() - started
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    diagnostics = [
        json.loads(line) for line in completed.stderr.splitlines() if line.startswith("{")
    ]
    return {
        "returncode": completed.returncode,
        "wall": wall,
        "records": records,
        "diagnostics": diagnostics,
        "stderr_tail": completed.stderr[-2000:],
    }


def _matrix() -> list[tuple[str, int, float]]:
    policies = ("adaptive", "fixed-tight", "hybrid-pdhcg-ipm")
    if QUICK:
        return [(policy, 100, 5.0) for policy in policies]
    return [
        (policy, intervals, deadline)
        for intervals in (100, 2000)
        for deadline in (5.0, 20.0)
        for policy in policies
    ]


@pytest.mark.parametrize(("policy", "intervals", "deadline"), _matrix())
def test_attempt_deadline_bounds_every_launched_pdhcg_attempt(
    policy: str, intervals: int, deadline: float
) -> None:
    session = run_session(policy, intervals, deadline, cap=1_000_000)
    assert session["returncode"] == 0, session["stderr_tail"]
    attempts = [record for record in session["records"] if record.get("case") == "g4_attempt"]
    ready = [record for record in session["records"] if record.get("case") == "g4_session_ready"]
    complete = [
        record for record in session["records"] if record.get("case") == "g4_session_complete"
    ]
    assert len(ready) == 1 and len(complete) == 1
    assert [(r["repeat_kind"], r["repeat"]) for r in attempts] == [
        ("warmup", 0),
        ("warmup", 1),
        *(("measured", index) for index in range(7)),
    ]
    launched = [record for record in attempts if record.get("launched") is True]
    # The group deadline (nine attempts plus 60 s) leaves room for every attempt, so the group
    # deadline must never prevent a launch. Attempts may legitimately be recorded as
    # deterministic replays (amendment v1.1 rule 1) only in the degenerate case where the
    # deadline is shorter than the solve preamble and warm-up/0, warm-up/1 and measured/0 all
    # timed out before the first PDHG iteration with identical zero-work traces.
    assert len(launched) >= 3, [record.get("disposition") for record in attempts]
    for record in attempts:
        if record.get("launched") is not True:
            assert record["disposition"] == "timeout_deterministic_replay", (
                record["disposition"],
                record.get("reason"),
            )
    for record in launched:
        elapsed = record["timing"]["elapsed_seconds"]
        assert elapsed <= deadline + ATTEMPT_GRACE_SECONDS, (
            f"{policy} N={intervals}: {record['attempt_id']} ran {elapsed:.2f} s against a "
            f"{deadline:.0f} s deadline"
        )
        # A cancelled attempt is an honest launched timeout, never an executor defect.
        assert record["disposition"] in {"timeout", "qualified", "unqualified", "numerical"}, (
            record["disposition"],
            record.get("reason"),
        )
        if record["disposition"] == "timeout":
            trace = record["trace"]
            assert trace["inner_iterations"] < 1_000_000
            assert trace["canonical_residual"] == trace["canonical_residual"]  # finite
    assert session["wall"] <= 9 * (deadline + ATTEMPT_GRACE_SECONDS) + SESSION_OVERHEAD_SECONDS, (
        f"{policy} N={intervals}: session wall {session['wall']:.1f} s"
    )
    # The per-stratum inner iteration cap is echoed by the executor and applied to every inner
    # limit, including the driver's re-solve floor (which must not exceed the cap).
    limits = [d for d in session["diagnostics"] if d.get("case") == "g4_inner_iteration_limits"]
    assert len(limits) == 1
    assert limits[0]["inner_iteration_cap"] == 1_000_000
    assert limits[0]["resolve_floor"] <= 1_000_000
    for key in ("repair", "progress", "refinement", "polish", "final_polish"):
        assert limits[0][key] <= 1_000_000


def test_claim_core_cap_bounds_every_inner_limit_including_resolve_floor() -> None:
    session = run_session("adaptive", 100, 5.0, cap=200_000)
    assert session["returncode"] == 0, session["stderr_tail"]
    limits = [d for d in session["diagnostics"] if d.get("case") == "g4_inner_iteration_limits"]
    assert len(limits) == 1
    assert limits[0]["inner_iteration_cap"] == 200_000
    assert limits[0]["polish"] == 200_000
    assert limits[0]["final_polish"] == 200_000
    assert limits[0]["refinement"] <= 200_000
    # Before the fix the identical-CQP re-solve escalated to max(limit, 350000) regardless of
    # the amendment cap; it must now stay at or below the claim-core cap.
    assert limits[0]["resolve_floor"] == 200_000
