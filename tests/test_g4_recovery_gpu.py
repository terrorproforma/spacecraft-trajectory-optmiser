"""Exercise corrected coefficients and real IPM recovery on a failed Lambda case."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = os.environ.get("SPACEPDHCG_G4_EXECUTOR", "")
pytestmark = pytest.mark.skipif(
    not EXECUTOR or not os.environ.get("SPACEPDHCG_QOCO_LIBRARY"),
    reason="requires a CUDA executor and corrected QOCO library",
)


def run(mode: str) -> str:
    fixture = ROOT / "tests/fixtures/gpu_recovery"
    provenance = json.loads((fixture / "provenance.json").read_text())
    result = subprocess.run(
        [
            EXECUTOR,
            mode,
            str(fixture / "low_thrust_n100_seed71.json"),
            provenance["policy_sha256"],
            provenance["matrix_sha256"],
            provenance["capability_sha256"],
        ],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr
    return result.stdout


def test_repeated_penalty_updates_preserve_cpu_transcription() -> None:
    # The executor compares coefficients with an independently assembled CPU
    # transcription after changing and restoring trust/penalty parameters.
    assert "PD3DATA dimensions" in run("--g4-dump")


def test_recovery_returns_verified_trajectory_and_stops_on_convergence() -> None:
    output = run("--g4-recovery")

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"Non-JSON numeric token: {value}")

    # Consumers must be able to read every diagnostic record, including a
    # first feasibility step whose reduction ratio is undefined.
    for line in output.splitlines():
        if line.startswith("{"):
            json.loads(line, parse_constant=reject_nonfinite)
    records = [
        json.loads(line)
        for line in output.splitlines()
        if line.startswith('{"case":"g4_sample"')
        or line.startswith('{"case":"production_outer"')
        or line.startswith('{"case":"gpu_recovery_profile"')
    ]
    by_case = {record["case"]: record for record in records}
    sample = by_case["g4_sample"]
    assert by_case["gpu_recovery_profile"]["official_g4_sample"] is False
    assert sample["policy"] == "gpu-ipm-recovery-v1"
    assert sample["qualified"] and sample["status"] == 0
    assert sample["canonical_residual"] <= 1e-8
    assert max(sample[key] for key in ("terminal", "dynamics", "path", "virtual")) <= 1e-6
    assert sample["hidden_cpu_fallback"] == 0
    assert sample["objective"] == pytest.approx(0.149993762, abs=1e-7)
    assert sample["qoco_workspace_creations"] >= 1
    assert 1 <= by_case["production_outer"]["outer_iterations"] < 100
