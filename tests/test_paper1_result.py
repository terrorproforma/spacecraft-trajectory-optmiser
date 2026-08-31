from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from spacepdhcg.experiments import (
    Paper1ResultError,
    read_paper1_result,
    validate_paper1_result,
    write_paper1_result,
)

_SHA256 = "a" * 64


def valid_result() -> dict[str, object]:
    artifact = {"location": "artifact://example", "sha256": _SHA256}
    return {
        "schema_version": "1.0.0",
        "identity": {
            "run_id": "run-001",
            "repository_commit": "b" * 40,
            "family": "P1-C-pd3",
            "instance_id": "pd3-n100-seed0",
            "solver": "spacepdhcg-persistent",
            "policy": "adaptive+polish",
            "status": "qualified",
            "hardware_id": "gpu-host-01",
            "precision": "float64",
            "warm_start": True,
            "cold_start": False,
        },
        "dimensions": {
            "intervals": 100,
            "scenarios": 1,
            "gpus": 1,
            "state_dimension": 7,
            "control_dimension": 4,
            "variables": 1200,
            "scalar_rows": 900,
            "affine_rows": 500,
            "q_nonzeros": 1200,
            "a_nonzeros": 10000,
            "f_nonzeros": 1500,
            "cone_inventory": {"soc": 100},
            "topology_bytes": 100000,
            "numeric_bytes": 200000,
        },
        "quality": {
            "qualified": True,
            "objective": 12.0,
            "reference_objective": 12.0001,
            "objective_gap": 0.0001,
            "canonical_primal_residual": 1.0e-7,
            "canonical_dual_residual": 2.0e-7,
            "canonical_cone_residual": 3.0e-7,
            "canonical_gap": 4.0e-7,
            "native_primal_residual": 1.0e-7,
            "native_dual_residual": 2.0e-7,
            "dynamics_residual": 1.0e-6,
            "path_residual": 2.0e-6,
            "terminal_residual": 3.0e-6,
            "virtual_control_residual": 4.0e-7,
            "nonanticipativity_residual": 0.0,
            "risk_epigraph_residual": 0.0,
            "ct_error_estimate": 5.0e-7,
            "requested_tolerance": 1.0e-6,
            "achieved_residual": 4.0e-7,
        },
        "timing": {
            "topology_seconds": 0.0,
            "coefficient_seconds": 0.01,
            "workspace_create_seconds": 0.0,
            "update_seconds": 0.001,
            "scaling_seconds": 0.0002,
            "h2d_seconds": 0.0001,
            "solve_seconds": 0.02,
            "residual_seconds": 0.001,
            "replay_seconds": 0.003,
            "acceptance_seconds": 0.0001,
            "d2h_seconds": 0.0001,
            "collective_seconds": 0.0,
            "cqp_total_seconds": 0.023,
            "scvx_total_seconds": 0.04,
            "accepted_trajectory_seconds": 0.04,
        },
        "work": {
            "outer_iterations": 5,
            "inner_iterations": 1000,
            "matvecs": 2000,
            "cone_projections": 1000,
            "factorisations": 1,
            "accepted_steps": 4,
            "rejected_steps": 1,
            "resolved_steps": 1,
            "polish_used": True,
            "scaling_refreshes": 1,
        },
        "resources": {
            "peak_device_bytes": 5000000,
            "reserved_device_bytes": 6000000,
            "h2d_bytes": 1000,
            "d2h_bytes": 128,
            "collective_bytes": 0,
            "collective_count": 0,
            "energy_joules": 2.5,
            "topology_allocation_count_after_create": 0,
            "load_imbalance": 1.0,
            "throughput_per_second": 25.0,
        },
        "aggregation": {
            "warmup_repeats": 2,
            "measured_repeats": 7,
            "statistic": "median_iqr",
            "median": 0.04,
            "q1": 0.039,
            "q3": 0.041,
            "minimum": 0.038,
            "maximum": 0.043,
            "coefficient_of_variation": 0.03,
            "censored_count": 0,
            "bootstrap_low": 1.1,
            "bootstrap_high": 1.3,
        },
        "artifacts": {
            "manifest": copy.deepcopy(artifact),
            "raw": copy.deepcopy(artifact),
            "stdout": copy.deepcopy(artifact),
            "stderr": copy.deepcopy(artifact),
            "nsys": None,
            "ncu": None,
            "compute_sanitizer": None,
            "energy_trace": None,
        },
        "notes": [],
    }


def test_valid_result_round_trip(tmp_path: Path) -> None:
    payload = valid_result()
    validate_paper1_result(payload)
    path = write_paper1_result(payload, tmp_path / "result.json")
    assert read_paper1_result(path) == json.loads(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value["identity"].update(family="unknown"), "unknown Paper 1 family"),
        (lambda value: value["dimensions"].update(gpus=0), "requires dimensions.gpus"),
        (
            lambda value: value["quality"].update(qualified=False),
            "requires quality.qualified=true",
        ),
        (
            lambda value: value["aggregation"].update(measured_repeats=4),
            "at least five repeats",
        ),
        (
            lambda value: value["aggregation"].update(q1=0.05),
            "not monotonically ordered",
        ),
        (
            lambda value: value["resources"].update(
                topology_allocation_count_after_create=None
            ),
            "must record post-create topology allocations",
        ),
        (
            lambda value: value["artifacts"]["raw"].update(sha256="not-a-hash"),
            "lowercase SHA-256",
        ),
    ],
)
def test_invalid_results_are_rejected(mutator: object, message: str) -> None:
    payload = valid_result()
    assert callable(mutator)
    mutator(payload)  # type: ignore[operator]
    with pytest.raises(Paper1ResultError, match=message):
        validate_paper1_result(payload)


def test_qualified_result_may_not_hide_missing_residual() -> None:
    payload = valid_result()
    payload["quality"]["canonical_gap"] = None  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match="may not omit"):
        validate_paper1_result(payload)


def test_scientific_nan_is_rejected_before_serialisation() -> None:
    payload = valid_result()
    payload["quality"]["objective"] = float("nan")  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match="must be finite"):
        validate_paper1_result(payload)
