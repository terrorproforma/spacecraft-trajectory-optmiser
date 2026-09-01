from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.experiments import (
    Paper1ResultError,
    read_paper1_result,
    timing_from_components,
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


def valid_g4_result() -> dict[str, object]:
    payload = valid_result()
    payload["identity"].update(  # type: ignore[union-attr]
        gate="G4",
        campaign="g4-primary",
        quality_tier="tight",
        conditioning=2.0,
        scaling_mode="refresh_if_needed",
        warm_mode="primal",
        seed=59,
        repeat_kind="measured",
        repeat=0,
        solver_order=2,
        failure_class="none",
        failure_reason=None,
    )
    payload["quality"].update(  # type: ignore[union-attr]
        dynamics_residual=1e-7,
        path_residual=1e-7,
        terminal_residual=1e-7,
        continuous_time_violation=1e-7,
        solver_status="converged",
        convergence_criteria_met=True,
        objective_equivalent=True,
        matched_quality_state="matched",
        independent_replay=True,
        path_inventory_complete=True,
        uses_solver_cached_residuals=False,
        path_inventory={
            name: {"violation": 1e-7, "independent": True}
            for name in ("thrust", "mass", "altitude", "glide_slope")
        },
    )
    payload["timing"] = {
        "topology_seconds": 0.0,
        **timing_from_components(
            {
                "coefficient_seconds": 0.01,
                "update_seconds": 0.001,
                "scaling_seconds": 0.0002,
                "h2d_seconds": 0.0001,
                "solve_seconds": 0.02,
                "recovery_seconds": 0.001,
                "residual_seconds": 0.001,
                "replay_seconds": 0.003,
                "acceptance_seconds": 0.0001,
                "d2h_seconds": 0.0001,
            }
        ),
    }
    payload["resources"].update(  # type: ignore[union-attr]
        energy_scope="GPU-only",
        energy_sampling_interval_milliseconds=50,
        energy_maximum_gap_seconds=0.05,
        energy_sampling_gaps=False,
        energy_valid=True,
        shared_or_display_gpu=False,
    )
    payload["aggregation"].update(  # type: ignore[union-attr]
        instance_count=20,
        evaluation_seed_count=20,
        paired_bootstrap_samples=10_000,
    )
    for artifact in payload["artifacts"].values():  # type: ignore[union-attr]
        if artifact is not None:
            artifact.update(
                immutable_uri="https://artifacts.example.invalid/run/content",
                internal_index_sha256=_SHA256,
                portable=True,
            )
    runtime = {
        "policy": "adaptive+polish",
        "quality_tier": "tight",
        "scaling_mode": "refresh_if_needed",
        "warm_mode": "primal",
    }
    payload["g4"] = {
        "policy_sha256": _SHA256,
        "runtime_requested": copy.deepcopy(runtime),
        "runtime_actual": copy.deepcopy(runtime),
        "outer_iterations": [
            {
                "phase": "polish",
                "requested_tolerance": 1e-8,
                "achieved_residual": 1e-8,
                "forcing_satisfied": True,
                "trust_before": 1.0,
                "trust_after": 1.8,
                "trust_action": "expand",
                "re_solved": False,
                "cqp_fingerprint": "abc",
                "resolve_fingerprint": "abc",
                "fingerprint_match": True,
                "scaling_refreshed": True,
                "predicted_reduction": 1.0,
                "actual_reduction": 0.9,
                "reduction_ratio": 0.9,
                "scaling_min": 0.1,
                "scaling_max": 10.0,
                "warm_mode_actual": "primal",
                "recovery_reason": "not-required",
                "polish_handoff": True,
                "accepted": True,
            }
        ],
        "hybrid_permutation": None,
        "hybrid_dual_disposition": None,
    }
    return payload


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
            lambda value: value["resources"].update(topology_allocation_count_after_create=None),
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


def test_primary_g4_result_passes_schema_and_semantic_contract() -> None:
    payload = valid_g4_result()
    schema = json.loads(
        (Path(__file__).parents[1] / "experiments/schema/paper1_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    validate_paper1_result(payload)


@pytest.mark.parametrize(
    ("section", "field", "message"),
    [
        ("identity", "quality_tier", "missing required keys"),
        ("quality", "continuous_time_violation", "missing required keys"),
        ("timing", "recovery_seconds", "missing required keys"),
        ("resources", "energy_valid", "missing required keys"),
        ("aggregation", "instance_count", "missing required keys"),
        ("g4", "outer_iterations", "missing required keys"),
    ],
)
def test_primary_g4_schema_gaps_are_rejected(section: str, field: str, message: str) -> None:
    payload = valid_g4_result()
    del payload[section][field]  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match=message):
        validate_paper1_result(payload)


def test_nonportable_local_evidence_cannot_be_qualified() -> None:
    payload = valid_g4_result()
    payload["artifacts"]["raw"].update(  # type: ignore[index]
        immutable_uri="file:///tmp/raw.tar",
        portable=False,
    )
    with pytest.raises(Paper1ResultError, match="not portable"):
        validate_paper1_result(payload)

    payload["identity"].update(  # type: ignore[union-attr]
        status="unqualified",
        failure_class="evidence",
        failure_reason="raw evidence is local-only",
    )
    payload["quality"].update(  # type: ignore[union-attr]
        qualified=False,
        matched_quality_state="unqualified",
    )
    validate_paper1_result(payload)


def test_primary_g4_semantics_reject_false_quality_flags() -> None:
    payload = valid_g4_result()
    payload["quality"]["reference_objective"] = 99.0  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match="not practically equivalent"):
        validate_paper1_result(payload)
    payload = valid_g4_result()
    payload["quality"]["path_inventory"]["thrust"]["violation"] = 1e-2  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match="path inventory check thrust"):
        validate_paper1_result(payload)


@pytest.mark.parametrize("status", ["not_applicable", "unsupported"])
def test_primary_g4_static_terminal_dispositions_are_strict_and_timed(status: str) -> None:
    payload = valid_g4_result()
    payload["identity"].update(  # type: ignore[union-attr]
        status=status,
        failure_class=status,
        failure_reason=f"authoritative {status} reason",
    )
    payload["quality"].update(  # type: ignore[union-attr]
        qualified=False,
        solver_status=status,
        matched_quality_state="unqualified",
    )
    payload["g4"]["outer_iterations"] = []  # type: ignore[index]
    validate_paper1_result(payload)
    payload["timing"]["cqp_total_seconds"] = None  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match=r"must be numeric|requires explicit timing"):
        validate_paper1_result(payload)


def test_hybrid_handoff_ineligible_is_not_max_iterations_or_unsupported() -> None:
    payload = valid_g4_result()
    payload["identity"].update(  # type: ignore[union-attr]
        solver="hybrid-pdhcg-ipm",
        policy="hybrid-pdhcg-ipm",
        status="hybrid_handoff_ineligible",
        failure_class="hybrid_handoff_ineligible",
        failure_reason="construction residual exceeded the frozen 1e-6 handoff gate",
    )
    payload["quality"].update(  # type: ignore[union-attr]
        qualified=False,
        solver_status="hybrid_handoff_ineligible",
        matched_quality_state="unqualified",
    )
    for name in ("runtime_requested", "runtime_actual"):
        payload["g4"][name]["policy"] = "hybrid-pdhcg-ipm"  # type: ignore[index]
    payload["g4"].update(  # type: ignore[union-attr]
        hybrid_permutation=[0, 1],
        hybrid_dual_disposition="discarded_unsupported",
    )
    validate_paper1_result(payload)


def test_raw_measured_result_scope_is_strict_but_not_a_publication_aggregate() -> None:
    payload = valid_g4_result()
    payload["identity"]["record_scope"] = "measured_attempt"  # type: ignore[index]
    payload["aggregation"].update(  # type: ignore[union-attr]
        warmup_repeats=0,
        measured_repeats=1,
        instance_count=1,
        evaluation_seed_count=1,
        paired_bootstrap_samples=0,
    )
    validate_paper1_result(payload)
    payload["aggregation"]["measured_repeats"] = 7  # type: ignore[index]
    with pytest.raises(Paper1ResultError, match="0 warmups and 1 measured"):
        validate_paper1_result(payload)
