from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.experiments.g4 import load_policy
from spacepdhcg.experiments.g4_execution_contract import (
    ExecutionGroup,
    iter_claim_core_groups,
    load_claim_core,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "g4_claim_core_decision", ROOT / "scripts/gpu/decide_g4_claim_core.py"
)
assert SPEC is not None and SPEC.loader is not None
DECIDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DECIDER)
COMMIT = "a" * 40
SOLVERS = {
    "fixed-tight": "spacepdhcg-persistent",
    "adaptive": "spacepdhcg-persistent",
    "pure-gpu-ipm": "qoco-gpu",
    "hybrid-pdhcg-ipm": "hybrid-pdhcg-ipm",
}
PATHS = {
    "P1-C-pd3": ("thrust", "mass", "altitude", "glide_slope"),
    "P1-E-low-thrust": ("thrust", "mass", "altitude"),
}
CQP = (
    "coefficient_seconds",
    "workspace_create_seconds",
    "update_seconds",
    "scaling_seconds",
    "h2d_seconds",
    "solve_seconds",
    "recovery_seconds",
    "residual_seconds",
    "d2h_seconds",
    "collective_seconds",
    "hybrid_conversion_seconds",
    "hybrid_setup_seconds",
    "polish_seconds",
)


def _core() -> dict[str, Any]:
    return load_claim_core(ROOT / "benchmarks/g4_h5_h6_claim_core.json").values


def _groups() -> dict[tuple[str, int, str, int], ExecutionGroup]:
    table = {}
    for group in iter_claim_core_groups(_core()):
        c = group.coordinate
        table[(c["family"], c["intervals"], c["policy"], c["seed"])] = group
    return table


def _measured(
    group: ExecutionGroup,
    repeat: int,
    seconds: float,
    *,
    disposition: str = "qualified",
    residual: float = 1e-8,
    forcing: bool = True,
) -> dict[str, Any]:
    """Synthesize one raw measured attempt shaped like the native executor output."""

    c = group.coordinate
    qualified = disposition == "qualified"
    failure_class = {
        "qualified": "none",
        "timeout": "timeout",
        "unqualified": "max_iterations",
        "hybrid_handoff_ineligible": "hybrid_handoff_ineligible",
    }[disposition]
    planned = group.attempts[2 + repeat]
    attempt_id = f"{group.group_id}-measured-{repeat}"
    solve = seconds * 0.9
    components = {name: 0.0 for name in CQP}
    components["solve_seconds"] = solve
    components["coefficient_seconds"] = seconds * 0.05
    replay, acceptance = seconds * 0.03, seconds * 0.02
    cqp_total = sum(components.values())
    scvx_total = cqp_total + replay + acceptance
    hybrid = c["policy"] == "hybrid-pdhcg-ipm"
    paper1 = {
        "schema_version": "1.0.0",
        "identity": {
            "run_id": attempt_id,
            "repository_commit": COMMIT,
            "family": c["family"],
            "instance_id": group.physical_instance_id,
            "solver": SOLVERS[c["policy"]],
            "policy": c["policy"],
            "status": disposition,
            "hardware_id": "cuda-device-0",
            "precision": "float64",
            "warm_start": True,
            "cold_start": False,
            "gate": "G4",
            "campaign": "g4-primary",
            "record_scope": "measured_attempt",
            "quality_tier": c["quality_tier"],
            "conditioning": c["conditioning"],
            "scaling_mode": c["scaling_mode"],
            "warm_mode": c["warm_mode"],
            "seed": c["seed"],
            "repeat_kind": "measured",
            "repeat": repeat,
            "solver_order": c["solver_order"],
            "failure_class": failure_class,
            "failure_reason": None if qualified else f"synthetic {disposition}",
        },
        "dimensions": {
            "intervals": c["intervals"],
            "scenarios": 1,
            "gpus": 1,
            "state_dimension": 7,
            "control_dimension": 3,
            "variables": 10 * c["intervals"],
            "scalar_rows": 4 * c["intervals"],
            "affine_rows": 7 * c["intervals"],
            "q_nonzeros": 10 * c["intervals"],
            "a_nonzeros": 70 * c["intervals"],
            "f_nonzeros": 0,
            "cone_inventory": {},
        },
        "quality": {
            "qualified": qualified,
            "objective": 1.5,
            "reference_objective": 1.5,
            "objective_gap": 0.0,
            "canonical_primal_residual": residual,
            "canonical_dual_residual": residual,
            "canonical_cone_residual": residual,
            "canonical_gap": residual,
            "native_primal_residual": residual,
            "native_dual_residual": residual,
            "dynamics_residual": residual,
            "path_residual": residual,
            "terminal_residual": residual,
            "virtual_control_residual": residual,
            "nonanticipativity_residual": 0.0,
            "risk_epigraph_residual": 0.0,
            "ct_error_estimate": residual,
            "requested_tolerance": c["quality_tolerance"],
            "achieved_residual": residual,
            "continuous_time_violation": residual,
            "solver_status": "converged" if qualified else "max_iterations",
            "convergence_criteria_met": qualified,
            "objective_equivalent": qualified,
            "matched_quality_state": "matched" if qualified else "unqualified",
            "independent_replay": True,
            "path_inventory_complete": True,
            "uses_solver_cached_residuals": False,
            "path_inventory": {
                name: {"violation": residual, "independent": True} for name in PATHS[c["family"]]
            },
        },
        "timing": {
            "topology_seconds": 0.0,
            **components,
            "replay_seconds": replay,
            "acceptance_seconds": acceptance,
            "cqp_total_seconds": cqp_total,
            "scvx_total_seconds": scvx_total,
            "accepted_trajectory_seconds": scvx_total,
            "cuda_startup_seconds": 0.4,
            "cuda_startup_included": False,
            "accepted_trajectory_count": 1,
            "accepted_timing_boundary": (
                "coefficient-generation-through-independent-replay-and-acceptance;"
                "cuda-startup-excluded"
            ),
            "cqp_total_identity": list(CQP),
            "scvx_total_identity": [*CQP, "replay_seconds", "acceptance_seconds"],
        },
        "work": {
            "outer_iterations": 3,
            "inner_iterations": 300,
            "matvecs": 1800,
            "cone_projections": 600,
            "factorisations": 0,
            "accepted_steps": 2,
            "rejected_steps": 1,
            "resolved_steps": 0,
            "polish_used": hybrid or c["policy"] == "pure-gpu-ipm",
        },
        "resources": {
            "peak_device_bytes": 1000,
            "reserved_device_bytes": 1200,
            "h2d_bytes": 10,
            "d2h_bytes": 10,
            "collective_bytes": 0,
            "collective_count": 0,
            "energy_joules": 12.0,
            "topology_allocation_count_after_create": 0,
            "energy_scope": "GPU-only",
            "energy_sampling_interval_milliseconds": 50,
            "energy_maximum_gap_seconds": 0.06,
            "energy_sampling_gaps": False,
            "energy_valid": True,
            "shared_or_display_gpu": True,
        },
        "aggregation": {
            "warmup_repeats": 0,
            "measured_repeats": 1,
            "statistic": "median_iqr",
            "median": scvx_total,
            "q1": scvx_total,
            "q3": scvx_total,
            "minimum": scvx_total,
            "maximum": scvx_total,
            "coefficient_of_variation": 0.0,
            "censored_count": 0 if qualified else 1,
            "instance_count": 1,
            "evaluation_seed_count": 1,
            "paired_bootstrap_samples": 0,
        },
        "artifacts": {
            name: {
                "location": f"g4-session://{group.group_id}/{name}",
                "sha256": "c" * 64,
                "immutable_uri": f"g4-session://{group.group_id}/{name}",
                "internal_index_sha256": "c" * 64,
                "portable": True,
            }
            for name in ("manifest", "raw", "stdout", "stderr")
        },
        "g4": {
            "policy_sha256": "9" * 64,
            "runtime_requested": {
                "policy": c["policy"],
                "quality_tier": c["quality_tier"],
                "scaling_mode": c["scaling_mode"],
                "warm_mode": c["warm_mode"],
            },
            "runtime_actual": {
                "policy": c["policy"],
                "quality_tier": c["quality_tier"],
                "scaling_mode": c["scaling_mode"],
                "warm_mode": c["warm_mode"],
            },
            "outer_iterations": [
                {
                    "phase": "refinement",
                    "requested_tolerance": c["quality_tolerance"],
                    "achieved_residual": residual,
                    "forcing_satisfied": forcing,
                    "trust_before": 1.0,
                    "trust_after": 1.0,
                    "trust_action": "retain",
                    "re_solved": False,
                    "cqp_fingerprint": "f" * 16,
                    "resolve_fingerprint": "none",
                    "fingerprint_match": True,
                    "scaling_refreshed": False,
                    "predicted_reduction": 0.1,
                    "actual_reduction": 0.1,
                    "reduction_ratio": 1.0,
                    "scaling_min": 1.0,
                    "scaling_max": 1.0,
                    "warm_mode_actual": c["warm_mode"],
                    "recovery_reason": "not-run",
                    "polish_handoff": hybrid,
                    "accepted": True,
                }
            ],
            "hybrid_permutation": [0, 1] if hybrid else None,
            "hybrid_dual_disposition": "discarded_unsupported" if hybrid else None,
        },
    }
    return {
        **_identity_fields(planned),
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "attempt_id": attempt_id,
        "launched": True,
        "disposition": disposition,
        "failure_class": failure_class,
        "reason": "synthetic qualified attempt" if qualified else f"synthetic {disposition}",
        "timing": {"elapsed_seconds": seconds},
        "case": "g4_attempt",
        "energy": {
            "source": "nvml-c-api",
            "scope": "GPU-only",
            "sampling_interval_milliseconds": 50,
            "sample_count": 40,
            "maximum_gap_seconds": 0.06,
            "gap_valid": True,
            "joules": 12.0,
            "errors": [],
        },
        "session": {
            "pid": 4242,
            "cuda_context_generation": 1,
            "workspace_generation": 1,
            "attempt_ordinal": 2 + repeat,
            "workspace_address": "0x1",
            "topology_fingerprint": "a" * 16,
            "topology_allocations_after_create": 0,
            "topology_index_copies_after_create": 0,
            "allocation_count": 3,
            "total_copy_count": 5,
            "reset_policy": "retain-primal-clear-dual-and-momentum",
            "warm_start_requested": c["warm_mode"],
            "warm_start_actual": c["warm_mode"],
            "dual_disposition": "discarded_unsupported" if hybrid else "not-produced",
            "qoco_workspace_creations": 1 if hybrid else 0,
            "qoco_numeric_updates": 3 if hybrid else 0,
        },
        "paper1_result": paper1,
    }


def _identity_fields(planned: dict[str, Any]) -> dict[str, Any]:
    """The raw-attempt schema is closed; only its identity fields come from the plan."""

    return {
        key: planned[key]
        for key in (
            "group_id",
            "family",
            "intervals",
            "policy",
            "instance",
            "seed",
            "repeat_kind",
            "repeat",
            "statistics_eligible",
        )
    }


def _warmup(group: ExecutionGroup, repeat: int) -> dict[str, Any]:
    return {
        **_identity_fields(group.attempts[repeat]),
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "attempt_id": f"{group.group_id}-warmup-{repeat}",
        "launched": True,
        "disposition": "qualified",
        "failure_class": "none",
        "reason": "synthetic warm-up",
        "timing": {"elapsed_seconds": 1.0},
    }


def _group_result(group: ExecutionGroup, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_kind": "execution_group_result",
        "group_id": group.group_id,
        "attempt_id": "x",
        "command": ["exe", "--g4-session", "m", "p" * 64, "m" * 64, "c" * 64],
        "returncode": 0,
        "raw_attempts": attempts,
    }


def _full_key(
    groups: dict[tuple[str, int, str, int], ExecutionGroup],
    family: str,
    intervals: int,
    policy: str,
    seconds: float,
    **overrides: Any,
) -> list[dict[str, Any]]:
    core = _core()
    measured = []
    for seed in core["evaluation_seeds"]:
        group = groups[(family, intervals, policy, seed)]
        for repeat in range(7):
            measured.append(_measured(group, repeat, seconds * (1.0 + 0.01 * repeat), **overrides))
    return measured


def test_synthetic_measured_attempt_passes_the_frozen_contracts() -> None:
    groups = _groups()
    group = groups[("P1-E-low-thrust", 100, "hybrid-pdhcg-ipm", 59)]
    raw_schema = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text())
    attempts = [_warmup(group, 0), _warmup(group, 1)]
    attempts.extend(_measured(group, repeat, 2.0) for repeat in range(7))
    measured = DECIDER.validate_group_evidence(
        group,
        _group_result(group, attempts),
        Draft202012Validator(raw_schema),
        ROOT / "experiments/schema/paper1_result.schema.json",
    )
    assert len(measured) == 7
    drifted = dict(attempts[4])
    drifted["seed"] = 71
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="field seed differs"):
        DECIDER.validate_group_evidence(
            group,
            _group_result(group, [*attempts[:4], drifted, *attempts[5:]]),
            Draft202012Validator(raw_schema),
            ROOT / "experiments/schema/paper1_result.schema.json",
        )


def test_publication_aggregate_is_strict_and_traces_all_twenty_instances() -> None:
    groups = _groups()
    key = ("P1-E-low-thrust", 100, "fixed-tight")
    measured = _full_key(groups, *key, 4.0)
    coordinate = groups[(*key, 59)].coordinate
    archive = {
        "location": "/tmp/campaign",
        "sha256": "d" * 64,
        "uri": "g4-claim-core://test",
        "index_sha256": "e" * 64,
        "note": "test",
    }
    record = DECIDER.build_publication_aggregate(
        key, measured, source_commit=COMMIT, group_coordinate=coordinate, archive=archive
    )
    identity, aggregation = record["identity"], record["aggregation"]
    assert identity["record_scope"] == "publication_aggregate"
    assert identity["status"] == "qualified"
    assert aggregation["instance_count"] == 20 and aggregation["evaluation_seed_count"] == 20
    assert aggregation["warmup_repeats"] == 2 and aggregation["measured_repeats"] == 7
    assert aggregation["censored_count"] == 0
    assert record["dimensions"]["trust_class"] == 1.0
    assert record["dimensions"]["transfer_class"] == "combined"
    assert abs(aggregation["median"] - 4.0 * 1.03) < 1e-9

    censored = _full_key(groups, *key, 4.0)
    for item in censored[:21]:
        item["disposition"] = "timeout"
        item["failure_class"] = "timeout"
        item["paper1_result"]["identity"]["status"] = "timeout"
        item["paper1_result"]["identity"]["failure_class"] = "timeout"
        item["paper1_result"]["identity"]["failure_reason"] = "synthetic timeout"
        item["paper1_result"]["quality"]["qualified"] = False
        item["paper1_result"]["quality"]["solver_status"] = "max_iterations"
        item["paper1_result"]["quality"]["convergence_criteria_met"] = False
        item["paper1_result"]["quality"]["objective_equivalent"] = False
        item["paper1_result"]["quality"]["matched_quality_state"] = "unqualified"
        item["paper1_result"]["aggregation"]["censored_count"] = 1
    partial = DECIDER.build_publication_aggregate(
        key, censored, source_commit=COMMIT, group_coordinate=coordinate, archive=archive
    )
    assert partial["identity"]["status"] == "timeout"
    assert partial["identity"]["failure_class"] == "timeout"
    assert partial["quality"]["qualified"] is False
    assert partial["aggregation"]["censored_count"] == 21


def _h5_inputs(
    groups: dict[tuple[str, int, str, int], ExecutionGroup],
    *,
    reduction: float,
) -> tuple[dict, dict]:
    core = _core()
    measured_by_key: dict = {}
    coordinates: dict = {}
    for family in core["families"]:
        for intervals in family["scales"]:
            for policy in family["policies"]:
                key = (family["family"], intervals, policy)
                coordinates[key] = groups[(*key, 59)].coordinate
                base = 10.0 * intervals / 100.0
                factor = (1.0 - reduction) if policy == "adaptive" else 1.0
                if policy == "hybrid-pdhcg-ipm":
                    factor = 0.85
                measured_by_key[key] = _full_key(groups, *key, base * factor)
    return measured_by_key, coordinates


def test_h5_rows_pair_qualified_attempts_and_reach_supported_decision() -> None:
    groups = _groups()
    policy = load_policy(ROOT / "benchmarks/g4_policy.json").values
    measured_by_key, coordinates = _h5_inputs(groups, reduction=0.25)
    rows = DECIDER.build_h5_rows(measured_by_key, coordinates, _core())
    assert [(row["family"], row["scale"]) for row in rows] == [
        ("P1-E-low-thrust", 100),
        ("P1-E-low-thrust", 500),
        ("P1-E-low-thrust", 2000),
        ("P1-C-pd3", 20),
        ("P1-C-pd3", 50),
        ("P1-C-pd3", 100),
    ]
    assert all(row["pair_count"] == 140 and "disposition" not in row for row in rows)
    assert all(row["attempts"] == 140 for row in rows)
    h6_rows = DECIDER.build_h6_rows(measured_by_key, coordinates, _core())
    assert [row["scale"] for row in h6_rows] == [100, 500, 2000]
    assert all(
        row["transfer_reliable"] and row["conversion_and_polish_included"] for row in h6_rows
    )
    assert all(row["unpolished_failed_tier"] is False for row in h6_rows)
    decision = DECIDER.g4_decision(rows, h6_rows, policy)
    assert decision["H5"]["decision"] == "supported"
    assert decision["H5"]["coordinates"][0]["low"] > 0
    # Hybrid is 15% faster but not one decade below the unpolished PDHCG residual (equal here).
    assert decision["H6"]["decision"] in {"mixed", "unresolved", "rejected"}


def test_h5_rows_censor_timeouts_and_never_impute() -> None:
    groups = _groups()
    measured_by_key, coordinates = _h5_inputs(groups, reduction=0.25)
    adaptive_key = ("P1-C-pd3", 50, "adaptive")
    for item in measured_by_key[adaptive_key]:
        item["disposition"] = "timeout"
        item["paper1_result"]["identity"]["status"] = "timeout"
    partial_key = ("P1-E-low-thrust", 500, "fixed-tight")
    for item in measured_by_key[partial_key][:10]:
        item["disposition"] = "unqualified"
        item["paper1_result"]["identity"]["status"] = "unqualified"
    rows = {
        (row["family"], row["scale"]): row
        for row in DECIDER.build_h5_rows(measured_by_key, coordinates, _core())
    }
    censored = rows[("P1-C-pd3", 50)]
    assert censored["disposition"] == "timeout"
    assert censored["pair_count"] == 0 and censored["candidate_failures"] == 140
    partial = rows[("P1-E-low-thrust", 500)]
    assert partial["pair_count"] == 130 and partial["baseline_failures"] == 10
    assert len(partial["fixed_tight_seconds"]) == len(partial["adaptive_seconds"]) == 130
    policy = load_policy(ROOT / "benchmarks/g4_policy.json").values
    decision = DECIDER.g4_decision(list(rows.values()), [], policy)
    assert decision["H5"]["censored_coordinates"] == 1
    assert decision["H5"]["decision"] != "supported"
