from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from spacepdhcg.experiments.g4 import (
    ACCEPTED_TIMING_BOUNDARY,
    G4ContractError,
    coverage_count,
    iter_coverage_ledger,
    load_policy,
    qualify_matched_quality,
    runtime_configuration,
    timing_from_components,
    validate_timing_identity,
    verify_artifact_payload,
)

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 64


def _module():
    path = ROOT / "scripts/gpu/run_g4_qualification.py"
    spec = importlib.util.spec_from_file_location("run_g4_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record() -> tuple[dict, dict]:
    loaded = load_policy(ROOT / "benchmarks/g4_policy.json")
    requested = runtime_configuration(
        loaded,
        family="P1-C-pd3",
        policy_name="adaptive",
        quality_tier="tight",
        scaling_mode="refresh_if_needed",
        warm_mode="primal",
    )
    behavior = {
        "policy_sha256": loaded.sha256,
        "requested": {
            key: requested[key]
            for key in (
                "policy",
                "quality_tier",
                "quality_tolerance",
                "scaling_mode",
                "warm_start_mode",
            )
        },
        "actual": {
            **{
                key: requested[key]
                for key in (
                    "policy",
                    "quality_tier",
                    "quality_tolerance",
                    "scaling_mode",
                    "warm_start_mode",
                )
            },
            "solver": requested["solver"],
            "resolve": requested["resolve"],
        },
    }
    path = {
        name: {"independent": True, "violation": 1e-8}
        for name in ("thrust", "mass", "altitude", "glide_slope")
    }
    artifacts = {
        name: {
            "immutable_uri": f"https://artifacts.example.invalid/{name}/sha256/{SHA}",
            "sha256": SHA,
            "internal_index_sha256": SHA,
            "portable": True,
        }
        for name in ("manifest", "raw", "stdout", "stderr")
    }
    record = {
        "family": "P1-C-pd3",
        "quality_tier": "tight",
        "solver_status": "converged",
        "convergence_criteria_met": True,
        "failure_class": "none",
        "quality": {
            "canonical_primal_residual": 1e-7,
            "canonical_dual_residual": 1e-7,
            "canonical_cone_residual": 1e-7,
            "canonical_gap": 1e-7,
            "dynamics_residual": 1e-7,
            "terminal_residual": 1e-7,
            "virtual_control_residual": 1e-7,
            "continuous_time_violation": 1e-7,
            "objective": 10.0,
            "reference_objective": 10.000001,
        },
        "independent_checks": {
            "replay_performed": True,
            "uses_solver_cached_residuals": False,
            "path_inventory_complete": True,
            "path_inventory": path,
        },
        "timing": timing_from_components(
            {
                "coefficient_seconds": 1.0,
                "solve_seconds": 2.0,
                "recovery_seconds": 0.1,
                "replay_seconds": 0.2,
                "acceptance_seconds": 0.3,
            }
        ),
        "runtime_configuration": requested,
        "runtime": behavior,
        "artifacts": artifacts,
    }
    return record, loaded.values


def test_parser_extracts_exactly_one_sample() -> None:
    module = _module()
    records = module.parse_json_lines(
        'noise\n{"case":"g4_iteration","outer":0}\n{"case":"g4_sample","qualified":false}\n'
    )
    assert module.sample_record(records)["qualified"] is False


def test_complete_matched_quality_record_qualifies() -> None:
    record, policy = _record()
    result = qualify_matched_quality(record, policy)
    assert result["qualified"] is True
    assert result["state"] == "matched"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda record: record.update(
                solver_status="max_iterations",
                convergence_criteria_met=False,
                failure_class="max_iterations",
            ),
            "maximum-iteration",
        ),
        (
            lambda record: record["quality"].update(objective=20.0),
            "not practically equivalent",
        ),
        (
            lambda record: record["quality"].update(continuous_time_violation=1e-3),
            "continuous-time violation",
        ),
        (
            lambda record: record.update(independent_checks={}),
            "independent replay",
        ),
        (
            lambda record: record["runtime"]["actual"].update(scaling_mode="reuse"),
            "actual scaling_mode differs",
        ),
        (
            lambda record: record.update(artifacts={}),
            "artifact evidence is missing",
        ),
    ],
)
def test_matched_quality_false_positives_are_rejected(mutation: object, reason: str) -> None:
    record, policy = _record()
    assert callable(mutation)
    mutation(record)  # type: ignore[operator]
    result = qualify_matched_quality(record, policy)
    assert result["qualified"] is False
    assert any(reason in message for message in result["reasons"])


def test_timing_identity_includes_common_boundary_components() -> None:
    timing = timing_from_components(
        {
            "coefficient_seconds": 1.0,
            "recovery_seconds": 2.0,
            "hybrid_conversion_seconds": 3.0,
            "hybrid_setup_seconds": 4.0,
            "polish_seconds": 5.0,
            "replay_seconds": 6.0,
            "acceptance_seconds": 7.0,
            "cuda_startup_seconds": 100.0,
        }
    )
    validate_timing_identity(timing)
    assert timing["cqp_total_seconds"] == 15.0
    assert timing["scvx_total_seconds"] == 28.0
    assert timing["accepted_timing_boundary"] == ACCEPTED_TIMING_BOUNDARY
    assert timing["cuda_startup_included"] is False
    altered = {**timing, "cqp_total_seconds": 14.0}
    with pytest.raises(G4ContractError, match="CQP timing sum identity"):
        validate_timing_identity(altered)


def _small_policy() -> dict:
    policy = json.loads((ROOT / "benchmarks/g4_policy.json").read_text(encoding="utf-8"))
    family = copy.deepcopy(policy["matrix"]["families"]["P1-C-pd3"])
    family["intervals"] = [20]
    family["dispersion_classes"] = [0.0]
    policy["matrix"]["families"] = {"P1-C-pd3": family}
    policy["matrix"]["quality_tiers"] = ["tight"]
    policy["matrix"]["conditioning_log10_spans"] = [0.0]
    policy["matrix"]["scaling_modes"] = ["reuse"]
    policy["matrix"]["warm_start_modes"] = ["cold"]
    policy["matrix"]["warmup_repeats"] = 1
    policy["matrix"]["measured_repeats"] = 1
    policy["matrix"]["randomised_instances_per_coordinate"] = 1
    return policy


def test_coverage_ledger_is_full_and_every_omission_is_explicit() -> None:
    policy = _small_policy()
    rows = list(
        iter_coverage_ledger(
            policy,
            [],
            supported_policies=("fixed-tight",),
        )
    )
    assert len(rows) == coverage_count(policy) == 12
    assert {row["disposition"] for row in rows} == {"unsupported", "unrun"}
    assert all(row["reason"] and "solver_order" in row for row in rows)
    assert all(
        {
            "family",
            "intervals",
            "policy",
            "quality_tier",
            "conditioning",
            "scaling_mode",
            "warm_mode",
            "dispersion_class",
            "seed",
            "instance",
            "repeat_kind",
            "repeat",
        }
        <= row.keys()
        for row in rows
    )


def test_incomplete_or_duplicate_ledger_rows_are_rejected() -> None:
    policy = _small_policy()
    incomplete = {"family": "P1-C-pd3"}
    with pytest.raises(KeyError):
        list(iter_coverage_ledger(policy, [incomplete]))


def test_portable_artifact_payload_and_internal_index_are_verified() -> None:
    payload = b"raw evidence"
    index = b'{"files":["raw evidence"]}'
    artifact = {
        "immutable_uri": "https://artifacts.example.invalid/content/immutable",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "internal_index_sha256": hashlib.sha256(index).hexdigest(),
        "portable": True,
    }
    verify_artifact_payload(artifact, payload, index)
    with pytest.raises(G4ContractError, match="content SHA-256 mismatch"):
        verify_artifact_payload(artifact, b"tampered", index)
