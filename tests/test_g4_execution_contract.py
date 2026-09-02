from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.experiments import (
    G4ContractError,
    claim_core_invocation_count,
    claim_core_may_populate_product,
    evaluate_applicability,
    iter_claim_core_groups,
    load_applicability,
    load_claim_core,
    physical_instance_id,
    solver_rotation_digest,
    validate_attempt_record,
    winner_eligible,
)
from spacepdhcg.experiments.g4 import POLICY_NAMES, coverage_count, load_policy
from spacepdhcg.experiments.g4_scheduler import (
    CampaignStore,
    execution_group_at,
    execution_group_count,
    scheduled_group_ordinal_at,
)
from spacepdhcg.paper1.aggregate import AggregationError, build_products

ROOT = Path(__file__).resolve().parents[1]
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "grouped_g4_runner",
    ROOT / "scripts/gpu/run_g4_campaign.py",
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def _locked(name: str, loader: object):
    digest = (ROOT / f"benchmarks/{name}.sha256").read_text(encoding="utf-8").split()[0]
    assert callable(loader)
    return loader(ROOT / f"benchmarks/{name}.json", expected_sha256=digest)


def _policy():
    return _locked("g4_policy", load_policy)


def _applicability():
    return _locked("g4_applicability", load_applicability)


def _core():
    return _locked("g4_h5_h6_claim_core", load_claim_core)


def _base_coordinate(**updates: object) -> dict[str, object]:
    coordinate: dict[str, object] = {
        "family": "P1-C-pd3",
        "intervals": 100,
        "dispersion_class": 0.05,
        "seed": 59,
        "policy": "fixed-tight",
        "quality_tier": "tight",
        "quality_tolerance": 1e-6,
        "conditioning": 4.0,
        "scaling_mode": "refresh_if_needed",
        "warm_mode": "primal",
        "solver_order": 0,
    }
    coordinate.update(updates)
    return coordinate


def test_applicability_contract_classifies_every_family_policy_and_axis() -> None:
    contract = _applicability().values
    classes = {
        "P1-C-pd3": {"dispersion_class": 0.05},
        "P1-D-pd6": {"attitude_class": 0.2, "rate_class": 0.05},
        "P1-E-low-thrust": {"trust_class": 1.0, "transfer_class": "combined"},
    }
    for family, policy_name in product(classes, POLICY_NAMES):
        coordinate = _base_coordinate(family=family, policy=policy_name)
        for axis in (
            "dispersion_class",
            "attitude_class",
            "rate_class",
            "trust_class",
            "transfer_class",
        ):
            coordinate.pop(axis, None)
        coordinate.update(classes[family])
        decision = evaluate_applicability(
            contract,
            coordinate,
        )
        assert decision.state == "executable"
        assert set(decision.axis_decisions) == {
            "dispersion_class",
            "attitude_class",
            "rate_class",
            "trust_class",
            "transfer_class",
            "quality_tier",
            "scaling_mode",
            "warm_mode",
        }
        assert all(item["reason"] for item in decision.axis_decisions.values())


def test_qoco_primal_dual_discards_dual_without_calling_solver_unsupported() -> None:
    contract = _applicability().values
    pure = evaluate_applicability(
        contract,
        _base_coordinate(policy="pure-gpu-ipm", warm_mode="primal_dual"),
    )
    assert pure.state == "executable"
    assert pure.effective_warm_mode == "primal"
    assert pure.dual_disposition == "discarded_unsupported"
    hybrid = evaluate_applicability(
        contract,
        _base_coordinate(policy="hybrid-pdhcg-ipm", warm_mode="primal_dual"),
    )
    assert hybrid.state == "executable"
    assert hybrid.effective_warm_mode == "primal_dual"
    assert hybrid.dual_disposition == "discarded_unsupported"


def test_complete_physical_axis_space_has_no_instance_id_collisions() -> None:
    policy = _policy().values
    seeds = policy["tuning_evaluation_split"]["evaluation_seeds"]
    identifiers: set[str] = set()
    expected = 0
    for family, values in policy["matrix"]["families"].items():
        if family == "P1-C-pd3":
            classes = (
                {"dispersion_class": dispersion} for dispersion in values["dispersion_classes"]
            )
        elif family == "P1-D-pd6":
            classes = (
                {"attitude_class": attitude, "rate_class": rate}
                for attitude, rate in product(
                    values["attitude_dispersion_radians"],
                    values["angular_rate_dispersion"],
                )
            )
        else:
            classes = (
                {"trust_class": trust, "transfer_class": transfer}
                for trust, transfer in product(
                    values["trust_radii"],
                    values["transfer_classes"],
                )
            )
        frozen_classes = tuple(classes)
        for intervals, class_values, seed in product(values["intervals"], frozen_classes, seeds):
            coordinate = {
                "family": family,
                "intervals": intervals,
                **class_values,
                "seed": seed,
            }
            identifiers.add(physical_instance_id(coordinate))
            expected += 1
    assert expected == 3_200
    assert len(identifiers) == expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("family", "P1-D-pd6"),
        ("intervals", 500),
        ("dispersion_class", 0.1),
        ("seed", 71),
        ("quality_tier", "ipm"),
        ("conditioning", 8.0),
        ("scaling_mode", "reuse"),
        ("warm_mode", "cold"),
    ],
)
def test_solver_rotation_digest_changes_for_every_coordinate_axis(
    field: str,
    replacement: object,
) -> None:
    original = _base_coordinate()
    changed = copy.deepcopy(original)
    changed[field] = replacement
    if field == "family":
        changed.pop("dispersion_class")
        changed.update(attitude_class=0.2, rate_class=0.05)
    assert solver_rotation_digest(20260901, original) != solver_rotation_digest(
        20260901,
        changed,
    )


def test_persistent_group_count_order_and_warmup_exclusion() -> None:
    policy = _policy().values
    assert coverage_count(policy) == 24_883_200
    assert execution_group_count(policy) == 2_764_800
    group = execution_group_at(policy, 0)
    assert len(group.attempts) == 9
    assert len({attempt["group_id"] for attempt in group.attempts}) == 1
    assert len({attempt["instance"] for attempt in group.attempts}) == 1
    assert [(item["repeat_kind"], item["repeat"]) for item in group.attempts] == [
        ("warmup", 0),
        ("warmup", 1),
        *[("measured", index) for index in range(7)],
    ]
    assert [item["statistics_eligible"] for item in group.attempts] == [
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    scheduled = [
        execution_group_at(policy, scheduled_group_ordinal_at(policy, index))
        for index in range(len(POLICY_NAMES))
    ]
    assert {group.coordinate["policy"] for group in scheduled} == set(POLICY_NAMES)
    assert [group.coordinate["solver_order"] for group in scheduled] == list(
        range(len(POLICY_NAMES))
    )


def test_policy_groups_share_paired_physical_instance_but_not_group_identity() -> None:
    first = _base_coordinate(policy="fixed-tight")
    second = _base_coordinate(policy="adaptive")
    from spacepdhcg.experiments import make_execution_group

    fixed = make_execution_group(first)
    adaptive = make_execution_group(second)
    assert fixed.physical_instance_id == adaptive.physical_instance_id
    assert fixed.group_id != adaptive.group_id


def _attempt(disposition: str, *, launched: bool = True, repeat_kind: str = "measured"):
    return {
        **_base_coordinate(policy="hybrid-pdhcg-ipm"),
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "group_id": "g4-group-v1-" + "a" * 64,
        "attempt_id": "attempt-1",
        "instance": "g4-instance-v2-" + "b" * 64,
        "repeat_kind": repeat_kind,
        "repeat": 0,
        "launched": launched,
        "statistics_eligible": repeat_kind == "measured",
        "disposition": disposition,
        "failure_class": disposition,
        "reason": f"authoritative {disposition} reason",
        "timing": {"elapsed_seconds": 1.0 if launched else 0.0},
    }


@pytest.mark.parametrize(
    "disposition",
    ["hybrid_handoff_ineligible", "not_applicable", "unsupported"],
)
def test_exact_terminal_dispositions_require_reason_timing_and_never_win(
    disposition: str,
) -> None:
    record = _attempt(disposition, launched=disposition != "not_applicable")
    validate_attempt_record(record)
    assert winner_eligible(record) is False
    missing_reason = dict(record, reason="")
    with pytest.raises(G4ContractError, match="reason"):
        validate_attempt_record(missing_reason)
    missing_timing = dict(record)
    missing_timing.pop("timing")
    with pytest.raises(G4ContractError, match="timing"):
        validate_attempt_record(missing_timing)


@pytest.mark.parametrize("disposition", ["timeout", "oom"])
def test_predictive_timeout_and_oom_are_forbidden(disposition: str) -> None:
    with pytest.raises(G4ContractError, match="actual launch"):
        validate_attempt_record(_attempt(disposition, launched=False))


def test_larger_groups_remain_pending_until_claimed(tmp_path: Path) -> None:
    policy = _policy()
    with CampaignStore(
        tmp_path,
        policy.values,
        policy.sha256,
        "a" * 40,
        grouped=True,
    ) as store:
        before = store.status()
        assert before["remaining"] == 2_764_800
        claim = store.claim()
        assert claim is not None
        after = store.status()
        assert after["running"] == 1
        assert after["completed"] == 0
        assert after["remaining"] == before["remaining"]
        rows = store.database.execute("SELECT COUNT(*) AS count FROM coordinates").fetchone()
        assert rows["count"] == 1


def test_claim_core_hash_count_pairing_and_product_prohibition() -> None:
    loaded = _core()
    definition = loaded.values
    assert loaded.sha256 == "40dc217467ffe32e919d4f901943e0200f69e302cf57cd15ccdfa88bfa0c8d0b"
    assert claim_core_invocation_count(definition) == 3_240
    groups = list(iter_claim_core_groups(definition))
    assert len(groups) == 360
    assert sum(len(group.attempts) for group in groups) == 3_240
    paired = {
        (
            group.coordinate["family"],
            group.coordinate["intervals"],
            group.coordinate["seed"],
        ): set()
        for group in groups
    }
    for group in groups:
        paired[
            (
                group.coordinate["family"],
                group.coordinate["intervals"],
                group.coordinate["seed"],
            )
        ].add(group.physical_instance_id)
    assert all(len(identifiers) == 1 for identifiers in paired.values())
    assert all(
        claim_core_may_populate_product(definition, product_id) is False
        for product_id in ("F04", "F10", "T05", "T07")
    )


def test_claim_core_cannot_enter_full_product_builder(tmp_path: Path) -> None:
    run = SimpleNamespace(
        run_id="claim-core-run",
        coordinate=("P1-C-pd3",),
        manifest=SimpleNamespace(experiment={"campaign_kind": "h5_h6_claim_resolution_core"}),
    )
    decisions = {f"H{index}": {} for index in range(1, 7)}
    with pytest.raises(AggregationError, match="cannot populate"):
        build_products([run], tmp_path, decisions=decisions)


@pytest.mark.parametrize(
    "schema_name",
    [
        "g4_raw_attempt.schema.json",
        "g4_execution_group.schema.json",
        "g4_h5_h6_claim_core.schema.json",
        "paper1_result.schema.json",
    ],
)
def test_new_and_extended_json_schemas_are_valid(schema_name: str) -> None:
    schema = json.loads((ROOT / "experiments/schema" / schema_name).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_claim_core_definition_matches_declarative_schema() -> None:
    schema = json.loads(
        (ROOT / "experiments/schema/g4_h5_h6_claim_core.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(_core().values)


def test_full_runner_requires_persistent_group_capability(tmp_path: Path) -> None:
    executable = tmp_path / "executor"
    executable.write_bytes(b"executor")
    capability_path = tmp_path / "capability.json"
    value = {
        "schema_version": 1,
        "source_commit": "a" * 40,
        "policy_sha256": "b" * 64,
        "matrix_sha256": "c" * 64,
        "executable_sha256": RUNNER.sha256_path(executable),
        "axes": {
            name: {
                "status": ("execution_only" if name in {"repeat", "solver_order"} else "applied")
            }
            for name in RUNNER.CAPABILITY_AXES
        },
        "timing_boundary": RUNNER.ACCEPTED_TIMING_BOUNDARY,
        "independent_replay": True,
    }
    value["capability_sha256"] = hashlib.sha256(RUNNER.canonical_bytes(value)).hexdigest()
    capability_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(G4ContractError, match="persistent nine-attempt"):
        RUNNER.load_capabilities(
            capability_path,
            executable,
            "b" * 64,
            "c" * 64,
            "a" * 40,
            require_persistent_group=True,
        )
    value.pop("capability_sha256")
    value["execution_contract"] = {
        "version": "g4-persistent-group-v1",
        "one_process_per_group": True,
        "persistent_session": True,
        "persistent_workspace": True,
        "separate_attempt_records": True,
        "policy_reset_between_attempts": True,
    }
    value["contract_hashes"] = {
        "applicability": RUNNER.sha256_path(ROOT / "benchmarks/g4_applicability.json"),
        "claim_core": RUNNER.sha256_path(ROOT / "benchmarks/g4_h5_h6_claim_core.json"),
        "execution_group_schema": RUNNER.sha256_path(
            ROOT / "experiments/schema/g4_execution_group.schema.json"
        ),
        "raw_attempt_schema": RUNNER.sha256_path(
            ROOT / "experiments/schema/g4_raw_attempt.schema.json"
        ),
        "paper1_result_schema": RUNNER.sha256_path(
            ROOT / "experiments/schema/paper1_result.schema.json"
        ),
    }
    value["session_probe"] = {
        "kind": "real_cuda_session",
        "attempt_count": 9,
        "warmup_count": 2,
        "measured_count": 7,
        "same_process": True,
        "same_context": True,
        "same_workspace": True,
        "zero_post_create_topology_allocations": True,
        "zero_post_create_topology_index_copies": True,
    }
    value["capability_sha256"] = hashlib.sha256(RUNNER.canonical_bytes(value)).hexdigest()
    capability_path.write_text(json.dumps(value), encoding="utf-8")
    loaded = RUNNER.load_capabilities(
        capability_path,
        executable,
        "b" * 64,
        "c" * 64,
        "a" * 40,
        require_persistent_group=True,
    )
    assert loaded["execution_contract"]["persistent_workspace"] is True
