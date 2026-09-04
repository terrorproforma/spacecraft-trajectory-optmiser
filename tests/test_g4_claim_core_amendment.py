"""Amendment single-gpu-v1.1: hash lock, sensitivity selection, schedule, replay, contamination.

Every rule the amendment introduces is checked here against the frozen JSON, the reference
implementation in ``g4_execution_contract`` and the scheduler/decision code that consumes it.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.campaign_scope import SCOPE_DEFINITIONS
from spacepdhcg.experiments.g4 import DISPOSITIONS, G4ContractError, load_policy
from spacepdhcg.experiments.g4_execution_contract import (
    AMENDMENT_ID,
    AMENDMENT_RECORD_FIELD,
    CLAIM_CORE_STRATUM,
    ORIGINAL_CENSORING,
    REPLAY_DISPOSITION,
    SENSITIVITY_STRATUM,
    amended_claim_core_groups,
    amended_schedule_sha256,
    censoring_sensitivity_group_ids,
    deterministic_replay_eligible,
    deterministic_trace_hash,
    deterministic_trace_string,
    fnv1a64,
    group_censoring,
    group_censoring_stratum,
    iter_claim_core_groups,
    load_claim_core,
    load_claim_core_amendment,
    sensitivity_group_for,
    validate_attempt_record,
    validate_claim_core_amendment,
)

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = ROOT / "benchmarks/g4_claim_core_amendment_v1_1.json"


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _module("amendment_runner", "scripts/gpu/run_g4_campaign.py")
DECIDER = _module("amendment_decider", "scripts/gpu/decide_g4_claim_core.py")
DECISION_TESTS = _module("amendment_decision_helpers", "tests/test_g4_claim_core_decision.py")


def _locked(name: str, loader):
    digest = (ROOT / f"benchmarks/{name}.sha256").read_text(encoding="utf-8").split()[0]
    return loader(ROOT / f"benchmarks/{name}.json", expected_sha256=digest)


def _core():
    return _locked("g4_h5_h6_claim_core", load_claim_core)


def _policy():
    return _locked("g4_policy", load_policy)


def _amendment():
    core, policy = _core(), _policy()
    lock = AMENDMENT_PATH.with_suffix(".sha256").read_text(encoding="utf-8").split()
    assert lock[1] == AMENDMENT_PATH.name
    return load_claim_core_amendment(
        AMENDMENT_PATH,
        core.values,
        claim_core_sha256=core.sha256,
        policy_sha256=policy.sha256,
        expected_sha256=lock[0],
    )


# ------------------------------------------------------------------------------------------
# Amendment document: hash lock, schema, registry references
# ------------------------------------------------------------------------------------------


def test_amendment_hash_lock_schema_and_registry_agree() -> None:
    amendment = _amendment()
    assert amendment.sha256 == hashlib.sha256(AMENDMENT_PATH.read_bytes()).hexdigest()
    schema = json.loads(
        (ROOT / "experiments/schema/g4_claim_core_amendment.schema.json").read_text("utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(amendment.values)
    assert amendment.values["amendment_id"] == AMENDMENT_ID == "single-gpu-v1.1"
    assert amendment.values["record_field"] == {
        "name": AMENDMENT_RECORD_FIELD,
        "value": AMENDMENT_ID,
    }
    assert amendment.values["censoring"]["original"] == ORIGINAL_CENSORING
    assert amendment.values["censoring"]["claim_core"] == {
        "attempt_deadline_seconds": 120,
        "inner_iteration_cap": 200_000,
    }
    # Registry: the single-gpu-v1 scope and its JSON mirror pin the amendment by hash.
    registered = SCOPE_DEFINITIONS["single-gpu-v1"]["amendments"]
    # v1.2 supersedes v1.1 (tests/test_g4_claim_core_amendment_v1_2.py); v1.1 stays registered.
    assert [item["amendment_id"] for item in registered] == [AMENDMENT_ID, "single-gpu-v1.2"]
    assert ROOT / registered[0]["path"] == AMENDMENT_PATH
    lock = (ROOT / registered[0]["lock"]).read_text("utf-8").split()
    assert lock == [amendment.sha256, AMENDMENT_PATH.name]
    assert (ROOT / registered[0]["document"]).is_file()
    mirror = json.loads((ROOT / "benchmarks/campaign_scopes/single-gpu-v1.json").read_text("utf-8"))
    assert mirror["amendments"] == registered
    contract = (ROOT / "docs/G4_EXECUTION_CONTRACT.md").read_text("utf-8")
    assert "Amendment `single-gpu-v1.1`" in contract
    assert "benchmarks/g4_claim_core_amendment_v1_1.sha256" in contract
    assert REPLAY_DISPOSITION in DISPOSITIONS


def test_amendment_validator_rejects_tampering() -> None:
    core, policy = _core(), _policy()
    amendment = _amendment()
    with pytest.raises(G4ContractError, match="hash drift"):
        load_claim_core_amendment(
            AMENDMENT_PATH,
            core.values,
            claim_core_sha256=core.sha256,
            policy_sha256=policy.sha256,
            expected_sha256="0" * 64,
        )
    tampered = copy.deepcopy(amendment.values)
    tampered["censoring"]["claim_core"]["attempt_deadline_seconds"] = 60
    with pytest.raises(G4ContractError, match="censoring"):
        validate_claim_core_amendment(
            tampered, core.values, claim_core_sha256=core.sha256, policy_sha256=policy.sha256
        )
    tampered = copy.deepcopy(amendment.values)
    tampered["censoring"]["original"]["attempt_deadline_seconds"] = 300
    with pytest.raises(G4ContractError, match="original"):
        validate_claim_core_amendment(
            tampered, core.values, claim_core_sha256=core.sha256, policy_sha256=policy.sha256
        )
    tampered = copy.deepcopy(amendment.values)
    stratum = tampered["censoring"]["sensitivity_stratum"]["group_ids"]
    first = next(iter(stratum))
    stratum[first] = list(reversed(stratum[first]))
    with pytest.raises(G4ContractError):
        validate_claim_core_amendment(
            tampered, core.values, claim_core_sha256=core.sha256, policy_sha256=policy.sha256
        )
    with pytest.raises(G4ContractError):
        validate_claim_core_amendment(
            amendment.values,
            core.values,
            claim_core_sha256="1" * 64,
            policy_sha256=policy.sha256,
        )


# ------------------------------------------------------------------------------------------
# Sensitivity stratum selection and schedule
# ------------------------------------------------------------------------------------------


def test_sensitivity_stratum_is_deterministic_and_stratified() -> None:
    core, amendment = _core(), _amendment()
    selected = censoring_sensitivity_group_ids(core.values, amendment.values)
    assert selected == amendment.values["censoring"]["sensitivity_stratum"]["group_ids"]
    assert len(selected) == 18 and all(len(ids) == 2 for ids in selected.values())
    assert sum(len(ids) for ids in selected.values()) == 36
    by_id = {group.group_id: group for group in iter_claim_core_groups(core.values)}
    for key, ids in selected.items():
        family, intervals, policy = key.split("/")
        for group_id in ids:
            coordinate = by_id[group_id].coordinate
            assert (coordinate["family"], str(coordinate["intervals"]), coordinate["policy"]) == (
                family,
                intervals,
                policy,
            )
    # Same inputs, same choice; different seed, different choice.
    assert censoring_sensitivity_group_ids(core.values, amendment.values) == selected
    other = copy.deepcopy(amendment.values)
    other["censoring"]["sensitivity_stratum"]["selection_seed"] = "another-seed"
    assert censoring_sensitivity_group_ids(core.values, other) != selected


def test_amended_schedule_orders_converging_policies_first_and_twins_after_cores() -> None:
    core, amendment = _core(), _amendment()
    groups = amended_claim_core_groups(core.values, amendment.values)
    assert len(groups) == 396 == amendment.values["schedule"]["group_count"]
    assert amended_schedule_sha256(groups) == amendment.values["schedule"]["schedule_sha256"]
    priority = amendment.values["schedule"]["policy_priority"]
    assert priority == ["pure-gpu-ipm", "adaptive", "hybrid-pdhcg-ipm", "fixed-tight"]
    positions = [priority.index(group.coordinate["policy"]) for group in groups]
    assert positions == sorted(positions)
    strata = Counter(group_censoring_stratum(group) for group in groups)
    assert strata == {CLAIM_CORE_STRATUM: 360, SENSITIVITY_STRATUM: 36}
    selected = {
        group_id
        for ids in amendment.values["censoring"]["sensitivity_stratum"]["group_ids"].values()
        for group_id in ids
    }
    for index, group in enumerate(groups):
        if group_censoring_stratum(group) == SENSITIVITY_STRATUM:
            previous = groups[index - 1]
            assert previous.group_id in selected
            assert sensitivity_group_for(previous).group_id == group.group_id
            assert {k: v for k, v in group.coordinate.items() if k != "censoring_stratum"} == (
                previous.coordinate
            )
            assert group.physical_instance_id == previous.physical_instance_id
            assert group_censoring(group, amendment.values) == {
                "attempt_deadline_seconds": 600,
                "inner_iteration_cap": 1_000_000,
                "group_deadline_seconds": 5460,
            }
        else:
            assert group_censoring(group, amendment.values) == {
                "attempt_deadline_seconds": 120,
                "inner_iteration_cap": 200_000,
                "group_deadline_seconds": 1140,
            }
    # Identities, solver_order values and repeat structure are those of the claim core.
    core_ids = [group.group_id for group in iter_claim_core_groups(core.values)]
    assert sorted(core_ids) == sorted(
        group.group_id for group in groups if group_censoring_stratum(group) == CLAIM_CORE_STRATUM
    )
    assert all(len(group.attempts) == 9 for group in groups)


# ------------------------------------------------------------------------------------------
# Deterministic replay: trace hash equality and difference fallback
# ------------------------------------------------------------------------------------------


def _trace(**updates: Any) -> dict[str, Any]:
    trace = {
        "inner_iterations": 200_000,
        "outer_iterations": 3,
        "canonical_residual": 1.2345678901234567e-3,
        "dynamics_residual": 4.5e-5,
        "path_residual": 0.0,
        "terminal_residual": 7.0e-6,
        "virtual_control_residual": 1.5e-4,
        "checkpoints": [
            ["progress", 1e-6, 2e-3, False, False],
            ["repair", 1e-6, 1.5e-3, False, True],
            ["progress", 1e-6, 1.2345678901234567e-3, False, False],
        ],
    }
    trace.update(updates)
    return trace


def _timeout(kind: str, repeat: int, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "repeat_kind": kind,
        "repeat": repeat,
        "launched": True,
        "disposition": "timeout",
        "trace": trace,
        "trace_hash": deterministic_trace_hash("timeout", trace),
    }


def test_fnv1a64_matches_reference_vectors() -> None:
    assert fnv1a64(b"") == 0xCBF29CE484222325
    assert fnv1a64(b"a") == 0xAF63DC4C8601EC8C
    assert fnv1a64(b"foobar") == 0x85944171F73967E8


def test_trace_hash_matches_the_native_executor_selftest_vector() -> None:
    """Vector emitted by ``device_scvx_integration_test --g4-amendment-selftest``."""

    trace = {
        "inner_iterations": 123456,
        "outer_iterations": 2,
        "canonical_residual": 1e-6,
        "dynamics_residual": 2.5e-7,
        "path_residual": 0.0,
        "terminal_residual": 3e-9,
        "virtual_control_residual": 0.1,
        "checkpoints": [["repair", 0.01, 0.005, True, False], ["polish", 1e-8, 7e-4, False, True]],
    }
    assert deterministic_trace_string("timeout", trace) == (
        "timeout|123456|2|9.9999999999999995e-07|2.4999999999999999e-07|0|3e-09|"
        "0.10000000000000001|repair:0.01:0.0050000000000000001:1:0;"
        "polish:1e-08:0.00069999999999999999:0:1"
    )
    assert deterministic_trace_hash("timeout", trace) == "decc750e9b3e902e"


def test_trace_string_is_canonical_and_hash_detects_any_difference() -> None:
    trace = _trace()
    text = deterministic_trace_string("timeout", trace)
    assert text.startswith("timeout|200000|3|0.0012345678901234567|4.5000000000000003e-05|0|")
    assert text.endswith("progress:9.9999999999999995e-07:0.0012345678901234567:0:0")
    assert len(deterministic_trace_hash("timeout", trace)) == 16
    assert deterministic_trace_hash("timeout", trace) == deterministic_trace_hash(
        "timeout", copy.deepcopy(trace)
    )
    for change in (
        {"inner_iterations": 199_999},
        {"outer_iterations": 4},
        {"canonical_residual": 1.2345678901235e-3},  # one ulp-scale change is a new trace
        {
            "checkpoints": [
                *trace["checkpoints"][:-1],
                ["progress", 1e-6, 1.2345678901234567e-3, True, False],
            ]
        },
        {
            "checkpoints": [
                *trace["checkpoints"][:-1],
                ["polish", 1e-6, 1.2345678901234567e-3, False, False],
            ]
        },
    ):
        assert deterministic_trace_hash("timeout", _trace(**change)) != deterministic_trace_hash(
            "timeout", trace
        )
    assert deterministic_trace_hash("qualified", trace) != deterministic_trace_hash(
        "timeout", trace
    )
    with pytest.raises(G4ContractError, match="finite"):
        deterministic_trace_hash("timeout", _trace(path_residual=float("nan")))


def test_replay_eligibility_requires_three_identical_timeouts() -> None:
    trace = _trace()
    eligible = [
        _timeout("warmup", 0, trace),
        _timeout("warmup", 1, trace),
        _timeout("measured", 0, trace),
    ]
    assert deterministic_replay_eligible(eligible) is True
    # Fallback: any trace difference executes all seven measured attempts.
    differing = copy.deepcopy(eligible)
    differing[2] = _timeout("measured", 0, _trace(inner_iterations=199_998))
    assert deterministic_replay_eligible(differing) is False
    # A non-timeout, an unlaunched attempt or the wrong repeat set is never eligible.
    qualified = copy.deepcopy(eligible)
    qualified[1]["disposition"] = "qualified"
    assert deterministic_replay_eligible(qualified) is False
    unlaunched = copy.deepcopy(eligible)
    unlaunched[0]["launched"] = False
    assert deterministic_replay_eligible(unlaunched) is False
    assert deterministic_replay_eligible(eligible[:2]) is False
    wrong_order = [eligible[2], eligible[0], eligible[1]]
    assert deterministic_replay_eligible(wrong_order) is False
    # A record whose hash does not match its own trace is a contract violation, not a replay.
    forged = copy.deepcopy(eligible)
    forged[2]["trace_hash"] = "0" * 16
    with pytest.raises(G4ContractError, match="trace_hash"):
        deterministic_replay_eligible(forged)


def test_replay_records_are_validated_by_the_raw_attempt_contract() -> None:
    core = _core()
    group = next(iter_claim_core_groups(core.values))
    planned = group.attempts[2 + 3]  # measured/3
    trace = _trace()
    record = {
        **DECISION_TESTS._identity_fields(planned),
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "attempt_id": f"{group.group_id}/measured-3",
        "launched": False,
        "disposition": REPLAY_DISPOSITION,
        "failure_class": "timeout",
        "reason": "deterministic replay of measured/0",
        "replay_source_attempt_id": f"{group.group_id}/measured-0",
        AMENDMENT_RECORD_FIELD: AMENDMENT_ID,
        "trace": trace,
        "trace_hash": deterministic_trace_hash("timeout", trace),
        "timing": {"elapsed_seconds": 0.0},
    }
    validate_attempt_record(record)
    schema = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text("utf-8"))
    validator = Draft202012Validator(schema)
    validator.validate(
        {
            **record,
            "amendment": {
                "censoring_stratum": CLAIM_CORE_STRATUM,
                "attempt_deadline_seconds": 120,
                "inner_iteration_cap": 200_000,
                "deterministic_replay": True,
            },
        }
    )
    launched = {**record, "launched": True}
    with pytest.raises(G4ContractError, match="without a launch"):
        validate_attempt_record(launched)
    without_amendment = dict(record)
    del without_amendment[AMENDMENT_RECORD_FIELD]
    with pytest.raises(G4ContractError, match="amendment"):
        validate_attempt_record(without_amendment)
    first = {**record, "repeat": 0, "attempt_id": f"{group.group_id}/measured-0"}
    with pytest.raises(G4ContractError, match="measured/0"):
        validate_attempt_record(first)


# ------------------------------------------------------------------------------------------
# Decision A: run-and-flag contamination in the scheduler
# ------------------------------------------------------------------------------------------


def _timeline_attempts() -> tuple[list[dict[str, Any]], list[tuple[float, str]]]:
    attempts = []
    timeline = [(90.0, "starting"), (100.0, json.dumps({"case": "g4_session_ready"}))]
    at = 100.0
    for kind, repeat in (("warmup", 0), ("warmup", 1), *(("measured", r) for r in range(7))):
        at += 10.0
        record = {
            "case": "g4_attempt",
            "repeat_kind": kind,
            "repeat": repeat,
            "launched": True,
            "disposition": "qualified",
        }
        attempts.append(dict(record))
        timeline.append((at, json.dumps(record)))
    # A replayed attempt was never launched and therefore can never be contaminated.
    attempts[-1]["launched"] = False
    attempts[-1]["disposition"] = REPLAY_DISPOSITION
    return attempts, timeline


def test_contamination_is_flagged_per_attempt_and_never_reruns() -> None:
    attempts, timeline = _timeline_attempts()
    windows = RUNNER.attempt_windows(timeline, 80.0)
    assert windows[("warmup", 0)] == (100.0, 110.0)  # session-ready resets the window start
    assert windows[("measured", 6)] == (180.0, 190.0)
    # Foreign compute observed once at t=145 (inside measured/2's window: 140..150).
    foreign = [{"monotonic": 145.0, "processes": ["foreign.exe"], "max_sm_percent": 37}]
    sample_times = [105.0 + 5.0 * i for i in range(16)]
    flagged = RUNNER.flag_contaminated_attempts(
        attempts, windows, foreign, sample_times, slack_seconds=0.0
    )
    assert flagged == 1
    contaminated = [(a["repeat_kind"], a["repeat"]) for a in attempts if a["contaminated"]]
    assert contaminated == [("measured", 2)]
    assert all("contaminated" in a for a in attempts)
    detail = attempts[4]["contamination"]
    assert detail["foreign_samples"] == 1 and detail["total_samples"] == 3
    assert detail["max_foreign_sm_percent"] == 37
    assert detail["foreign_processes"] == ["foreign.exe"]
    assert (detail["window_start_monotonic"], detail["window_end_monotonic"]) == (140.0, 150.0)
    # Dispositions are retained: run-and-flag never rewrites or re-runs the attempt.
    assert attempts[4]["disposition"] == "qualified"
    # Slack widens the window: with 6 s slack the neighbours also overlap the sample.
    attempts2, _ = _timeline_attempts()
    flagged2 = RUNNER.flag_contaminated_attempts(
        attempts2, windows, foreign, sample_times, slack_seconds=6.0
    )
    assert flagged2 == 3
    # A late sample overlapping the replayed attempt's window flags nothing: it was not launched.
    attempts3, _ = _timeline_attempts()
    late = [{"monotonic": 185.0, "processes": ["foreign.exe"], "max_sm_percent": 90}]
    assert (
        RUNNER.flag_contaminated_attempts(attempts3, windows, late, sample_times, slack_seconds=0.0)
        == 0
    )
    assert (
        attempts3[-1]["contaminated"] is False
        and attempts3[-1]["contamination"]["total_samples"] == 0
    )
    # No foreign samples: nothing flagged, and the dispositions are untouched.
    attempts4, _ = _timeline_attempts()
    assert (
        RUNNER.flag_contaminated_attempts(attempts4, windows, [], sample_times, slack_seconds=0.0)
        == 0
    )
    assert [a["disposition"] for a in attempts4][:8] == ["qualified"] * 8


def test_sensitivity_and_replay_env_and_deadlines_follow_the_stratum() -> None:
    core, amendment = _core(), _amendment()
    groups = amended_claim_core_groups(core.values, amendment.values)
    twin = next(g for g in groups if group_censoring_stratum(g) == SENSITIVITY_STRATUM)
    core_group = next(g for g in groups if group_censoring_stratum(g) == CLAIM_CORE_STRATUM)
    assert group_censoring(twin, amendment.values)["attempt_deadline_seconds"] == 600
    assert group_censoring(core_group, amendment.values)["attempt_deadline_seconds"] == 120
    assert RUNNER.SHARED_GPU_LOCK_FILE == Path("/home/angus/.spacepdhcg-gpu.lock")
    assert amendment.values["contamination"]["detection"]["lock_file"] == str(
        RUNNER.SHARED_GPU_LOCK_FILE
    )


# ------------------------------------------------------------------------------------------
# Decision code: contamination exclusion with honest n, replay censoring, acceptance rule
# ------------------------------------------------------------------------------------------


def _mark(item: dict[str, Any], disposition: str) -> None:
    item["disposition"] = disposition
    item["paper1_result"]["identity"]["status"] = disposition
    if disposition != "qualified":
        item["failure_class"] = "timeout"
        item["paper1_result"]["identity"]["failure_class"] = "timeout"
        item["paper1_result"]["identity"]["failure_reason"] = f"synthetic {disposition}"
        item["paper1_result"]["quality"]["qualified"] = False
        item["paper1_result"]["quality"]["convergence_criteria_met"] = False
        item["paper1_result"]["quality"]["objective_equivalent"] = False
        item["paper1_result"]["quality"]["matched_quality_state"] = "unqualified"
    if disposition == REPLAY_DISPOSITION:
        item["launched"] = False
        item["replay_source_attempt_id"] = item["attempt_id"].rsplit("-", 1)[0] + "-0"


def test_contaminated_pairs_leave_timing_but_stay_in_disposition_counts() -> None:
    groups = DECISION_TESTS._groups()
    measured_by_key, coordinates = DECISION_TESTS._h5_inputs(groups, reduction=0.25)
    key = ("P1-C-pd3", 50, "adaptive")
    for item in measured_by_key[key][:12]:
        item["contaminated"] = True
    rows = {
        (row["family"], row["scale"]): row
        for row in DECIDER.build_h5_rows(measured_by_key, coordinates, DECISION_TESTS._core())
    }
    row = rows[("P1-C-pd3", 50)]
    assert row["pair_count"] == 128 and len(row["adaptive_seconds"]) == 128
    assert row["candidate_failures"] == 0  # contaminated attempts are not failures
    assert row["adaptive_dispositions"] == {"qualified": 140}
    assert row["contamination"] == {
        "baseline_contaminated": 0,
        "candidate_contaminated": 12,
        "contaminated_pairs_excluded": 12,
        "baseline_replayed": 0,
        "candidate_replayed": 0,
    }
    assert "disposition" not in row
    # Fully contaminated coordinate: no pair, disposition "contaminated", never imputed.
    for item in measured_by_key[("P1-C-pd3", 20, "fixed-tight")]:
        item["contaminated"] = True
    rows = {
        (row["family"], row["scale"]): row
        for row in DECIDER.build_h5_rows(measured_by_key, coordinates, DECISION_TESTS._core())
    }
    assert rows[("P1-C-pd3", 20)]["pair_count"] == 0
    assert rows[("P1-C-pd3", 20)]["disposition"] == "contaminated"
    policy = _policy().values
    decision = DECIDER.g4_decision(list(rows.values()), [], policy)
    contaminated_row = next(
        item
        for item in decision["H5"]["coordinates"]
        if item.get("terminal_disposition") == "contaminated"
    )
    assert contaminated_row["eligible"] is False and contaminated_row["censored"] is False


def test_replayed_timeouts_are_censoring_in_rows_and_aggregates() -> None:
    groups = DECISION_TESTS._groups()
    measured_by_key, coordinates = DECISION_TESTS._h5_inputs(groups, reduction=0.25)
    key = ("P1-C-pd3", 100, "adaptive")
    for item in measured_by_key[key]:
        _mark(item, "timeout" if item["repeat"] == 0 else REPLAY_DISPOSITION)
    rows = {
        (row["family"], row["scale"]): row
        for row in DECIDER.build_h5_rows(measured_by_key, coordinates, DECISION_TESTS._core())
    }
    row = rows[("P1-C-pd3", 100)]
    assert row["disposition"] == "timeout"
    assert row["adaptive_dispositions"] == {"timeout": 20, REPLAY_DISPOSITION: 120}
    assert row["contamination"]["candidate_replayed"] == 120
    policy = _policy().values
    decision = DECIDER.g4_decision(list(rows.values()), [], policy)
    assert decision["H5"]["censored_coordinates"] == 1
    archive = {
        "location": "/tmp/campaign",
        "sha256": "d" * 64,
        "uri": "g4-claim-core://test",
        "index_sha256": "e" * 64,
        "note": "test",
    }
    aggregate = DECIDER.build_publication_aggregate(
        key,
        measured_by_key[key],
        source_commit=DECISION_TESTS.COMMIT,
        group_coordinate=coordinates[key],
        archive=archive,
        amendment=_amendment(),
    )
    assert aggregate["identity"]["status"] == REPLAY_DISPOSITION
    assert aggregate["identity"]["failure_class"] == "timeout"
    assert aggregate["aggregation"]["censored_count"] == 140
    assert aggregate["aggregation"]["median"] is None
    assert any("policy_amendment: single-gpu-v1.1" in note for note in aggregate["notes"])
    assert any("timeout_deterministic_replay=120" in note for note in aggregate["notes"])


def test_contaminated_attempts_are_excluded_from_aggregate_timing_with_visible_n() -> None:
    groups = DECISION_TESTS._groups()
    key = ("P1-E-low-thrust", 100, "fixed-tight")
    measured = DECISION_TESTS._full_key(groups, *key, 4.0)
    for item in measured:
        if item["repeat"] == 6:  # the slowest repeat of every instance is contaminated
            item["contaminated"] = True
    archive = {
        "location": "/tmp/campaign",
        "sha256": "d" * 64,
        "uri": "g4-claim-core://test",
        "index_sha256": "e" * 64,
        "note": "test",
    }
    record = DECIDER.build_publication_aggregate(
        key,
        measured,
        source_commit=DECISION_TESTS.COMMIT,
        group_coordinate=groups[(*key, 59)].coordinate,
        archive=archive,
    )
    assert record["identity"]["status"] == "qualified"
    assert record["aggregation"]["censored_count"] == 0
    assert abs(record["aggregation"]["maximum"] - 4.0 * 1.05) < 1e-9  # repeat 6 (x1.06) excluded
    assert record["resources"]["energy_valid"] is False
    assert any(
        "n=120 qualified uncontaminated attempts of 140; contaminated=20" in n
        for n in record["notes"]
    )


def test_censoring_acceptance_rule_detects_600s_qualification_censored_at_120s() -> None:
    core, amendment = _core(), _amendment()
    groups = amended_claim_core_groups(core.values, amendment.values)
    twins = {g.group_id: g for g in groups if group_censoring_stratum(g) == SENSITIVITY_STRATUM}
    cores = {g.group_id: g for g in groups if group_censoring_stratum(g) == CLAIM_CORE_STRATUM}
    twin_to_core = {}
    for index, group in enumerate(groups):
        if group.group_id in twins:
            twin_to_core[group.group_id] = groups[index - 1].group_id
    twin_id, twin = next(iter(twins.items()))
    core_group = cores[twin_to_core[twin_id]]

    def measured(group, disposition: str) -> list[dict[str, Any]]:
        records = [DECISION_TESTS._measured(group, r, 3.0) for r in range(7)]
        for record in records:
            if disposition != "qualified":
                _mark(record, disposition)
        return records

    # Both qualify: accepted once every twin has been compared.
    result = DECIDER.censoring_acceptance(
        {core_group.group_id: measured(core_group, "qualified")},
        {twin_id: measured(twin, "qualified")},
        twins,
        twin_to_core,
        expected_twins=1,
    )
    assert result["status"] == "accepted" and result["violations"] == []
    assert result["qualified_twin_attempts_compared"] == 7
    # Pending while twins are outstanding.
    assert (
        DECIDER.censoring_acceptance(
            {core_group.group_id: measured(core_group, "qualified")},
            {twin_id: measured(twin, "qualified")},
            twins,
            twin_to_core,
            expected_twins=36,
        )["status"]
        == "pending"
    )
    # Core timed out (replayed) where the 600 s twin qualified: amendment invalid.
    core_records = measured(core_group, "qualified")
    _mark(core_records[0], "timeout")
    for record in core_records[1:]:
        _mark(record, REPLAY_DISPOSITION)
    result = DECIDER.censoring_acceptance(
        {core_group.group_id: core_records},
        {twin_id: measured(twin, "qualified")},
        twins,
        twin_to_core,
        expected_twins=1,
    )
    assert result["status"] == "invalid"
    assert len(result["violations"]) == 7
    assert result["violations"][0]["core_disposition"] == "timeout"
    assert result["violations"][1]["core_disposition"] == REPLAY_DISPOSITION
    # Twin also timed out: no evidence of censoring-induced loss, accepted.
    result = DECIDER.censoring_acceptance(
        {core_group.group_id: core_records},
        {twin_id: measured(twin, "timeout")},
        twins,
        twin_to_core,
        expected_twins=1,
    )
    assert result["status"] == "accepted" and result["qualified_twin_attempts_compared"] == 0
    # Core qualified but contaminated still counts as qualified for acceptance.
    contaminated_core = measured(core_group, "qualified")
    for record in contaminated_core:
        record["contaminated"] = True
    result = DECIDER.censoring_acceptance(
        {core_group.group_id: contaminated_core},
        {twin_id: measured(twin, "qualified")},
        twins,
        twin_to_core,
        expected_twins=1,
    )
    assert result["status"] == "accepted"


def test_group_evidence_requires_amendment_echo_and_matching_censoring() -> None:
    core, amendment = _core(), _amendment()
    groups = amended_claim_core_groups(core.values, amendment.values)
    group = next(g for g in groups if group_censoring_stratum(g) == CLAIM_CORE_STRATUM)
    raw_schema = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text())
    attempts = [DECISION_TESTS._warmup(group, 0), DECISION_TESTS._warmup(group, 1)]
    attempts.extend(DECISION_TESTS._measured(group, repeat, 2.0) for repeat in range(7))
    result = DECISION_TESTS._group_result(group, attempts)
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="amendment echo"):
        DECIDER.validate_group_evidence(
            group,
            result,
            Draft202012Validator(raw_schema),
            ROOT / "experiments/schema/paper1_result.schema.json",
            amendment,
        )
    result[AMENDMENT_RECORD_FIELD] = AMENDMENT_ID
    result["policy_amendment_sha256"] = amendment.sha256
    result["censoring"] = {
        "stratum": CLAIM_CORE_STRATUM,
        "attempt_deadline_seconds": 600,
        "inner_iteration_cap": 200_000,
    }
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="wrong censoring"):
        DECIDER.validate_group_evidence(
            group,
            result,
            Draft202012Validator(raw_schema),
            ROOT / "experiments/schema/paper1_result.schema.json",
            amendment,
        )
    result["censoring"]["attempt_deadline_seconds"] = 120
    for item in attempts:
        item[AMENDMENT_RECORD_FIELD] = AMENDMENT_ID
        item["amendment"] = {
            "censoring_stratum": CLAIM_CORE_STRATUM,
            "attempt_deadline_seconds": 120,
            "inner_iteration_cap": 200_000,
            "deterministic_replay": True,
        }
        item["trace_hash"] = "0" * 16
        item["trace"] = _trace()
    measured = DECIDER.validate_group_evidence(
        group,
        result,
        Draft202012Validator(raw_schema),
        ROOT / "experiments/schema/paper1_result.schema.json",
        amendment,
    )
    assert len(measured) == 7
    # Amended records without --amendment are refused rather than silently accepted.
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="require --amendment"):
        DECIDER.validate_group_evidence(
            group,
            result,
            Draft202012Validator(raw_schema),
            ROOT / "experiments/schema/paper1_result.schema.json",
        )
