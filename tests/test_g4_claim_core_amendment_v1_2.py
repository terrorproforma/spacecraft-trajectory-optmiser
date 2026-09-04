"""Amendment single-gpu-v1.2: hash lock, supersession, rule A/B/C selection and recording.

Rule A (IPM equilibration selection and the ``not_applicable_ipm_native`` scaling_mode record),
rule B (measured wall time past the attempt deadline is ``timeout`` for every backend, never
``numerical``) and rule C (hard bound recorded unchanged) are checked against the frozen JSON, the
reference implementation in ``g4_execution_contract`` and the scheduler/decision code.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.campaign_scope import SCOPE_DEFINITIONS
from spacepdhcg.experiments.g4 import G4ContractError, load_policy
from spacepdhcg.experiments.g4_execution_contract import (
    AMENDMENT_ID,
    AMENDMENT_ID_V1_2,
    AMENDMENT_RECORD_FIELD,
    AMENDMENT_V1_1_SHA256,
    DEADLINE_CLASSIFICATION_RULE,
    EXECUTOR_DEFECT_DISPOSITION,
    IPM_NATIVE_SCALING_MODE,
    NO_EQUILIBRATION_DIAGNOSTIC_STRATUM,
    REPLAY_DISPOSITION,
    SUPPORTED_AMENDMENT_IDS,
    amended_claim_core_groups,
    amended_schedule_sha256,
    classify_launched_attempt,
    deterministic_trace_hash,
    expected_ipm_equilibration,
    iter_claim_core_groups,
    load_claim_core,
    load_claim_core_amendment,
    policy_uses_ipm,
    recorded_scaling_mode,
    validate_attempt_record,
    validate_claim_core_amendment,
)
from spacepdhcg.experiments.g4_scheduler import CampaignStore

ROOT = Path(__file__).resolve().parents[1]
V1_1_PATH = ROOT / "benchmarks/g4_claim_core_amendment_v1_1.json"
V1_2_PATH = ROOT / "benchmarks/g4_claim_core_amendment_v1_2.json"


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _module("amendment_v1_2_runner", "scripts/gpu/run_g4_campaign.py")
DECIDER = _module("amendment_v1_2_decider", "scripts/gpu/decide_g4_claim_core.py")


def _locked(name: str, loader):
    digest = (ROOT / f"benchmarks/{name}.sha256").read_text(encoding="utf-8").split()[0]
    return loader(ROOT / f"benchmarks/{name}.json", expected_sha256=digest)


def _core():
    return _locked("g4_h5_h6_claim_core", load_claim_core)


def _policy():
    return _locked("g4_policy", load_policy)


def _amendment(path: Path = V1_2_PATH):
    core, policy = _core(), _policy()
    lock = path.with_suffix(".sha256").read_text(encoding="utf-8").split()
    assert lock[1] == path.name
    return load_claim_core_amendment(
        path,
        core.values,
        claim_core_sha256=core.sha256,
        policy_sha256=policy.sha256,
        expected_sha256=lock[0],
    )


# ------------------------------------------------------------------------------------------
# Document: hash lock, supersession, schema reuse, registry references, preregistration
# ------------------------------------------------------------------------------------------


def test_v1_2_hash_lock_supersession_schema_and_registry_agree() -> None:
    amendment = _amendment()
    values = amendment.values
    assert amendment.sha256 == hashlib.sha256(V1_2_PATH.read_bytes()).hexdigest()
    assert values["amendment_id"] == AMENDMENT_ID_V1_2 == "single-gpu-v1.2"
    assert values["record_field"] == {"name": AMENDMENT_RECORD_FIELD, "value": AMENDMENT_ID_V1_2}
    # Supersedes v1.1 by hash and inherits its sections byte-for-byte.
    v1_1 = json.loads(V1_1_PATH.read_text("utf-8"))
    assert hashlib.sha256(V1_1_PATH.read_bytes()).hexdigest() == AMENDMENT_V1_1_SHA256
    assert values["supersedes"] == {
        "amendment_id": AMENDMENT_ID,
        "path": "benchmarks/g4_claim_core_amendment_v1_1.json",
        "sha256": AMENDMENT_V1_1_SHA256,
    }
    for section in values["inherited_from_v1_1"]:
        assert values[section] == v1_1[section], section
    # Schedule identity is unchanged: same groups, same hash as v1.1.
    core = _core()
    groups = amended_claim_core_groups(core.values, values)
    assert len(groups) == 396
    assert amended_schedule_sha256(groups) == v1_1["schedule"]["schedule_sha256"]
    # Preregistration: frozen before any amended group ran; timestamp precedes the v1.2 records.
    assert values["preregistered_before_results"] is True
    assert values["frozen_at"] == "2026-09-03T06:45:00Z"
    assert values["frozen_at"] > v1_1["frozen_at"]
    # Schema reuse: the single amendment schema validates both documents.
    schema = json.loads(
        (ROOT / "experiments/schema/g4_claim_core_amendment.schema.json").read_text("utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(values)
    validator.validate(v1_1)
    # v1.2 sections are required for v1.2 and forbidden for v1.1.
    missing = copy.deepcopy(values)
    del missing["ipm_equilibration"]
    assert not validator.is_valid(missing)
    leaked = copy.deepcopy(v1_1)
    leaked["hard_bound"] = values["hard_bound"]
    assert not validator.is_valid(leaked)
    # Registry: scope, JSON mirror, contract document.
    registered = SCOPE_DEFINITIONS["single-gpu-v1"]["amendments"]
    assert [item["amendment_id"] for item in registered] == [AMENDMENT_ID, AMENDMENT_ID_V1_2]
    entry = registered[1]
    assert entry["supersedes"] == AMENDMENT_ID
    assert ROOT / entry["path"] == V1_2_PATH
    assert (ROOT / entry["lock"]).read_text("utf-8").split() == [amendment.sha256, V1_2_PATH.name]
    assert (ROOT / entry["document"]).is_file()
    mirror = json.loads((ROOT / "benchmarks/campaign_scopes/single-gpu-v1.json").read_text("utf-8"))
    assert mirror["amendments"] == registered
    contract = (ROOT / "docs/G4_EXECUTION_CONTRACT.md").read_text("utf-8")
    assert "Amendment `single-gpu-v1.2`" in contract
    assert "benchmarks/g4_claim_core_amendment_v1_2.sha256" in contract
    assert NO_EQUILIBRATION_DIAGNOSTIC_STRATUM in contract
    report = (ROOT / "docs/G4_GATE_REPORT.md").read_text("utf-8")
    assert NO_EQUILIBRATION_DIAGNOSTIC_STRATUM in report


def test_v1_2_validator_rejects_tampering_with_rules_a_b_c() -> None:
    core, policy = _core(), _policy()
    amendment = _amendment()

    def check(mutate, match: str) -> None:
        tampered = copy.deepcopy(amendment.values)
        mutate(tampered)
        with pytest.raises(G4ContractError, match=match):
            validate_claim_core_amendment(
                tampered,
                core.values,
                claim_core_sha256=core.sha256,
                policy_sha256=policy.sha256,
            )

    check(lambda a: a["supersedes"].update(sha256="0" * 64), "supersede")
    check(lambda a: a["ipm_equilibration"].update(ruiz_iterations=5), "ruiz_iterations drift")
    check(lambda a: a["ipm_equilibration"].update(mode="qoco_native_ruiz"), "equilibration mode")
    check(
        lambda a: a["ipm_equilibration"]["recorded_scaling_mode"].update(
            {"pure-gpu-ipm": "identity"}
        ),
        "scaling_mode rule",
    )
    check(lambda a: a["ipm_equilibration"]["probe_evidence"].update(probes=[]), "probe evidence")
    check(lambda a: a["deadline_classification"].update(never="timeout"), "exclude numerical")
    check(lambda a: a["deadline_classification"].update(applies_to="ipm only"), "every backend")
    check(lambda a: a["hard_bound"].update(unchanged=False), "hard bound")
    check(
        lambda a: a["diagnostic_strata"][NO_EQUILIBRATION_DIAGNOSTIC_STRATUM].update(
            excluded_from=[]
        ),
        "excluded from H6",
    )
    # Inherited sections are still enforced (v1.1 rules survive the supersession).
    check(lambda a: a["censoring"]["claim_core"].update(attempt_deadline_seconds=60), "censoring")


# ------------------------------------------------------------------------------------------
# Rule A: equilibration selection and the not_applicable scaling_mode record
# ------------------------------------------------------------------------------------------


def test_rule_a_selects_qoco_native_default_for_ipm_policies_only() -> None:
    values = _amendment().values
    selection = values["ipm_equilibration"]
    assert selection["mode"] == "qoco_native_default"
    assert selection["ruiz_iterations"] == 0
    assert selection["qoco_default_at_pinned_commit"]["ruiz_iters"] == 0
    assert [p for p in ("pure-gpu-ipm", "hybrid-pdhcg-ipm") if policy_uses_ipm(p)] == [
        "pure-gpu-ipm",
        "hybrid-pdhcg-ipm",
    ]
    for policy in ("adaptive", "fixed-tight", "fixed-loose", "adaptive+polish"):
        assert not policy_uses_ipm(policy)
        assert expected_ipm_equilibration(policy, values) is None
    for policy in ("pure-gpu-ipm", "hybrid-pdhcg-ipm"):
        assert expected_ipm_equilibration(policy, values) == {
            "mode": "qoco_native_default",
            "ruiz_iterations": 0,
            "requested_ruiz_iterations": 0,
            "scaling_mode": IPM_NATIVE_SCALING_MODE,
        }
    # Under v1.1 no selection is recorded at all (that is why v1.1 IPM records are a stratum).
    v1_1 = json.loads(V1_1_PATH.read_text("utf-8"))
    assert expected_ipm_equilibration("pure-gpu-ipm", v1_1) is None
    # The probe evidence documents a genuine IPM negative rather than a rescued coordinate.
    labels = {probe["label"] for probe in selection["probe_evidence"]["probes"]}
    assert {"campaign_v1_1_ruiz0_cond4", "ruiz5_cond4_pinned", "ruiz0_cond0_patched"} <= labels
    assert "genuine IPM negative" in selection["probe_evidence"]["conclusion"]
    assert len(selection["qoco_cuda_ruiz_defects"]) == 2


def test_rule_a_records_not_applicable_scaling_mode_for_pure_ipm_only() -> None:
    for axis in ("refresh_if_needed", "always_refresh", "reuse"):
        assert recorded_scaling_mode("pure-gpu-ipm", axis, AMENDMENT_ID_V1_2) == (
            IPM_NATIVE_SCALING_MODE
        )
        # Hybrid keeps the axis because its PDHCG stage consumes it.
        assert recorded_scaling_mode("hybrid-pdhcg-ipm", axis, AMENDMENT_ID_V1_2) == axis
        assert recorded_scaling_mode("adaptive", axis, AMENDMENT_ID_V1_2) == axis
        # Before v1.2 the coordinate axis was recorded verbatim for every policy.
        assert recorded_scaling_mode("pure-gpu-ipm", axis, AMENDMENT_ID) == axis
        assert recorded_scaling_mode("pure-gpu-ipm", axis, None) == axis
    schema = json.loads((ROOT / "experiments/schema/paper1_result.schema.json").read_text("utf-8"))
    text = json.dumps(schema)
    assert IPM_NATIVE_SCALING_MODE in text
    raw = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text("utf-8"))
    raw_text = json.dumps(raw)
    assert AMENDMENT_ID_V1_2 in raw_text
    assert "ipm_equilibration" in raw_text and "deadline_classification" in raw_text


# ------------------------------------------------------------------------------------------
# Rule B: timeout classification by measured wall time, every backend, never numerical
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("solver", "elapsed", "expected"),
    [
        ("numerical", 128.1, "timeout"),
        ("qualified", 155.8, "timeout"),
        ("unqualified", 120.001, "timeout"),
        ("timeout", 120.0, "timeout"),
        ("numerical", 120.0, "numerical"),
        ("qualified", 92.0, "qualified"),
        ("unqualified", 5.0, "unqualified"),
        ("infeasible", 300.0, "timeout"),
        (EXECUTOR_DEFECT_DISPOSITION, 500.0, EXECUTOR_DEFECT_DISPOSITION),
        ("unrun", 0.0, "unrun"),
        (REPLAY_DISPOSITION, 0.0, REPLAY_DISPOSITION),
    ],
)
def test_rule_b_classifies_wall_past_deadline_as_timeout(
    solver: str, elapsed: float, expected: str
) -> None:
    assert (
        classify_launched_attempt(
            solver_disposition=solver,
            elapsed_seconds=elapsed,
            attempt_deadline_seconds=120.0,
            amendment_id=AMENDMENT_ID_V1_2,
        )
        == expected
    )
    # Before v1.2 the solver's own disposition stood (how the v1.1 stratum was recorded).
    assert (
        classify_launched_attempt(
            solver_disposition=solver,
            elapsed_seconds=elapsed,
            attempt_deadline_seconds=120.0,
            amendment_id=AMENDMENT_ID,
        )
        == solver
    )


def test_rule_b_never_yields_numerical_past_the_deadline_and_rejects_bad_inputs() -> None:
    for solver in ("numerical", "qualified", "unqualified", "infeasible", "timeout"):
        assert (
            classify_launched_attempt(
                solver_disposition=solver,
                elapsed_seconds=121.0,
                attempt_deadline_seconds=120.0,
                amendment_id=AMENDMENT_ID_V1_2,
            )
            != "numerical"
        )
    with pytest.raises(G4ContractError, match="unknown solver disposition"):
        classify_launched_attempt(
            solver_disposition="crashed",
            elapsed_seconds=1.0,
            attempt_deadline_seconds=120.0,
            amendment_id=AMENDMENT_ID_V1_2,
        )
    with pytest.raises(G4ContractError, match="deadline must be positive"):
        classify_launched_attempt(
            solver_disposition="qualified",
            elapsed_seconds=1.0,
            attempt_deadline_seconds=0.0,
            amendment_id=AMENDMENT_ID_V1_2,
        )
    values = _amendment().values
    assert values["deadline_classification"]["criterion"] == DEADLINE_CLASSIFICATION_RULE
    assert values["deadline_classification"]["never"] == "numerical"


# ------------------------------------------------------------------------------------------
# Scheduler: the record echo the executor must produce under v1.2
# ------------------------------------------------------------------------------------------


def _record(
    *,
    policy: str,
    disposition: str,
    solver_disposition: str,
    elapsed: float,
    repeat_kind: str = "measured",
    repeat: int = 0,
    launched: bool = True,
) -> dict:
    values = _amendment().values
    ipm = expected_ipm_equilibration(policy, values)
    if ipm is not None:
        ipm = {**ipm, "qoco_status_code": 1}
    record = {
        "case": "g4_attempt",
        "repeat_kind": repeat_kind,
        "repeat": repeat,
        "launched": launched,
        "disposition": disposition,
        AMENDMENT_RECORD_FIELD: AMENDMENT_ID_V1_2,
        "amendment": {
            "censoring_stratum": "claim_core",
            "attempt_deadline_seconds": 120.0,
            "inner_iteration_cap": 200_000,
            "deterministic_replay": True,
            "ipm_equilibration": ipm,
            "deadline_classification": {
                "rule": DEADLINE_CLASSIFICATION_RULE,
                "wall_exceeded_deadline": elapsed > 120.0,
                "attempt_deadline_seconds": 120.0,
                "measured_wall_seconds": elapsed,
                "solver_disposition": solver_disposition,
            },
        },
        "timing": {"elapsed_seconds": elapsed},
        "trace": {
            "inner_iterations": 10,
            "outer_iterations": 1,
            "canonical_residual": 1e-9,
            "dynamics_residual": 0.0,
            "path_residual": 0.0,
            "terminal_residual": 0.0,
            "virtual_control_residual": 0.0,
            "checkpoints": [],
        },
        "paper1_result": {
            "identity": {
                "scaling_mode": recorded_scaling_mode(
                    policy, "refresh_if_needed", AMENDMENT_ID_V1_2
                )
            }
        },
    }
    from spacepdhcg.experiments.g4_execution_contract import deterministic_trace_hash

    record["trace_hash"] = deterministic_trace_hash(disposition, record["trace"])
    return record


def _expected(policy: str) -> dict:
    values = _amendment().values
    return {
        "amendment_id": AMENDMENT_ID_V1_2,
        "censoring_stratum": "claim_core",
        "attempt_deadline_seconds": 120.0,
        "inner_iteration_cap": 200_000,
        "ipm_equilibration": expected_ipm_equilibration(policy, values),
        "scaling_mode": recorded_scaling_mode(policy, "refresh_if_needed", AMENDMENT_ID_V1_2),
    }


def test_runner_accepts_a_conformant_v1_2_ipm_record_and_rejects_drift() -> None:
    good = _record(
        policy="pure-gpu-ipm", disposition="timeout", solver_disposition="numerical", elapsed=128.1
    )
    assert RUNNER.validate_v1_2_record_echo(good, good["amendment"], _expected("pure-gpu-ipm")) is (
        None
    )
    # A wall past the deadline recorded as numerical is exactly what rule B forbids.
    bad = _record(
        policy="pure-gpu-ipm",
        disposition="numerical",
        solver_disposition="numerical",
        elapsed=128.1,
    )
    problem = RUNNER.validate_v1_2_record_echo(bad, bad["amendment"], _expected("pure-gpu-ipm"))
    assert problem is not None and "rule B reference 'timeout'" in problem
    # Wrong equilibration echo (Ruiz on) is rejected.
    ruiz = _record(
        policy="pure-gpu-ipm", disposition="qualified", solver_disposition="qualified", elapsed=9.0
    )
    ruiz["amendment"]["ipm_equilibration"]["ruiz_iterations"] = 5
    problem = RUNNER.validate_v1_2_record_echo(ruiz, ruiz["amendment"], _expected("pure-gpu-ipm"))
    assert problem is not None and "ipm_equilibration echo" in problem
    # Missing QOCO status code is rejected (the adapter must report it).
    nostatus = _record(
        policy="pure-gpu-ipm", disposition="qualified", solver_disposition="qualified", elapsed=9.0
    )
    del nostatus["amendment"]["ipm_equilibration"]["qoco_status_code"]
    assert (
        RUNNER.validate_v1_2_record_echo(nostatus, nostatus["amendment"], _expected("pure-gpu-ipm"))
        is not None
    )
    # pure-gpu-ipm must record the not_applicable scaling_mode in its Paper 1 identity.
    axis = _record(
        policy="pure-gpu-ipm", disposition="qualified", solver_disposition="qualified", elapsed=9.0
    )
    axis["paper1_result"]["identity"]["scaling_mode"] = "refresh_if_needed"
    problem = RUNNER.validate_v1_2_record_echo(axis, axis["amendment"], _expected("pure-gpu-ipm"))
    assert problem is not None and "scaling_mode" in problem
    # A PDHCG policy carries a null equilibration echo and keeps the axis.
    pdhcg = _record(
        policy="adaptive", disposition="timeout", solver_disposition="qualified", elapsed=130.0
    )
    assert pdhcg["amendment"]["ipm_equilibration"] is None
    assert pdhcg["paper1_result"]["identity"]["scaling_mode"] == "refresh_if_needed"
    assert RUNNER.validate_v1_2_record_echo(pdhcg, pdhcg["amendment"], _expected("adaptive")) is (
        None
    )
    # Executor defects keep their disposition even past the deadline.
    defect = _record(
        policy="adaptive",
        disposition=EXECUTOR_DEFECT_DISPOSITION,
        solver_disposition=EXECUTOR_DEFECT_DISPOSITION,
        elapsed=400.0,
    )
    assert (
        RUNNER.validate_v1_2_record_echo(defect, defect["amendment"], _expected("adaptive")) is None
    )


def test_runner_end_to_end_amendment_records_check_under_v1_2() -> None:
    records = [
        _record(
            policy="pure-gpu-ipm",
            disposition="qualified",
            solver_disposition="qualified",
            elapsed=8.0,
            repeat_kind="warmup",
            repeat=index,
        )
        for index in range(2)
    ]
    records.extend(
        _record(
            policy="pure-gpu-ipm",
            disposition="qualified",
            solver_disposition="qualified",
            elapsed=8.0 + index,
            repeat=index,
        )
        for index in range(7)
    )
    assert RUNNER.validate_amendment_records(records, _expected("pure-gpu-ipm")) is None
    # A v1.1 record inside a v1.2 campaign is refused.
    stale = copy.deepcopy(records)
    stale[0][AMENDMENT_RECORD_FIELD] = AMENDMENT_ID
    assert "lacks policy_amendment=single-gpu-v1.2" in RUNNER.validate_amendment_records(
        stale, _expected("pure-gpu-ipm")
    )
    # Records without the rule B echo are refused.
    missing = copy.deepcopy(records)
    del missing[3]["amendment"]["deadline_classification"]
    assert "deadline_classification" in RUNNER.validate_amendment_records(
        missing, _expected("pure-gpu-ipm")
    )
    assert RUNNER.AMENDMENT_IN_FORCE_PATH == "benchmarks/g4_claim_core_amendment_v1_2.json"


# ------------------------------------------------------------------------------------------
# Deterministic replay under v1.2 (inherited verbatim from v1.1)
# ------------------------------------------------------------------------------------------


_IDENTITY_KEYS = (
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


def test_replay_records_are_valid_under_v1_2_and_refused_under_unknown_amendments() -> None:
    """Regression for g4-claim-core-46bc895 ordinals 45/47/49/57.

    The raw-attempt contract accepted a ``timeout_deterministic_replay`` record only when it was
    stamped ``single-gpu-v1.1``; every conformant v1.2 replay group was therefore quarantined as
    ``invalid_evidence`` by the scheduler (and would have been refused by the decision step).
    v1.2 inherits the ``deterministic_replay`` section verbatim, so a replay is valid under every
    supported amendment; anything else is still refused.
    """

    group = next(iter_claim_core_groups(_core().values))
    planned = group.attempts[2 + 3]  # measured/3
    trace = {
        # An IPM attempt that never completes an outer iteration: the campaign's actual shape.
        "inner_iterations": 200,
        "outer_iterations": 0,
        "canonical_residual": 0.0,
        "dynamics_residual": 0.0,
        "path_residual": 0.0,
        "terminal_residual": 0.0,
        "virtual_control_residual": 0.0,
        "checkpoints": [],
    }
    record = {
        **{key: planned[key] for key in _IDENTITY_KEYS},
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "attempt_id": f"{group.group_id}/measured-3",
        "launched": False,
        "disposition": REPLAY_DISPOSITION,
        "failure_class": "timeout",
        "reason": "deterministic replay of the measured/0 timeout",
        "replay_source_attempt_id": f"{group.group_id}/measured-0",
        AMENDMENT_RECORD_FIELD: AMENDMENT_ID_V1_2,
        "trace": trace,
        "trace_hash": deterministic_trace_hash(REPLAY_DISPOSITION, trace),
        "timing": {"elapsed_seconds": 0.0},
    }
    validate_attempt_record(record)
    validate_attempt_record({**record, AMENDMENT_RECORD_FIELD: AMENDMENT_ID})
    assert SUPPORTED_AMENDMENT_IDS == (AMENDMENT_ID, AMENDMENT_ID_V1_2)
    for foreign in ("single-gpu-v1", "single-gpu-v9.9", None):
        with pytest.raises(G4ContractError, match="exist only under amendments"):
            validate_attempt_record({**record, AMENDMENT_RECORD_FIELD: foreign})
    without = dict(record)
    del without[AMENDMENT_RECORD_FIELD]
    with pytest.raises(G4ContractError, match="exist only under amendments"):
        validate_attempt_record(without)


def test_runner_accepts_a_replayed_group_under_v1_2() -> None:
    """Three launched timeouts with identical traces plus six unlaunched replays (the shape the
    executor emitted for the quarantined N=2000 pure-gpu-ipm groups) pass every scheduler check
    a v1.2 group must pass: the raw-attempt contract and the amendment consistency rules."""

    group = next(
        item
        for item in iter_claim_core_groups(_core().values)
        if item.coordinate["policy"] == "pure-gpu-ipm"
    )
    records = []
    for kind, repeat in (("warmup", 0), ("warmup", 1), ("measured", 0)):
        executed = _record(
            policy="pure-gpu-ipm",
            disposition="timeout",
            solver_disposition="numerical",
            elapsed=272.3,
            repeat_kind=kind,
            repeat=repeat,
        )
        records.append(executed)
    for repeat in range(1, 7):
        replay = _record(
            policy="pure-gpu-ipm",
            disposition=REPLAY_DISPOSITION,
            solver_disposition=REPLAY_DISPOSITION,
            elapsed=0.0,
            repeat=repeat,
            launched=False,
        )
        replay["replay_source_attempt_id"] = f"{group.group_id}/measured-0"
        records.append(replay)
    for record in records:
        index = record["repeat"] if record["repeat_kind"] == "warmup" else 2 + record["repeat"]
        record.update({key: group.attempts[index][key] for key in _IDENTITY_KEYS})
        record.update(
            {
                "schema_version": "1.0.0",
                "record_kind": "raw_attempt",
                "attempt_id": f"{group.group_id}/{record['repeat_kind']}-{record['repeat']}",
                "failure_class": "timeout",
                "reason": "synthetic deadline outcome",
            }
        )
        validate_attempt_record(record)
    assert RUNNER.deterministic_replay_eligible(records[:3]) is True
    assert RUNNER.validate_amendment_records(records, _expected("pure-gpu-ipm")) is None
    # The replay must repeat measured/0's trace; a drifted replay is still refused.
    drifted = copy.deepcopy(records)
    drifted[5]["trace"]["inner_iterations"] = 199
    drifted[5]["trace_hash"] = deterministic_trace_hash(REPLAY_DISPOSITION, drifted[5]["trace"])
    assert "does not repeat the measured/0 trace" in RUNNER.validate_amendment_records(
        drifted, _expected("pure-gpu-ipm")
    )


def test_migrate_can_leave_quarantined_rows_behind_for_re_run(tmp_path: Path) -> None:
    """``migrate --skip-quarantined``: quarantined rows stay in the source ledger (records
    retained) and the target campaign re-runs them first; the default still carries them."""

    core, policy, amendment = _core(), _policy(), _amendment()
    groups = amended_claim_core_groups(core.values, amendment.values)
    schedule = amended_schedule_sha256(groups)
    extra = {
        AMENDMENT_RECORD_FIELD: AMENDMENT_ID_V1_2,
        "policy_amendment_sha256": amendment.sha256,
        "claim_core_sha256": core.sha256,
    }

    def open_store(root: Path, commit: str) -> CampaignStore:
        return CampaignStore(
            root,
            policy.values,
            policy.sha256,
            commit,
            grouped=True,
            groups=groups,
            schedule_sha256=schedule,
            extra_metadata=extra,
        )

    source_root = tmp_path / "source"
    with open_store(source_root, "a" * 40) as source:
        valid = source.claim()
        assert valid is not None
        source.finish(
            valid,
            disposition="completed_group",
            reason="all raw attempts and measured Paper 1 results validated",
            record={"raw_attempts": []},
            valid=True,
        )
        broken = source.claim()
        assert broken is not None
        source.finish(
            broken,
            disposition="invalid_evidence",
            reason="strict measured-result validation failed: synthetic",
            record={"raw_attempts": []},
            valid=False,
        )
        assert (source.status()["completed"], source.status()["quarantined"]) == (1, 1)

    with open_store(tmp_path / "rerun", "b" * 40) as target:
        outcome = RUNNER.migrate_terminal_rows(target, source_root, include_quarantined=False)
        assert outcome == {"imported": 1, "already_present": 0, "skipped_quarantined": 1}
        assert (target.status()["completed"], target.status()["quarantined"]) == (1, 0)
        claim = target.claim()
        assert claim is not None and claim.ordinal == broken.ordinal
    with open_store(tmp_path / "carry", "c" * 40) as target:
        outcome = RUNNER.migrate_terminal_rows(target, source_root)
        assert outcome == {"imported": 2, "already_present": 0, "skipped_quarantined": 0}
        assert (target.status()["completed"], target.status()["quarantined"]) == (1, 1)
        claim = target.claim()
        assert claim is not None and claim.ordinal not in {valid.ordinal, broken.ordinal}


# ------------------------------------------------------------------------------------------
# Decision: v1.2 attempt checks and the diagnostic-stratum citation
# ------------------------------------------------------------------------------------------


def test_decider_validates_v1_2_attempts_and_requires_the_stratum_citation() -> None:
    amendment = _amendment()
    groups = amended_claim_core_groups(_core().values, amendment.values)
    by_policy = {}
    for group in groups:
        by_policy.setdefault(group.coordinate["policy"], group)
    censoring = {"attempt_deadline_seconds": 120.0, "inner_iteration_cap": 200_000}
    good = _record(
        policy="pure-gpu-ipm", disposition="timeout", solver_disposition="numerical", elapsed=128.1
    )
    DECIDER.validate_v1_2_attempt(good, by_policy["pure-gpu-ipm"], amendment, censoring)
    bad = _record(
        policy="pure-gpu-ipm",
        disposition="numerical",
        solver_disposition="numerical",
        elapsed=128.1,
    )
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match=r"rule B|reference|timeout"):
        DECIDER.validate_v1_2_attempt(bad, by_policy["pure-gpu-ipm"], amendment, censoring)
    ruiz = _record(
        policy="hybrid-pdhcg-ipm",
        disposition="qualified",
        solver_disposition="qualified",
        elapsed=9.0,
    )
    ruiz["amendment"]["ipm_equilibration"]["mode"] = "qoco_native_ruiz"
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="amendment selects"):
        DECIDER.validate_v1_2_attempt(ruiz, by_policy["hybrid-pdhcg-ipm"], amendment, censoring)
    leak = _record(
        policy="adaptive", disposition="qualified", solver_disposition="qualified", elapsed=9.0
    )
    leak["amendment"]["ipm_equilibration"] = {"mode": "qoco_native_default"}
    with pytest.raises(DECIDER.ClaimCoreDecisionError, match="PDHCG-only"):
        DECIDER.validate_v1_2_attempt(leak, by_policy["adaptive"], amendment, censoring)
    source = (ROOT / "scripts/gpu/decide_g4_claim_core.py").read_text("utf-8")
    assert "must cite the ipm_no_equilibration_v1_1 diagnostic stratum" in source
