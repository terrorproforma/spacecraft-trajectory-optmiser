"""Capture synthetic native readbacks without loading or executing native code."""

from __future__ import annotations

import ctypes
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from spacepdhcg.gtoc12 import gpu_completion as g
from spacepdhcg.gtoc12.completion_capture import CompletionRecorder
from spacepdhcg.gtoc12.refinement_admission import (
    AdmissionCandidate,
    FixedCargoRequest,
    RefinementAdmissionQueue,
)
from spacepdhcg.gtoc12.search import PlannedLeg, RoutePlan, RouteSearch, SearchSettings, _Partial


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("capture contract tests cannot execute native code")

    monkeypatch.setattr(ctypes, "CDLL", blocked)


def case(body=101, *, failed=True):
    cargo = 10.0 * 1000.0 / 365.25
    first = PlannedLeg(0, body, 64328.0, 65000.0, 1.0, 1.0, "earth_out")
    last = PlannedLeg(body, 0, 66000.0, 66500.0, 2.0, 1.0, "earth_return")
    partial = _Partial([first], body, 65000.0, 600.0, [(body, 65000.0)])
    row = (partial, {body: 65000.0}, {body: 66000.0}, [last], True)
    results = np.zeros(1, g.RESULT)
    results[0] = (
        5 if failed else 0,
        -1,
        -1,
        1,
        2440.0,
        499.0 + cargo if failed else 501.0 + cargo,
        cargo,
        -1.0 if failed else 1.0,
    )
    details = np.zeros(1, g.LEG_RESULT)
    details[0] = (
        4,
        1,
        600.0,
        cargo,
        600.0 + cargo,
        float(results[0]["final_mass"]),
        20.0,
        1.2,
        600.0 + cargo - float(results[0]["final_mass"]),
        500.0,
    )
    collected = np.asarray([cargo], dtype=np.float64)
    policy = np.zeros(1, g.POLICY)
    policy["initial_mass"], policy["dry_mass"], policy["miner_mass"] = 3000, 500, 40
    candidates = np.zeros(1, g.CANDIDATE)
    candidates[0] = (0, 1, 0, 1, 600.0)
    deploys = np.zeros(1, g.DEPLOY)
    deploys[0] = (65000.0, 66000.0, 1, 0)
    legs = np.zeros(1, g.LEG)
    legs["role"], legs["source_deploy"] = 4, 0
    legs["departure"], legs["arrival"], legs["dv"] = 66000, 66500, 2
    return row, results, details, collected, (policy, candidates, deploys, legs)


def record(recorder, values, model=None):
    row, results, details, collected, inputs = values
    recorder(row, results[0], details, collected, inputs, model)


def test_complete_mass_rejection_survives_workspace_reuse_as_immutable_data():
    captured = []
    recorder = CompletionRecorder(
        "a" * 64, lambda envelope: captured.append(envelope) or "retained"
    )
    values = case()
    record(recorder, values)
    envelope = captured[0]
    snapshot = (
        envelope.plan_summary,
        envelope.raw_result,
        envelope.raw_legs,
        envelope.raw_collected,
        envelope.input_sha256,
        envelope.observation.readback_sha256,
    )
    row, results, details, collected, inputs = values
    results.fill(0)
    details.fill(0)
    collected.fill(0)
    for array in inputs:
        array.fill(0)
    row[1][101] += 100
    row[3].clear()
    assert snapshot == (
        envelope.plan_summary,
        envelope.raw_result,
        envelope.raw_legs,
        envelope.raw_collected,
        envelope.input_sha256,
        envelope.observation.readback_sha256,
    )
    summary = json.loads(envelope.plan_summary)
    assert not summary["feasible_proxy"]
    assert envelope.observation.failure == "mass_below_dry_plus_collected"
    assert FixedCargoRequest.from_summary(summary) == envelope.request
    assert (
        hashlib.sha256(
            envelope.raw_result
            + envelope.raw_legs
            + envelope.raw_collected
            + envelope.input_sha256.encode()
        ).hexdigest()
        == envelope.observation.readback_sha256
    )
    assert recorder.dispositions == {"retained": 1}


@pytest.mark.parametrize("failure", [1, 2, 3, 4, 6])
def test_earlier_failures_cannot_reach_the_uncertainty_consumer(failure):
    captured = []
    recorder = CompletionRecorder(
        "a" * 64, lambda envelope: captured.append(envelope) or "retained"
    )
    values = case()
    values[1]["failure"] = failure
    record(recorder, values)
    assert not captured
    assert recorder.dispositions["earlier_completion_gate"] == 1


@pytest.mark.parametrize(
    "corruption", ["inflation", "fuel", "collected", "collected_length", "stage", "incomplete"]
)
def test_corrupted_or_incomplete_readbacks_are_not_escalated(corruption):
    captured, blocked = [], []
    recorder = CompletionRecorder(
        "a" * 64,
        lambda envelope: captured.append(envelope) or "retained",
        on_blocked=lambda *args: blocked.append(args),
    )
    row, results, details, collected, inputs = case()
    if corruption == "inflation":
        details["inflation"] = -1.0
    elif corruption == "fuel":
        details["propellant"] = float("nan")
    elif corruption == "collected":
        collected[0] -= 1.0
    elif corruption == "collected_length":
        collected = np.asarray([], dtype=np.float64)
    elif corruption == "stage":
        details["stage"] = 2
    else:
        results["processed_legs"] = 0
    recorder(row, results[0], details, collected, inputs)
    assert not captured and len(blocked) == 1


def test_different_native_model_signatures_bind_distinct_readbacks_for_same_prescription():
    captured = []
    recorder = CompletionRecorder(
        "a" * 64, lambda envelope: captured.append(envelope) or "retained"
    )
    record(recorder, case(), model="b" * 64)
    record(recorder, case(), model="c" * 64)
    first, second = captured
    assert first.request == second.request
    assert first.raw_result == second.raw_result
    assert first.input_sha256 != second.input_sha256
    assert first.observation.readback_sha256 != second.observation.readback_sha256


def test_invalid_prescription_is_retained_as_blocked_metadata_not_a_route():
    captured, blocked = [], []
    recorder = CompletionRecorder(
        "a" * 64,
        lambda envelope: captured.append(envelope) or "retained",
        on_blocked=lambda *args: blocked.append(args),
    )
    values = case()
    values[0][2][101] = 66100.0  # collection occurs after departure
    record(recorder, values)
    assert not captured and blocked[0][0] == "unsupported_prescription"
    assert len(blocked[0][2]) == len(blocked[0][3]) == 64


def test_opt_in_capture_does_not_change_the_normal_failed_completion_result():
    envelopes = []
    recorder = CompletionRecorder(
        "a" * 64, lambda envelope: envelopes.append(envelope) or "retained"
    )
    values = case()
    before = values[1].tobytes()
    record(recorder, values)
    assert values[1].tobytes() == before
    assert int(values[1][0]["failure"]) == 5
    assert not RoutePlan.from_summary(json.loads(envelopes[0].plan_summary)).feasible


def test_bounded_consumer_keeps_only_current_queue_prescriptions_and_can_rebuild_them():
    queue = RefinementAdmissionQueue()
    plans = {}

    def consume(envelope):
        candidate = AdmissionCandidate(envelope.request, envelope.observation, 1.0, 15000.0, 23)
        disposition = queue.offer(candidate)
        plans[envelope.request.sha256] = envelope.plan_summary
        retained = {item.request.sha256 for item in queue.retained()}
        for key in list(plans):
            if key not in retained:
                del plans[key]
        assert len(plans) <= 48
        return disposition

    recorder = CompletionRecorder("a" * 64, consume)
    for body in range(1, 51):
        record(recorder, case(body, failed=body % 2 == 0))
    assert len(plans) == len(queue.retained()) == 48
    claims = []
    while (claimed := queue.claim_next()) is not None:
        claims.append(claimed)
        plan = RoutePlan.from_summary(json.loads(plans[claimed.request.sha256]))
        assert FixedCargoRequest.from_summary(plan.summary()) == claimed.request
    assert len(claims) == 4 and sum(x.uncertain for x in claims) == 2
    assert sum(recorder.dispositions.values()) == 50


def test_capture_requires_loaded_binary_identity_and_a_journal_disposition():
    with pytest.raises(ValueError, match="producer"):
        CompletionRecorder("", lambda x: "ignored")
    recorder = CompletionRecorder("a" * 64, lambda envelope: None)
    with pytest.raises(TypeError, match="journal"):
        record(recorder, case())


def test_production_loop_captures_failed_and_successful_slices_without_promoting_failure(
    monkeypatch,
):
    """Exercise the real reconstruction loop; only the native evaluate call is a stub."""
    monkeypatch.setenv("SPACEPDHCG_TEST_GTOC12_COMPLETION_NATIVE_MODEL", "0")
    captured = []
    values = [case(101, failed=True), case(102, failed=False)]
    requests = [(*value[0][:4], False) for value in values]
    search = RouteSearch(None, np.asarray([101, 102], dtype=np.int64), SearchSettings())
    search.completion_capture = CompletionRecorder(
        "a" * 64, lambda envelope: captured.append(envelope) or "retained"
    )
    workspace = object.__new__(g.GpuCompletion)
    workspace.gpu = SimpleNamespace(_owned=lambda: None, telemetry={})
    workspace.handle = None
    workspace.native_model = None
    workspace.inputs = tuple(
        np.zeros(n, dtype)
        for n, dtype in ((1, g.POLICY), (2, g.CANDIDATE), (2, g.DEPLOY), (2, g.LEG))
    )
    workspace.results = np.concatenate([value[1] for value in values])
    workspace.leg_results = np.concatenate([value[2] for value in values])
    workspace.collected = np.concatenate([value[3] for value in values])
    workspace.stats = np.zeros(1, g.STATS)
    calls = []

    def evaluate(*args):
        calls.append(args)
        return 0

    workspace.evaluate = evaluate
    rows = workspace.run(search, requests)
    assert len(calls) == 1 and len(captured) == 2
    assert rows[0] == (None, "mass_below_dry_plus_collected")
    assert rows[1][0] is not None and rows[1][1] == ""
    assert [json.loads(x.plan_summary)["asteroids"] for x in captured] == [[101], [102]]
    assert captured[0].observation.failure and not captured[1].observation.failure
    assert captured[0].raw_result == values[0][1][0].tobytes()
    assert captured[1].raw_result == values[1][1][0].tobytes()
    assert search.completion_capture.dispositions["retained"] == 2
