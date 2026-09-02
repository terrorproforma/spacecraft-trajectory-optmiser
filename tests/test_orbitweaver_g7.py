from __future__ import annotations

import threading
from pathlib import Path

import pytest

from spacepdhcg.orbitweaver import (
    ArcFidelity,
    ArcRequest,
    ArcResult,
    ArcStatus,
    BoundedScheduler,
    CertificationChecks,
    Checkpoint,
    IndependentCertifier,
    LogicalRankOwnership,
    RiskMeasure,
    ScenarioOutcome,
    SchedulerConfig,
    TopologyKey,
    aggregate_risk,
    deterministic_top_k,
    expand_promising_scenarios,
    load_frozen_paper2_matrix,
)


def request(identifier: int, topology: int = 11) -> ArcRequest:
    return ArcRequest(
        deterministic_id=identifier,
        from_target=0,
        to_target=1,
        departure_epoch=0.0,
        arrival_epoch=10.0,
        initial_mass=100.0,
        spacecraft=0,
        scenario_count=1,
        fidelity=ArcFidelity.COARSE_CONVEX,
        requested_tolerance=1.0e-4,
        model_identifier="test",
        topology=TopologyKey(topology, ArcFidelity.COARSE_CONVEX, 8, 1),
    )


class Backend:
    def evaluate(
        self,
        topology: TopologyKey,
        requests: list[ArcRequest],
        owner: object,
        cancelled: threading.Event,
    ) -> list[ArcResult]:
        del topology, owner, cancelled
        return [
            ArcResult(
                item.deterministic_id,
                ArcStatus.FEASIBLE if item.deterministic_id != 3 else ArcStatus.INFEASIBLE,
                item.fidelity,
                cost=float(item.deterministic_id + 1),
                lower_bound=1.0,
                duration=10.0,
                delta_v=1.0,
                propellant=5.0,
                final_mass=95.0,
                terminal_error=0.0,
                path_violation=0.0,
                uncertainty_violation=0.0,
                canonical_residual=0.0,
                replay_residual=0.0,
                warm_token=100 + item.deterministic_id,
            )
            for item in requests
        ]


def test_scheduler_groups_bounds_and_retains_failures() -> None:
    scheduler = BoundedScheduler(
        Backend(),
        ownership=LogicalRankOwnership((0, 1)),
        config=SchedulerConfig(2, 8, 64, 128),
    )
    results = scheduler.run([request(3, 22), request(2), request(1)])
    assert [item.deterministic_id for item in results] == [1, 2, 3]
    assert scheduler.telemetry.batches == 3
    assert scheduler.telemetry.estimated_peak_buffer_bytes == 64
    selected = deterministic_top_k(results, 3)
    assert [item.deterministic_id for item in selected] == [1, 2, 3]
    assert selected[-1].status is ArcStatus.INFEASIBLE


def test_scheduler_applies_backpressure_and_failure_propagation() -> None:
    scheduler = BoundedScheduler(Backend(), config=SchedulerConfig(1, 1, 1, 1))
    with pytest.raises(BufferError):
        scheduler.run([request(1), request(2)])


def test_risk_nonanticipativity_expected_worst_and_cvar() -> None:
    outcomes = [
        ScenarioOutcome(1, 0.75, 2.0, 1.0, (1.0,)),
        ScenarioOutcome(0, 0.25, 6.0, 2.0, (1.0,)),
    ]
    assert aggregate_risk(outcomes, RiskMeasure.EXPECTED).objective == 3.0
    assert aggregate_risk(outcomes, RiskMeasure.WORST_CASE).objective == 6.0
    assert aggregate_risk(outcomes, RiskMeasure.CVAR, cvar_alpha=0.5).objective >= 3.0
    invalid = [
        ScenarioOutcome(0, 0.5, 1.0, 0.0, (1.0,)),
        ScenarioOutcome(1, 0.5, 2.0, 0.0, (1.1,)),
    ]
    assert not aggregate_risk(invalid, RiskMeasure.EXPECTED).feasible


def test_scenario_expansion_and_checkpoint_restart(tmp_path: Path) -> None:
    expanded = expand_promising_scenarios([request(2), request(1)], scenario_count=3, top_k=1)
    assert [item.scenario_index for item in expanded] == [0, 1, 2]
    assert all(item.fidelity is ArcFidelity.ROBUST_SCVX for item in expanded)
    checkpoint = Checkpoint(
        1,
        "test",
        "a" * 64,
        "78c4e33e4aabcd85d63ba3f1e03aa2214b3ab207e680bcaaf347516802b2f6a2",
        42,
        0,
        2,
        4.0,
        1.0,
        (1, 2),
        (101,),
    )
    path = tmp_path / "checkpoint.json"
    checkpoint.write(path)
    assert Checkpoint.read(path) == checkpoint


def test_independent_certification_rejects_optimizer_status() -> None:
    result = Backend().evaluate(request(1).topology, [request(1)], object(), threading.Event())[0]
    accepted = IndependentCertifier(
        lambda _: CertificationChecks(0.0, 0.0, 0.0, 0.0, 1.0e-7),
        backend_identifier="independent-rk4",
        tolerance=1.0e-6,
    )
    rejected = IndependentCertifier(
        lambda _: CertificationChecks(0.0, 0.0, 1.0e-2, 0.0, 0.0),
        backend_identifier="independent-rk4",
        tolerance=1.0e-6,
    )
    assert accepted.certify(result).accepted
    assert not rejected.certify(result).accepted
    result.status = ArcStatus.INFEASIBLE
    assert not accepted.certify(result).accepted


def test_frozen_paper2_matrix() -> None:
    matrix = load_frozen_paper2_matrix(
        Path(__file__).resolve().parents[1] / "benchmarks" / "paper2_matrix.json"
    )
    # P2-F (historical GTOC replay) was appended by the comparative solver campaign
    # specification; the frozen P2-A..P2-E prefix is unchanged.
    assert [family["id"] for family in matrix["families"]][:5] == [
        "P2-A",
        "P2-B",
        "P2-C",
        "P2-D",
        "P2-E",
    ]
    assert [family["id"] for family in matrix["families"]][5:] in ([], ["P2-F"])
