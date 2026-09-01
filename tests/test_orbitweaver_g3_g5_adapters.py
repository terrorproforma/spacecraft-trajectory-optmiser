from __future__ import annotations

import threading
from pathlib import Path

import pytest

from spacepdhcg.orbitweaver import (
    PAPER2_MATRIX_SHA256,
    ArcFidelity,
    ArcRequest,
    ArcStatus,
    BoundedScheduler,
    CertificationChecks,
    CollectiveStatus,
    G3Solve,
    G3Status,
    G3TrajectoryOracleAdapter,
    G5DistributedAdapter,
    G5Partition,
    G5Restart,
    G5WorkItem,
    IndependentCertifier,
    LogicalCollective,
    OrbitWeaverAdapterFlow,
    Ownership,
    RiskMeasure,
    RouteDefinition,
    RunManifest,
    SchedulerConfig,
    TopologyKey,
    flow_result_record,
)


class DeterministicG3Fixture:
    """Clearly marked unit fixture; never admissible as solver evidence."""

    def __init__(self, topology: TopologyKey, owner: Ownership) -> None:
        self.topology_fingerprint = topology.topology_fingerprint
        self.intervals = topology.intervals
        self.scenario_count = topology.scenario_count
        self.owner = owner
        self.updates: list[int] = []
        self.imports: list[tuple[object, int]] = []
        self.closed = False

    def update_numeric_in_place(self, request: ArcRequest) -> None:
        self.updates.append(request.deterministic_id)

    def import_warm_state(self, state: object, source_intervals: int) -> bool:
        self.imports.append((state, source_intervals))
        return source_intervals <= self.intervals

    def solve(self, request: ArcRequest, cancelled: threading.Event) -> G3Solve:
        if cancelled.is_set():
            return G3Solve(G3Status.CANCELLED)
        status = {
            "infeasible": G3Status.INFEASIBLE,
            "unsupported": G3Status.UNSUPPORTED,
            "numerical": G3Status.NUMERICAL_FAILURE,
            "timeout": G3Status.TIMEOUT,
            "oom": G3Status.OOM,
            "censored": G3Status.ITERATION_LIMIT,
        }.get(request.model_identifier, G3Status.CONVERGED)
        usable = request.model_identifier != "censored"
        cost = 10.0 + request.route_index + request.scenario_index
        controls = (float(request.route_index),)
        return G3Solve(
            status,
            objective=cost,
            lower_bound=cost - 1.0,
            duration=request.arrival_epoch - request.departure_epoch,
            delta_v=2.0,
            propellant=5.0,
            final_mass=request.initial_mass - 5.0,
            canonical_residual=2.0e-8,
            replay_residual=3.0e-8,
            path_violation=4.0e-8,
            terminal_error=5.0e-8,
            uncertainty_violation=6.0e-8,
            nonanticipative_controls=controls,
            warm_state={"request": request.deterministic_id},
            certifiable_candidate=usable,
            diagnostic=f"deterministic fixture: {status.value}",
        )

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class FixtureFactory:
    def __init__(self) -> None:
        self.drivers: list[DeterministicG3Fixture] = []
        self.mismatch = False

    def __call__(
        self, topology: TopologyKey, owner: Ownership
    ) -> DeterministicG3Fixture:
        driver = DeterministicG3Fixture(topology, owner)
        if self.mismatch:
            driver.topology_fingerprint += 1
        self.drivers.append(driver)
        return driver


def arc(
    identifier: int,
    fidelity: ArcFidelity,
    topology: int,
    intervals: int,
    *,
    route: int = 0,
    trajectory_arc: int = 0,
    scenario: int = 0,
    scenario_count: int = 1,
    model: str = "fixture",
    warm: int | None = None,
) -> ArcRequest:
    return ArcRequest(
        identifier,
        route,
        route + 10,
        0.0,
        10.0,
        100.0,
        route,
        scenario_count,
        fidelity,
        1.0e-5,
        model,
        TopologyKey(topology, fidelity, intervals, scenario_count),
        warm_token=warm,
        route_index=route,
        trajectory_arc_index=trajectory_arc,
        scenario_index=scenario,
    )


def scheduler(backend: object, ownership: object | None = None) -> BoundedScheduler:
    return BoundedScheduler(
        backend,
        ownership=ownership,
        config=SchedulerConfig(2, 16, 64, 128),
    )


def test_g3_reuses_workspace_updates_values_and_warm_state() -> None:
    factory = FixtureFactory()
    adapter = G3TrajectoryOracleAdapter(factory, maximum_workspaces=2, maximum_warm_tokens=2)
    coarse = scheduler(adapter).run(
        [
            arc(2, ArcFidelity.COARSE_CONVEX, 11, 8),
            arc(1, ArcFidelity.COARSE_CONVEX, 11, 8),
        ]
    )
    assert adapter.workspace_creations == 1
    assert adapter.numeric_updates == 2
    assert factory.drivers[0].updates == [1, 2]
    refined = scheduler(adapter).run(
        [
            arc(
                3,
                ArcFidelity.REFINED_SCVX,
                22,
                16,
                warm=coarse[0].warm_token,
            )
        ]
    )
    assert refined[0].status is ArcStatus.FEASIBLE
    assert factory.drivers[1].imports == [({"request": 1}, 8)]
    assert refined[0].canonical_residual == 2.0e-8
    assert refined[0].replay_residual == 3.0e-8
    assert refined[0].path_violation == 4.0e-8
    assert refined[0].terminal_error == 5.0e-8


def test_g3_explicit_failure_and_incompatibility_classification() -> None:
    factory = FixtureFactory()
    adapter = G3TrajectoryOracleAdapter(factory)
    token = scheduler(adapter).run(
        [arc(1, ArcFidelity.COARSE_CONVEX, 11, 8)]
    )[0].warm_token
    incompatible = arc(
        2,
        ArcFidelity.REFINED_SCVX,
        22,
        16,
        route=9,
        warm=token,
    )
    results = scheduler(adapter).run(
        [
            incompatible,
            arc(3, ArcFidelity.REFINED_SCVX, 22, 16, model="timeout"),
            arc(4, ArcFidelity.REFINED_SCVX, 22, 16, model="oom"),
            arc(5, ArcFidelity.REFINED_SCVX, 22, 16, model="unsupported"),
            arc(6, ArcFidelity.REFINED_SCVX, 22, 16, model="infeasible"),
            arc(7, ArcFidelity.REFINED_SCVX, 22, 16, model="numerical"),
            arc(8, ArcFidelity.REFINED_SCVX, 22, 16, model="censored"),
        ]
    )
    assert [item.status for item in results] == [
        ArcStatus.WARM_START_INCOMPATIBLE,
        ArcStatus.TIMEOUT,
        ArcStatus.OOM,
        ArcStatus.UNSUPPORTED,
        ArcStatus.INFEASIBLE,
        ArcStatus.NUMERICAL_FAILURE,
        ArcStatus.CENSORED,
    ]
    mismatched = FixtureFactory()
    mismatched.mismatch = True
    result = scheduler(G3TrajectoryOracleAdapter(mismatched)).run(
        [arc(9, ArcFidelity.COARSE_CONVEX, 99, 8)]
    )
    assert result[0].status is ArcStatus.TOPOLOGY_MISMATCH


def test_g5_partition_restart_and_backpressure(tmp_path: Path) -> None:
    work = [
        G5WorkItem(100 + index, index // 4, index // 2, index % 2, index + 1)
        for index in range(8)
    ]
    first = G5Partition.build(work, (2, 5, 7))
    second = G5Partition.build(reversed(work), (2, 5, 7))
    assert first.fingerprint == second.fingerprint
    assert first.owners == second.owners
    restart = G5Restart(
        1,
        first.fingerprint,
        (100, 101),
        {0: (11,), 1: (12,), 2: ()},
    )
    path = tmp_path / "rank-restart.json"
    restart.write(path, first)
    assert G5Restart.read(path, first) == restart
    incompatible = G5Partition.build(work, (2, 5))
    with pytest.raises(ValueError):
        G5Restart.read(path, incompatible)

    adapter = G3TrajectoryOracleAdapter(FixtureFactory())
    too_small = BoundedScheduler(adapter, config=SchedulerConfig(1, 1, 1, 1))
    with pytest.raises(BufferError):
        too_small.run(
            [
                arc(1, ArcFidelity.COARSE_CONVEX, 11, 8),
                arc(2, ArcFidelity.COARSE_CONVEX, 11, 8),
            ]
        )


def test_full_coarse_refined_scenario_master_certification_flow() -> None:
    factory = FixtureFactory()
    backend = G3TrajectoryOracleAdapter(factory)
    scenario_ids = [1000, 1001, 1010, 1011]
    partition = G5Partition.build(
        [
            G5WorkItem(identifier, identifier // 10 % 2, 0, identifier % 10, 10)
            for identifier in scenario_ids
        ],
        (0, 1),
    )
    collective = LogicalCollective()
    distributed = G5DistributedAdapter(
        lambda ownership: scheduler(backend, ownership),
        partition,
        collective,
    )
    certifier = IndependentCertifier(
        lambda result: CertificationChecks(
            result.replay_residual,
            result.path_violation,
            result.terminal_error,
            result.uncertainty_violation,
            result.canonical_residual,
        ),
        backend_identifier="independent-cpu-replay-fixture",
        tolerance=1.0e-6,
    )
    flow = OrbitWeaverAdapterFlow(
        scheduler(backend),
        scheduler(backend),
        distributed,
        certifier,
        top_k=2,
    )
    coarse = [
        arc(0, ArcFidelity.COARSE_CONVEX, 11, 8, route=0),
        arc(1, ArcFidelity.COARSE_CONVEX, 11, 8, route=1),
    ]

    def refine(request: ArcRequest, result: object) -> ArcRequest:
        return arc(
            request.deterministic_id,
            ArcFidelity.REFINED_SCVX,
            22,
            16,
            route=request.route_index,
            warm=result.warm_token,
        )

    def scenarios(
        request: ArcRequest, result: object
    ) -> list[tuple[ArcRequest, float]]:
        return [
            (
                arc(
                    1000 + request.route_index * 10 + index,
                    ArcFidelity.ROBUST_SCVX,
                    33,
                    24,
                    route=request.route_index,
                    scenario=index,
                    scenario_count=2,
                ),
                0.5,
            )
            for index in range(2)
        ]

    routes = [
        RouteDefinition(10, 0, (0,), ((0, 0),)),
        RouteDefinition(20, 1, (1,), ((1, 0),)),
    ]
    result = flow.run(
        coarse,
        refine,
        scenarios,
        routes,
        (0, 1),
        RiskMeasure.CVAR,
        cvar_alpha=0.5,
    )
    assert len(result.coarse) == 2
    assert len(result.refined) == 2
    assert len(result.scenarios.results) == 4
    assert result.scenarios.global_status is CollectiveStatus.HEALTHY
    assert result.scenarios.scheduler.batches == 2
    assert result.master.certified
    assert len(result.master.incumbent) == 2
    assert result.master.lower_bound <= result.master.cost
    assert all(column.certified for column in result.columns)
    assert {call["kind"] for call in result.scenarios.collective_calls} == {
        "status_max",
        "risk",
    }
    manifest = RunManifest(
        1,
        "adapter-flow",
        "1" * 40,
        42,
        "g3-public-device-scvx",
        "logical_rank_mock",
        (0, 1),
        "a" * 64,
        PAPER2_MATRIX_SHA256,
        1,
        {"python": "3.12", "compiler": "gcc", "cmake": "cmake", "cuda": "12.8"},
        {"os": "Linux", "cpu": "fixture", "gpus": []},
        "cpu_correctness_tested",
    )
    record = flow_result_record(result, manifest, 0)
    assert record.certified and record.status == "converged"
    record.validate(manifest)


def test_nonanticipativity_and_failed_rank_block_incumbent() -> None:
    factory = FixtureFactory()
    backend = G3TrajectoryOracleAdapter(factory)
    work = [G5WorkItem(1000, 0, 0, 0, 1), G5WorkItem(1001, 0, 0, 1, 1)]
    partition = G5Partition.build(work, (0, 1))
    distributed = G5DistributedAdapter(
        lambda ownership: scheduler(backend, ownership),
        partition,
        LogicalCollective(),
    )
    failed = distributed.execute(
        [
            arc(
                1000,
                ArcFidelity.ROBUST_SCVX,
                33,
                24,
                scenario=0,
                scenario_count=2,
            ),
            arc(
                1001,
                ArcFidelity.ROBUST_SCVX,
                33,
                24,
                scenario=1,
                scenario_count=2,
                model="numerical",
            ),
        ]
    )
    assert failed.global_status is CollectiveStatus.FAILED
