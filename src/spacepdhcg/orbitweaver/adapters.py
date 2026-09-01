"""Concrete G3 trajectory and G5 ownership adapters for OrbitWeaver G7.

The adapters depend only on public lifecycle contracts.  Tests may inject a marked
deterministic fixture, while production supplies a binding around the public G3 device
SCvx C API and the public G5 runtime/workspace API.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from .g7 import (
    ArcRequest,
    ArcResult,
    ArcStatus,
    BoundedScheduler,
    CertificationChecks,
    CertificationRecord,
    IndependentCertifier,
    Ownership,
    ResultRecord,
    RiskMeasure,
    RiskResult,
    RunManifest,
    ScenarioOutcome,
    SchedulerTelemetry,
    TopologyKey,
    aggregate_risk,
    deterministic_top_k,
)


class G3Status(StrEnum):
    CONVERGED = "converged"
    ITERATION_LIMIT = "iteration_limit"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    NUMERICAL_FAILURE = "numerical_failure"
    TIMEOUT = "timeout"
    OOM = "oom"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class G3Solve:
    """Host diagnostics copied from one public G3 SCvx result."""

    status: G3Status
    objective: float = math.inf
    lower_bound: float = 0.0
    duration: float = 0.0
    delta_v: float = 0.0
    propellant: float = 0.0
    final_mass: float = 0.0
    canonical_residual: float = math.inf
    replay_residual: float = math.inf
    path_violation: float = math.inf
    terminal_error: float = math.inf
    uncertainty_violation: float = math.inf
    nonanticipative_controls: tuple[float, ...] = ()
    warm_state: object | None = None
    certifiable_candidate: bool = True
    diagnostic: str = ""


class G3PersistentScvxDriver(Protocol):
    """Public G3 lifecycle exposed by the native binding."""

    @property
    def topology_fingerprint(self) -> int: ...

    @property
    def intervals(self) -> int: ...

    @property
    def scenario_count(self) -> int: ...

    def update_numeric_in_place(self, request: ArcRequest) -> None: ...

    def import_warm_state(self, state: object, source_intervals: int) -> bool: ...

    def solve(self, request: ArcRequest, cancelled: threading.Event) -> G3Solve: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


G3DriverFactory = Callable[[TopologyKey, Ownership], G3PersistentScvxDriver]


@dataclass(frozen=True, slots=True)
class _WarmRecord:
    model_identifier: str
    from_target: int
    to_target: int
    spacecraft: int
    scenario_count: int
    source_intervals: int
    state: object

    def compatible(self, request: ArcRequest) -> bool:
        return (
            self.model_identifier == request.model_identifier
            and self.from_target == request.from_target
            and self.to_target == request.to_target
            and self.spacecraft == request.spacecraft
            and self.scenario_count == request.scenario_count
        )


class G3TrajectoryOracleAdapter:
    """Bounded persistent G3 backend used directly by :class:`BoundedScheduler`.

    A workspace is retained per topology/fidelity/rank/device.  Numerical buffers are
    updated in place for every arc.  Warm states are opaque, bounded, compatibility
    checked and imported only when the target driver explicitly accepts remeshing.
    """

    def __init__(
        self,
        factory: G3DriverFactory,
        *,
        maximum_workspaces: int = 16,
        maximum_warm_tokens: int = 1024,
    ) -> None:
        if maximum_workspaces <= 0 or maximum_warm_tokens <= 0:
            raise ValueError("G3 adapter bounds must be positive")
        self._factory = factory
        self._maximum_workspaces = maximum_workspaces
        self._maximum_warm_tokens = maximum_warm_tokens
        self._workspaces: OrderedDict[
            tuple[TopologyKey, Ownership], G3PersistentScvxDriver
        ] = OrderedDict()
        self._warm: OrderedDict[int, _WarmRecord] = OrderedDict()
        self._next_token = 1
        self.numeric_updates = 0
        self.workspace_creations = 0

    def close(self) -> None:
        for driver in self._workspaces.values():
            driver.close()
        self._workspaces.clear()
        self._warm.clear()

    def _driver(self, topology: TopologyKey, owner: Ownership) -> G3PersistentScvxDriver:
        key = (topology, owner)
        driver = self._workspaces.pop(key, None)
        if driver is not None:
            self._workspaces[key] = driver
            return driver
        driver = self._factory(topology, owner)
        self.workspace_creations += 1
        if (
            driver.topology_fingerprint != topology.topology_fingerprint
            or driver.intervals != topology.intervals
            or driver.scenario_count != topology.scenario_count
        ):
            driver.close()
            raise _TopologyMismatch("G3 driver topology does not match scheduled group")
        while len(self._workspaces) >= self._maximum_workspaces:
            _, evicted = self._workspaces.popitem(last=False)
            evicted.close()
        self._workspaces[key] = driver
        return driver

    def _apply_warm(self, driver: G3PersistentScvxDriver, request: ArcRequest) -> None:
        if request.warm_token is None:
            return
        record = self._warm.get(request.warm_token)
        if record is None or not record.compatible(request):
            raise _WarmIncompatible("warm token is missing or incompatible with the arc")
        if not driver.import_warm_state(record.state, record.source_intervals):
            raise _WarmIncompatible("G3 rejected warm state remeshing for this topology")
        self._warm.move_to_end(request.warm_token)

    def _store_warm(self, request: ArcRequest, solve: G3Solve) -> int | None:
        if solve.warm_state is None:
            return None
        token = self._next_token
        self._next_token += 1
        self._warm[token] = _WarmRecord(
            request.model_identifier,
            request.from_target,
            request.to_target,
            request.spacecraft,
            request.scenario_count,
            request.topology.intervals,
            solve.warm_state,
        )
        while len(self._warm) > self._maximum_warm_tokens:
            self._warm.popitem(last=False)
        return token

    def evaluate(
        self,
        topology: TopologyKey,
        requests: Sequence[ArcRequest],
        owner: Ownership,
        cancelled: threading.Event,
    ) -> Sequence[ArcResult]:
        try:
            driver = self._driver(topology, owner)
        except _TopologyMismatch as error:
            return [
                _failure(item, ArcStatus.TOPOLOGY_MISMATCH, str(error)) for item in requests
            ]
        output: list[ArcResult] = []
        for request in requests:
            if cancelled.is_set():
                driver.cancel()
                output.append(_failure(request, ArcStatus.CANCELLED, "execution cancelled"))
                continue
            try:
                self._apply_warm(driver, request)
                driver.update_numeric_in_place(request)
                self.numeric_updates += 1
                solve = driver.solve(request, cancelled)
                output.append(self._convert(request, solve))
            except _WarmIncompatible as error:
                output.append(
                    _failure(request, ArcStatus.WARM_START_INCOMPATIBLE, str(error))
                )
            except MemoryError as error:
                output.append(_failure(request, ArcStatus.OOM, str(error) or "G3 OOM"))
            except TimeoutError as error:
                output.append(_failure(request, ArcStatus.TIMEOUT, str(error) or "G3 timeout"))
            except NotImplementedError as error:
                output.append(
                    _failure(request, ArcStatus.UNSUPPORTED, str(error) or "unsupported")
                )
            except (ArithmeticError, FloatingPointError) as error:
                output.append(
                    _failure(
                        request,
                        ArcStatus.NUMERICAL_FAILURE,
                        str(error) or "numerical failure",
                    )
                )
            except Exception as error:
                output.append(_failure(request, ArcStatus.BACKEND_FAILURE, str(error)))
        return output

    def _convert(self, request: ArcRequest, solve: G3Solve) -> ArcResult:
        status = {
            G3Status.INFEASIBLE: ArcStatus.INFEASIBLE,
            G3Status.UNSUPPORTED: ArcStatus.UNSUPPORTED,
            G3Status.NUMERICAL_FAILURE: ArcStatus.NUMERICAL_FAILURE,
            G3Status.TIMEOUT: ArcStatus.TIMEOUT,
            G3Status.OOM: ArcStatus.OOM,
            G3Status.CANCELLED: ArcStatus.CANCELLED,
        }.get(solve.status)
        if status is not None:
            return _failure(request, status, solve.diagnostic or solve.status.value)
        if solve.status not in {G3Status.CONVERGED, G3Status.ITERATION_LIMIT}:
            return _failure(request, ArcStatus.BACKEND_FAILURE, "unknown G3 status")
        if solve.status is G3Status.ITERATION_LIMIT and not solve.certifiable_candidate:
            return _failure(
                request,
                ArcStatus.CENSORED,
                solve.diagnostic or "G3 iteration limit without a replayable candidate",
            )
        token = self._store_warm(request, solve)
        return ArcResult(
            request.deterministic_id,
            ArcStatus.FEASIBLE,
            request.fidelity,
            cost=solve.objective,
            lower_bound=max(request.inherited_lower_bound, solve.lower_bound),
            duration=solve.duration,
            delta_v=solve.delta_v,
            propellant=solve.propellant,
            final_mass=solve.final_mass,
            terminal_error=solve.terminal_error,
            path_violation=solve.path_violation,
            uncertainty_violation=solve.uncertainty_violation,
            canonical_residual=solve.canonical_residual,
            replay_residual=solve.replay_residual,
            nonanticipative_controls=solve.nonanticipative_controls,
            warm_token=token,
            diagnostic=solve.diagnostic or solve.status.value,
        )


class _WarmIncompatible(RuntimeError):
    pass


class _TopologyMismatch(RuntimeError):
    pass


def _failure(request: ArcRequest, status: ArcStatus, diagnostic: str) -> ArcResult:
    return ArcResult(
        request.deterministic_id,
        status,
        request.fidelity,
        diagnostic=diagnostic or status.value,
    )


@dataclass(frozen=True, order=True, slots=True)
class G5WorkItem:
    deterministic_id: int
    route_index: int
    trajectory_arc_index: int
    scenario_index: int
    work_units: int

    def validate(self) -> None:
        if min(asdict(self).values()) < 0 or self.work_units == 0:
            raise ValueError("invalid G5 work item")


@dataclass(frozen=True, slots=True)
class G5Partition:
    world_size: int
    devices: tuple[int, ...]
    owners: Mapping[int, Ownership]
    rank_items: tuple[tuple[G5WorkItem, ...], ...]
    predicted_rank_work: tuple[int, ...]
    fingerprint: str

    @classmethod
    def build(cls, items: Iterable[G5WorkItem], devices: Sequence[int]) -> G5Partition:
        ordered = list(items)
        if not ordered or not devices or len(set(devices)) != len(devices):
            raise ValueError("G5 partition requires work and unique rank devices")
        for item in ordered:
            item.validate()
        if len({item.deterministic_id for item in ordered}) != len(ordered):
            raise ValueError("G5 deterministic IDs must be unique")
        ranked = sorted(
            ordered,
            key=lambda item: (
                -item.work_units,
                item.route_index,
                item.trajectory_arc_index,
                item.scenario_index,
                item.deterministic_id,
            ),
        )
        rank_items: list[list[G5WorkItem]] = [[] for _ in devices]
        loads = [0 for _ in devices]
        owners: dict[int, Ownership] = {}
        for item in ranked:
            rank = min(range(len(devices)), key=lambda value: (loads[value], value))
            rank_items[rank].append(item)
            loads[rank] += item.work_units
            owners[item.deterministic_id] = Ownership(rank, devices[rank])
        normalized = tuple(
            tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.route_index,
                        item.trajectory_arc_index,
                        item.scenario_index,
                        item.deterministic_id,
                    ),
                )
            )
            for values in rank_items
        )
        payload = {
            "devices": list(devices),
            "rank_items": [[asdict(item) for item in values] for values in normalized],
            "predicted_rank_work": loads,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            len(devices),
            tuple(devices),
            owners,
            normalized,
            tuple(loads),
            fingerprint,
        )


class G5OwnershipAdapter:
    """Deterministic route/arc/scenario ownership backed by a frozen partition."""

    def __init__(self, partition: G5Partition) -> None:
        self.partition = partition

    def owner(self, request: ArcRequest, batch: int) -> Ownership:
        del batch
        try:
            return self.partition.owners[request.deterministic_id]
        except KeyError as error:
            raise ValueError("request is absent from the frozen G5 partition") from error


class CollectiveStatus(StrEnum):
    HEALTHY = "healthy"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RANK_LOST = "rank_lost"


class G5Collective(Protocol):
    def synchronize_status(
        self, rank_status: Mapping[int, CollectiveStatus]
    ) -> CollectiveStatus: ...

    def record(self, kind: str, elements: int, purpose: str) -> None: ...


class LogicalCollective:
    """CPU-only collective fixture; it is not physical multi-GPU evidence."""

    def __init__(self) -> None:
        self.calls: list[dict[str, int | str]] = []

    def synchronize_status(
        self, rank_status: Mapping[int, CollectiveStatus]
    ) -> CollectiveStatus:
        order = {
            CollectiveStatus.HEALTHY: 0,
            CollectiveStatus.CANCELLED: 1,
            CollectiveStatus.FAILED: 2,
            CollectiveStatus.RANK_LOST: 3,
        }
        return max(rank_status.values(), key=order.__getitem__)

    def record(self, kind: str, elements: int, purpose: str) -> None:
        if not kind or not purpose or elements < 0:
            raise ValueError("invalid collective telemetry")
        self.calls.append({"kind": kind, "elements": elements, "purpose": purpose})


@dataclass(frozen=True, slots=True)
class G5Execution:
    results: tuple[ArcResult, ...]
    global_status: CollectiveStatus
    scheduler: SchedulerTelemetry
    partition_fingerprint: str
    collective_calls: tuple[dict[str, int | str], ...]


class G5DistributedAdapter:
    """Rank-local ownership, propagation and collective-telemetry adapter."""

    def __init__(
        self,
        scheduler_factory: Callable[[G5OwnershipAdapter], BoundedScheduler],
        partition: G5Partition,
        collective: G5Collective,
    ) -> None:
        self.partition = partition
        self.collective = collective
        self.scheduler = scheduler_factory(G5OwnershipAdapter(partition))
        self.cancelled = threading.Event()

    def cancel(self) -> None:
        self.cancelled.set()
        self.scheduler.cancelled.set()

    def execute(self, requests: Iterable[ArcRequest]) -> G5Execution:
        results = self.scheduler.run(requests)
        statuses = {rank: CollectiveStatus.HEALTHY for rank in range(self.partition.world_size)}
        for result in results:
            if result.status is ArcStatus.CANCELLED:
                statuses[result.owner_rank] = CollectiveStatus.CANCELLED
            elif not result.feasible:
                statuses[result.owner_rank] = CollectiveStatus.FAILED
        global_status = self.collective.synchronize_status(statuses)
        self.collective.record(
            "status_max",
            self.partition.world_size,
            "cancellation and failure propagation",
        )
        self.collective.record(
            "risk",
            len(results),
            "expected, worst-case, or CVaR scenario reduction",
        )
        calls = tuple(getattr(self.collective, "calls", ()))
        return G5Execution(
            tuple(results),
            global_status,
            self.scheduler.telemetry,
            self.partition.fingerprint,
            calls,
        )


@dataclass(frozen=True, slots=True)
class G5Restart:
    schema_version: int
    partition_fingerprint: str
    completed_ids: tuple[int, ...]
    rank_warm_tokens: Mapping[int, tuple[int, ...]]

    def validate(self, partition: G5Partition) -> None:
        if (
            self.schema_version != 1
            or self.partition_fingerprint != partition.fingerprint
            or tuple(sorted(self.completed_ids)) != self.completed_ids
            or set(self.rank_warm_tokens) - set(range(partition.world_size))
        ):
            raise ValueError("G5 restart is incompatible with the frozen partition")

    def write(self, path: Path, partition: G5Partition) -> None:
        self.validate(partition)
        payload = {
            "schema_version": self.schema_version,
            "partition_fingerprint": self.partition_fingerprint,
            "completed_ids": list(self.completed_ids),
            "rank_warm_tokens": {
                str(rank): list(tokens)
                for rank, tokens in sorted(self.rank_warm_tokens.items())
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path, partition: G5Partition) -> G5Restart:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {
            "schema_version",
            "partition_fingerprint",
            "completed_ids",
            "rank_warm_tokens",
        }:
            raise ValueError("G5 restart schema drift")
        result = cls(
            payload["schema_version"],
            payload["partition_fingerprint"],
            tuple(payload["completed_ids"]),
            {
                int(rank): tuple(tokens)
                for rank, tokens in payload["rank_warm_tokens"].items()
            },
        )
        result.validate(partition)
        return result


@dataclass(frozen=True, slots=True)
class RouteDefinition:
    identifier: int
    spacecraft: int
    targets: tuple[int, ...]
    arc_keys: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class CertifiedRouteColumn:
    definition: RouteDefinition
    cost: float
    lower_bound: float
    propellant: float
    certifications: tuple[CertificationRecord, ...]

    @property
    def certified(self) -> bool:
        return bool(self.certifications) and all(item.accepted for item in self.certifications)


@dataclass(frozen=True, slots=True)
class RouteMasterResult:
    incumbent: tuple[CertifiedRouteColumn, ...]
    cost: float
    lower_bound: float
    certified: bool


def build_route_columns(
    definitions: Iterable[RouteDefinition],
    arc_results: Mapping[tuple[int, int], ArcResult],
    certifier: IndependentCertifier,
) -> list[CertifiedRouteColumn]:
    columns: list[CertifiedRouteColumn] = []
    for definition in sorted(definitions, key=lambda item: item.identifier):
        arcs = [arc_results[key] for key in definition.arc_keys]
        if not arcs or not all(item.feasible for item in arcs):
            continue
        columns.append(
            CertifiedRouteColumn(
                definition,
                sum(item.cost for item in arcs),
                sum(item.lower_bound for item in arcs),
                sum(item.propellant for item in arcs),
                tuple(certifier.certify(item) for item in arcs),
            )
        )
    return columns


def solve_certified_route_master(
    columns: Iterable[CertifiedRouteColumn],
    required_targets: Iterable[int],
) -> RouteMasterResult:
    """Exact bounded route master; uncertified combinations cannot become incumbents."""

    candidates = sorted(columns, key=lambda item: item.definition.identifier)
    required = frozenset(required_targets)
    best: tuple[CertifiedRouteColumn, ...] = ()
    best_cost = math.inf
    best_bound = math.inf

    def search(
        index: int,
        selected: tuple[CertifiedRouteColumn, ...],
        covered: frozenset[int],
        spacecraft: frozenset[int],
    ) -> None:
        nonlocal best, best_cost, best_bound
        if covered == required:
            cost = sum(item.cost for item in selected)
            if all(item.certified for item in selected) and cost < best_cost:
                best = selected
                best_cost = cost
                best_bound = sum(item.lower_bound for item in selected)
            return
        if index == len(candidates):
            return
        search(index + 1, selected, covered, spacecraft)
        item = candidates[index]
        targets = frozenset(item.definition.targets)
        if (
            item.certified
            and item.definition.spacecraft not in spacecraft
            and not covered.intersection(targets)
            and targets.issubset(required)
        ):
            search(
                index + 1,
                (*selected, item),
                covered.union(targets),
                spacecraft.union({item.definition.spacecraft}),
            )

    search(0, (), frozenset(), frozenset())
    return RouteMasterResult(best, best_cost, best_bound, bool(best))


@dataclass(frozen=True, slots=True)
class AdapterFlowResult:
    coarse: tuple[ArcResult, ...]
    refined: tuple[ArcResult, ...]
    scenarios: G5Execution
    risk_by_arc: Mapping[tuple[int, int], RiskResult]
    columns: tuple[CertifiedRouteColumn, ...]
    master: RouteMasterResult


class OrbitWeaverAdapterFlow:
    """Bounded coarse -> refined -> scenario -> master -> certification flow."""

    def __init__(
        self,
        coarse_scheduler: BoundedScheduler,
        refined_scheduler: BoundedScheduler,
        distributed: G5DistributedAdapter,
        certifier: IndependentCertifier,
        *,
        top_k: int,
    ) -> None:
        if top_k <= 0:
            raise ValueError("flow top-K must be positive")
        self.coarse_scheduler = coarse_scheduler
        self.refined_scheduler = refined_scheduler
        self.distributed = distributed
        self.certifier = certifier
        self.top_k = top_k

    def run(
        self,
        coarse_requests: Iterable[ArcRequest],
        refine: Callable[[ArcRequest, ArcResult], ArcRequest],
        scenarios: Callable[[ArcRequest, ArcResult], Sequence[tuple[ArcRequest, float]]],
        routes: Iterable[RouteDefinition],
        required_targets: Iterable[int],
        risk_measure: RiskMeasure,
        *,
        cvar_alpha: float = 0.9,
    ) -> AdapterFlowResult:
        coarse_input = list(coarse_requests)
        coarse = self.coarse_scheduler.run(coarse_input)
        selected = deterministic_top_k(coarse, self.top_k, retain_failures=False)
        by_id = {item.deterministic_id: item for item in coarse_input}
        refined_input = [refine(by_id[item.deterministic_id], item) for item in selected]
        refined = self.refined_scheduler.run(refined_input)
        scenario_input: list[ArcRequest] = []
        probabilities: dict[int, float] = {}
        parents: dict[int, tuple[int, int]] = {}
        refined_by_id = {item.deterministic_id: item for item in refined_input}
        for result in refined:
            if not result.feasible:
                continue
            parent = refined_by_id[result.deterministic_id]
            for request, probability in scenarios(parent, result):
                scenario_input.append(request)
                probabilities[request.deterministic_id] = probability
                parents[request.deterministic_id] = (
                    request.route_index,
                    request.trajectory_arc_index,
                )
        execution = self.distributed.execute(scenario_input)
        grouped: dict[tuple[int, int], list[ScenarioOutcome]] = {}
        result_requests = {item.deterministic_id: item for item in scenario_input}
        for result in execution.results:
            request = result_requests[result.deterministic_id]
            grouped.setdefault(parents[result.deterministic_id], []).append(
                ScenarioOutcome(
                    request.scenario_index,
                    probabilities[result.deterministic_id],
                    result.cost,
                    result.lower_bound,
                    result.nonanticipative_controls,
                    result.status,
                )
            )
        risk_by_arc = {
            key: aggregate_risk(values, risk_measure, cvar_alpha=cvar_alpha)
            for key, values in grouped.items()
        }
        aggregate_arcs: dict[tuple[int, int], ArcResult] = {}
        for key, risk in risk_by_arc.items():
            members = [
                item
                for item in execution.results
                if parents[item.deterministic_id] == key
            ]
            if not risk.feasible:
                continue
            template = members[0]
            aggregate_arcs[key] = replace(
                template,
                cost=risk.objective,
                lower_bound=risk.lower_bound,
                terminal_error=max(item.terminal_error for item in members),
                path_violation=max(item.path_violation for item in members),
                uncertainty_violation=max(item.uncertainty_violation for item in members),
                canonical_residual=max(item.canonical_residual for item in members),
                replay_residual=max(item.replay_residual for item in members),
                propellant=max(item.propellant for item in members),
                final_mass=min(item.final_mass for item in members),
            )
        columns = build_route_columns(routes, aggregate_arcs, self.certifier)
        master = solve_certified_route_master(columns, required_targets)
        return AdapterFlowResult(
            tuple(coarse),
            tuple(refined),
            execution,
            risk_by_arc,
            tuple(columns),
            master,
        )


def flow_result_record(
    flow: AdapterFlowResult,
    manifest: RunManifest,
    repeat_index: int,
) -> ResultRecord:
    """Create and immediately validate the strict bf9d10 result contract."""

    failures = tuple(
        {
            "status": _record_failure_status(item.status),
            "diagnostic": item.diagnostic or item.status.value,
            "deterministic_id": item.deterministic_id,
        }
        for item in flow.scenarios.results
        if not item.feasible
    )
    checks = [
        certification.checks
        for column in flow.master.incumbent
        for certification in column.certifications
        if certification.checks is not None
    ]
    certification = None
    if checks:
        certification = CertificationRecord(
            flow.master.certified,
            CertificationChecks(
                max(item.dynamics_defect for item in checks),
                max(item.path_violation for item in checks),
                max(item.terminal_error for item in checks),
                max(item.uncertainty_violation for item in checks),
                max(item.integration_error for item in checks),
            ),
            "independent-route-certification",
            "all selected route arcs independently certified"
            if flow.master.certified
            else "selected route certification rejected",
        )
    status = "converged"
    if not flow.master.certified:
        status = (
            "failed"
            if any(item["status"] == "failed" for item in failures)
            else failures[0]["status"]
            if failures
            else "censored"
        )
    if not flow.master.certified and not failures:
        failures = (
            {
                "status": "censored",
                "diagnostic": "no independently certified route incumbent",
                "deterministic_id": None,
            },
        )
    telemetry = asdict(flow.scenarios.scheduler)
    record = ResultRecord(
        schema_version=1,
        run_id=manifest.run_id,
        manifest_sha256=manifest.sha256(),
        paper2_matrix_sha256=manifest.paper2_matrix_sha256,
        seed=manifest.seed,
        repeat_index=repeat_index,
        status=status,
        incumbent=flow.master.cost if flow.master.certified else None,
        lower_bound=flow.master.lower_bound if flow.master.certified else None,
        optimality_gap=(
            max(0.0, flow.master.cost - flow.master.lower_bound)
            if flow.master.certified
            else None
        ),
        certified=flow.master.certified,
        certification=certification,
        telemetry=telemetry,
        failures=failures,
    )
    record.validate(manifest)
    return record


def _record_failure_status(status: ArcStatus) -> str:
    return {
        ArcStatus.INFEASIBLE: "infeasible",
        ArcStatus.UNSUPPORTED: "unsupported",
        ArcStatus.TIMEOUT: "timeout",
        ArcStatus.OOM: "oom",
        ArcStatus.CANCELLED: "cancelled",
        ArcStatus.CENSORED: "censored",
    }.get(status, "failed")
