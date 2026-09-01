"""Bounded, deterministic OrbitWeaver G7 orchestration contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

PAPER2_MATRIX_SHA256 = "78c4e33e4aabcd85d63ba3f1e03aa2214b3ab207e680bcaaf347516802b2f6a2"


class ArcFidelity(StrEnum):
    ANALYTICAL = "analytical"
    LAMBERT = "lambert"
    COARSE_CONVEX = "coarse_convex"
    REFINED_SCVX = "refined_scvx"
    ROBUST_SCVX = "robust_scvx"
    CERTIFIED = "certified"


class ArcStatus(StrEnum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNSUPPORTED = "unsupported"
    INVALID_INPUT = "invalid_input"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_FAILURE = "backend_failure"
    CANCELLED = "cancelled"
    CERTIFICATION_REJECTED = "certification_rejected"


class RiskMeasure(StrEnum):
    EXPECTED = "expected"
    WORST_CASE = "worst_case"
    CVAR = "cvar"


@dataclass(frozen=True, order=True, slots=True)
class TopologyKey:
    topology_fingerprint: int
    fidelity: ArcFidelity
    intervals: int
    scenario_count: int

    def validate(self) -> None:
        if self.topology_fingerprint <= 0 or self.intervals < 2 or self.scenario_count < 1:
            raise ValueError("invalid topology/fidelity group")


@dataclass(frozen=True, slots=True)
class ArcRequest:
    deterministic_id: int
    from_target: int
    to_target: int
    departure_epoch: float
    arrival_epoch: float
    initial_mass: float
    spacecraft: int
    scenario_count: int
    fidelity: ArcFidelity
    requested_tolerance: float
    model_identifier: str
    topology: TopologyKey
    inherited_lower_bound: float = 0.0
    warm_token: int | None = None
    route_index: int = 0
    trajectory_arc_index: int = 0
    scenario_index: int = 0
    time_node_index: int = 0

    def validate(self) -> None:
        self.topology.validate()
        valid_numbers = (
            math.isfinite(self.departure_epoch)
            and math.isfinite(self.arrival_epoch)
            and self.arrival_epoch > self.departure_epoch
            and math.isfinite(self.initial_mass)
            and self.initial_mass > 0.0
            and math.isfinite(self.requested_tolerance)
            and self.requested_tolerance > 0.0
            and math.isfinite(self.inherited_lower_bound)
            and self.inherited_lower_bound >= 0.0
        )
        if (
            self.deterministic_id < 0
            or self.from_target == self.to_target
            or not self.model_identifier
            or not valid_numbers
            or self.topology.fidelity is not self.fidelity
            or self.topology.scenario_count != self.scenario_count
        ):
            raise ValueError("invalid G7 arc request")


@dataclass(slots=True)
class ArcResult:
    deterministic_id: int
    status: ArcStatus
    fidelity: ArcFidelity
    cost: float = math.inf
    lower_bound: float = 0.0
    duration: float = 0.0
    delta_v: float = 0.0
    propellant: float = 0.0
    final_mass: float = 0.0
    terminal_error: float = math.inf
    path_violation: float = math.inf
    uncertainty_violation: float = math.inf
    warm_token: int | None = None
    owner_rank: int = 0
    owner_device: int = 0
    batch_sequence: int = 0
    diagnostic: str = ""

    @property
    def feasible(self) -> bool:
        return self.status is ArcStatus.FEASIBLE

    def validate(self, request: ArcRequest) -> None:
        if self.deterministic_id != request.deterministic_id:
            raise ValueError("backend changed deterministic identity")
        if not self.feasible:
            return
        values = (
            self.cost,
            self.lower_bound,
            self.duration,
            self.delta_v,
            self.propellant,
            self.final_mass,
            self.terminal_error,
            self.path_violation,
            self.uncertainty_violation,
        )
        if (
            not all(math.isfinite(value) and value >= 0.0 for value in values)
            or self.lower_bound > self.cost
            or not math.isclose(
                self.propellant + self.final_mass,
                request.initial_mass,
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
        ):
            raise ValueError("feasible arc result is invalid")


@dataclass(frozen=True, slots=True)
class Ownership:
    rank: int
    device: int


class OwnershipPolicy(Protocol):
    def owner(self, request: ArcRequest, batch: int) -> Ownership: ...


@dataclass(frozen=True, slots=True)
class SingleGpuOwnership:
    device: int = 0

    def owner(self, request: ArcRequest, batch: int) -> Ownership:
        del request, batch
        return Ownership(0, self.device)


@dataclass(frozen=True, slots=True)
class LogicalRankOwnership:
    """Logical rank mock only; never physical scaling evidence."""

    devices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.devices:
            raise ValueError("logical ownership needs devices")

    def owner(self, request: ArcRequest, batch: int) -> Ownership:
        rank = (request.deterministic_id + batch) % len(self.devices)
        return Ownership(rank, self.devices[rank])


class BatchBackend(Protocol):
    def evaluate(
        self,
        topology: TopologyKey,
        requests: Sequence[ArcRequest],
        owner: Ownership,
        cancelled: threading.Event,
    ) -> Sequence[ArcResult]: ...


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    maximum_batch_size: int = 128
    maximum_buffered_arcs: int = 1024
    bytes_per_arc_budget: int = 1 << 20
    maximum_workspace_bytes: int = 1 << 30

    def validate(self) -> None:
        if (
            min(
                self.maximum_batch_size,
                self.maximum_buffered_arcs,
                self.bytes_per_arc_budget,
                self.maximum_workspace_bytes,
            )
            <= 0
            or self.maximum_batch_size > self.maximum_buffered_arcs
            or self.maximum_batch_size * self.bytes_per_arc_budget > self.maximum_workspace_bytes
        ):
            raise ValueError("invalid fixed-memory scheduler limits")


@dataclass(slots=True)
class SchedulerTelemetry:
    submitted: int = 0
    completed: int = 0
    feasible: int = 0
    failed: int = 0
    cancelled: int = 0
    batches: int = 0
    maximum_observed_batch: int = 0
    estimated_peak_buffer_bytes: int = 0
    group_batches: dict[str, int] = field(default_factory=dict)
    ownership_batches: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdaptiveFidelity:
    coarse_intervals: int = 16
    refined_intervals: int = 64
    robust_intervals: int = 96
    refinement_gap: float = 1.0e-2
    robust_gap: float = 1.0e-3

    def promote(self, result: ArcResult) -> tuple[ArcFidelity, int]:
        if not result.feasible:
            return result.fidelity, self.coarse_intervals
        gap = max(0.0, result.cost - result.lower_bound)
        if result.fidelity is ArcFidelity.COARSE_CONVEX and gap > self.refinement_gap:
            return ArcFidelity.REFINED_SCVX, self.refined_intervals
        if result.fidelity is ArcFidelity.REFINED_SCVX and gap > self.robust_gap:
            return ArcFidelity.ROBUST_SCVX, self.robust_intervals
        return result.fidelity, self.refined_intervals


class BoundedScheduler:
    def __init__(
        self,
        backend: BatchBackend,
        *,
        ownership: OwnershipPolicy | None = None,
        config: SchedulerConfig | None = None,
    ) -> None:
        ownership = ownership or SingleGpuOwnership()
        config = config or SchedulerConfig()
        config.validate()
        self.backend = backend
        self.ownership = ownership
        self.config = config
        self.cancelled = threading.Event()
        self.telemetry = SchedulerTelemetry()

    def run(self, requests: Iterable[ArcRequest]) -> list[ArcResult]:
        pending = list(requests)
        if len(pending) > self.config.maximum_buffered_arcs:
            raise BufferError("G7 scheduler backpressure limit exceeded")
        for request in pending:
            request.validate()
        pending.sort(key=lambda item: (item.topology, item.deterministic_id))
        self.telemetry = SchedulerTelemetry(submitted=len(pending))
        output: list[ArcResult] = []
        cursor = 0
        sequence = 0
        while cursor < len(pending):
            topology = pending[cursor].topology
            group_end = cursor
            while group_end < len(pending) and pending[group_end].topology == topology:
                group_end += 1
            stop = min(cursor + self.config.maximum_batch_size, group_end)
            batch = pending[cursor:stop]
            owner = self.ownership.owner(batch[0], sequence)
            evaluated = self._evaluate(topology, batch, owner)
            for request, result in zip(batch, evaluated, strict=True):
                result.owner_rank = owner.rank
                result.owner_device = owner.device
                result.batch_sequence = sequence
                try:
                    result.validate(request)
                except ValueError as error:
                    result = ArcResult(
                        request.deterministic_id,
                        ArcStatus.BACKEND_FAILURE,
                        request.fidelity,
                        owner_rank=owner.rank,
                        owner_device=owner.device,
                        batch_sequence=sequence,
                        diagnostic=str(error),
                    )
                self.telemetry.completed += 1
                if result.feasible:
                    self.telemetry.feasible += 1
                elif result.status is ArcStatus.CANCELLED:
                    self.telemetry.cancelled += 1
                else:
                    self.telemetry.failed += 1
                output.append(result)
            group_key = (
                f"{topology.topology_fingerprint}:{topology.fidelity}:"
                f"{topology.intervals}:{topology.scenario_count}"
            )
            owner_key = f"{owner.rank}:{owner.device}"
            self.telemetry.group_batches[group_key] = (
                self.telemetry.group_batches.get(group_key, 0) + 1
            )
            self.telemetry.ownership_batches[owner_key] = (
                self.telemetry.ownership_batches.get(owner_key, 0) + 1
            )
            self.telemetry.batches += 1
            self.telemetry.maximum_observed_batch = max(
                self.telemetry.maximum_observed_batch, len(batch)
            )
            self.telemetry.estimated_peak_buffer_bytes = max(
                self.telemetry.estimated_peak_buffer_bytes,
                len(batch) * self.config.bytes_per_arc_budget,
            )
            cursor = stop
            sequence += 1
        return sorted(output, key=lambda item: item.deterministic_id)

    def _evaluate(
        self,
        topology: TopologyKey,
        batch: Sequence[ArcRequest],
        owner: Ownership,
    ) -> list[ArcResult]:
        if self.cancelled.is_set():
            return [
                ArcResult(item.deterministic_id, ArcStatus.CANCELLED, item.fidelity)
                for item in batch
            ]
        try:
            result = list(self.backend.evaluate(topology, batch, owner, self.cancelled))
            if len(result) != len(batch):
                raise RuntimeError("backend returned mismatched batch length")
            return result
        except Exception as error:
            return [
                ArcResult(
                    item.deterministic_id,
                    ArcStatus.BACKEND_FAILURE,
                    item.fidelity,
                    diagnostic=str(error),
                )
                for item in batch
            ]


def deterministic_top_k(
    results: Iterable[ArcResult],
    count: int,
    *,
    retain_failures: bool = True,
) -> list[ArcResult]:
    if count <= 0:
        raise ValueError("top-K count must be positive")
    ordered = sorted(
        results,
        key=lambda item: (
            not item.feasible,
            item.cost,
            item.lower_bound,
            item.deterministic_id,
        ),
    )
    if not retain_failures:
        ordered = [item for item in ordered if item.feasible]
    return ordered[:count]


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    scenario: int
    probability: float
    cost: float
    lower_bound: float
    nonanticipative_controls: tuple[float, ...]
    status: ArcStatus = ArcStatus.FEASIBLE


@dataclass(frozen=True, slots=True)
class RiskResult:
    feasible: bool
    objective: float
    lower_bound: float
    nonanticipativity_violation: float
    cvar_threshold: float | None = None


def aggregate_risk(
    outcomes: Iterable[ScenarioOutcome],
    measure: RiskMeasure,
    *,
    cvar_alpha: float = 0.9,
    nonanticipativity_tolerance: float = 1.0e-10,
) -> RiskResult:
    scenarios = sorted(outcomes, key=lambda item: item.scenario)
    if not scenarios or (measure is RiskMeasure.CVAR and not 0.0 < cvar_alpha < 1.0):
        raise ValueError("invalid risk configuration")
    if any(
        item.status is not ArcStatus.FEASIBLE
        or not math.isfinite(item.probability)
        or item.probability < 0.0
        or not math.isfinite(item.cost)
        or item.lower_bound > item.cost
        for item in scenarios
    ):
        return RiskResult(False, math.inf, math.inf, math.inf)
    if not math.isclose(sum(item.probability for item in scenarios), 1.0, abs_tol=1.0e-12):
        raise ValueError("scenario probabilities must sum to one")
    reference = scenarios[0].nonanticipative_controls
    if any(len(item.nonanticipative_controls) != len(reference) for item in scenarios):
        raise ValueError("non-anticipative prefixes disagree in size")
    violation = max(
        (
            abs(value - reference[index])
            for item in scenarios
            for index, value in enumerate(item.nonanticipative_controls)
        ),
        default=0.0,
    )
    if violation > nonanticipativity_tolerance:
        return RiskResult(False, math.inf, math.inf, violation)
    expected = sum(item.probability * item.cost for item in scenarios)
    expected_bound = sum(item.probability * item.lower_bound for item in scenarios)
    if measure is RiskMeasure.EXPECTED:
        return RiskResult(True, expected, expected_bound, violation)
    if measure is RiskMeasure.WORST_CASE:
        return RiskResult(
            True,
            max(item.cost for item in scenarios),
            max(item.lower_bound for item in scenarios),
            violation,
        )
    ordered = sorted(scenarios, key=lambda item: (item.cost, item.scenario))
    cumulative = 0.0
    threshold = ordered[-1].cost
    for item in ordered:
        cumulative += item.probability
        if cumulative >= cvar_alpha:
            threshold = item.cost
            break
    objective = threshold + sum(
        item.probability * max(0.0, item.cost - threshold) for item in scenarios
    ) / (1.0 - cvar_alpha)
    return RiskResult(True, objective, min(objective, expected_bound), violation, threshold)


@dataclass(frozen=True, slots=True)
class CertificationChecks:
    dynamics_defect: float
    path_violation: float
    terminal_error: float
    uncertainty_violation: float
    integration_error: float

    @property
    def maximum(self) -> float:
        return max(asdict(self).values())


@dataclass(frozen=True, slots=True)
class CertificationRecord:
    accepted: bool
    checks: CertificationChecks | None
    backend_identifier: str
    diagnostic: str


class IndependentCertifier:
    def __init__(self, callback: Any, *, backend_identifier: str, tolerance: float):
        if not backend_identifier or not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("invalid independent certifier")
        self.callback = callback
        self.backend_identifier = backend_identifier
        self.tolerance = tolerance

    def certify(self, result: ArcResult) -> CertificationRecord:
        if not result.feasible:
            return CertificationRecord(
                False, None, self.backend_identifier, "optimizer status is not certification"
            )
        checks = self.callback(result)
        accepted = (
            all(math.isfinite(value) and value >= 0.0 for value in asdict(checks).values())
            and checks.maximum <= self.tolerance
        )
        return CertificationRecord(
            accepted,
            checks,
            self.backend_identifier,
            "independent certification accepted"
            if accepted
            else "independent certification rejected incumbent",
        )


@dataclass(frozen=True, slots=True)
class Checkpoint:
    schema_version: int
    seed: int
    completed_batches: int
    incumbent: float | None
    lower_bound: float | None
    completed_arc_ids: tuple[int, ...]
    warm_tokens: tuple[int, ...]

    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.seed < 0
            or self.completed_batches < 0
            or tuple(sorted(set(self.completed_arc_ids))) != self.completed_arc_ids
        ):
            raise ValueError("invalid deterministic checkpoint")

    def write(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(asdict(self), stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    @classmethod
    def read(cls, path: str | Path) -> Checkpoint:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(
            value["schema_version"],
            value["seed"],
            value["completed_batches"],
            value["incumbent"],
            value["lower_bound"],
            tuple(value["completed_arc_ids"]),
            tuple(value["warm_tokens"]),
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    run_id: str
    repository_commit: str
    seed: int
    backend: str
    ownership: str
    device_ids: tuple[int, ...]
    config_sha256: str
    evidence_level: str = "implemented_compiled"

    def validate(self) -> None:
        levels = {
            "implemented_compiled",
            "cpu_correctness_tested",
            "one_gpu_correctness_tested",
            "physical_multi_gpu_tested",
        }
        if (
            self.schema_version != 1
            or not self.run_id
            or len(self.repository_commit) != 40
            or self.seed < 0
            or not self.backend
            or not self.ownership
            or not self.device_ids
            or len(self.config_sha256) != 64
            or self.evidence_level not in levels
        ):
            raise ValueError("invalid G7 run manifest")


@dataclass(frozen=True, slots=True)
class ResultRecord:
    schema_version: int
    run_id: str
    status: str
    incumbent: float | None
    lower_bound: float | None
    optimality_gap: float | None
    certified: bool
    certification: CertificationRecord | None
    telemetry: dict[str, Any]
    failures: tuple[dict[str, Any], ...]

    def validate(self) -> None:
        if self.schema_version != 1 or not self.run_id or not self.status:
            raise ValueError("invalid G7 result header")
        if self.certified and (self.certification is None or not self.certification.accepted):
            raise ValueError("optimizer status alone may not certify a result")
        if (
            self.incumbent is not None
            and self.lower_bound is not None
            and self.lower_bound > self.incumbent
        ):
            raise ValueError("result lower bound exceeds incumbent")


def expand_promising_scenarios(
    requests: Iterable[ArcRequest], *, scenario_count: int, top_k: int
) -> list[ArcRequest]:
    if scenario_count <= 0 or top_k <= 0:
        raise ValueError("scenario expansion limits must be positive")
    selected = sorted(requests, key=lambda item: item.deterministic_id)[:top_k]
    result: list[ArcRequest] = []
    for parent in selected:
        topology = TopologyKey(
            parent.topology.topology_fingerprint,
            ArcFidelity.ROBUST_SCVX,
            parent.topology.intervals,
            scenario_count,
        )
        for scenario in range(scenario_count):
            result.append(
                replace(
                    parent,
                    deterministic_id=len(result),
                    scenario_count=scenario_count,
                    fidelity=ArcFidelity.ROBUST_SCVX,
                    topology=topology,
                    scenario_index=scenario,
                )
            )
    return result


def load_frozen_paper2_matrix(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PAPER2_MATRIX_SHA256:
        raise ValueError(f"Paper 2 matrix hash mismatch: {digest}")
    value = json.loads(raw)
    if value.get("schema_version") != 1 or value.get("programme") != "OrbitWeaver Paper 2":
        raise ValueError("Paper 2 matrix schema is incompatible")
    return value
