"""Bounded, deterministic OrbitWeaver G7 orchestration contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from spacepdhcg.campaign_scope import (
    ACTIVE_SINGLE_GPU_SCOPE_ID,
    HISTORICAL_FULL_SCOPE_ID,
    SCOPE_IDS,
)

from .contracts import PAPER2_MATRIX_SHA256, validate_named


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
    WARM_START_INCOMPATIBLE = "warm_start_incompatible"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    NUMERICAL_FAILURE = "numerical_failure"
    BACKEND_FAILURE = "backend_failure"
    TIMEOUT = "timeout"
    OOM = "oom"
    CENSORED = "censored"
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
    canonical_residual: float = math.inf
    replay_residual: float = math.inf
    nonanticipative_controls: tuple[float, ...] = ()
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
            self.canonical_residual,
            self.replay_residual,
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
        del batch
        rank = request.deterministic_id % len(self.devices)
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
        pending.sort(
            key=lambda item: (
                item.topology,
                self.ownership.owner(item, 0).rank,
                self.ownership.owner(item, 0).device,
                item.deterministic_id,
            )
        )
        self.telemetry = SchedulerTelemetry(submitted=len(pending))
        output: list[ArcResult] = []
        cursor = 0
        sequence = 0
        while cursor < len(pending):
            topology = pending[cursor].topology
            owner = self.ownership.owner(pending[cursor], 0)
            group_end = cursor
            while (
                group_end < len(pending)
                and pending[group_end].topology == topology
                and self.ownership.owner(pending[group_end], 0) == owner
            ):
                group_end += 1
            stop = min(cursor + self.config.maximum_batch_size, group_end)
            batch = pending[cursor:stop]
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
    run_id: str
    manifest_sha256: str
    paper2_matrix_sha256: str
    seed: int
    repeat_index: int
    completed_batches: int
    incumbent: float | None
    lower_bound: float | None
    completed_arc_ids: tuple[int, ...]
    warm_tokens: tuple[int, ...]

    def validate(self, manifest: RunManifest | None = None) -> None:
        validate_named(self.to_dict(), "checkpoint")
        if tuple(sorted(self.completed_arc_ids)) != self.completed_arc_ids:
            raise ValueError("invalid deterministic checkpoint")
        if (
            self.incumbent is not None
            and self.lower_bound is not None
            and self.lower_bound > self.incumbent
        ):
            raise ValueError("checkpoint lower bound exceeds incumbent")
        if manifest is not None:
            manifest.validate()
            if (
                self.run_id != manifest.run_id
                or self.manifest_sha256 != manifest.sha256()
                or self.paper2_matrix_sha256 != manifest.paper2_matrix_sha256
                or self.seed != manifest.seed
                or self.repeat_index >= manifest.repeat_count
            ):
                raise ValueError("checkpoint seed/repeat/pins disagree with manifest")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "completed_arc_ids": list(self.completed_arc_ids),
            "warm_tokens": list(self.warm_tokens),
        }

    def write(self, path: str | Path, manifest: RunManifest | None = None) -> None:
        self.validate(manifest)
        _write_json(path, self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        manifest: RunManifest | None = None,
    ) -> Checkpoint:
        validate_named(value, "checkpoint")
        result = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            manifest_sha256=value["manifest_sha256"],
            paper2_matrix_sha256=value["paper2_matrix_sha256"],
            seed=value["seed"],
            repeat_index=value["repeat_index"],
            completed_batches=value["completed_batches"],
            incumbent=value["incumbent"],
            lower_bound=value["lower_bound"],
            completed_arc_ids=tuple(value["completed_arc_ids"]),
            warm_tokens=tuple(value["warm_tokens"]),
        )
        result.validate(manifest)
        return result

    @classmethod
    def read(
        cls,
        path: str | Path,
        manifest: RunManifest | None = None,
    ) -> Checkpoint:
        return cls.from_dict(_read_json_object(path), manifest)


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
    paper2_matrix_sha256: str
    repeat_count: int
    toolchain: dict[str, str | None]
    hardware: dict[str, Any]
    evidence_level: str = "implemented_compiled"
    campaign_scope_id: str | None = None

    def validate(self) -> None:
        validate_named(self.to_dict(), "manifest")
        if self.schema_version == 1:
            if self.campaign_scope_id is not None:
                raise ValueError("historical G7 manifest may not override campaign scope")
        elif self.schema_version == 2:
            if self.campaign_scope_id not in SCOPE_IDS:
                raise ValueError("scoped G7 manifest requires a known campaign scope")
        else:
            raise ValueError("unsupported G7 manifest schema version")
        if self.campaign_scope_id == ACTIVE_SINGLE_GPU_SCOPE_ID and (
            self.ownership != "single_gpu"
            or len(self.device_ids) != 1
            or self.evidence_level == "physical_multi_gpu_tested"
        ):
            raise ValueError("single-gpu-v1 manifest contains cross-scope physical evidence")
        if (
            self.campaign_scope_id == HISTORICAL_FULL_SCOPE_ID
            and self.ownership == "g5_distributed"
            and len(self.device_ids) < 2
            and self.evidence_level == "physical_multi_gpu_tested"
        ):
            raise ValueError("physical multi-GPU evidence requires at least two devices")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "repository_commit": self.repository_commit,
            "seed": self.seed,
            "repeat_count": self.repeat_count,
            "backend": self.backend,
            "ownership": self.ownership,
            "device_ids": list(self.device_ids),
            "config_sha256": self.config_sha256,
            "paper2_matrix_sha256": self.paper2_matrix_sha256,
            "toolchain": dict(self.toolchain),
            "hardware": dict(self.hardware),
            "evidence_level": self.evidence_level,
        }
        if self.campaign_scope_id is not None:
            result["campaign_scope_id"] = self.campaign_scope_id
        return result

    def write(self, path: str | Path) -> None:
        self.validate()
        _write_json(path, self.to_dict())

    def sha256(self) -> str:
        self.validate()
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunManifest:
        validate_named(value, "manifest")
        result = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            repository_commit=value["repository_commit"],
            seed=value["seed"],
            backend=value["backend"],
            ownership=value["ownership"],
            device_ids=tuple(value["device_ids"]),
            config_sha256=value["config_sha256"],
            paper2_matrix_sha256=value["paper2_matrix_sha256"],
            repeat_count=value["repeat_count"],
            toolchain=dict(value["toolchain"]),
            hardware=dict(value["hardware"]),
            evidence_level=value["evidence_level"],
            campaign_scope_id=value.get("campaign_scope_id"),
        )
        result.validate()
        return result

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        return cls.from_dict(_read_json_object(path))

    @classmethod
    def capture(
        cls,
        *,
        run_id: str,
        repository: str | Path,
        config_path: str | Path,
        matrix_path: str | Path,
        backend: str,
        ownership: str,
        device_ids: Sequence[int],
        evidence_level: str = "implemented_compiled",
        campaign_scope_id: str = ACTIVE_SINGLE_GPU_SCOPE_ID,
    ) -> RunManifest:
        config_raw = Path(config_path).read_bytes()
        config = json.loads(config_raw)
        validate_named(config, "config")
        load_frozen_paper2_matrix(matrix_path)
        repository_path = Path(repository).resolve()
        commit = _command(["git", "-C", str(repository_path), "rev-parse", "HEAD"])
        if commit is None:
            raise ValueError("unable to capture repository commit")
        gpu_text = _command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,compute_cap,driver_version",
                "--format=csv,noheader",
            ]
        )
        result = cls(
            schema_version=2,
            run_id=run_id,
            repository_commit=commit,
            seed=config["seed"],
            backend=backend,
            ownership=ownership,
            device_ids=tuple(device_ids),
            config_sha256=hashlib.sha256(config_raw).hexdigest(),
            paper2_matrix_sha256=PAPER2_MATRIX_SHA256,
            repeat_count=config["repeat_count"],
            toolchain={
                "python": sys.version.split()[0],
                "compiler": _first_line(_command(["c++", "--version"])),
                "cmake": _first_line(_command(["cmake", "--version"])),
                "cuda": _first_line(_command(["nvcc", "--version"])),
            },
            hardware={
                "os": platform.platform(),
                "cpu": platform.processor() or None,
                "gpus": [] if gpu_text is None else gpu_text.splitlines(),
            },
            evidence_level=evidence_level,
            campaign_scope_id=campaign_scope_id,
        )
        result.validate()
        return result


@dataclass(frozen=True, slots=True)
class ResultRecord:
    schema_version: int
    run_id: str
    manifest_sha256: str
    paper2_matrix_sha256: str
    seed: int
    repeat_index: int
    status: str
    incumbent: float | None
    lower_bound: float | None
    optimality_gap: float | None
    certified: bool
    certification: CertificationRecord | None
    telemetry: dict[str, Any]
    failures: tuple[dict[str, Any], ...]

    def validate(self, manifest: RunManifest | None = None) -> None:
        payload = self.to_dict()
        validate_named(payload, "result")
        if manifest is not None:
            manifest.validate()
            if (
                self.run_id != manifest.run_id
                or self.manifest_sha256 != manifest.sha256()
                or self.paper2_matrix_sha256 != manifest.paper2_matrix_sha256
                or self.seed != manifest.seed
                or self.repeat_index >= manifest.repeat_count
            ):
                raise ValueError("result seed/repeat/pin fields disagree with manifest")
        if self.certified and (self.certification is None or not self.certification.accepted):
            raise ValueError("optimizer status alone may not certify a result")
        if (
            self.incumbent is not None
            and self.lower_bound is not None
            and self.lower_bound > self.incumbent
        ):
            raise ValueError("result lower bound exceeds incumbent")
        expected_gap = (
            None
            if self.incumbent is None or self.lower_bound is None
            else max(0.0, self.incumbent - self.lower_bound)
        )
        if (self.optimality_gap is None) != (expected_gap is None) or (
            expected_gap is not None
            and not math.isclose(
                self.optimality_gap,
                expected_gap,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError("result optimality gap is inconsistent with its bounds")
        no_incumbent = {"infeasible", "unsupported"}
        if self.status in no_incumbent and any(
            value is not None for value in (self.incumbent, self.lower_bound, self.optimality_gap)
        ):
            raise ValueError(f"{self.status} result may not contain incumbent bounds")
        terminal = {
            "infeasible",
            "cancelled",
            "failed",
            "censored",
            "unsupported",
            "oom",
            "timeout",
        }
        if self.status in terminal and not any(
            failure["status"] == self.status for failure in self.failures
        ):
            raise ValueError("terminal status requires a matching failure record")
        submitted = self.telemetry["submitted"]
        completed = self.telemetry["completed"]
        if (
            completed
            != self.telemetry["feasible"] + self.telemetry["failed"] + self.telemetry["cancelled"]
            or completed > submitted
        ):
            raise ValueError("result telemetry counters are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        certification = (
            None
            if self.certification is None
            else {
                "accepted": self.certification.accepted,
                "checks": (
                    None if self.certification.checks is None else asdict(self.certification.checks)
                ),
                "backend_identifier": self.certification.backend_identifier,
                "diagnostic": self.certification.diagnostic,
            }
        )
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "manifest_sha256": self.manifest_sha256,
            "paper2_matrix_sha256": self.paper2_matrix_sha256,
            "seed": self.seed,
            "repeat_index": self.repeat_index,
            "status": self.status,
            "incumbent": self.incumbent,
            "lower_bound": self.lower_bound,
            "optimality_gap": self.optimality_gap,
            "certified": self.certified,
            "certification": certification,
            "telemetry": dict(self.telemetry),
            "failures": [dict(value) for value in self.failures],
        }

    def write(self, path: str | Path, manifest: RunManifest | None = None) -> None:
        self.validate(manifest)
        _write_json(path, self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        manifest: RunManifest | None = None,
    ) -> ResultRecord:
        validate_named(value, "result")
        raw_certification = value["certification"]
        certification = None
        if raw_certification is not None:
            raw_checks = raw_certification["checks"]
            checks = None if raw_checks is None else CertificationChecks(**raw_checks)
            certification = CertificationRecord(
                raw_certification["accepted"],
                checks,
                raw_certification["backend_identifier"],
                raw_certification["diagnostic"],
            )
        result = cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            manifest_sha256=value["manifest_sha256"],
            paper2_matrix_sha256=value["paper2_matrix_sha256"],
            seed=value["seed"],
            repeat_index=value["repeat_index"],
            status=value["status"],
            incumbent=value["incumbent"],
            lower_bound=value["lower_bound"],
            optimality_gap=value["optimality_gap"],
            certified=value["certified"],
            certification=certification,
            telemetry=dict(value["telemetry"]),
            failures=tuple(dict(item) for item in value["failures"]),
        )
        result.validate(manifest)
        return result

    @classmethod
    def read(
        cls,
        path: str | Path,
        manifest: RunManifest | None = None,
    ) -> ResultRecord:
        return cls.from_dict(_read_json_object(path), manifest)


SINGLE_GPU_G7_STAGES = frozenset(
    {
        "coarse_convex",
        "refined_scvx",
        "scenario",
        "pricing_master",
        "certification",
        "visualisation",
    }
)


def single_gpu_completion_record(
    manifest: RunManifest,
    records: Iterable[ResultRecord],
    completed_stages: Iterable[str],
) -> dict[str, Any]:
    """Accept one-GPU G7 completion without implying deferred scaling evidence."""

    manifest.validate()
    if manifest.campaign_scope_id != ACTIVE_SINGLE_GPU_SCOPE_ID:
        raise ValueError("one-GPU completion requires campaign scope single-gpu-v1")
    stages = frozenset(completed_stages)
    missing = sorted(SINGLE_GPU_G7_STAGES - stages)
    unknown = sorted(stages - SINGLE_GPU_G7_STAGES)
    if missing or unknown:
        raise ValueError(f"G7 stage inventory invalid; missing={missing}, unknown={unknown}")
    ordered = sorted(records, key=lambda item: item.repeat_index)
    if not ordered:
        raise ValueError("G7 completion requires result records")
    for record in ordered:
        record.validate(manifest)
        if record.status not in {"converged", "iteration_limit"} or not record.certified:
            raise ValueError("G7 completion requires independently certified one-GPU results")
    return {
        "schema_version": "1.0.0",
        "campaign_scope_id": ACTIVE_SINGLE_GPU_SCOPE_ID,
        "status": "complete-in-scope",
        "completed_stages": sorted(stages),
        "run_id": manifest.run_id,
        "result_count": len(ordered),
        "deferred_claims": [
            "physical multi-GPU scaling",
            "distributed route-by-scenario scaling",
            "energy reduction",
            "memory crossover",
            "throughput or tractability-frontier improvement",
        ],
        "statement": (
            "One-GPU correctness and simulation completion only; no physical scaling, "
            "energy, crossover, or throughput claim."
        ),
    }


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json(path: str | Path, value: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _read_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("G7 record must be a JSON object")
    return value


def _command(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None


def _first_line(value: str | None) -> str | None:
    return None if value is None else value.splitlines()[0]


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
