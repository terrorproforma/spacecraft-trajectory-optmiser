"""Bounded refinement priority for complete, uncertain fuel-proxy requests.

An admission is never a feasible RoutePlan or a trajectory certificate. The caller
must retain the completed request before a failed completion is discarded, run a
fixed-cargo refiner, and verify the emitted fleet with both checkers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from . import constants as C


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class FixedCargoRequest:
    """Immutable event/cargo prescription; excludes the proxy's acceptance flag."""

    payload: bytes

    @classmethod
    def from_summary(cls, plan: dict, cargo: dict | None = None) -> FixedCargoRequest:
        flights = [
            (int(x["from"]), int(x["to"]), float(x["t0"]), float(x["tf"]))
            for x in plan["legs"]
            if x["role"] != "camp"
        ]
        deploy = {int(k): float(v) for k, v in plan["deploy_epochs"].items()}
        collect = {int(k): float(v) for k, v in plan["collect_epochs"].items()}
        masses = {
            int(k): float(v)
            for k, v in (plan["collected_mass_kg"] if cargo is None else cargo).items()
        }
        if plan.get("foreign_deploy_epochs") or set(deploy) != set(collect):
            raise ValueError("admission currently requires an independent closed miner inventory")
        if set(masses) != set(collect) or not 0 < len(deploy) <= C.MAX_MINERS_PER_SHIP:
            raise ValueError("invalid prescribed inventory")
        if not flights or flights[0][0] != 0 or flights[-1][1] != 0:
            raise ValueError("prescription must depart and return to Earth")
        if flights[0][2] < C.MISSION_START_MJD or flights[-1][3] > C.MISSION_END_MJD:
            raise ValueError("prescription outside mission window")
        stays: dict[int, list[tuple[float, float]]] = {}
        for i, (origin, target, departure, arrival) in enumerate(flights):
            if not all(math.isfinite(x) for x in (departure, arrival)) or arrival <= departure:
                raise ValueError("invalid flight epochs")
            if origin == target or origin < 0 or target < 0:
                raise ValueError("invalid flight endpoints")
            if i:
                previous = flights[i - 1]
                if origin != previous[1] or departure < previous[3]:
                    raise ValueError("discontinuous prescription")
                stays.setdefault(origin, []).append((previous[3], departure))
            if target and target not in deploy:
                raise ValueError("unrecorded asteroid visit")
        for body, epoch in deploy.items():
            if not any(b == body and arrival == epoch for _, b, _, arrival in flights):
                raise ValueError("deployment is not an arrival event")
            end = collect[body]
            if not any(arrival <= end <= departure for arrival, departure in stays.get(body, [])):
                raise ValueError("collection is not an on-body event")
            duration = end - epoch
            if duration < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 1e-6:
                raise ValueError("minimum mining stay violated")
            mass = masses[body]
            if not math.isfinite(mass) or mass < 0 or mass > C.maximum_collected_mass(duration):
                raise ValueError("prescribed cargo violates mining production")
        return cls(
            _json(
                {
                    "flights": flights,
                    "deploy": sorted(deploy.items()),
                    "collect": sorted(collect.items()),
                    "cargo": sorted(masses.items()),
                }
            )
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def cargo_kg(self) -> float:
        return sum(m for _, m in json.loads(self.payload)["cargo"])


@dataclass(frozen=True, slots=True)
class CompletionObservation:
    """Copied completion readback bound to its prescription and frozen producer.

    ``stages`` includes camps. Successful native stages are CAMP=1 and COSTED=4.
    An earlier authority/mining/inflation failure cannot enter the uncertain lane.
    Native request capture must also retain the full per-leg readback externally.
    """

    request_sha256: str
    producer_sha256: str
    readback_sha256: str
    failure: str
    expected_legs: int
    processed_legs: int
    stages: tuple[int, ...]
    final_mass_kg: float
    collected_kg: float
    finite_nonnegative_costs: bool

    def blocker(self, request: FixedCargoRequest) -> str:
        if self.request_sha256 != request.sha256:
            return "request_readback_mismatch"
        for digest in (self.producer_sha256, self.readback_sha256):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                return "missing_readback_provenance"
        if self.failure not in ("", "mass_below_dry_plus_collected"):
            return "earlier_completion_gate"
        if (
            self.expected_legs <= 0
            or self.processed_legs != self.expected_legs
            or len(self.stages) != self.expected_legs
            or any(stage not in (1, 4) for stage in self.stages)
            or 4 not in self.stages
        ):
            return "incomplete_forward_pass"
        if (
            not self.finite_nonnegative_costs
            or not math.isfinite(self.final_mass_kg)
            or self.final_mass_kg <= 0
            or not math.isfinite(self.collected_kg)
            or not math.isclose(self.collected_kg, request.cargo_kg, rel_tol=0, abs_tol=2e-10)
        ):
            return "invalid_completion_readback"
        failed = self.final_mass_kg < C.DRY_MASS_KG + self.collected_kg
        if failed != bool(self.failure):
            return "inconsistent_mass_classification"
        return ""

    @property
    def deficit_kg(self) -> float:
        return max(0.0, C.DRY_MASS_KG + self.collected_kg - self.final_mass_kg)


@dataclass(frozen=True, slots=True)
class AdmissionCandidate:
    request: FixedCargoRequest
    observation: CompletionObservation
    weighted_gain_kg: float
    proposed_fleet_raw_kg: float
    proposed_ship_count: int

    @property
    def uncertain(self) -> bool:
        return bool(self.observation.failure)

    @property
    def priority(self) -> tuple[float, float, str]:
        # This is a priority estimate, not a calibrated probability or fuel bound.
        return (self.observation.deficit_kg, -self.weighted_gain_kg, self.request.sha256)

    def blocker(self, *, replay_controls: bool = False) -> str:
        failure = self.observation.blocker(self.request)
        if failure:
            return failure
        if not math.isfinite(self.weighted_gain_kg) or self.weighted_gain_kg < 0:
            return "objective_not_preserved"
        if self.weighted_gain_kg == 0 and not replay_controls:
            return "no_weighted_improvement"
        if (
            not isinstance(self.proposed_ship_count, int)
            or isinstance(self.proposed_ship_count, bool)
            or not 1 <= self.proposed_ship_count <= C.MAX_SHIPS
            or not math.isfinite(self.proposed_fleet_raw_kg)
            or self.proposed_fleet_raw_kg < 0
            or self.proposed_ship_count
            > C.maximum_ship_count(self.proposed_fleet_raw_kg / self.proposed_ship_count)
        ):
            return "raw_fleet_ship_limit"
        return ""


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    retained_requests: int = 48
    refinement_budget: int = 4
    uncertainty_budget: int = 2

    def __post_init__(self):
        if not 0 <= self.uncertainty_budget <= self.refinement_budget <= self.retained_requests:
            raise ValueError("invalid admission budgets")
        if self.retained_requests > 48 or self.refinement_budget > 4 or self.uncertainty_budget > 2:
            raise ValueError("admission exceeds the reviewed finite budget")


class RefinementAdmissionQueue:
    """Retain at most 48 prescriptions; reserve at most two of four slots for uncertainty.

    Offers return a disposition which the caller must journal. No candidate here
    is accepted, certified, or included in a master. Claiming consumes budget even
    if its subsequent solver call fails; new offers are forbidden after claiming.
    """

    def __init__(self, policy: AdmissionPolicy | None = None, *, replay_controls: bool = False):
        self.policy = policy or AdmissionPolicy()
        self.replay_controls = replay_controls
        self._pending: dict[str, AdmissionCandidate] = {}
        self._selected: tuple[AdmissionCandidate, ...] | None = None
        self.claimed = 0

    def offer(self, candidate: AdmissionCandidate) -> str:
        if self._selected is not None:
            raise RuntimeError("admission is closed once any refinement is claimed")
        reason = candidate.blocker(replay_controls=self.replay_controls)
        if reason:
            return reason
        key = candidate.request.sha256
        previous = self._pending.get(key)
        if previous is not None:
            # A prescription is one solver job, even if several proxy models priced it.
            if previous == candidate:
                return "duplicate_prescription"
            self._pending[key] = min(
                (previous, candidate), key=lambda x: (*x.priority, x.observation.readback_sha256)
            )
            return "duplicate_prescription_other_proxy"
        self._pending[key] = candidate
        ranked = sorted(self._pending.values(), key=lambda x: x.priority)
        reserved = [x for x in ranked if x.uncertain][: self.policy.uncertainty_budget]
        keys = {x.request.sha256 for x in reserved}
        retained = (
            reserved
            + [x for x in ranked if x.request.sha256 not in keys][
                : self.policy.retained_requests - len(reserved)
            ]
        )
        self._pending = {x.request.sha256: x for x in retained}
        return "refinement_eligible" if key in self._pending else "retention_budget"

    def shortlist(self) -> tuple[AdmissionCandidate, ...]:
        if self._selected is not None:
            return self._selected
        ordered = sorted(self._pending.values(), key=lambda x: x.priority)
        uncertain = [x for x in ordered if x.uncertain][: self.policy.uncertainty_budget]
        regular = [x for x in ordered if not x.uncertain]
        return tuple(regular[: self.policy.refinement_budget - len(uncertain)] + uncertain)

    def retained(self) -> tuple[AdmissionCandidate, ...]:
        """Immutable bounded inventory for pruning the caller's captured-plan storage."""
        return tuple(sorted(self._pending.values(), key=lambda x: x.priority))

    def claim_next(self) -> AdmissionCandidate | None:
        if self.replay_controls:
            raise RuntimeError("archival replay queues cannot launch refinement")
        if self._selected is None:
            self._selected = self.shortlist()
        if self.claimed >= len(self._selected):
            return None
        result = self._selected[self.claimed]
        self.claimed += 1
        return result


def promotion_blockers(
    request: FixedCargoRequest,
    returned_request: FixedCargoRequest,
    *,
    all_legs_certified: bool,
    result_sha256: str,
    independent: dict,
    official: dict,
    incumbent_weighted_kg: float,
) -> tuple[str, ...]:
    """Additional identity/score guard after a fixed-cargo solve and both fleet checks.

    These reports must come from running the checkers, never from proxy readbacks
    or archived control flags. This helper does not execute or replace a checker.
    """
    failures = []
    if request != returned_request:
        failures.append("prescribed_events_or_cargo_changed")
    if not all_legs_certified:
        failures.append("uncertified_route")
    if len(result_sha256) != 64 or any(c not in "0123456789abcdef" for c in result_sha256):
        failures.append("missing_result_identity")
    for name, report in (("independent", independent), ("official", official)):
        if report.get("result_sha256") != result_sha256:
            failures.append(f"{name}_result_mismatch")
        if report.get("ok") is not True:
            failures.append(f"{name}_rejected")
    score = independent.get("weighted_score_fixed_bonus_kg")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not math.isfinite(incumbent_weighted_kg)
        or score <= incumbent_weighted_kg
    ):
        failures.append("no_verified_weighted_improvement")
    count = independent.get("ships", independent.get("ship_count"))
    raw = independent.get("total_mass_kg")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= C.MAX_SHIPS
        or not isinstance(raw, (int, float))
        or isinstance(raw, bool)
        or not math.isfinite(raw)
        or raw < 0
        or count > C.maximum_ship_count(raw / count)
    ):
        failures.append("verified_raw_fleet_ship_limit")
    return tuple(failures)
