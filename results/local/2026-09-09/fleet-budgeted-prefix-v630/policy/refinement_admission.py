"""Bounded refinement priority for complete, uncertain fuel-proxy requests.

An admission is never a feasible RoutePlan or a trajectory certificate. The caller
must retain the completed request before a failed completion is discarded, run a
fixed-cargo refiner, and verify the emitted fleet with both checkers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

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
class ReplacementBudget:
    """Prescribed fleet arithmetic, never a trajectory or fleet certificate."""

    raw_gain_kg: float
    objective_gain_kg: float
    proposed_fleet_raw_kg: float
    ship_count: int
    score_kind: str

    @property
    def blocker(self) -> str:
        if self.objective_gain_kg <= 0:
            return "no_objective_improvement"
        if self.ship_count > C.maximum_ship_count(self.proposed_fleet_raw_kg / self.ship_count):
            return "raw_fleet_ship_limit"
        return ""


@dataclass(frozen=True, slots=True)
class FleetBudgetContext:
    """Immutable ledgers derived from the exact incumbent checked by both verifiers.

    A replacement must have independent miner inventory. Other retained ships may
    cooperate: their entire visited footprint is excluded from the replacement.
    Each proposal spends the same incumbent budget; combining proposals requires
    a new joint ledger and both checks on the emitted fleet.
    """

    result_sha256: str
    # ship id, all visited asteroids, collected cargo, independently replaceable
    ships: tuple[tuple[int, tuple[int, ...], tuple[tuple[int, float], ...], bool], ...]
    bonus_weights: tuple[tuple[int, float], ...] | None
    _weights: Mapping[int, float] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_weights",
            None if self.bonus_weights is None else MappingProxyType(dict(self.bonus_weights)),
        )

    @classmethod
    def from_verified_result(cls, result: bytes, *, independent, official, weights=None):
        from .solution import parse_solution

        digest = hashlib.sha256(result).hexdigest()
        solution = parse_solution(result.decode("ascii"))
        for name, report in (("independent", independent), ("official", official)):
            if report.get("ok") is not True or report.get("result_sha256") != digest:
                raise ValueError(f"{name} checker is not bound to this accepted Result")
            if report.get("ships", report.get("ship_count")) != solution.ship_count:
                raise ValueError(f"{name} ship count differs from Result")
        bonus = (
            None
            if weights is None
            else tuple(sorted((int(k), float(v)) for k, v in weights.items()))
        )
        if bonus is not None and any(not math.isfinite(v) or v < 0 for _, v in bonus):
            raise ValueError("invalid fixed bonus weights")
        visits = {}
        for ship in solution.ships:
            for event in ship.asteroid_visits():
                visits.setdefault(event.event_id, []).append((ship.ship_id, event))
        cargo = {ship.ship_id: [] for ship in solution.ships}
        independent_ids = set(cargo)
        if len(cargo) != solution.ship_count:
            raise ValueError("duplicate ship identifiers")
        for body, events in visits.items():
            events.sort(key=lambda x: x[1].epoch)
            if len(events) == 1:
                # Retained ships may leave an uncollected miner; it is not cargo
                # and that dependent/open slot is outside this replacement API.
                independent_ids.discard(events[0][0])
                continue
            if len(events) != 2:
                raise ValueError("incumbent must have a complete collected inventory")
            (deployer, _), (collector, event) = events
            mass = event.after.mass - event.before.mass
            if not math.isfinite(mass) or mass < 0:
                raise ValueError("invalid incumbent cargo ledger")
            if deployer != collector:
                independent_ids.difference_update((deployer, collector))
            ship = next(s for s in solution.ships if s.ship_id == collector)
            if not any(
                e.event_id == C.EVENT_EARTH_FLYBY and e.epoch >= event.epoch
                for e in ship.events[1:]
            ):
                raise ValueError("incumbent cargo has no subsequent Earth unload")
            cargo[collector].append((body, mass))
        rows = tuple(
            (
                s.ship_id,
                tuple(sorted({e.event_id for e in s.asteroid_visits()})),
                tuple(sorted(cargo[s.ship_id])),
                s.ship_id in independent_ids,
            )
            for s in solution.ships
        )
        context = cls(digest, rows, bonus)
        totals = [("total_mass_kg", context.raw_kg)]
        if bonus is not None:
            totals.append(("weighted_score_fixed_bonus_kg", context.objective_kg))
        for key, actual in totals:
            reported = independent.get(key)
            if (
                not isinstance(reported, (float, int))
                or isinstance(reported, bool)
                or not math.isfinite(reported)
                or not math.isclose(actual, reported, rel_tol=0, abs_tol=1e-8)
            ):
                raise ValueError(f"incumbent ledger differs from verified {key}")
        if not 1 <= len(rows) <= C.MAX_SHIPS or len(rows) > C.maximum_ship_count(
            context.raw_kg / len(rows)
        ):
            raise ValueError("incumbent violates raw fleet ship limit")
        return context

    @property
    def score_kind(self):
        return "raw_kg" if self.bonus_weights is None else "weighted_fixed_bonus_kg"

    @property
    def raw_kg(self):
        return sum(m for _, _, cargo, _ in self.ships for _, m in cargo)

    def _value(self, cargo):
        if self.bonus_weights is None:
            return sum(m for _, m in cargo)
        # Missing weights are an error, never an implicit switch to raw mass.
        try:
            value = sum(self._weights[a] * m for a, m in cargo)
        except KeyError as error:
            raise ValueError(f"missing fixed bonus for asteroid {error.args[0]}") from error
        if not math.isfinite(value):
            raise ValueError("nonfinite fixed bonus objective")
        return value

    @property
    def objective_kg(self):
        return self._value(tuple(item for _, _, cargo, _ in self.ships for item in cargo))

    def _ship(self, ship_id):
        for row in self.ships:
            if row[0] == ship_id:
                if not row[3]:
                    raise ValueError("replacement ship has dependent miner inventory")
                return row
        raise ValueError("replacement ship is absent from incumbent")

    def excluded_asteroids(self, ship_id):
        self._ship(ship_id)
        return frozenset(
            a for sid, footprint, _, _ in self.ships if sid != ship_id for a in footprint
        )

    def replacement(self, ship_id, request: FixedCargoRequest) -> ReplacementBudget:
        old = self._ship(ship_id)[2]
        new = tuple((int(a), float(m)) for a, m in json.loads(request.payload)["cargo"])
        if self.excluded_asteroids(ship_id).intersection(a for a, _ in new):
            raise ValueError("replacement conflicts with retained asteroid inventory")
        delta = sum(m for _, m in new) - sum(m for _, m in old)
        return ReplacementBudget(
            delta,
            self._value(new) - self._value(old),
            self.raw_kg + delta,
            len(self.ships),
            self.score_kind,
        )


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
        self._plans: dict[str, bytes] = {}
        self._replacement_ships: dict[str, int] = {}
        self._fleet_context: FleetBudgetContext | None = None
        self.claimed = 0

    @property
    def plan_lookup(self):
        """Bounded immutable plan bytes for ``run_refinement_queue``."""
        return MappingProxyType(self._plans)

    def replacement_ship(self, request_sha256: str) -> int:
        return self._replacement_ships[request_sha256]

    def completion_consumer(self, context: FleetBudgetContext, ship_id: int):
        """Bind a production CompletionRecorder to ledger-derived fleet admission.

        Producer/source identity stays the recorder's responsibility. Its copied
        envelope is admitted before the native completion adapter drops a mass
        rejection. Captured summaries remain bounded with the queue, ready for
        the fixed-cargo executor and an explicit full-fleet verifier callback.
        """
        context._ship(ship_id)
        if context.bonus_weights is None:
            raise ValueError("the fixed refinement queue currently requires a weighted objective")
        if self._fleet_context is not None and self._fleet_context != context:
            raise ValueError("a queue cannot mix incumbent or objective contexts")
        self._fleet_context = context

        def consume(envelope):
            if self._selected is not None:
                raise RuntimeError("admission is closed once any refinement is claimed")
            summary = bytes(envelope.plan_summary)
            if FixedCargoRequest.from_summary(json.loads(summary)) != envelope.request:
                return "captured_prescription_mismatch"
            key = envelope.request.sha256
            if key in self._replacement_ships and self._replacement_ships[key] != ship_id:
                return "ambiguous_replacement_ship"
            try:
                budget = context.replacement(ship_id, envelope.request)
            except ValueError as error:
                return f"fleet_context_rejected: {error}"
            if budget.blocker:
                return budget.blocker
            candidate = AdmissionCandidate(
                envelope.request,
                envelope.observation,
                budget.objective_gain_kg,
                budget.proposed_fleet_raw_kg,
                budget.ship_count,
            )
            disposition = self.offer(candidate)
            if self._pending.get(key) == candidate:
                self._plans[key] = summary
                self._replacement_ships[key] = ship_id
            for stale in self._plans.keys() - self._pending.keys():
                del self._plans[stale]
                del self._replacement_ships[stale]
            return disposition

        return consume

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
