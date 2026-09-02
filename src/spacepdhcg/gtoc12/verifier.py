"""Independent GTOC12 solution verifier and scorer.

This re-implements the checks performed by the organisers' ``GTOC12_Verify`` program from the
problem statement, the submission-format document and the program's own diagnostic catalogue
(``Error001``..``Error901``, ``ErrorA00``..``ErrorA23``).  Between two consecutive events the ship
state and mass are propagated numerically; coast segments use the exact two-body solution and
burning arcs integrate the thrust obtained by cubic Lagrange interpolation of the daily samples.
The acceptance test for this module is exact reproduction of the official verifier's ship count,
mined-asteroid list, per-asteroid collected masses and total mass on the archived reference
solutions.

The *unweighted* total ``sum M_i`` is what the offline verifier prints.  The *fixed
post-competition* weighted score applies the frozen ``bonus_coefficients.txt``; the *dynamic*
competition score depends on the leaderboard history and can only be recomputed from a supplied
already-mined-mass table.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from . import constants as C
from .data import AsteroidCatalogue, BonusTable
from .ephemeris import asteroid_state, planet_state, propagate_kepler
from .solution import BurnArc, Event, ShipTrajectory, Solution, parse_solution

FloatArray = NDArray[np.float64]

# Reference files print epochs with 17 significant digits, so consecutive daily samples differ by
# 1 + O(1e-11) days; the official verifier accepts them.  One microday (86 ms) of slack keeps the
# one-day rule while ignoring representation noise.
SAMPLE_INTERVAL_SLACK_DAYS = 1.0e-6


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    ship_id: int | None
    line: int | None
    message: str
    value: float | None = None

    def __str__(self) -> str:
        where = "" if self.ship_id is None else f" ship {self.ship_id}"
        line = "" if self.line is None else f" line {self.line}"
        return f"{self.code}{where}{line}: {self.message}"


@dataclass(frozen=True, slots=True)
class LegCheck:
    """Propagation from one event's after-state to the next event's before-state."""

    ship_id: int
    from_event: int
    to_event: int
    from_line: int
    to_line: int
    start_epoch: float
    end_epoch: float
    burn_count: int
    position_error_km: float
    velocity_error_km_s: float
    mass_error_kg: float
    minimum_sun_distance_au: float

    @property
    def passed(self) -> bool:
        return (
            self.position_error_km <= C.TOLERANCE_POSITION_KM
            and self.velocity_error_km_s <= C.TOLERANCE_VELOCITY_KM_S
            and self.mass_error_kg <= C.TOLERANCE_MASS_KG
            and self.minimum_sun_distance_au >= C.MIN_SUN_DISTANCE_AU
        )


@dataclass(frozen=True, slots=True)
class AsteroidVisit:
    asteroid_id: int
    ship_id: int
    epoch: float
    mass_before: float
    mass_after: float
    line: int


@dataclass(frozen=True, slots=True)
class MinedAsteroid:
    asteroid_id: int
    deploy_ship: int
    deploy_epoch: float
    collect_ship: int
    collect_epoch: float
    collected_mass_kg: float
    unloaded: bool
    unload_epoch: float | None


@dataclass(slots=True)
class VerificationReport:
    ok: bool
    ship_count: int
    violations: list[Violation]
    legs: list[LegCheck]
    mined: dict[int, MinedAsteroid]
    total_mass_kg: float
    ship_limit: float
    weighted_score_fixed_bonus_kg: float | None = None
    weighted_score_dynamic_kg: float | None = None
    ship_unloaded_mass: dict[int, float] = field(default_factory=dict)

    @property
    def mined_asteroid_count(self) -> int:
        return sum(1 for item in self.mined.values() if item.unloaded)

    @property
    def scored_masses(self) -> dict[int, float]:
        return {
            key: item.collected_mass_kg for key, item in sorted(self.mined.items()) if item.unloaded
        }

    @property
    def max_position_error_km(self) -> float:
        return max((leg.position_error_km for leg in self.legs), default=0.0)

    @property
    def max_velocity_error_km_s(self) -> float:
        return max((leg.velocity_error_km_s for leg in self.legs), default=0.0)

    @property
    def max_mass_error_kg(self) -> float:
        return max((leg.mass_error_kg for leg in self.legs), default=0.0)

    def summary(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "ships": self.ship_count,
            "mined_asteroids": self.mined_asteroid_count,
            "total_mass_kg": self.total_mass_kg,
            "weighted_score_fixed_bonus_kg": self.weighted_score_fixed_bonus_kg,
            "weighted_score_dynamic_kg": self.weighted_score_dynamic_kg,
            "ship_limit": self.ship_limit,
            "max_position_error_km": self.max_position_error_km,
            "max_velocity_error_km_s": self.max_velocity_error_km_s,
            "max_mass_error_kg": self.max_mass_error_kg,
            "violations": [str(item) for item in self.violations],
        }


# --- thrust interpolation ---------------------------------------------------------------------


class LagrangeThrust:
    """Cubic Lagrange interpolation of burn samples, the official verifier's thrust model.

    For ``t`` in ``[t_j, t_{j+1}]`` the four samples ``j-1..j+2`` (clamped to the arc) define the
    interpolating polynomial; arcs with fewer than four samples use all of them.
    """

    def __init__(self, epochs_s: FloatArray, thrust_n: FloatArray, order: int = 3) -> None:
        epochs = np.asarray(epochs_s, dtype=np.float64)
        thrust = np.asarray(thrust_n, dtype=np.float64)
        if epochs.ndim != 1 or thrust.shape != (epochs.shape[0], 3):
            raise ValueError("thrust samples must be (n,) epochs and (n, 3) vectors")
        if np.any(np.diff(epochs) < 0.0):
            raise ValueError("burn sample epochs must not decrease")
        # Reference files contain zero-length "partial day" samples (two interior lines sharing
        # one epoch); they carry no time, so keep the first occurrence of each epoch.
        keep = np.concatenate(([True], np.diff(epochs) > 0.0))
        self.epochs = epochs[keep]
        self.thrust = thrust[keep]
        if self.epochs.shape[0] < 2:
            raise ValueError("at least two distinct sample epochs are required")
        self.points = min(order + 1, self.epochs.shape[0])

    def stencil(self, t: float) -> int:
        n = self.epochs.shape[0]
        j = int(np.searchsorted(self.epochs, t, side="right")) - 1
        j = min(max(j, 0), n - 2)
        start = j - (self.points - 1) // 2
        return min(max(start, 0), n - self.points)

    def __call__(self, t: float) -> FloatArray:
        start = self.stencil(t)
        xs = self.epochs[start : start + self.points]
        ys = self.thrust[start : start + self.points]
        result = np.zeros(3)
        for k in range(self.points):
            weight = 1.0
            xk = xs[k]
            for m in range(self.points):
                if m != k:
                    weight *= (t - xs[m]) / (xk - xs[m])
            result += weight * ys[k]
        return result

    def sample(self, times: FloatArray) -> FloatArray:
        return np.asarray([self(float(t)) for t in np.asarray(times, dtype=np.float64)])


def _thrust_dynamics(interpolant: LagrangeThrust):
    mu = C.MU_SUN_KM3_S2
    flow = C.MASS_FLOW_PER_NEWTON_KG_S

    def fun(t: float, y: FloatArray) -> FloatArray:
        r = y[0:3]
        radius = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        thrust = interpolant(t)
        magnitude = math.sqrt(thrust[0] ** 2 + thrust[1] ** 2 + thrust[2] ** 2)
        mass = y[6]
        scale = -mu / radius**3
        accel = 1.0e-3 / mass  # N / kg -> km/s^2
        return np.array(
            [
                y[3],
                y[4],
                y[5],
                scale * r[0] + accel * thrust[0],
                scale * r[1] + accel * thrust[1],
                scale * r[2] + accel * thrust[2],
                -flow * magnitude,
            ]
        )

    return fun


@dataclass(slots=True)
class PropagatedHistory:
    """Dense propagated samples (MJD, km, km/s, kg, N) for viewer export and diagnostics."""

    epochs_mjd: list[float] = field(default_factory=list)
    positions_km: list[FloatArray] = field(default_factory=list)
    velocities_km_s: list[FloatArray] = field(default_factory=list)
    masses_kg: list[float] = field(default_factory=list)
    thrust_n: list[FloatArray] = field(default_factory=list)

    def append(self, t: float, r: FloatArray, v: FloatArray, m: float, thrust: FloatArray) -> None:
        self.epochs_mjd.append(float(t))
        self.positions_km.append(np.asarray(r, dtype=np.float64).copy())
        self.velocities_km_s.append(np.asarray(v, dtype=np.float64).copy())
        self.masses_kg.append(float(m))
        self.thrust_n.append(np.asarray(thrust, dtype=np.float64).copy())

    def arrays(self) -> dict[str, FloatArray]:
        return {
            "epochs_mjd": np.asarray(self.epochs_mjd),
            "positions_km": np.asarray(self.positions_km).reshape(-1, 3),
            "velocities_km_s": np.asarray(self.velocities_km_s).reshape(-1, 3),
            "masses_kg": np.asarray(self.masses_kg),
            "thrust_n": np.asarray(self.thrust_n).reshape(-1, 3),
        }


def propagate_burn(
    epoch0_mjd: float,
    position: FloatArray,
    velocity: FloatArray,
    mass: float,
    arc: BurnArc,
    *,
    rtol: float = 1e-12,
    history: PropagatedHistory | None = None,
    sample_days: float = 1.0,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Integrate one burning arc; returns ``(r, v, m, minimum radius km)`` at the arc end."""

    epochs_days, thrust = arc.interior_arrays()
    epochs_s = (epochs_days - epoch0_mjd) * C.DAY_S
    t0, t1 = float(epochs_s[0]), float(epochs_s[-1])
    y0 = np.concatenate((position, velocity, [mass]))
    if t1 <= t0:
        return position.copy(), velocity.copy(), mass, float(np.linalg.norm(position))
    interpolant = LagrangeThrust(epochs_s, thrust, C.THRUST_INTERPOLATION_ORDER)
    samples = np.arange(t0, t1, sample_days * C.DAY_S)
    t_eval = np.unique(np.concatenate((samples, [t1])))
    result = solve_ivp(
        _thrust_dynamics(interpolant),
        (t0, t1),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=np.array([1e-7, 1e-7, 1e-7, 1e-10, 1e-10, 1e-10, 1e-9]),
        t_eval=t_eval,
    )
    if not result.success:
        raise RuntimeError(f"burn integration failed: {result.message}")
    states = result.y.T
    radii = np.linalg.norm(states[:, 0:3], axis=1)
    if history is not None:
        for t, y in zip(result.t, states, strict=True):
            history.append(epoch0_mjd + t / C.DAY_S, y[0:3], y[3:6], y[6], interpolant(float(t)))
    final = states[-1]
    return final[0:3].copy(), final[3:6].copy(), float(final[6]), float(np.min(radii))


def propagate_coast(
    epoch0_mjd: float,
    position: FloatArray,
    velocity: FloatArray,
    mass: float,
    epoch1_mjd: float,
    *,
    history: PropagatedHistory | None = None,
    sample_days: float = 5.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Exact two-body coast, optionally sampled into the history; returns state and min radius."""

    duration = (epoch1_mjd - epoch0_mjd) * C.DAY_S
    if duration <= 0.0:
        return position.copy(), velocity.copy(), float(np.linalg.norm(position))
    offsets = np.arange(0.0, duration, sample_days * C.DAY_S)
    offsets = np.unique(np.concatenate((offsets, [duration])))
    r, v = propagate_kepler(
        np.repeat(position[None, :], len(offsets), 0),
        np.repeat(velocity[None, :], len(offsets), 0),
        offsets,
    )
    radii = np.linalg.norm(r, axis=1)
    if history is not None:
        zero = np.zeros(3)
        for offset, ri, vi in zip(offsets, r, v, strict=True):
            history.append(epoch0_mjd + offset / C.DAY_S, ri, vi, mass, zero)
    return r[-1].copy(), v[-1].copy(), float(np.min(radii))


# --- verifier ----------------------------------------------------------------------------------


class Gtoc12Verifier:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        *,
        bonus: BonusTable | None = None,
        rtol: float = 1e-12,
        history: dict[int, PropagatedHistory] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.bonus = bonus
        self.rtol = rtol
        self.history = history

    # -- public API --

    def verify_file(self, path: str | Path) -> VerificationReport:
        return self.verify(Solution.read(path))

    def verify_text(self, text: str) -> VerificationReport:
        return self.verify(parse_solution(text))

    def verify(
        self,
        solution: Solution,
        *,
        dynamic_already_mined: Mapping[int, float] | None = None,
    ) -> VerificationReport:
        violations: list[Violation] = []
        legs: list[LegCheck] = []
        visits: list[AsteroidVisit] = []
        unload_events: list[tuple[int, float, float, int]] = []  # ship, epoch, mass, line
        for ship in solution.ships:
            self._verify_ship(ship, violations, legs, visits, unload_events)
        mined = self._mining_bookkeeping(solution, visits, unload_events, violations)
        total = sum(item.collected_mass_kg for item in mined.values() if item.unloaded)
        ship_count = solution.ship_count
        mean = total / ship_count if ship_count else 0.0
        limit = C.maximum_ship_count(mean)
        if ship_count > limit:
            violations.append(
                Violation(
                    "Error301",
                    None,
                    None,
                    f"{ship_count} ships exceed the allowable maximum {limit:.6f}",
                    float(limit),
                )
            )
        per_ship_unloaded = {ship_id: 0.0 for ship_id in range(1, ship_count + 1)}
        for ship_id, _epoch, mass, _line in unload_events:
            per_ship_unloaded[ship_id] += mass
        report = VerificationReport(
            ok=not violations,
            ship_count=ship_count,
            violations=violations,
            legs=legs,
            mined=mined,
            total_mass_kg=total,
            ship_limit=limit,
            ship_unloaded_mass=per_ship_unloaded,
        )
        if self.bonus is not None:
            report.weighted_score_fixed_bonus_kg = sum(
                self.bonus.for_asteroid(key) * mass for key, mass in report.scored_masses.items()
            )
        if dynamic_already_mined is not None:
            report.weighted_score_dynamic_kg = sum(
                C.bonus_coefficient(float(dynamic_already_mined.get(key, 0.0))) * mass
                for key, mass in report.scored_masses.items()
            )
        return report

    # -- per ship --

    def _verify_ship(
        self,
        ship: ShipTrajectory,
        violations: list[Violation],
        legs: list[LegCheck],
        visits: list[AsteroidVisit],
        unload_events: list[tuple[int, float, float, int]],
    ) -> None:
        sid = ship.ship_id
        history = None
        if self.history is not None:
            history = self.history.setdefault(sid, PropagatedHistory())
        events = ship.events
        if not events or events[0].event_id != C.EVENT_LAUNCH:
            violations.append(
                Violation("ErrorA14", sid, None, "the event ID of the launch is wrong")
            )
            return
        last_epoch = -math.inf
        for event in events:
            if event.epoch < C.MISSION_START_MJD:
                violations.append(
                    Violation(
                        "Error001",
                        sid,
                        event.before.line_number,
                        "epoch earlier than 64328 MJD",
                        event.epoch,
                    )
                )
            if event.epoch > C.MISSION_END_MJD:
                violations.append(
                    Violation(
                        "Error002",
                        sid,
                        event.before.line_number,
                        "epoch later than 69807 MJD",
                        event.epoch,
                    )
                )
            if event.epoch < last_epoch:
                violations.append(
                    Violation(
                        "Error005",
                        sid,
                        event.before.line_number,
                        "event earlier than the previous event",
                    )
                )
            last_epoch = event.epoch
        for arc in ship.burns:
            for sample in arc.samples:
                if sample.epoch < C.MISSION_START_MJD:
                    violations.append(
                        Violation(
                            "Error001",
                            sid,
                            sample.line_number,
                            "burn epoch earlier than 64328 MJD",
                            sample.epoch,
                        )
                    )
                if sample.epoch > C.MISSION_END_MJD:
                    violations.append(
                        Violation(
                            "Error002",
                            sid,
                            sample.line_number,
                            "burn epoch later than 69807 MJD",
                            sample.epoch,
                        )
                    )
            self._check_arc_samples(sid, arc, violations)

        carried = 0.0
        miners_deployed = 0
        launch = events[0]
        self._check_launch(sid, launch, violations)
        # walk items in order, tracking the propagated state
        state_epoch = launch.after.epoch
        position = launch.after.position.copy()
        velocity = launch.after.velocity.copy()
        mass = launch.after.mass
        if history is not None:
            history.append(state_epoch, position, velocity, mass, np.zeros(3))
        pending_burns: list[BurnArc] = []
        previous_event = launch
        for item in ship.items[1:]:
            if isinstance(item, BurnArc):
                if item.start < previous_event.epoch:
                    violations.append(
                        Violation(
                            "Error003",
                            sid,
                            item.samples[0].line_number,
                            "burning arc starts before the previous event",
                            item.start,
                        )
                    )
                pending_burns.append(item)
                continue
            event = item
            for arc in pending_burns:
                if arc.end > event.epoch:
                    violations.append(
                        Violation(
                            "Error004",
                            sid,
                            arc.samples[-1].line_number,
                            "burning arc ends after the subsequent event",
                            arc.end,
                        )
                    )
            leg = self._propagate_leg(
                sid,
                state_epoch,
                position,
                velocity,
                mass,
                pending_burns,
                previous_event,
                event,
                history,
            )
            legs.append(leg)
            if leg.position_error_km > C.TOLERANCE_POSITION_KM:
                violations.append(
                    Violation(
                        "Error201",
                        sid,
                        event.before.line_number,
                        f"position error {leg.position_error_km:.3f} km > 1000 km",
                        leg.position_error_km,
                    )
                )
            if leg.velocity_error_km_s > C.TOLERANCE_VELOCITY_KM_S:
                violations.append(
                    Violation(
                        "Error202",
                        sid,
                        event.before.line_number,
                        f"velocity error {leg.velocity_error_km_s * 1e3:.4f} m/s > 1 m/s",
                        leg.velocity_error_km_s,
                    )
                )
            if leg.mass_error_kg > C.TOLERANCE_MASS_KG:
                violations.append(
                    Violation(
                        "Error203",
                        sid,
                        event.before.line_number,
                        f"mass error {leg.mass_error_kg:.6f} kg > 0.001 kg",
                        leg.mass_error_kg,
                    )
                )
            if leg.minimum_sun_distance_au < C.MIN_SUN_DISTANCE_AU:
                violations.append(
                    Violation(
                        "Error101",
                        sid,
                        event.before.line_number,
                        "distance to the Sun below 0.3 AU",
                        leg.minimum_sun_distance_au,
                    )
                )
            pending_burns = []
            # event-specific checks and mass bookkeeping
            if event.kind == "rendezvous":
                self._check_rendezvous(sid, event, violations)
                delta = event.after.mass - event.before.mass
                visits.append(
                    AsteroidVisit(
                        event.event_id,
                        sid,
                        event.epoch,
                        event.before.mass,
                        event.after.mass,
                        event.before.line_number,
                    )
                )
                if delta < 0.0:
                    miners_deployed += 1
                else:
                    carried += delta
            elif event.kind == "flyby":
                unloaded = self._check_flyby(sid, event, carried, violations)
                if unloaded is not None:
                    unload_events.append((sid, event.epoch, unloaded, event.before.line_number))
                    carried = max(0.0, carried - unloaded)
            elif event.event_id == C.EVENT_LAUNCH:
                violations.append(
                    Violation(
                        "ErrorA13",
                        sid,
                        event.before.line_number,
                        "second launch in one ship section",
                    )
                )
            if event.after.mass < C.DRY_MASS_KG + carried - C.TOLERANCE_MASS_KG:
                violations.append(
                    Violation(
                        "Error901",
                        sid,
                        event.after.line_number,
                        f"mass {event.after.mass:.3f} kg below dry mass + carried "
                        f"resources {C.DRY_MASS_KG + carried:.3f} kg",
                        event.after.mass,
                    )
                )
            state_epoch = event.after.epoch
            position = event.after.position.copy()
            velocity = event.after.velocity.copy()
            mass = event.after.mass
            previous_event = event
        for arc in pending_burns:
            violations.append(
                Violation(
                    "Error004",
                    sid,
                    arc.samples[0].line_number,
                    "burning arc after the last event has no terminating event",
                )
            )
        if miners_deployed > C.MAX_MINERS_PER_SHIP:
            violations.append(
                Violation(
                    "Error807",
                    sid,
                    None,
                    f"{miners_deployed} miners exceed {C.MAX_MINERS_PER_SHIP}",
                    float(miners_deployed),
                )
            )
        final_mass = events[-1].after.mass
        if final_mass < C.DRY_MASS_KG + carried - C.TOLERANCE_MASS_KG:
            violations.append(
                Violation(
                    "Error901",
                    sid,
                    events[-1].after.line_number,
                    "final mass below dry mass plus carried resources",
                    final_mass,
                )
            )

    def _check_arc_samples(self, sid: int, arc: BurnArc, violations: list[Violation]) -> None:
        samples = arc.samples
        if len(samples) < 4:
            violations.append(
                Violation(
                    "ErrorA03",
                    sid,
                    samples[0].line_number,
                    "number of lines in this burning arc is unreasonable",
                )
            )
        if samples[0].magnitude != 0.0:
            violations.append(
                Violation("ErrorA04", sid, samples[0].line_number, "first line thrust is not 0")
            )
        if samples[-1].magnitude != 0.0:
            violations.append(
                Violation("ErrorA06", sid, samples[-1].line_number, "last line thrust is not 0")
            )
        for previous, sample in itertools.pairwise(samples):
            gap = sample.epoch - previous.epoch
            if gap < 0.0:
                violations.append(
                    Violation("ErrorA07", sid, sample.line_number, "time interval less than 0", gap)
                )
            if gap > C.MAX_BURN_SAMPLE_INTERVAL_DAYS + SAMPLE_INTERVAL_SLACK_DAYS:
                violations.append(
                    Violation(
                        "ErrorA08", sid, sample.line_number, "time interval greater than 1 day", gap
                    )
                )
        for sample in samples:
            if sample.magnitude > C.THRUST_MAX_N + 1e-9:
                violations.append(
                    Violation(
                        "Error401",
                        sid,
                        sample.line_number,
                        f"thrust {sample.magnitude:.9f} N exceeds {C.THRUST_MAX_N} N",
                        sample.magnitude,
                    )
                )

    def _check_pair(
        self, sid: int, event: Event, codes: tuple[str, str, str], violations: list[Violation]
    ) -> None:
        time_code, position_code, mass_code = codes
        if event.before.epoch != event.after.epoch:
            violations.append(
                Violation(
                    time_code,
                    sid,
                    event.after.line_number,
                    "the time of two adjacent lines is different",
                )
            )
        if np.linalg.norm(event.before.position - event.after.position) > C.TOLERANCE_POSITION_KM:
            violations.append(
                Violation(
                    position_code,
                    sid,
                    event.after.line_number,
                    "the position of two adjacent lines is different",
                )
            )
        if mass_code and abs(event.before.mass - event.after.mass) > C.TOLERANCE_MASS_KG:
            violations.append(
                Violation(
                    mass_code,
                    sid,
                    event.after.line_number,
                    "the mass of two adjacent lines is different",
                )
            )

    def _check_launch(self, sid: int, event: Event, violations: list[Violation]) -> None:
        self._check_pair(sid, event, ("Error501", "Error502", "Error503"), violations)
        if event.before.mass > C.MAX_INITIAL_MASS_KG + C.TOLERANCE_MASS_KG:
            violations.append(
                Violation(
                    "Error504",
                    sid,
                    event.before.line_number,
                    f"initial mass {event.before.mass:.3f} kg exceeds 3000 kg",
                    event.before.mass,
                )
            )
        earth_r, earth_v = planet_state(C.EARTH, event.epoch)
        position_error = float(np.linalg.norm(event.before.position - earth_r))
        velocity_error = float(np.linalg.norm(event.before.velocity - earth_v))
        if position_error > C.TOLERANCE_POSITION_KM:
            violations.append(
                Violation(
                    "Error506",
                    sid,
                    event.before.line_number,
                    f"launch position differs from Earth by {position_error:.3f} km",
                    position_error,
                )
            )
        if velocity_error > C.TOLERANCE_VELOCITY_KM_S:
            violations.append(
                Violation(
                    "Error507",
                    sid,
                    event.before.line_number,
                    f"launch line velocity differs from Earth by {velocity_error * 1e3:.4f} m/s",
                    velocity_error,
                )
            )
        vinf = float(np.linalg.norm(event.after.velocity - earth_v))
        if vinf > C.MAX_VINF_EARTH_KM_S + C.TOLERANCE_VELOCITY_KM_S:
            violations.append(
                Violation(
                    "Error505",
                    sid,
                    event.after.line_number,
                    f"launch v-infinity {vinf:.6f} km/s exceeds 6 km/s",
                    vinf,
                )
            )

    def _check_rendezvous(self, sid: int, event: Event, violations: list[Violation]) -> None:
        self._check_pair(sid, event, ("Error701", "Error702", ""), violations)
        if np.linalg.norm(event.before.velocity - event.after.velocity) > C.TOLERANCE_VELOCITY_KM_S:
            violations.append(
                Violation(
                    "Error703",
                    sid,
                    event.after.line_number,
                    "the velocity of two adjacent lines is different",
                )
            )
        r, v = asteroid_state(self.catalogue, event.event_id, event.epoch)
        position_error = float(np.linalg.norm(event.before.position - r))
        velocity_error = float(np.linalg.norm(event.before.velocity - v))
        if position_error > C.TOLERANCE_POSITION_KM:
            violations.append(
                Violation(
                    "Error704",
                    sid,
                    event.before.line_number,
                    f"position differs from asteroid {event.event_id} by {position_error:.3f} km",
                    position_error,
                )
            )
        if velocity_error > C.TOLERANCE_VELOCITY_KM_S:
            violations.append(
                Violation(
                    "Error705",
                    sid,
                    event.before.line_number,
                    f"velocity differs from asteroid {event.event_id} by "
                    f"{velocity_error * 1e3:.4f} m/s",
                    velocity_error,
                )
            )

    def _check_flyby(
        self, sid: int, event: Event, carried: float, violations: list[Violation]
    ) -> float | None:
        planet = C.PLANETS[event.event_id]
        self._check_pair(sid, event, ("Error601", "Error602", ""), violations)
        r, v = planet_state(planet, event.epoch)
        position_error = float(np.linalg.norm(event.before.position - r))
        if position_error > C.TOLERANCE_POSITION_KM:
            violations.append(
                Violation(
                    "Error604",
                    sid,
                    event.before.line_number,
                    f"position differs from {planet.name} by {position_error:.3f} km",
                    position_error,
                )
            )
        vinf_in = event.before.velocity - v
        vinf_out = event.after.velocity - v
        speed_in = float(np.linalg.norm(vinf_in))
        speed_out = float(np.linalg.norm(vinf_out))
        if abs(speed_in - speed_out) > C.TOLERANCE_VELOCITY_KM_S:
            violations.append(
                Violation(
                    "Error607",
                    sid,
                    event.after.line_number,
                    f"v-infinity magnitude changes {speed_in:.6f} -> {speed_out:.6f} km/s",
                    speed_out - speed_in,
                )
            )
        if speed_in > 0.0 and speed_out > 0.0:
            cosine = float(np.clip(np.dot(vinf_in, vinf_out) / (speed_in * speed_out), -1.0, 1.0))
            turn = math.acos(cosine)
            ratio = planet.gravitational_parameter_km3_s2 / planet.minimum_pericentre_radius_km
            maximum = 2.0 * math.asin(ratio / (speed_in * speed_in + ratio))
            if turn > maximum + 1e-9:
                violations.append(
                    Violation(
                        "Error605",
                        sid,
                        event.after.line_number,
                        f"turn angle {math.degrees(turn):.4f} deg exceeds "
                        f"{math.degrees(maximum):.4f} deg",
                        turn,
                    )
                )
        delta_mass = event.before.mass - event.after.mass
        if event.event_id != C.EVENT_EARTH_FLYBY:
            if abs(delta_mass) > C.TOLERANCE_MASS_KG:
                violations.append(
                    Violation(
                        "Error603",
                        sid,
                        event.after.line_number,
                        "mass changes at a Venus/Mars flyby",
                        delta_mass,
                    )
                )
            return None
        if speed_in <= C.MAX_VINF_EARTH_KM_S + C.TOLERANCE_VELOCITY_KM_S:
            # unloading flyby: all carried resources leave the ship
            if abs(delta_mass - carried) > C.TOLERANCE_MASS_KG:
                violations.append(
                    Violation(
                        "Error801",
                        sid,
                        event.after.line_number,
                        f"unloaded {delta_mass:.6f} kg but the ship carries {carried:.6f} kg",
                        delta_mass - carried,
                    )
                )
            return (
                carried
                if abs(delta_mass - carried) <= C.TOLERANCE_MASS_KG
                else max(delta_mass, 0.0)
            )
        if abs(delta_mass) > C.TOLERANCE_MASS_KG:
            violations.append(
                Violation(
                    "Error806",
                    sid,
                    event.after.line_number,
                    f"resources dropped with v-infinity {speed_in:.6f} km/s > 6 km/s",
                    speed_in,
                )
            )
            return max(delta_mass, 0.0)
        return None

    def _propagate_leg(
        self,
        sid: int,
        epoch: float,
        position: FloatArray,
        velocity: FloatArray,
        mass: float,
        burns: list[BurnArc],
        previous: Event,
        target: Event,
        history: PropagatedHistory | None,
    ) -> LegCheck:
        minimum_radius = float(np.linalg.norm(position))
        current_epoch = epoch
        r, v, m = position.copy(), velocity.copy(), mass
        for arc in sorted(burns, key=lambda item: item.start):
            if arc.start > current_epoch:
                r, v, radius = propagate_coast(current_epoch, r, v, m, arc.start, history=history)
                minimum_radius = min(minimum_radius, radius)
                current_epoch = arc.start
            r, v, m, radius = propagate_burn(
                current_epoch, r, v, m, arc, rtol=self.rtol, history=history
            )
            minimum_radius = min(minimum_radius, radius)
            current_epoch = arc.end
        if target.epoch > current_epoch:
            r, v, radius = propagate_coast(current_epoch, r, v, m, target.epoch, history=history)
            minimum_radius = min(minimum_radius, radius)
        return LegCheck(
            ship_id=sid,
            from_event=previous.event_id,
            to_event=target.event_id,
            from_line=previous.after.line_number,
            to_line=target.before.line_number,
            start_epoch=epoch,
            end_epoch=target.epoch,
            burn_count=len(burns),
            position_error_km=float(np.linalg.norm(r - target.before.position)),
            velocity_error_km_s=float(np.linalg.norm(v - target.before.velocity)),
            mass_error_kg=abs(m - target.before.mass),
            minimum_sun_distance_au=minimum_radius / C.AU_KM,
        )

    # -- mining --

    def _mining_bookkeeping(
        self,
        solution: Solution,
        visits: Iterable[AsteroidVisit],
        unload_events: list[tuple[int, float, float, int]],
        violations: list[Violation],
    ) -> dict[int, MinedAsteroid]:
        by_asteroid: dict[int, list[AsteroidVisit]] = {}
        for visit in visits:
            by_asteroid.setdefault(visit.asteroid_id, []).append(visit)
        unloads_by_ship: dict[int, list[float]] = {}
        for ship_id, epoch, _mass, _line in unload_events:
            unloads_by_ship.setdefault(ship_id, []).append(epoch)
        mined: dict[int, MinedAsteroid] = {}
        for asteroid_id, items in sorted(by_asteroid.items()):
            items.sort(key=lambda item: item.epoch)
            if len(items) > 2:
                violations.append(
                    Violation(
                        "Error805",
                        items[2].ship_id,
                        items[2].line,
                        f"asteroid {asteroid_id} rendezvoused more than twice",
                    )
                )
            first = items[0]
            if abs((first.mass_before - first.mass_after) - C.MINER_MASS_KG) > C.TOLERANCE_MASS_KG:
                violations.append(
                    Violation(
                        "Error803",
                        first.ship_id,
                        first.line,
                        f"first rendezvous of asteroid {asteroid_id} changes mass by "
                        f"{first.mass_before - first.mass_after:.6f} kg, not one miner",
                    )
                )
            if len(items) < 2:
                continue
            second = items[1]
            stay_days = second.epoch - first.epoch
            collected = second.mass_after - second.mass_before
            if stay_days < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS:
                violations.append(
                    Violation(
                        "Error802",
                        second.ship_id,
                        second.line,
                        f"asteroid {asteroid_id} revisited after {stay_days:.3f} days < one year",
                        stay_days,
                    )
                )
            if (
                collected < -C.TOLERANCE_MASS_KG
                or collected > C.maximum_collected_mass(stay_days) + C.TOLERANCE_MASS_KG
            ):
                violations.append(
                    Violation(
                        "Error804",
                        second.ship_id,
                        second.line,
                        f"collected {collected:.6f} kg on asteroid {asteroid_id} is unreasonable "
                        f"for a {stay_days:.3f} day stay",
                        collected,
                    )
                )
            later_unloads = [
                epoch for epoch in unloads_by_ship.get(second.ship_id, []) if epoch >= second.epoch
            ]
            unloaded = bool(later_unloads)
            mined[asteroid_id] = MinedAsteroid(
                asteroid_id=asteroid_id,
                deploy_ship=first.ship_id,
                deploy_epoch=first.epoch,
                collect_ship=second.ship_id,
                collect_epoch=second.epoch,
                collected_mass_kg=max(collected, 0.0),
                unloaded=unloaded,
                unload_epoch=min(later_unloads) if unloaded else None,
            )
        return mined


def verify_solution_file(
    path: str | Path,
    catalogue: AsteroidCatalogue,
    *,
    bonus: BonusTable | None = None,
    rtol: float = 1e-12,
    history: dict[int, PropagatedHistory] | None = None,
) -> VerificationReport:
    return Gtoc12Verifier(catalogue, bonus=bonus, rtol=rtol, history=history).verify_file(path)
