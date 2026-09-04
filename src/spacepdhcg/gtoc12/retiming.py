"""Joint per-ship re-timing of a mining route and chain re-extension.

The beam search picks leg durations from a coarse grid and schedules the collection tour
backwards from the window end with a propellant-first criterion, so certified routes end with a
long camp, an early Earth return and 200-400 kg of unspent propellant while the chain cannot grow
because "the collection tour does not fit" (``docs/GTOC12_TRACK.md``, section 8).  This module
trades that margin for time:

* :func:`retime_plan` keeps the *visit order* of a plan fixed and re-optimises every departure
  epoch, leg duration and camp jointly by dynamic programming on a 15-day lattice covering the
  whole mission.  The objective is the (bonus-weighted) collected mass minus a Lagrangian price
  on propellant; the price is raised until the forward mass bookkeeping closes, and the thrust
  authority of every leg is checked with the forward masses (the DP uses the previous mass
  profile and iterates to a fixed point).  Because mined mass is linear in the deploy and collect
  epochs, the objective decomposes stage by stage and the DP is exact for the fixed order.
* :func:`extend_plan` turns the slack a re-timed plan exposes (camp + end gap) into more asteroids
  by resuming the beam search from the deploy phase and scheduling a new collection tour, then
  re-timing again; :func:`improve_plan` alternates the two until no proxy improvement remains.

Everything here is at proxy level (impulsive Lambert x finite-thrust inflation); the pipeline
re-flies and certifies every leg with SCvx afterwards, and the caller keeps the original plan as a
fallback when a re-timed leg turns out not to be flyable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .collectdp import RETURN_SWEEP_REACH as _RETURN_SWEEP_REACH
from .cooperative import MinerPool, orphan_credit_kg
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .low_thrust import ScvxSettings
from .pipeline import RefinedRoute, refine_route
from .screening import (
    lambert_hops,
    low_thrust_inflation,
    propellant_for_delta_v,
    return_inflation_model,
    thrust_authority_km_s,
)
from .search import (
    EARTH_ID,
    PlannedLeg,
    RoutePlan,
    RouteSearch,
    SearchSettings,
    element_deviations,
)

FloatArray = NDArray[np.float64]
# grid cells within this many lattice steps (departure or TOF) of an SCvx-swept return cell are
# priced by the nearest swept cell's measured inflation (2 steps = 30 days); the collect DP's
# pair table applies the same reach (``collectdp.RETURN_SWEEP_REACH``)
RETURN_SWEEP_REACH = _RETURN_SWEEP_REACH


@dataclass(frozen=True, slots=True)
class RetimeSettings:
    step_days: float = 15.0
    hop_tof_days: tuple[float, float] = (60.0, 720.0)
    # every certified Earth leg so far lasted >= 400 days; the v∞ credit makes the Lambert ΔV of
    # faster Earth legs understate the propulsive effort (a 330-day return at ratio 0.34 failed)
    earth_tof_days: tuple[float, float] = (400.0, 900.0)
    camp_max_days: float = 720.0  # camps at ordinary visits
    long_camp_max_days: float = 8.0 * C.YEAR_DAYS  # the deploy+collect (camp) asteroid
    end_margin_days: float = 2.0
    # Propellant inflation (true low-thrust ΔV / zero-revolution Lambert ΔV) and authority-ratio
    # limits per leg role; ``None`` inherits the calibrated ``SearchSettings`` values.  The hop
    # ratio is tightened to 0.45 here because the re-timer deliberately pushes hops towards the
    # limit (every certified hop had ratio <= 0.49; half of those at >= 0.49 failed SCvx).
    earth_out_inflation: float | None = None
    earth_return_inflation: float | None = None
    hop_inflation: float | None = None
    # Ratio-dependent hop inflation ``floor + slope x (Lambert ΔV / full authority)`` (see
    # ``screening.low_thrust_inflation``; fitted on 1674 certified hops).  ``None`` keeps the
    # flat ``hop_inflation``.  With the model, a slow hop (ratio 0.15) costs 1.15x Lambert and a
    # hop at the limit (0.5) 1.38x, so the DP stops treating fast and slow hops alike and the
    # margin it spends buys cheaper, longer hops; per-pair calibration becomes a residual factor.
    hop_inflation_slope: float | None = 0.65
    hop_inflation_floor: float = 1.05
    hop_authority_ratio: float | None = 0.45
    # Earth return priced with the archive's TOF/ratio model (``screening.return_inflation_model``)
    # times the pair's calibrated residual, instead of one flat factor for every TOF: the flat
    # factor made a 420-day return look as cheap as a 540-day one although SCvx charges 1.30x vs
    # 0.96x Lambert, so the DP spent its margin on a late, short, expensive return.  ``None``
    # inherits ``SearchSettings.earth_return_tof_model``.
    return_tof_model: bool | None = None
    earth_out_authority_ratio: float | None = None
    earth_return_authority_ratio: float | None = None
    # a leg SCvx proved infeasible bans ratios >= ban_factor x its ratio for that body pair
    ban_factor: float = 0.9
    # a certified leg calibrates its pair's inflation to (SCvx ΔV / Lambert ΔV) x this margin
    calibration_margin: float = 1.03
    propellant_price: float = 0.15  # kg of objective per kg of propellant (start value)
    # cooperative collection: fleet value credited per kg another ship could mine from a miner
    # this ship deploys and leaves (orphan), and the end-of-mission margin assumed for that.
    # Off by default: at 0.5 the fleet6_coop_v1 run left nine orphans nobody collected (the
    # next ship's beam lands in another cluster and cross-cluster collect hops are DP-infeasible)
    # and lost 103 kg against the self-cleaning fleet.  With 0 the extension still collects
    # existing orphans (foreign collects) but never speculates on leaving its own.
    orphan_credit: float = 0.0
    orphan_margin_days: float = 400.0
    price_growth: float = 2.0
    max_price_rounds: int = 8
    max_mass_rounds: int = 4
    lambert_chunk: int = 4096


@dataclass(frozen=True, slots=True)
class Visit:
    body: int
    deploy: bool
    collect: bool
    role_out: str  # role of the leg departing this visit ("" for the final Earth visit)
    # cooperative collection: the (fixed) epoch another ship deployed the miner collected here
    foreign_deploy_epoch: float | None = None
    # a deploy another ship already collects at this epoch: the re-timing must arrive exactly then
    pinned_arrival: float | None = None


@dataclass(slots=True)
class RetimeResult:
    plan: RoutePlan | None
    original: RoutePlan
    objective_before: float
    objective_after: float
    price: float
    mass_rounds: int
    price_rounds: int
    lambert_evaluations: int
    wall_seconds: float
    failure: str = ""

    @property
    def improved(self) -> bool:
        return self.plan is not None and self.objective_after > self.objective_before + 1e-9

    def summary(self) -> dict[str, Any]:
        return {
            "improved": self.improved,
            "objective_before_kg": self.objective_before,
            "objective_after_kg": self.objective_after,
            "collected_before_kg": self.original.total_collected_kg,
            "collected_after_kg": None if self.plan is None else self.plan.total_collected_kg,
            "final_mass_before_kg": self.original.final_mass_proxy_kg,
            "final_mass_after_kg": None if self.plan is None else self.plan.final_mass_proxy_kg,
            "deploy_phase_days_before": deploy_phase_days(self.original),
            "deploy_phase_days_after": None if self.plan is None else deploy_phase_days(self.plan),
            "camp_days_before": camp_days(self.original),
            "camp_days_after": None if self.plan is None else camp_days(self.plan),
            "end_gap_days_before": C.MISSION_END_MJD - self.original.legs[-1].arrival_epoch,
            "end_gap_days_after": None
            if self.plan is None
            else C.MISSION_END_MJD - self.plan.legs[-1].arrival_epoch,
            "propellant_price": self.price,
            "mass_rounds": self.mass_rounds,
            "price_rounds": self.price_rounds,
            "lambert_evaluations": self.lambert_evaluations,
            "wall_seconds": self.wall_seconds,
            "failure": self.failure,
        }


def deploy_phase_days(plan: RoutePlan) -> float:
    deploys = [leg for leg in plan.legs if leg.role in ("earth_out", "deploy_hop")]
    return deploys[-1].arrival_epoch - deploys[0].arrival_epoch


def camp_days(plan: RoutePlan) -> float:
    return sum(leg.tof_days for leg in plan.legs if leg.role == "camp")


def weighted_collected(plan: RoutePlan, weights: dict[int, float] | None) -> float:
    if weights is None:
        return plan.total_collected_kg
    return sum(weights.get(a, 1.0) * m for a, m in plan.collected_mass.items())


def plan_value(plan: RoutePlan, retimer: Retimer) -> float:
    """Bonus-weighted collected mass plus the credited fleet value of orphaned miners."""

    s = retimer.settings
    return weighted_collected(plan, retimer.weights) + orphan_credit_kg(
        plan, retimer.weights, s.orphan_credit, s.orphan_margin_days
    )


def visits_of(plan: RoutePlan) -> tuple[list[Visit], list[float], list[float]]:
    """Fixed visit order of a plan with its arrival and departure epochs (camps folded in)."""

    legs = [leg for leg in plan.legs if leg.role != "camp"]
    visits: list[Visit] = [Visit(EARTH_ID, False, False, legs[0].role)]
    arrivals: list[float] = [legs[0].departure_epoch]
    departures: list[float] = [legs[0].departure_epoch]
    for index, leg in enumerate(legs):
        body = leg.to_id
        arrival = leg.arrival_epoch
        departure = legs[index + 1].departure_epoch if index + 1 < len(legs) else arrival
        if body == EARTH_ID:
            visits.append(Visit(EARTH_ID, False, False, ""))
        else:
            deploy = body in plan.deploy_epochs and abs(plan.deploy_epochs[body] - arrival) < 1e-6
            collect = body in plan.collect_epochs and (
                abs(plan.collect_epochs[body] - departure) < 1e-6
                or abs(plan.collect_epochs[body] - arrival) < 1e-6
            )
            visits.append(Visit(body, deploy, collect, legs[index + 1].role))
        arrivals.append(arrival)
        departures.append(departure)
    return visits, arrivals, departures


class _Lattice:
    def __init__(self, step: float, end_margin: float) -> None:
        self.step = step
        self.start = C.MISSION_START_MJD
        self.count = int(np.floor((C.MISSION_END_MJD - end_margin - self.start) / step)) + 1
        self.epochs = self.start + step * np.arange(self.count)

    def index(self, epoch: float) -> int:
        return round((epoch - self.start) / self.step)

    def exact_index(self, epoch: float, tolerance: float = 1e-6) -> int | None:
        """Lattice index of an epoch that lies on the lattice (within ``tolerance`` days)."""

        k = self.index(epoch)
        if k < 0 or k >= self.count or abs(self.epochs[k] - epoch) > tolerance:
            return None
        return k


class Retimer:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        search_settings: SearchSettings | None = None,
        settings: RetimeSettings | None = None,
        weights: dict[int, float] | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.search_settings = search_settings or SearchSettings()
        self.settings = settings or RetimeSettings()
        self.weights = weights
        self.lambert_evaluations = 0
        self.bans: dict[tuple[int, int], float] = {}
        self.inflations: dict[tuple[int, int], float] = {}
        self._tables: dict[tuple[int, int, str], tuple[FloatArray, NDArray[np.bool_]]] = {}
        self.lattice = _Lattice(self.settings.step_days, self.settings.end_margin_days)
        # Earth-out TOF floor (days): set from a certified, continuously optimised Earth leg so
        # the DP may only keep or lengthen it.  Earth legs are nearly thrust-saturated and their
        # Lambert proxy barely depends on TOF, so without the floor the DP shortens the leg to
        # buy hop time and SCvx then costs 60-200 kg more (earthleg.py).
        self.earth_out_tof_floor: float | None = None
        # SCvx sweeps of the Earth return per departure asteroid (``returnsweep.ReturnSweep``):
        # where swept, the DP prices the return with the measured ΔV and SCvx's own feasibility
        # instead of the Lambert proxy; unswept cells take the nearest swept cell's inflation
        self.return_sweeps: dict[int, Any] = {}
        self._return_tables: dict[int, tuple[FloatArray, FloatArray]] = {}

    def release_caches(self) -> int:
        """Drop the memoised Lambert leg tables (one ``(n_t, n_tof)`` pair per leg and role;
        recomputed on demand).  Bans, inflations and sweeps - the re-timer's learnt state - stay.
        Returns the number of tables released."""

        released = len(self._tables) + len(self._return_tables)
        self._tables.clear()
        self._return_tables.clear()
        return released

    # -- Lambert tables ------------------------------------------------------------------

    def _tofs(self, role: str) -> FloatArray:
        s = self.settings
        lo, hi = s.earth_tof_days if role in ("earth_out", "earth_return") else s.hop_tof_days
        # the grid must sit on the lattice: the DP realises a TOF as ``round(tof / step)`` steps,
        # so an off-lattice bound (400 d on a 15- or 30-day lattice) makes the DP price and
        # authority-check a leg at one TOF and fly it at another - the forward pass then refuses
        # legs the DP accepted and the mass rounds cannot converge
        lo = s.step_days * math.ceil(lo / s.step_days - 1e-9)
        hi = s.step_days * math.floor(hi / s.step_days + 1e-9)
        if role == "earth_out" and self.earth_out_tof_floor is not None:
            # snap the floor onto the lattice so the certified TOF itself stays admissible
            floor = s.step_days * math.floor(self.earth_out_tof_floor / s.step_days + 1e-9)
            lo = max(lo, floor)
        hi = max(hi, lo)
        return np.arange(lo, hi + 1e-9, s.step_days)

    def protect_earth_leg(self, plan: RoutePlan) -> None:
        """Keep the DP's Earth-out TOF at or above the plan's (certified) Earth-leg TOF."""

        first = plan.legs[0] if plan.legs else None
        if first is None or first.role != "earth_out":
            return
        floor = float(first.tof_days)
        if self.earth_out_tof_floor is None or floor != self.earth_out_tof_floor:
            self.earth_out_tof_floor = floor
            self._tables = {k: v for k, v in self._tables.items() if k[2] != "earth_out"}

    def set_return_sweep(self, sweep: Any) -> None:
        """Price the Earth return from ``sweep.asteroid`` with the SCvx-measured sweep."""

        self.return_sweeps[int(sweep.asteroid)] = sweep
        self._return_tables.pop(int(sweep.asteroid), None)

    def _return_override(self, asteroid: int) -> tuple[FloatArray, FloatArray] | None:
        """``(inflation, ok)`` on the DP's return grid from the asteroid's SCvx sweep.

        ``inflation`` is the measured (SCvx ΔV / Lambert ΔV) of the nearest swept grid cell
        (index distance on the lattice x TOF grid) and ``ok`` its SCvx feasibility, so every
        cell is priced by what SCvx did on the closest flown return rather than by the model;
        the flown cells themselves carry their own measurement.  ``None`` without a sweep.
        """

        sweep = self.return_sweeps.get(asteroid)
        if sweep is None:
            return None
        if asteroid in self._return_tables:
            return self._return_tables[asteroid]
        tofs = self._tofs("earth_return")
        step = self.settings.step_days
        dv_table, _ = self.leg_table(asteroid, EARTH_ID, "earth_return")
        cells: list[tuple[int, int, float, bool]] = []
        for i, departure in enumerate(sweep.departures):
            k = self.lattice.exact_index(float(departure))
            if k is None:
                continue
            for j, tof in enumerate(sweep.tofs):
                t = round((float(tof) - tofs[0]) / step)
                if not (0 <= t < tofs.shape[0]) or abs(tofs[t] - tof) > 1e-6:
                    continue
                if not bool(sweep.attempted[i, j]):
                    continue  # arrival past the window: not flown, not a refusal
                certified = bool(sweep.certified[i, j])
                lambert = float(dv_table[k, t])
                measured = float(sweep.delta_v_km_s[i, j])
                inflation = measured / lambert if certified and lambert > 1e-9 else np.nan
                cells.append((k, t, inflation, certified and np.isfinite(inflation)))
        if not cells:
            return None
        ks = np.asarray([c[0] for c in cells])
        ts = np.asarray([c[1] for c in cells])
        infl = np.asarray([c[2] for c in cells])
        oks = np.asarray([c[3] for c in cells])
        grid_k, grid_t = np.meshgrid(
            np.arange(self.lattice.count), np.arange(tofs.shape[0]), indexing="ij"
        )
        # nearest swept cell in index space (departure steps count like TOF steps: both are
        # one lattice step of 15 days); ties -> the earlier sweep entry (deterministic).  Cells
        # farther than ``RETURN_SWEEP_REACH`` steps from every swept cell keep the model (NaN).
        distance = (grid_k[:, :, None] - ks[None, None, :]) ** 2 + (
            grid_t[:, :, None] - ts[None, None, :]
        ) ** 2
        nearest = np.argmin(distance, axis=2)
        reached = np.min(distance, axis=2) <= RETURN_SWEEP_REACH**2
        inflation = np.where(reached & oks[nearest], infl[nearest], np.nan)
        # strict: with a sweep in hand the return may only be re-timed onto (or right next to)
        # a cell SCvx certified - cells beyond the sweep's reach and cells next to a refusal are
        # infeasible.  Pricing them by the model instead is what the first experiment did, and
        # SCvx refused the model's pick (28079 -> Earth, 450 d at ratio 0.34).
        ok = reached & oks[nearest]
        table = (inflation, ok)
        self._return_tables[asteroid] = table
        return table

    def refuse_return(self, asteroid: int, departure: float, tof_days: float) -> bool:
        """Mark the swept return cell nearest ``(departure, tof)`` as refused (SCvx did not
        certify the re-flown leg there) so the next re-timing avoids it; False without a sweep."""

        sweep = self.return_sweeps.get(int(asteroid))
        if sweep is None:
            return False
        i = int(np.argmin(np.abs(sweep.departures - float(departure))))
        j = int(np.argmin(np.abs(sweep.tofs - float(tof_days))))
        if not bool(sweep.certified[i, j]):
            return False
        sweep.certified[i, j] = False
        sweep.propellant_kg[i, j] = np.inf
        sweep.diagnostics.append(
            {"cell": [float(sweep.departures[i]), float(sweep.tofs[j])], "refused": "re-flight"}
        )
        self._return_tables.pop(int(asteroid), None)
        return True

    def _state(self, body: int, epochs: FloatArray) -> tuple[FloatArray, FloatArray]:
        if body == EARTH_ID:
            return earth_state(epochs)
        return asteroid_state(self.catalogue, np.full(epochs.shape[0], body), epochs)

    def leg_table(self, from_body: int, to_body: int, role: str) -> tuple[FloatArray, NDArray]:
        """Proxy ``dv[departure lattice index, tof index]`` for one body pair (cached)."""

        key = (from_body, to_body, role)
        if key in self._tables:
            return self._tables[key]
        tofs = self._tofs(role)
        departures = self.lattice.epochs
        d_idx, t_idx = np.meshgrid(
            np.arange(departures.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
        )
        d_idx, t_idx = d_idx.ravel(), t_idx.ravel()
        dv = np.full(d_idx.shape[0], np.inf)
        feasible = np.zeros(d_idx.shape[0], dtype=bool)
        chunk = self.settings.lambert_chunk
        for start in range(0, d_idx.shape[0], chunk):
            sl = slice(start, start + chunk)
            t_dep = departures[d_idx[sl]]
            t_arr = t_dep + tofs[t_idx[sl]]
            r1, v1 = self._state(from_body, t_dep)
            r2, v2 = self._state(to_body, t_arr)
            hop = lambert_hops(
                r1,
                v1,
                r2,
                v2,
                t_dep,
                tofs[t_idx[sl]],
                departure_allowance_km_s=C.MAX_VINF_EARTH_KM_S if from_body == EARTH_ID else 0.0,
                arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S if to_body == EARTH_ID else 0.0,
            )
            self.lambert_evaluations += 2 * t_dep.shape[0]
            dv[sl] = np.where(hop.feasible, hop.total_delta_v, np.inf)
            feasible[sl] = hop.feasible & np.isfinite(hop.total_delta_v)
        shape = (departures.shape[0], tofs.shape[0])
        self._tables[key] = (dv.reshape(shape), feasible.reshape(shape))
        return self._tables[key]

    # -- dynamic programme ----------------------------------------------------------------

    def _limits(self, role: str, from_body: int, to_body: int) -> tuple[float, float]:
        """(propellant inflation, authority ratio limit) for a leg; bans tighten the ratio."""

        s = self.settings
        inflation, ratio = self.search_settings_limits(role)
        if role == "earth_out":
            overrides = (s.earth_out_inflation, s.earth_out_authority_ratio)
        elif role == "earth_return":
            overrides = (s.earth_return_inflation, s.earth_return_authority_ratio)
        else:
            overrides = (s.hop_inflation, s.hop_authority_ratio)
        inflation = inflation if overrides[0] is None else overrides[0]
        ratio = ratio if overrides[1] is None else overrides[1]
        if not self._modelled(role):
            inflation = self.inflations.get((from_body, to_body), inflation)
        return inflation, min(ratio, self.bans.get((from_body, to_body), np.inf))

    def _ratio_model(self, role: str) -> bool:
        """True when hops of this role are priced with the ratio-dependent inflation model."""

        return self.settings.hop_inflation_slope is not None and role in (
            "deploy_hop",
            "collect_hop",
        )

    def _return_model(self, role: str) -> bool:
        """True when the Earth return is priced with the TOF/ratio model."""

        if role != "earth_return":
            return False
        flag = self.settings.return_tof_model
        return self.search_settings.earth_return_tof_model if flag is None else flag

    def _modelled(self, role: str) -> bool:
        """Roles whose pair calibration is a residual on a model rather than a flat factor."""

        return self._ratio_model(role) or self._return_model(role)

    def leg_inflation(
        self,
        role: str,
        from_body: int,
        to_body: int,
        delta_v_km_s: FloatArray | float,
        mass_kg: float,
        tof_days: FloatArray | float,
    ) -> FloatArray | float:
        """Propellant inflation of a leg: flat per role (Earth-out, legacy hops) or a model -
        ratio-dependent for hops, TOF/ratio for the return - times the pair's calibrated
        residual."""

        base, _ = self._limits(role, from_body, to_body)
        if self._return_model(role):
            authority = thrust_authority_km_s(mass_kg, tof_days, 1.0)
            dv = np.asarray(delta_v_km_s, dtype=np.float64)
            ratio = np.where(np.isfinite(dv), dv, 0.0) / np.maximum(authority, 1e-12)
            model = return_inflation_model(tof_days, ratio)
            return model * self.inflations.get((from_body, to_body), 1.0)
        if not self._ratio_model(role):
            return base
        s = self.settings
        model = low_thrust_inflation(
            delta_v_km_s,
            mass_kg,
            tof_days,
            floor=s.hop_inflation_floor,
            slope=s.hop_inflation_slope,
        )
        return model * self.inflations.get((from_body, to_body), 1.0)

    def search_settings_limits(self, role: str) -> tuple[float, float]:
        s = self.search_settings
        if role == "earth_out":
            return s.earth_out_inflation, s.earth_out_authority_ratio
        if role == "earth_return":
            return s.earth_return_inflation, s.earth_return_authority_ratio
        return s.hop_inflation, s.hop_authority_ratio

    def ban(self, from_body: int, to_body: int, ratio: float) -> None:
        """Record that a leg of this body pair at authority ratio ``ratio`` was not flyable."""

        limit = self.settings.ban_factor * ratio
        self.bans[(from_body, to_body)] = min(self.bans.get((from_body, to_body), np.inf), limit)

    def calibrate(
        self,
        from_body: int,
        to_body: int,
        measured_inflation: float,
        *,
        authority_ratio: float | None = None,
        tof_days: float | None = None,
    ) -> None:
        """Use the SCvx-measured ΔV ratio (x safety margin) as this pair's propellant inflation.

        With the ratio-dependent hop model the pair's entry is the *residual* measured / model
        at the flown ``authority_ratio`` (floored at 0.95), so a hop flown fast does not inflate
        the same pair's slow hops by its own finite-thrust penalty.  Likewise a return under the
        TOF model calibrates the residual at the flown ``tof_days`` and ratio, so a 420-day
        return flown at 1.3x does not price the same asteroid's 540-day return at 1.3x.
        """

        s = self.settings
        hop = from_body != EARTH_ID and to_body != EARTH_ID
        if hop and authority_ratio is not None and s.hop_inflation_slope is not None:
            model = s.hop_inflation_floor + s.hop_inflation_slope * max(authority_ratio, 0.0)
            value = max(measured_inflation / model, 0.95) * s.calibration_margin
        elif (
            to_body == EARTH_ID
            and from_body != EARTH_ID
            and authority_ratio is not None
            and tof_days is not None
            and self._return_model("earth_return")
        ):
            model = float(return_inflation_model(tof_days, max(authority_ratio, 0.0)))
            value = max(measured_inflation / model, 0.95) * s.calibration_margin
        else:
            value = max(measured_inflation, 1.0) * s.calibration_margin
        previous = self.inflations.get((from_body, to_body))
        self.inflations[(from_body, to_body)] = value if previous is None else max(previous, value)

    @staticmethod
    def authority_ratio(delta_v_km_s: float, mass_kg: float, tof_days: float) -> float:
        """Lambert ΔV over the full-thrust authority ``T_max / m x TOF``."""

        return delta_v_km_s / float(thrust_authority_km_s(mass_kg, tof_days, 1.0))

    def _dp(
        self,
        visits: list[Visit],
        masses: list[float],
        price: float,
    ) -> tuple[list[int], list[int], float] | None:
        """Exact DP over the lattice for a fixed visit order and a fixed per-leg mass profile.

        Returns lattice indices of arrivals and departures per visit and the objective.
        """

        s = self.settings
        lat = self.lattice
        n = lat.count
        step = s.step_days
        rate = C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS
        neg_inf = -np.inf
        value = np.full(n, neg_inf)
        value[0:] = 0.0  # launch may happen at any lattice epoch
        back_arrival: list[NDArray[np.int64]] = []  # per stage j>=1: arrival index -> dep idx(j-1)
        back_tof: list[NDArray[np.int64]] = []
        back_camp: list[NDArray[np.int64]] = []  # per stage j>=0: departure index -> arrival idx
        # bodies this ship collects (at the deploy visit itself or at a later visit): their deploy
        # epoch costs the full mining rate; a deploy nobody in this plan collects is an orphan
        collected_here = {visit.body for visit in visits if visit.collect}
        for j, visit in enumerate(visits[:-1]):
            weight = 1.0 if self.weights is None else self.weights.get(visit.body, 1.0)
            # -- camp layer: arrival t -> departure d
            if visit.body == EARTH_ID:
                camp_max = 0
                camp_min = 0
            elif visit.deploy and visit.collect:
                camp_max = int(s.long_camp_max_days // step)
                camp_min = int(np.ceil(C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS / step))
            else:
                camp_max = int(s.camp_max_days // step)
                camp_min = 0
            arrival_term = value.copy()
            if visit.deploy and visit.body in collected_here:
                arrival_term -= weight * rate * lat.epochs
            elif visit.deploy:
                # orphan left for another ship: the fleet gains (credit x) mass mined from here
                arrival_term -= s.orphan_credit * weight * rate * lat.epochs
            departure_value = np.full(n, neg_inf)
            departure_back = np.zeros(n, dtype=np.int64)
            for camp in range(camp_min, camp_max + 1):
                if camp >= n:
                    break
                shifted = np.full(n, neg_inf)
                shifted[camp:] = arrival_term[: n - camp]
                better = shifted > departure_value
                departure_value = np.where(better, shifted, departure_value)
                departure_back = np.where(better, np.arange(n) - camp, departure_back)
            if visit.collect:
                departure_value += weight * rate * lat.epochs
                if visit.foreign_deploy_epoch is not None:
                    # another ship's miner: collect at least one year after *its* deploy epoch
                    earliest = visit.foreign_deploy_epoch + C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
                    departure_value[lat.epochs < earliest - 1e-9] = neg_inf
            back_camp.append(departure_back)
            # -- leg layer: departure d -> next arrival t'
            role = visit.role_out
            nxt = visits[j + 1]
            dv, feasible = self.leg_table(visit.body, nxt.body, role)
            tofs = self._tofs(role)
            _flat, ratio_limit = self._limits(role, visit.body, nxt.body)
            mass = masses[j]
            authority = thrust_authority_km_s(mass, tofs, 1.0)
            ok = feasible & (dv <= ratio_limit * authority[None, :])
            safe_dv = np.where(np.isfinite(dv), dv, 0.0)
            infl_p = self.leg_inflation(role, visit.body, nxt.body, safe_dv, mass, tofs[None, :])
            override = self._return_override(visit.body) if role == "earth_return" else None
            if override is not None:
                # swept return: SCvx's feasibility replaces the authority check and the measured
                # inflation replaces the model where a swept cell is near enough
                swept_inflation, swept_ok = override
                measured = ~np.isnan(swept_inflation)
                ok = np.where(measured, feasible & swept_ok, ok & swept_ok)
                infl_p = np.where(measured, swept_inflation, infl_p)
            propellant = propellant_for_delta_v(mass, dv * infl_p)
            leg_value = np.where(ok, departure_value[:, None] - price * propellant, neg_inf)
            next_value = np.full(n, neg_inf)
            next_back = np.zeros(n, dtype=np.int64)
            next_tof = np.zeros(n, dtype=np.int64)
            for t_index, tof in enumerate(tofs):
                shift = round(tof / step)
                if shift >= n:
                    break
                shifted = np.full(n, neg_inf)
                shifted[shift:] = leg_value[: n - shift, t_index]
                better = shifted > next_value
                next_value = np.where(better, shifted, next_value)
                next_back = np.where(better, np.arange(n) - shift, next_back)
                next_tof = np.where(better, t_index, next_tof)
            back_arrival.append(next_back)
            back_tof.append(next_tof)
            if nxt.pinned_arrival is not None:
                # another ship collects this miner against exactly this deploy epoch: the only
                # admissible arrival is the pinned lattice epoch (off-lattice pins are infeasible)
                k = lat.exact_index(nxt.pinned_arrival)
                if k is None:
                    return None
                pinned_value = np.full(n, neg_inf)
                pinned_value[k] = next_value[k]
                next_value = pinned_value
            value = next_value
        if not np.any(np.isfinite(value)):
            return None
        final_index = int(np.argmax(value))  # the Earth return arrival; ties -> earliest
        objective = float(value[final_index])
        arrivals = [0] * len(visits)
        departures = [0] * len(visits)
        arrivals[-1] = final_index
        departures[-1] = final_index
        for j in range(len(visits) - 2, -1, -1):
            departures[j] = int(back_arrival[j][arrivals[j + 1]])
            arrivals[j] = int(back_camp[j][departures[j]])
        return arrivals, departures, objective

    # -- forward bookkeeping ------------------------------------------------------------

    def _forward(
        self, visits: list[Visit], arrivals: list[float], departures: list[float]
    ) -> tuple[RoutePlan | None, list[float], str]:
        """Rebuild a RoutePlan with forward masses; returns (plan, per-leg masses, failure)."""

        legs: list[PlannedLeg] = []
        deploy: dict[int, float] = {}
        collect: dict[int, float] = {}
        foreign: dict[int, float] = {}
        for visit, arrival, departure in zip(visits, arrivals, departures, strict=True):
            if visit.deploy:
                deploy[visit.body] = arrival
            if visit.collect:
                collect[visit.body] = departure
                if visit.body not in deploy:  # deploy visits precede collect visits
                    if visit.foreign_deploy_epoch is None:
                        return None, [], "collect_without_deploy"
                    foreign[visit.body] = visit.foreign_deploy_epoch
        for asteroid in collect:
            deployed_at = deploy[asteroid] if asteroid in deploy else foreign[asteroid]
            if collect[asteroid] - deployed_at < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS - 1e-6:
                return None, [], "stay_too_short"
        collected = {
            a: C.maximum_collected_mass(collect[a] - (deploy[a] if a in deploy else foreign[a]))
            for a in collect
        }
        mass = self.search_settings.initial_mass
        masses: list[float] = []
        propellant_total = 0.0
        for j, visit in enumerate(visits[:-1]):
            nxt = visits[j + 1]
            if departures[j] > arrivals[j] + 1e-9:
                legs.append(
                    PlannedLeg(visit.body, visit.body, arrivals[j], departures[j], 0.0, 1.0, "camp")
                )
            if visit.collect:
                mass += collected[visit.body]
            role = visit.role_out
            dv_table, _ = self.leg_table(visit.body, nxt.body, role)
            tofs = self._tofs(role)
            d_index = self.lattice.index(departures[j])
            t_index = round((arrivals[j + 1] - departures[j]) / self.settings.step_days)
            t_index -= round(tofs[0] / self.settings.step_days)
            if not (0 <= t_index < tofs.shape[0]):
                return None, [], "tof_outside_grid"
            dv = float(dv_table[d_index, t_index])
            if not np.isfinite(dv):
                return None, [], "leg_infeasible"
            _flat, ratio_limit = self._limits(role, visit.body, nxt.body)
            tof = arrivals[j + 1] - departures[j]
            override = self._return_override(visit.body) if role == "earth_return" else None
            swept = override is not None and not np.isnan(override[0][d_index, t_index])
            if override is not None and not override[1][d_index, t_index]:
                return None, [], "leg_infeasible"  # SCvx refused this (or the nearest) return
            if not swept and self.authority_ratio(dv, mass, tof) > ratio_limit:
                # include the refused leg's departure mass: it is the entry of the DP's profile
                # that was too optimistic, and the mass rounds correct exactly that entry
                return None, [*masses, mass], "leg_authority"
            masses.append(mass)
            if swept:
                infl_p = float(override[0][d_index, t_index])  # type: ignore[index]
            else:
                infl_p = float(self.leg_inflation(role, visit.body, nxt.body, dv, mass, tof))
            propellant = float(propellant_for_delta_v(mass, dv * infl_p))
            propellant_total += propellant
            mass -= propellant
            legs.append(
                PlannedLeg(visit.body, nxt.body, departures[j], arrivals[j + 1], dv, infl_p, role)
            )
            if nxt.deploy:
                mass -= C.MINER_MASS_KG
        if mass < C.DRY_MASS_KG + sum(collected.values()) - 1e-9:
            return None, masses, "mass_below_dry_plus_collected"
        plan = RoutePlan(tuple(legs), deploy, collect, collected, propellant_total, mass, foreign)
        return plan, masses, ""

    # -- driver -------------------------------------------------------------------------

    def retime_order(
        self,
        deploy_order: list[int],
        collect_order: list[int],
        profile: list[float],
        *,
        before: float = -np.inf,
        original: RoutePlan | None = None,
        foreign: dict[int, float] | None = None,
        pinned: dict[int, float] | None = None,
    ) -> RetimeResult:
        """Re-time the visit order ``Earth -> deploys -> collects -> Earth`` from scratch.

        ``profile`` is the per-leg mass guess the DP starts from; the propellant price is raised
        geometrically until the forward bookkeeping closes (or lowered geometrically while it
        keeps closing) and then bisected twice towards the last failing price so the margin is
        spent rather than left over.  ``foreign`` gives the deploy epochs of miners another ship
        deployed (cooperative collection); ``pinned`` the deploy epochs of this ship's miners that
        another ship collects, which the re-timed plan must reproduce exactly.
        """

        started = time.perf_counter()
        s = self.settings
        try:
            visits = build_visits(deploy_order, collect_order, foreign, pinned)
        except ValueError as error:
            # an ill-formed order (duplicate visit, collect without a deployer) is a failed
            # re-timing, not a crash of the pricing worker
            return RetimeResult(
                None,
                original if original is not None else _empty_plan(),
                before,
                before,
                s.propellant_price,
                0,
                0,
                self.lambert_evaluations,
                time.perf_counter() - started,
                f"invalid_order: {error}",
            )
        if len(profile) != len(visits) - 1:
            raise ValueError("mass profile must have one entry per leg")
        best: RoutePlan | None = None
        best_objective = -np.inf
        failure = ""
        price_rounds = 0
        mass_rounds = 0
        price_lo: float | None = None  # highest price known not to close the mass budget
        price_hi: float | None = None  # lowest price known to close it
        price = s.propellant_price
        best_price = price
        bisections = 0
        profile = list(profile)
        while price_rounds < s.max_price_rounds:
            price_rounds += 1
            # the mass-profile corrections (forward masses heavier than guessed) describe the visit
            # order, not the price: carry them over so each price round does not restart the
            # leg-by-leg convergence from the original guess
            candidate, failure, rounds, profile = self._solve_at_price(visits, profile, price)
            mass_rounds = max(mass_rounds, rounds)
            if candidate is None:
                if failure not in ("mass_below_dry_plus_collected", "leg_authority"):
                    break
                # a higher propellant price also makes the DP pick less aggressive legs, which
                # is the remedy when the mass-profile iteration did not converge
                price_lo = price
                if price_hi is None:
                    price *= s.price_growth
                    continue
            else:
                objective = plan_value(candidate, self)
                if objective > best_objective:
                    best, best_objective, best_price = candidate, objective, price
                price_hi = price if price_hi is None else min(price_hi, price)
                if price_lo is None:
                    # closed without ever failing: keep lowering the price (spending more of the
                    # margin on faster hops) until the mass budget stops closing, then bisect
                    price /= s.price_growth
                    continue
            if bisections >= 2:
                break
            bisections += 1
            price = 0.5 * (price_lo + price_hi)  # type: ignore[operator]
        return RetimeResult(
            best,
            original if original is not None else best if best is not None else _empty_plan(),
            before,
            best_objective if best is not None else before,
            best_price if best is not None else price,
            mass_rounds,
            price_rounds,
            self.lambert_evaluations,
            time.perf_counter() - started,
            failure if best is None else "",
        )

    def _solve_at_price(
        self, visits: list[Visit], profile: list[float], price: float
    ) -> tuple[RoutePlan | None, str, int, list[float]]:
        """DP + forward bookkeeping at one price; returns (plan, failure, rounds, last profile)."""

        s = self.settings
        failure = ""
        for mass_round in range(s.max_mass_rounds):
            dp = self._dp(visits, profile, price)
            if dp is None:
                return None, "dp_infeasible", mass_round + 1, profile
            a_idx, d_idx, _objective = dp
            arrivals = [float(self.lattice.epochs[i]) for i in a_idx]
            departures = [float(self.lattice.epochs[i]) for i in d_idx]
            candidate, forward_masses, failure = self._forward(visits, arrivals, departures)
            if candidate is not None:
                return candidate, "", mass_round + 1, profile
            if failure == "leg_authority" and forward_masses:
                # the DP used an optimistic mass profile: replace the leading masses with the
                # forward ones and make the rest at least as heavy (conservative -> converges)
                scale = forward_masses[-1] / max(profile[len(forward_masses) - 1], 1e-9)
                profile = forward_masses + [
                    max(m, m * scale) for m in profile[len(forward_masses) :]
                ]
                continue
            return None, failure, mass_round + 1, profile
        return None, failure, s.max_mass_rounds, profile

    def retime(self, plan: RoutePlan) -> RetimeResult:
        """Re-time an existing plan keeping its visit order."""

        deploy_order, collect_order = orders_of(plan)
        return self.retime_order(
            deploy_order,
            collect_order,
            self._plan_masses(plan),
            before=plan_value(plan, self),
            original=plan,
            foreign=plan.foreign_deploy_epochs,
        )

    def _plan_masses(self, plan: RoutePlan) -> list[float]:
        """Per-leg (non-camp) initial masses of a plan under the re-timer's own inflations.

        Using the re-timer's inflations (not the plan's) keeps the DP's mass profile consistent
        with its forward bookkeeping; otherwise the cheaper Earth leg leaves the ship ~100 kg
        heavier than the profile and every hop at the authority limit fails the forward check.
        """

        mass = self.search_settings.initial_mass
        masses: list[float] = []
        for leg in plan.legs:
            if leg.role == "camp":
                continue
            if leg.from_id != EARTH_ID and leg.from_id in plan.collect_epochs:
                if abs(plan.collect_epochs[leg.from_id] - leg.departure_epoch) < 1e-6:
                    mass += plan.collected_mass[leg.from_id]
            masses.append(mass)
            inflation = float(
                self.leg_inflation(
                    leg.role, leg.from_id, leg.to_id, leg.delta_v_proxy_km_s, mass, leg.tof_days
                )
            )
            mass -= float(propellant_for_delta_v(mass, leg.delta_v_proxy_km_s * inflation))
            if leg.to_id != EARTH_ID and leg.to_id in plan.deploy_epochs:
                if abs(plan.deploy_epochs[leg.to_id] - leg.arrival_epoch) < 1e-6:
                    mass -= C.MINER_MASS_KG
        return masses


def _empty_plan() -> RoutePlan:
    return RoutePlan((), {}, {}, {}, 0.0, 0.0)


def orders_of(plan: RoutePlan) -> tuple[list[int], list[int]]:
    """Deploy order and collect order (visit sequence) of a plan (cooperative plans included)."""

    deploy_order = [leg.to_id for leg in plan.legs if leg.role in ("earth_out", "deploy_hop")]
    collect_order = [
        leg.from_id
        for leg in plan.legs
        if leg.role in ("collect_hop", "earth_return")
        and leg.from_id in plan.collect_epochs
        # the forward tour leaves its camp asteroid once without collecting (revisited later)
        and abs(plan.collect_epochs[leg.from_id] - leg.departure_epoch) < 1e-6
    ]
    return deploy_order, collect_order


def build_visits(
    deploy_order: list[int],
    collect_order: list[int],
    foreign: dict[int, float] | None = None,
    pinned: dict[int, float] | None = None,
) -> list[Visit]:
    """Visit sequence for ``Earth -> deploy_order -> collect_order -> Earth``.

    When the last deployed asteroid is also the first collected the ship camps there (one visit
    that deploys on arrival and collects at departure); otherwise it deploys and leaves.
    Asteroids collected but not deployed here must have their (other ship's) deploy epoch in
    ``foreign``; asteroids deployed but not collected are left as orphans for another ship.
    ``pinned`` fixes the deploy epoch of the listed deploys - miners another ship of the bundle
    collects against exactly that epoch - so re-timing this ship cannot strand that collector.
    """

    if not deploy_order or not collect_order:
        raise ValueError("deploy and collect orders must be non-empty")
    if len(set(deploy_order)) != len(deploy_order) or len(set(collect_order)) != len(collect_order):
        raise ValueError("an asteroid appears twice in a visit order")
    foreign = foreign or {}
    pinned = pinned or {}
    for asteroid in collect_order:
        if asteroid not in deploy_order and asteroid not in foreign:
            raise ValueError(f"asteroid {asteroid} is collected but deployed by nobody")
    visits: list[Visit] = [Visit(EARTH_ID, False, False, "earth_out")]
    for asteroid in deploy_order[:-1]:
        visits.append(
            Visit(asteroid, True, False, "deploy_hop", pinned_arrival=pinned.get(asteroid))
        )
    camp = deploy_order[-1]
    merged = collect_order[0] == camp
    remaining = collect_order[1:] if merged else list(collect_order)
    role = "earth_return" if not remaining else "collect_hop"
    visits.append(
        Visit(
            camp,
            True,
            merged,
            role if merged else "collect_hop",
            pinned_arrival=pinned.get(camp),
        )
    )
    for index, asteroid in enumerate(remaining):
        last = index == len(remaining) - 1
        visits.append(
            Visit(
                asteroid,
                False,
                True,
                "earth_return" if last else "collect_hop",
                foreign.get(asteroid) if asteroid not in deploy_order else None,
            )
        )
    visits.append(Visit(EARTH_ID, False, False, ""))
    return visits


# -- chain extension -----------------------------------------------------------------------


@dataclass(slots=True)
class ImproveResult:
    plan: RoutePlan
    original: RoutePlan
    steps: list[dict[str, Any]] = field(default_factory=list)
    wall_seconds: float = 0.0
    lambert_evaluations: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "asteroids_before": len(self.original.asteroids),
            "asteroids_after": len(self.plan.asteroids),
            "collected_before_kg": self.original.total_collected_kg,
            "collected_after_kg": self.plan.total_collected_kg,
            "final_mass_before_kg": self.original.final_mass_proxy_kg,
            "final_mass_after_kg": self.plan.final_mass_proxy_kg,
            "deploy_phase_days_before": deploy_phase_days(self.original),
            "deploy_phase_days_after": deploy_phase_days(self.plan),
            "camp_days_before": camp_days(self.original),
            "camp_days_after": camp_days(self.plan),
            "steps": self.steps,
            "wall_seconds": self.wall_seconds,
            "lambert_evaluations": self.lambert_evaluations,
        }


def insertion_candidates(
    plan: RoutePlan, search: RouteSearch, count: int, banned: set[int] | None = None
) -> list[int]:
    """Asteroids worth inserting after the last deploy: nearest in position space at that epoch."""

    deploy_order, _ = orders_of(plan)
    camp = deploy_order[-1]
    epoch = plan.deploy_epochs[camp]
    excluded = set(plan.asteroids) | set(search.excluded) | (banned or set())
    ranked = search.candidates(camp, epoch)
    return [int(a) for a in ranked if int(a) not in excluded][:count]


def orphan_candidates(
    plan: RoutePlan,
    pool: MinerPool,
    catalogue: AsteroidCatalogue,
    count: int,
    settings: SearchSettings | None = None,
) -> list[int]:
    """Pool orphans co-moving with the collect tour (smallest scaled element deviation).

    Position distance at one epoch is misleading here: an orphan on a different orbit may pass
    close by and still cost a cluster-to-cluster hop, so the ranking uses the same a / e-vector /
    i-vector bands as the search, taking the best match over the tour's asteroids.
    """

    orphans = pool.orphans()
    if not orphans:
        return []
    s = settings or SearchSettings()
    _, collect_order = orders_of(plan)
    ids = np.asarray(sorted(orphans), dtype=np.int64)
    best = np.full(ids.shape[0], np.inf)
    for asteroid in collect_order:
        da, de, di = element_deviations(catalogue, asteroid, ids)
        best = np.minimum(best, da / s.band_a_au + de / s.band_e + di / s.band_i_deg)
    order = np.lexsort((ids, best))
    return [int(ids[i]) for i in order[:count]]


def extend_plan(
    plan: RoutePlan,
    search: RouteSearch,
    retimer: Retimer,
    *,
    candidates: int = 12,
    pool: MinerPool | None = None,
    foreign_candidates: int = 4,
    harvest: bool = False,
) -> tuple[list[RetimeResult], list[dict[str, Any]]]:
    """Insert one more asteroid and re-time each variant.

    Three kinds of insertion are tried: a fresh asteroid deployed last and collected first
    (self-cleaning); with a ``pool``, a fresh asteroid deployed last and *left* for another ship
    (orphan, valued at the re-timer's orphan credit); and a pool orphan (another ship's miner)
    inserted into the collect tour first or next to the tour asteroid it co-moves with best.
    With ``harvest`` (collector itineraries) *every* pool orphan is tried at *every* position of
    the collect tour: a miner another ship dropped years earlier has accumulated mass at the
    official rate since, so a late collector picks up 60-90 kg for one hop and no miner - the
    cooperative structure the references use.  Returns the closing variants sorted by plan
    value and the failure records of the others (retained for the report).
    """

    deploy_order, collect_order = orders_of(plan)
    profile = retimer._plan_masses(plan)
    camp_index = len(deploy_order)  # index of the leg departing the old camp asteroid
    before = plan_value(plan, retimer)
    results: list[RetimeResult] = []
    failures: list[dict[str, Any]] = []
    foreign = dict(plan.foreign_deploy_epochs)

    def attempt(
        kind: str,
        asteroid: int,
        new_deploy: list[int],
        new_collect: list[int],
        new_profile: list[float],
        new_foreign: dict[int, float],
    ) -> None:
        # a foreign collect placed first un-merges the camp visit (deploy, leave, come back):
        # the visit list then has one leg more than the naive insertion count
        try:
            needed = len(build_visits(new_deploy, new_collect, new_foreign)) - 1
        except ValueError as error:
            failures.append(
                {
                    "kind": kind,
                    "asteroid": asteroid,
                    "reason": f"invalid_order:{error}",
                    "deploy_order": list(new_deploy),
                    "collect_order": list(new_collect),
                    "plan_legs": [(leg.from_id, leg.to_id, leg.role) for leg in plan.legs],
                }
            )
            return
        while len(new_profile) < needed:
            new_profile.insert(camp_index, new_profile[camp_index])
        while len(new_profile) > needed:
            new_profile.pop(camp_index)
        result = retimer.retime_order(
            new_deploy, new_collect, new_profile, before=before, original=plan, foreign=new_foreign
        )
        if result.plan is not None and result.plan.feasible:
            results.append(result)
        else:
            failures.append(
                {
                    "kind": kind,
                    "asteroid": asteroid,
                    "reason": result.failure or "not_feasible",
                    "price": result.price,
                }
            )

    guess = profile[camp_index] if camp_index < len(profile) else profile[-1]
    banned = pool.touched() if pool is not None else set()
    for asteroid in insertion_candidates(plan, search, candidates, banned):
        # new legs: old camp -> new (deploy hop) and new -> old camp (first collect hop)
        attempt(
            "self_cleaning",
            asteroid,
            [*deploy_order, asteroid],
            [asteroid, *collect_order],
            profile[:camp_index]
            + [guess - C.MINER_MASS_KG, guess - C.MINER_MASS_KG]
            + [m - C.MINER_MASS_KG for m in profile[camp_index:]],
            foreign,
        )
        if pool is not None and retimer.settings.orphan_credit > 0.0:
            # deploy-only: the new miner is left for another ship (collect tour unchanged)
            attempt(
                "orphan",
                asteroid,
                [*deploy_order, asteroid],
                list(collect_order),
                profile[:camp_index]
                + [guess - C.MINER_MASS_KG, guess - C.MINER_MASS_KG]
                + [m - C.MINER_MASS_KG for m in profile[camp_index:]],
                foreign,
            )
    if pool is not None:
        count = len(pool.orphans()) if harvest else foreign_candidates
        for asteroid in orphan_candidates(plan, pool, search.catalogue, count, search.settings):
            if asteroid in plan.asteroids:
                continue
            deploy_epoch = pool.orphans()[asteroid]
            # insert next to the tour asteroid it co-moves with best (before / after), plus first;
            # a harvesting collector tries every position
            da, de, di = element_deviations(
                search.catalogue, asteroid, np.asarray(collect_order, dtype=np.int64)
            )
            ss = search.settings
            nearest = int(np.argmin(da / ss.band_a_au + de / ss.band_e + di / ss.band_i_deg))
            positions = (
                range(len(collect_order) + 1) if harvest else sorted({0, nearest, nearest + 1})
            )
            for position in positions:
                new_collect = [*collect_order[:position], asteroid, *collect_order[position:]]
                # one extra collect leg; the ship is heavier by the collected mass afterwards
                leg = camp_index + position
                new_profile = [
                    *profile[:leg],
                    profile[min(leg, len(profile) - 1)],
                    *profile[leg:],
                ]
                attempt(
                    "foreign_collect",
                    asteroid,
                    list(deploy_order),
                    new_collect,
                    new_profile,
                    {**foreign, asteroid: deploy_epoch},
                )
    results.sort(
        key=lambda item: (
            -plan_value(item.plan, retimer),  # type: ignore[arg-type]
            item.plan.propellant_proxy_kg,  # type: ignore[union-attr]
        )
    )
    return results, failures


def improve_plan(
    plan: RoutePlan,
    search: RouteSearch,
    retimer: Retimer,
    *,
    max_rounds: int = 6,
    candidates: int = 12,
    time_budget_seconds: float = float("inf"),
    pool: MinerPool | None = None,
    harvest: bool = False,
) -> ImproveResult:
    """Re-time, then insert asteroids one at a time while the plan value grows.

    With a ``pool`` the insertions include cooperative variants (orphans left for other ships and
    other ships' orphans collected here; ``harvest`` tries every orphan at every position).
    """

    started = time.perf_counter()
    steps: list[dict[str, Any]] = []
    current = plan
    first = retimer.retime(current)
    steps.append({"kind": "retime", **first.summary()})
    if first.improved and first.plan is not None and first.plan.feasible:
        current = first.plan
    for _round in range(max_rounds):
        if time.perf_counter() - started > time_budget_seconds:
            steps.append({"kind": "stop", "reason": "time budget"})
            break
        if len(current.asteroids) >= search.settings.max_deploys:
            steps.append({"kind": "stop", "reason": "max deploys"})
            break
        variants, failures = extend_plan(
            current, search, retimer, candidates=candidates, pool=pool, harvest=harvest
        )
        value = plan_value(current, retimer)
        step: dict[str, Any] = {
            "kind": "extend",
            "asteroids": len(current.asteroids),
            "closing_variants": len(variants),
            "failures": failures,
            "value_before_kg": value,
        }
        accepted = None
        for variant in variants:
            if plan_value(variant.plan, retimer) > value + 1e-9:  # type: ignore[arg-type]
                accepted = variant
                break
        if accepted is None:
            step["result"] = "no improving insertion"
            steps.append(step)
            break
        previous = current
        current = accepted.plan  # type: ignore[assignment]
        inserted = [a for a in current.asteroids if a not in previous.asteroids]
        step["result"] = {
            "inserted": inserted[-1] if inserted else None,
            "kind": "foreign_collect"
            if inserted and inserted[-1] in current.foreign_deploy_epochs
            else "orphan"
            if inserted and inserted[-1] in current.orphaned
            else "self_cleaning",
            "value_after_kg": plan_value(current, retimer),
            "collected_kg": current.total_collected_kg,
            "final_mass_kg": current.final_mass_proxy_kg,
            "camp_days": camp_days(current),
        }
        steps.append(step)
    return ImproveResult(
        current, plan, steps, time.perf_counter() - started, retimer.lambert_evaluations
    )


def calibrate_from_route(retimer: Retimer, route: RefinedRoute) -> int:
    """Calibrate the re-timer's pair inflations from every certified leg of a refined route.

    Returns the number of legs used.  The measured inflation is SCvx ΔV / planned Lambert ΔV; the
    leg's authority ratio (planned ΔV over the full thrust authority at its initial mass) lets
    the ratio-dependent hop model store a residual instead of the raw factor.
    """

    used = 0
    for leg in route.legs:
        if not leg.certified or leg.solution is None or leg.planned.delta_v_proxy_km_s <= 0:
            continue
        planned = leg.planned
        retimer.calibrate(
            planned.from_id,
            planned.to_id,
            leg.solution.delta_v_km_s / planned.delta_v_proxy_km_s,
            authority_ratio=retimer.authority_ratio(
                planned.delta_v_proxy_km_s, leg.mass_before, planned.tof_days
            ),
            tof_days=planned.tof_days,
        )
        used += 1
    return used


@dataclass(slots=True)
class CertifiedImprovement:
    route: RefinedRoute | None  # best certified re-timed route (None if none certified)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    wall_seconds: float = 0.0
    certified_routes: list[RefinedRoute] = field(default_factory=list)  # all, for the master

    def summary(self) -> dict[str, Any]:
        return {
            "certified": self.route is not None,
            "certified_variants": len(self.certified_routes),
            "attempts": self.attempts,
            "wall_seconds": self.wall_seconds,
            "route": None if self.route is None else self.route.summary(),
        }


def improve_and_certify(
    plan: RoutePlan,
    search: RouteSearch,
    retimer: Retimer,
    catalogue: AsteroidCatalogue,
    *,
    scvx: ScvxSettings | None = None,
    max_attempts: int = 4,
    max_rounds: int = 6,
    candidates: int = 12,
    minimum_gain_kg: float = 0.0,
    time_budget_seconds: float = float("inf"),
    pool: MinerPool | None = None,
    refine=None,
    harvest: bool = False,
) -> CertifiedImprovement:
    """SCvx-in-the-loop re-timing: improve at proxy level, re-fly, ban what does not fly, repeat.

    Each attempt re-times and extends ``plan`` under the current bans, refines every leg with
    SCvx and, when a leg is not certifiable, tightens the authority ratio allowed for that body
    pair to ``ban_factor`` x the failing ratio before trying again.  Only fully certified routes
    are returned; the caller keeps its previously certified route otherwise.  Every certified
    variant is retained in ``certified_routes`` as a column for the fleet master.  ``refine``
    replaces :func:`refine_route` (tests inject a proxy-trusting stand-in).
    """

    started = time.perf_counter()
    refine = refine or (lambda candidate: refine_route(candidate, catalogue, scvx=scvx))
    attempts: list[dict[str, Any]] = []
    best: RefinedRoute | None = None
    certified_routes: list[RefinedRoute] = []
    reference = plan_value(plan, retimer)
    for attempt in range(max_attempts):
        remaining = time_budget_seconds - (time.perf_counter() - started)
        if remaining <= 0.0:
            attempts.append({"attempt": attempt, "stopped": "time budget"})
            break
        improved = improve_plan(
            plan,
            search,
            retimer,
            max_rounds=max_rounds,
            candidates=candidates,
            time_budget_seconds=remaining,
            pool=pool,
            harvest=harvest,
        )
        record: dict[str, Any] = {
            "attempt": attempt,
            "proxy": improved.summary(),
            "bans": {f"{a}->{b}": ratio for (a, b), ratio in sorted(retimer.bans.items())},
            "calibrated_pairs": len(retimer.inflations),
        }
        gain = plan_value(improved.plan, retimer) - reference
        if improved.plan is plan or gain <= minimum_gain_kg:
            record["result"] = "no proxy improvement"
            attempts.append(record)
            break
        refined = refine(improved.plan)
        record["refined"] = {
            "certified": refined.certified,
            "collected_kg": refined.total_collected_kg,
            "final_mass_kg": refined.final_mass_kg,
            "refined_arcs": refined.refined_arc_count,
            "wall_seconds": refined.wall_seconds,
            "failures": refined.failures,
        }
        # every certified leg calibrates its pair's propellant inflation for the next attempt
        calibrate_from_route(retimer, refined)
        if refined.certified:
            certified_routes.append(refined)
            if best is None or refined.total_collected_kg > best.total_collected_kg:
                best = refined
            record["result"] = "certified"
            attempts.append(record)
            # the calibrated inflations may free enough mass for one more asteroid: try again
            # from the certified plan
            plan = improved.plan
            reference = plan_value(plan, retimer)
            continue
        failing = next((leg for leg in refined.legs if not leg.certified), None)
        if failing is None:
            record["result"] = "refinement did not certify (mass budget)"
            attempts.append(record)
            break
        ratio = retimer.authority_ratio(
            failing.planned.delta_v_proxy_km_s, failing.mass_before, failing.planned.tof_days
        )
        retimer.ban(failing.planned.from_id, failing.planned.to_id, ratio)
        record["result"] = {
            "banned": f"{failing.planned.from_id}->{failing.planned.to_id}",
            "ratio": ratio,
            "tof_days": failing.planned.tof_days,
            "departure": failing.planned.departure_epoch,
            "mass_before": failing.mass_before,
        }
        if failing.planned.role == "earth_return":
            # a swept return is priced from its cell, not the ratio ban: refuse the cell itself
            record["result"]["sweep_cell_refused"] = retimer.refuse_return(
                failing.planned.from_id,
                failing.planned.departure_epoch,
                failing.planned.tof_days,
            )
        attempts.append(record)
    return CertifiedImprovement(best, attempts, time.perf_counter() - started, certified_routes)
