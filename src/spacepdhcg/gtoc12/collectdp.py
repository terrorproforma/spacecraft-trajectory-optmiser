"""Exact collect-tour pricing: order *and* timing of a ship's collection phase.

The beam search (``search.py``) builds the deploy chain forwards; what a chain is worth is
decided by the collection tour that re-flies the family years later, when the relative phase
drift has changed every pair cost.  The heuristic tours (greedy-backward, reverse, forward) pick
an order from a few hundred Lambert evaluations at guessed epochs and left our collect hops at
90-110 kg against 66 kg in the archived solutions.  This module prices the collect phase exactly
for the leg model the beam and the re-timer use:

* :class:`CollectPairTable` holds, per ordered asteroid pair, the zero-revolution Lambert ΔV on
  an absolute epoch lattice x collect TOF grid (the *actual* collect epochs of the mission, not
  a single look-ahead epoch), computed lazily in one batched Lambert call per pair and kept in a
  bounded cache shared by every partial of a search; the same for the Earth return of each
  asteroid.  Costs become propellant with the certified-hop inflation model of ``screening``.
* :func:`plan_collect_tour` runs a Held-Karp dynamic programme over ``(collected set, location,
  lattice epoch)``: every collect order is allowed (the deploy order is *not* imposed), the
  ship may leave its camp without collecting and revisit it, may camp anywhere, and every
  collect epoch is chosen with the official mining-rate bookkeeping (``rate x stay``, one-year
  minimum stay, collection on departure) traded against the propellant of the hops it moves.
  The objective is ``Σ collected - weight x propellant``; the caller re-prices the chosen tour
  with its exact forward mass pass.

Complexity is ``O(k^2 2^k n_t n_tof)`` vector operations for ``k`` deployed asteroids: a few
seconds at ``k = 10`` on the 30-day lattice, milliseconds at ``k <= 6``.  Everything is
deterministic (ties break on lattice index, TOF index and asteroid ID).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .clusters import mean_longitude
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .hopcalib import InflationFit
from .screening import (
    lambert_hops,
    low_thrust_inflation,
    propellant_for_delta_v,
    return_inflation_model,
    thrust_authority_km_s,
)

FloatArray = NDArray[np.float64]
EARTH_ID = 0
# a swept return cell certifies the grid nodes within this many index steps of it (departure
# and TOF steps count alike); shared with the re-timer's ``_return_override``
RETURN_SWEEP_REACH = 2


@dataclass(frozen=True, slots=True)
class CollectDPSettings:
    # departure lattice: every lattice epoch is a candidate departure, so the relative-phase
    # windows (zero-crossings of the pair's phase drift) are found by enumeration; 15 days
    # resolves a 60-180 d hop's window (the phase of co-moving pairs drifts < 1 deg/month)
    step_days: float = 15.0
    # collect hop TOFs (multiples of ``step_days``); the references' collect hops are 180 d
    tofs: tuple[float, ...] = (
        60.0,
        90.0,
        120.0,
        150.0,
        180.0,
        210.0,
        240.0,
        300.0,
        360.0,
        450.0,
        600.0,
    )
    # Earth return TOFs: 30-day grid over the certified return envelope (240-720 d)
    return_tofs: tuple[float, ...] = tuple(float(x) for x in range(240, 721, 30))
    max_asteroids: int = 10
    end_margin_days: float = 2.0
    # objective: collected kg minus this many kg per kg of propellant (the propellant is a hard
    # constraint through the final mass; 1.0 lets ~10 kg/yr of extra mining pay for a 10 kg
    # cheaper hop and nothing more)
    propellant_weight: float = 1.0
    hop_inflation: float = 1.2
    hop_inflation_slope: float | None = None
    hop_inflation_floor: float = 1.05
    hop_authority_ratio: float = 0.667
    return_inflation: float = 1.6
    return_authority_ratio: float = 0.5
    # price the Earth return with the TOF/ratio model measured on the certified archive
    # (``screening.return_inflation_model``: 1.38x Lambert at 420 d, 0.98x at 540 d) instead of
    # the flat ``return_inflation``; the flat factor cannot see that the long return is cheaper
    return_tof_model: bool = True
    cache_pairs: int = 20_000  # bounded pair cache (float32 (n_t, n_tof) per pair)
    # bounded LRU of pair geometries (Δa, Δλ(t)); keyed by the epoch slice, so one pair is
    # re-derived for every distinct deploy timing - a few vector ops, not worth the 80k-entry
    # dict (~200 MB) the previous clear-on-overflow policy let a beam grow
    cache_geometry: int = 2_048
    # per-DP LRU of move-mass propellant fractions (see ``plan_collect_tour``); 512 x ~20 KB
    fraction_cache_entries: int = 512
    # calibrated inflation model (``hopcalib.InflationFit``): when set, hop propellant uses
    # ``f(r, TOF, Δa, Δλ)`` fitted on the certified archive instead of the flat/ratio factor
    inflation_fit: InflationFit | None = None
    # harvest-phase prior (``harvestphase.HarvestPhasePrior``, tenth iteration): every collect
    # hop departing with the pair misaligned beyond the references' p75 of |Δλ| is charged
    # ``phase_weight x penalty_kg(|Δλ| at departure)`` in the DP objective (priced like
    # propellant, kept out of the propellant accounting), so an aligned 180-day hop beats a
    # misaligned 210-day one when their surrogate propellant is comparable
    harvest_phase: Any = None
    phase_weight: float = 1.0

    def __post_init__(self) -> None:
        for tof in self.tofs:
            if abs(tof / self.step_days - round(tof / self.step_days)) > 1e-9:
                raise ValueError(
                    f"collect TOF {tof} is not a multiple of the {self.step_days} d step"
                )


class CollectPairTable:
    """Lazily built, bounded table of pair and Earth-return costs on the epoch lattice.

    ``hop(i, j)[t, k]`` is the Lambert ΔV (km/s, ``inf`` when infeasible) of departing ``i`` at
    ``epochs[t]`` and arriving at ``j`` after ``tofs[k]`` days; ``earth_return(i)[t, k]`` the same
    towards Earth with the 6 km/s arrival allowance and the end-of-window constraint applied.
    """

    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        settings: CollectDPSettings | None = None,
        *,
        start_epoch: float | None = None,
    ) -> None:
        self.catalogue = catalogue
        self.settings = settings or CollectDPSettings()
        s = self.settings
        start = C.MISSION_START_MJD if start_epoch is None else float(start_epoch)
        end = C.MISSION_END_MJD - s.end_margin_days
        self.epochs: FloatArray = start + np.arange(0.0, end - start + 1e-9, s.step_days)
        self.tofs: FloatArray = np.asarray(s.tofs, dtype=np.float64)
        self.tof_steps: NDArray[np.int64] = np.rint(self.tofs / s.step_days).astype(np.int64)
        self.return_tofs: FloatArray = np.asarray(s.return_tofs, dtype=np.float64)
        self._hops: OrderedDict[tuple[int, int], NDArray[np.float32]] = OrderedDict()
        self._returns: dict[int, NDArray[np.float32]] = {}
        self._geometry: OrderedDict[tuple[int, int, float, int], tuple[float, FloatArray]] = (
            OrderedDict()
        )
        # SCvx return sweeps (``returnsweep.ReturnSweep``) per camp asteroid, and the derived
        # (inflation, ok) override tables on the return grid; see ``set_return_sweep``
        self.return_sweeps: dict[int, Any] = {}
        self._return_overrides: dict[int, tuple[FloatArray, NDArray[np.bool_]]] = {}
        self.lambert_evaluations = 0

    def release_caches(self) -> int:
        """Drop the cached pair, return and geometry tables (recomputed on demand); returns the
        number of pair tables released.  A parked slot keeps its table for the orphan repair,
        which needs a handful of pairs, not the beam's thousands (14 kB each on the 15-day
        lattice: ~110 MB of live growth per slot in a 35-member family)."""

        released = len(self._hops)
        self._hops.clear()
        self._returns.clear()
        self._geometry.clear()
        self._return_overrides.clear()  # rebuilt from the kept sweeps on demand
        return released

    # -- certified return cells ------------------------------------------------------------

    def set_return_sweep(self, sweep: Any) -> None:
        """Price the DP's Earth return from ``sweep.asteroid`` with the SCvx-measured sweep.

        With a sweep in hand the return out of that camp is priced strictly from certified
        cells (:meth:`return_override`): the DP may end the tour on any cell SCvx certified (or
        within ``RETURN_SWEEP_REACH`` grid steps of one) and nowhere else - the model is not
        consulted for that asteroid.  Tours through other camps keep the model.
        """

        self.return_sweeps[int(sweep.asteroid)] = sweep
        self._return_overrides.pop(int(sweep.asteroid), None)

    def return_override(self, asteroid: int) -> tuple[FloatArray, NDArray[np.bool_]] | None:
        """``(inflation, ok)`` on the table's ``(epochs, return_tofs)`` grid from the asteroid's
        sweep, or ``None`` without one.

        Each flown sweep cell is placed on its nearest grid node; its inflation is the measured
        SCvx ΔV over the zero-revolution Lambert ΔV of the cell's own (departure, TOF).  Every
        grid node within ``RETURN_SWEEP_REACH`` index steps of a certified cell takes the
        nearest cell's inflation (ties -> the earlier sweep entry); nodes nearest to a refused
        cell, or beyond the reach, are infeasible.
        """

        key = int(asteroid)
        sweep = self.return_sweeps.get(key)
        if sweep is None:
            return None
        cached = self._return_overrides.get(key)
        if cached is not None:
            return cached
        step = self.settings.step_days
        n_t, n_k = self.epochs.shape[0], self.return_tofs.shape[0]
        cells: list[tuple[int, int, float, bool]] = []
        flown: list[tuple[int, int, float, float, float]] = []  # (k, t, departure, tof, dv)
        for i, departure in enumerate(sweep.departures):
            k = round((float(departure) - self.epochs[0]) / step)
            if not (0 <= k < n_t) or abs(self.epochs[k] - departure) > step / 2.0 + 1e-6:
                continue
            for j, tof in enumerate(sweep.tofs):
                if not bool(sweep.attempted[i, j]):
                    continue  # arrival past the window: not flown, not a refusal
                t = int(np.argmin(np.abs(self.return_tofs - float(tof))))
                certified = bool(sweep.certified[i, j])
                flown.append((k, t, float(departure), float(tof), float(sweep.delta_v_km_s[i, j])))
                cells.append((k, t, np.nan, certified))
        if not cells:
            return None
        # Lambert ΔV of the flown cells themselves (their own geometry, not the grid node's)
        deps = np.asarray([f[2] for f in flown])
        tofs = np.asarray([f[3] for f in flown])
        r_s, v_s = asteroid_state(self.catalogue, np.full(deps.shape[0], key), deps)
        r_e, v_e = earth_state(deps + tofs)
        hop = lambert_hops(
            r_s, v_s, r_e, v_e, deps, tofs, arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S
        )
        self.lambert_evaluations += 2 * deps.shape[0]
        lambert = np.where(hop.feasible, hop.total_delta_v, np.inf)
        measured = np.asarray([f[4] for f in flown])
        oks = np.asarray([c[3] for c in cells]) & np.isfinite(lambert) & (lambert > 1e-9)
        oks &= np.isfinite(measured)
        infl = np.where(oks, measured / np.where(lambert > 1e-9, lambert, 1.0), np.nan)
        ks = np.asarray([c[0] for c in cells])
        ts = np.asarray([c[1] for c in cells])
        grid_k, grid_t = np.meshgrid(np.arange(n_t), np.arange(n_k), indexing="ij")
        distance = (grid_k[:, :, None] - ks[None, None, :]) ** 2 + (
            grid_t[:, :, None] - ts[None, None, :]
        ) ** 2
        nearest = np.argmin(distance, axis=2)
        reached = np.min(distance, axis=2) <= RETURN_SWEEP_REACH**2
        ok = reached & oks[nearest]
        inflation = np.where(ok, infl[nearest], np.nan)
        table = (inflation, ok)
        self._return_overrides[key] = table
        return table

    # -- lattice -------------------------------------------------------------------------

    def index_at_or_after(self, epoch: float) -> int:
        """Lattice index of the first node at or after ``epoch`` (``len`` when past the end)."""

        return int(np.searchsorted(self.epochs, epoch - 1e-9, side="left"))

    # -- costs ---------------------------------------------------------------------------

    def hop(self, source: int, target: int) -> NDArray[np.float32]:
        key = (int(source), int(target))
        cached = self._hops.get(key)
        if cached is not None:
            self._hops.move_to_end(key)
            return cached
        n_t, n_k = self.epochs.shape[0], self.tofs.shape[0]
        t_idx, k_idx = np.meshgrid(np.arange(n_t), np.arange(n_k), indexing="ij")
        t_idx, k_idx = t_idx.ravel(), k_idx.ravel()
        departures = self.epochs[t_idx]
        tofs = self.tofs[k_idx]
        arrivals = departures + tofs
        valid = arrivals <= self.epochs[-1] + 1e-9
        dv = np.full(departures.shape[0], np.inf)
        if np.any(valid):
            dep, tof, arr = departures[valid], tofs[valid], arrivals[valid]
            r_s, v_s = asteroid_state(self.catalogue, np.full(dep.shape[0], key[0]), dep)
            r_t, v_t = asteroid_state(self.catalogue, np.full(dep.shape[0], key[1]), arr)
            hop = lambert_hops(r_s, v_s, r_t, v_t, dep, tof)
            self.lambert_evaluations += 2 * dep.shape[0]
            dv[valid] = np.where(hop.feasible, hop.total_delta_v, np.inf)
        table = dv.reshape(n_t, n_k).astype(np.float32)
        self._hops[key] = table
        while len(self._hops) > self.settings.cache_pairs:
            self._hops.popitem(last=False)
        return table

    def earth_return(self, source: int) -> NDArray[np.float32]:
        key = int(source)
        cached = self._returns.get(key)
        if cached is not None:
            return cached
        n_t, n_k = self.epochs.shape[0], self.return_tofs.shape[0]
        t_idx, k_idx = np.meshgrid(np.arange(n_t), np.arange(n_k), indexing="ij")
        t_idx, k_idx = t_idx.ravel(), k_idx.ravel()
        departures = self.epochs[t_idx]
        tofs = self.return_tofs[k_idx]
        arrivals = departures + tofs
        valid = arrivals <= C.MISSION_END_MJD - self.settings.end_margin_days + 1e-9
        dv = np.full(departures.shape[0], np.inf)
        if np.any(valid):
            dep, tof, arr = departures[valid], tofs[valid], arrivals[valid]
            r_s, v_s = asteroid_state(self.catalogue, np.full(dep.shape[0], key), dep)
            r_e, v_e = earth_state(arr)
            hop = lambert_hops(
                r_s, v_s, r_e, v_e, dep, tof, arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S
            )
            self.lambert_evaluations += 2 * dep.shape[0]
            dv[valid] = np.where(hop.feasible, hop.total_delta_v, np.inf)
        table = dv.reshape(n_t, n_k).astype(np.float32)
        self._returns[key] = table
        return table

    @property
    def cached_pairs(self) -> int:
        return len(self._hops)

    # -- propellant model ----------------------------------------------------------------

    def pair_geometry(
        self, source: int, target: int, epochs: FloatArray
    ) -> tuple[float, FloatArray]:
        """``(Δa in AU, Δλ(t) in rad wrapped to [-π, π])`` of the pair at ``epochs``."""

        epochs = np.asarray(epochs, dtype=np.float64)
        key = (source, target, float(epochs[0]) if epochs.shape[0] else 0.0, epochs.shape[0])
        cached = self._geometry.get(key)
        if cached is not None:
            self._geometry.move_to_end(key)
            return cached
        cat = self.catalogue
        src = int(np.searchsorted(cat.ids, source))
        tgt = int(np.searchsorted(cat.ids, target))
        delta_a = float(cat.semi_major_axis_km[tgt] - cat.semi_major_axis_km[src]) / C.AU_KM
        epochs = np.asarray(epochs, dtype=np.float64)
        n_s = float(np.sqrt(C.MU_SUN_KM3_S2 / cat.semi_major_axis_km[src] ** 3) * C.DAY_S)
        n_t = float(np.sqrt(C.MU_SUN_KM3_S2 / cat.semi_major_axis_km[tgt] ** 3) * C.DAY_S)
        ref = float(epochs[0]) if epochs.shape[0] else C.MISSION_START_MJD
        l_s = float(mean_longitude(cat, np.asarray([src]), ref)[0])
        l_t = float(mean_longitude(cat, np.asarray([tgt]), ref)[0])
        delta = (l_t - l_s) + (n_t - n_s) * (epochs - ref)
        result = (delta_a, (delta + np.pi) % (2.0 * np.pi) - np.pi)
        self._geometry[key] = result
        while len(self._geometry) > self.settings.cache_geometry:
            self._geometry.popitem(last=False)
        return result

    def phase_deg(self, source: int, target: int, epochs: FloatArray) -> FloatArray:
        """``|Δλ|`` (deg) of the pair at every departure epoch (from :meth:`pair_geometry`)."""

        _delta_a, delta_l = self.pair_geometry(int(source), int(target), epochs)
        return np.degrees(np.abs(delta_l))

    def phase_penalty(self, source: int, target: int, epochs: FloatArray) -> FloatArray:
        """Kilograms the harvest-phase prior charges a hop of the pair departing at each of
        ``epochs`` (zeros without a prior); ``phase_weight`` applied."""

        s = self.settings
        epochs = np.asarray(epochs, dtype=np.float64)
        if s.harvest_phase is None or s.phase_weight <= 0.0:
            return np.zeros(epochs.shape[0])
        penalty = np.asarray(s.harvest_phase.penalty_kg(self.phase_deg(source, target, epochs)))
        return s.phase_weight * penalty

    def hop_propellant(
        self,
        dv: FloatArray,
        mass: float,
        tofs: FloatArray,
        *,
        pair: tuple[int, int] | None = None,
        epochs: FloatArray | None = None,
    ) -> FloatArray:
        """Propellant (kg) of hops ``dv`` at ``mass`` over ``tofs``; ``inf`` when not flyable.

        With a calibrated ``inflation_fit`` and the pair + departure ``epochs`` (rows of ``dv``)
        given, the inflation is the fitted ``f(r, TOF, Δa, Δλ)``; otherwise the flat or
        ratio-only factor of the settings.
        """

        s = self.settings
        dv = np.asarray(dv, dtype=np.float64)
        authority = thrust_authority_km_s(mass, tofs, 1.0)
        ok = np.isfinite(dv) & (dv <= s.hop_authority_ratio * authority)
        fit = getattr(s, "inflation_fit", None)
        if fit is not None and pair is not None and epochs is not None and dv.ndim == 2:
            delta_a, delta_l = self.pair_geometry(pair[0], pair[1], epochs)
            inflation = fit.inflation(
                np.where(ok, dv, 0.0),
                mass,
                np.broadcast_to(np.asarray(tofs, dtype=np.float64)[None, :], dv.shape),
                delta_a,
                np.broadcast_to(delta_l[:, None], dv.shape),
            )
        elif s.hop_inflation_slope is None:
            inflation = np.full(dv.shape, s.hop_inflation)
        else:
            inflation = low_thrust_inflation(
                np.where(ok, dv, 0.0),
                mass,
                tofs,
                floor=s.hop_inflation_floor,
                slope=s.hop_inflation_slope,
            )
        propellant = propellant_for_delta_v(mass, np.where(ok, dv, 0.0) * inflation)
        return np.where(ok, propellant, np.inf)

    def harvest_window_cost(
        self,
        a: int,
        b: int,
        mass: float,
        *,
        window: tuple[float, float],
        max_tof_days: float,
    ) -> float:
        """Cheapest calibrated collect-hop propellant (kg) between ``a`` and ``b`` - either
        direction - departing inside ``window`` (MJD) with a TOF of at most ``max_tof_days``.

        This is what the pair will cost the collect DP if both asteroids end up in the same
        tour: the DP picks the epoch and direction, so the window minimum (not the cost at one
        epoch) is the deploy-time predictor of the harvest.  ``inf`` when the pair cannot be
        re-flown inside the window.
        """

        lo = self.index_at_or_after(window[0])
        hi = max(self.index_at_or_after(window[1]), lo)
        keep = self.tofs <= max_tof_days + 1e-9
        if lo >= hi or not np.any(keep):
            return float("inf")
        best = float("inf")
        for source, target in ((int(a), int(b)), (int(b), int(a))):
            dv = self.hop(source, target)[lo:hi][:, keep].astype(np.float64)
            cost = self.hop_propellant(
                dv, mass, self.tofs[keep], pair=(source, target), epochs=self.epochs[lo:hi]
            )
            best = min(best, float(np.min(cost)))
        return best

    def return_inflation(
        self, dv: FloatArray | float, mass: float, tofs: FloatArray | float
    ) -> FloatArray:
        """Inflation of an Earth return: the TOF/ratio model or the flat setting; vectorised."""

        s = self.settings
        dv = np.asarray(dv, dtype=np.float64)
        if not s.return_tof_model:
            return np.full(dv.shape, s.return_inflation)
        authority = thrust_authority_km_s(mass, tofs, 1.0)
        ratio = np.where(np.isfinite(dv), dv, 0.0) / np.maximum(authority, 1e-12)
        return return_inflation_model(tofs, ratio)

    def return_propellant(
        self,
        dv: FloatArray,
        mass: float,
        tofs: FloatArray,
        *,
        asteroid: int | None = None,
        t0: int = 0,
    ) -> FloatArray:
        """Propellant (kg) of the Earth returns ``dv`` (rows = lattice epochs from ``t0``,
        columns = ``tofs``) at ``mass``; ``inf`` where not flyable.

        With ``asteroid`` given and a return sweep set for it, the rows are priced strictly from
        the certified cells (:meth:`return_override`) - measured inflation, no authority model,
        infeasible off the certified cells; otherwise the TOF/ratio model with the authority
        limit.
        """

        s = self.settings
        dv = np.asarray(dv, dtype=np.float64)
        override = None if asteroid is None else self.return_override(asteroid)
        if override is not None and dv.ndim == 2:
            inflation, ok = override
            rows = slice(t0, t0 + dv.shape[0])
            inflation, ok = inflation[rows], ok[rows]
            ok = ok & np.isfinite(dv)
            propellant = propellant_for_delta_v(
                mass, np.where(ok, dv, 0.0) * np.where(ok, inflation, 1.0)
            )
            return np.where(ok, propellant, np.inf)
        authority = thrust_authority_km_s(mass, tofs, 1.0)
        ok = np.isfinite(dv) & (dv <= s.return_authority_ratio * authority)
        inflation = self.return_inflation(dv, mass, tofs)
        propellant = propellant_for_delta_v(mass, np.where(ok, dv, 0.0) * inflation)
        return np.where(ok, propellant, np.inf)

    def return_inflation_at(
        self, asteroid: int, departure: float, tof: float, dv: float, mass: float
    ) -> float:
        """Inflation the table priced the return ``asteroid -> Earth`` at ``(departure, tof)``
        with: the certified cell's measurement when a sweep covers it, else the model."""

        override = self.return_override(int(asteroid))
        if override is not None:
            inflation, ok = override
            k = round((float(departure) - self.epochs[0]) / self.settings.step_days)
            t = int(np.argmin(np.abs(self.return_tofs - float(tof))))
            if 0 <= k < inflation.shape[0] and ok[k, t]:
                return float(inflation[k, t])
        return float(self.return_inflation(dv, mass, tof)[()])


@dataclass(slots=True)
class CollectTour:
    """A collect tour: collect order with epochs, hop legs and the Earth return."""

    order: tuple[int, ...]  # asteroids in collect order
    collect_epochs: dict[int, float]
    # (from, to, departure, tof_days, lambert dv) - collect hops in flight order, the initial
    # repositioning hop (if any) first, then the Earth return
    hops: list[tuple[int, int, float, float, float]]
    reposition: bool  # the ship left the camp without collecting and came back later
    objective_kg: float
    collected_proxy_kg: float
    propellant_proxy_kg: float
    return_departure: float
    return_tof: float
    return_dv: float
    dp_states: int = 0
    diagnostics: dict[str, object] = field(default_factory=dict)
    # model propellant (kg) of every entry of ``hops`` in flight order (return excluded)
    hop_propellant_kg: list[float] = field(default_factory=list)
    # harvest-phase prior (when the table carries one): |Δλ| (deg) of every hop's pair at its
    # departure and the kg the prior charged the tour (inside ``objective_kg``, outside
    # ``propellant_proxy_kg``)
    hop_phase_deg: list[float] = field(default_factory=list)
    phase_penalty_kg: float = 0.0


def plan_collect_tour(
    table: CollectPairTable,
    deployed: list[tuple[int, float]] | tuple[tuple[int, float], ...],
    camp: int,
    camp_epoch: float,
    mass_after_deploys: float,
    *,
    weights: dict[int, float] | None = None,
    banned_pairs: set[tuple[int, int]] | frozenset[tuple[int, int]] | None = None,
    propellant_weight: float | None = None,
    burn_per_hop: float | None = None,
) -> CollectTour | None:
    """Best collect tour over ``deployed`` for a ship sitting at ``camp`` at ``camp_epoch``.

    ``deployed`` are ``(asteroid, deploy epoch)`` pairs (the camp asteroid among them);
    ``mass_after_deploys`` the ship mass at the camp.  ``propellant_weight`` overrides the
    settings' objective weight.  ``burn_per_hop`` (kg) fixes the burn schedule of the mass
    model and skips the heavy first pass (a sweep over weights reuses the first tour's).
    Returns ``None`` when no tour closes.

    Propellant is priced as ``mass x f`` with ``f = 1 - exp(-inflation x ΔV / v_e)`` tabulated
    per pair and move mass.  The mass of a hop (feasibility, inflation model and propellant)
    is the camp mass plus the miners collected so far (mined to the window end) minus the
    propellant burnt on the hops flown so far.  The DP state does not carry the burnt mass, so
    the tour is solved twice: pass 1 with no burn credit (heavy ship), pass 2 with the burn
    schedule of the pass-1 tour (mean hop propellant x hops flown).  Pricing every move at the
    heavy pass-1 mass put the certified tours' returns (7.4 km/s at 1120 kg, ratio 0.36) over
    the 0.5 authority limit (ratio 0.60 at 1900 kg) and made the DP settle for tours 100 kg
    worse than the ones the re-timer later certified.
    """

    s = table.settings
    w = s.propellant_weight if propellant_weight is None else float(propellant_weight)
    ids = [int(a) for a, _ in deployed]
    deploy_epoch = {int(a): float(e) for a, e in deployed}
    k = len(ids)
    if k == 0 or k > s.max_asteroids or camp not in deploy_epoch:
        return None
    if len(set(ids)) != k:
        raise ValueError("deployed asteroids must be distinct")
    weights = weights or {}
    banned = banned_pairs or set()
    pos = {a: i for i, a in enumerate(ids)}
    camp_i = pos[camp]
    t0 = table.index_at_or_after(camp_epoch)
    if t0 >= table.epochs.shape[0]:
        return None
    epochs = table.epochs[t0:]  # local lattice: the collect phase starts at the camp
    n_t = epochs.shape[0]
    full = (1 << k) - 1
    min_stay = C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS
    # mined mass on departure at every lattice epoch, per asteroid (-inf where the stay is short)
    mined = np.full((k, n_t), -np.inf)
    for i, a in enumerate(ids):
        stay = epochs - deploy_epoch[a]
        ok = stay >= min_stay - 1e-9
        mined[i, ok] = weights.get(a, 1.0) * C.MINING_RATE_KG_PER_YEAR * stay[ok] / C.YEAR_DAYS
    mined_end = np.array(
        [C.maximum_collected_mass(max(epochs[-1] - deploy_epoch[a], 0.0)) for a in ids]
    )
    popcount = np.array([bin(m).count("1") for m in range(full + 1)])
    mined_by_subset = np.array(
        [sum(mined_end[i] for i in range(k) if m >> i & 1) for m in range(full + 1)]
    )
    floor_mass = C.DRY_MASS_KG + 1.0

    # The fraction depends on the move mass (inflation is a function of the authority ratio),
    # and the mass is the mined total of the collected subset: every (state, successor) of the
    # Held-Karp pass asks for a distinct key, so an unbounded cache held one (n_t x n_tof)
    # float64 table per expansion - 2^k x k^2 tables, the ~350 MB transient of a 26-member
    # family's beam.  Reuse only exists within a state's expansion (and between the two burn
    # passes for equal masses), which a small LRU captures; the tables are recomputed
    # otherwise (a few vector ops on ~20 KB, ~10 % of the DP time), bit-for-bit identical.
    fractions: OrderedDict[tuple[int, int, int], FloatArray] = OrderedDict()
    hop_tables: dict[tuple[int, int], FloatArray] = {}
    # harvest-phase penalty (kg) per departure epoch of the local lattice, per ordered pair
    penalties: dict[tuple[int, int], FloatArray] = {}

    def phase_penalty(j: int, l_i: int) -> FloatArray:
        cached = penalties.get((j, l_i))
        if cached is None:
            cached = table.phase_penalty(ids[j], ids[l_i], epochs)
            penalties[(j, l_i)] = cached
        return cached

    def fraction(j: int, l_i: int, mass: float) -> FloatArray:
        """Propellant per kg of ship mass for hop ``j -> l`` on the local lattice x TOF grid,
        flyable and priced at the mass the ship has on that move."""

        key = (j, l_i, round(mass))
        cached = fractions.get(key)
        if cached is not None:
            fractions.move_to_end(key)
            return cached
        dv = hop_tables.get((j, l_i))
        if dv is None:
            dv = table.hop(ids[j], ids[l_i])[t0:].astype(np.float64)
            hop_tables[(j, l_i)] = dv
        cost = table.hop_propellant(dv, mass, table.tofs, pair=(ids[j], ids[l_i]), epochs=epochs)
        cached = cost / mass
        fractions[key] = cached
        while len(fractions) > s.fraction_cache_entries:
            fractions.popitem(last=False)
        return cached

    def solve(burn_per_hop: float) -> CollectTour | None:
        return _solve_collect_dp(
            table,
            ids,
            camp_i,
            t0,
            epochs,
            mined,
            mined_by_subset,
            popcount,
            weights,
            banned,
            w,
            mass_after_deploys,
            burn_per_hop,
            floor_mass,
            fraction,
            deploy_epoch,
            phase_penalty if (s.harvest_phase is not None and s.phase_weight > 0.0) else None,
        )

    # a caller's burn schedule that is not a number (the mean of a tour whose hop propellant
    # came back NaN: cluster_fleet_v9 family 10 crashed in the substitution pass on it) falls
    # back to the two-pass schedule instead of poisoning every move mass
    if burn_per_hop is not None and not np.isfinite(float(burn_per_hop)):
        burn_per_hop = None
    if burn_per_hop is not None:
        return solve(max(float(burn_per_hop), 0.0))
    first = solve(0.0)
    if first is None:
        # nothing closes for the heavy ship: retry once with a nominal burn schedule (a 2 km/s
        # hop at 1.2x costs 6 % of the mass); the forward mass pass judges the result
        return solve(0.06 * mass_after_deploys)
    if not first.hop_propellant_kg:
        return first
    burn_per_hop = float(np.mean(first.hop_propellant_kg))
    if not np.isfinite(burn_per_hop) or burn_per_hop <= 0.0:
        return first
    second = solve(burn_per_hop)
    if second is None:
        return first
    second.diagnostics["pass1_objective_kg"] = first.objective_kg
    second.diagnostics["burn_per_hop_kg"] = burn_per_hop
    return second


def _solve_collect_dp(
    table: CollectPairTable,
    ids: list[int],
    camp_i: int,
    t0: int,
    epochs: FloatArray,
    mined: FloatArray,
    mined_by_subset: FloatArray,
    popcount: NDArray[np.int64],
    weights: dict[int, float],
    banned: set[tuple[int, int]] | frozenset[tuple[int, int]],
    w: float,
    mass_after_deploys: float,
    burn_per_hop: float,
    floor_mass: float,
    fraction,
    deploy_epoch: dict[int, float],
    phase_penalty=None,
) -> CollectTour | None:
    """One Held-Karp pass with the mass schedule ``camp mass + mined(S) - burn x hops flown``
    (the hop out of the ``h``-th collected asteroid is the ``h``-th hop).  ``phase_penalty(j,
    l)`` (kg per local departure epoch) is charged on every move like propellant."""

    k = len(ids)
    n_t = epochs.shape[0]
    full = (1 << k) - 1
    tof_steps = table.tof_steps
    n_tof = tof_steps.shape[0]
    # mass on the move out of location j once the set S (j included) is collected: hop number
    # popcount(S) - 1 hops have been flown before it
    mass_by_subset = np.maximum(
        mass_after_deploys + mined_by_subset - burn_per_hop * np.maximum(popcount - 1, 0),
        floor_mass,
    )

    # DP tables: value on *arrival* at location j with collected set S, per lattice epoch
    neg = -np.inf
    arrive: dict[tuple[int, int], FloatArray] = {}
    # back-pointers on arrival: (previous location [+k when the camp was left uncollected],
    # departure index, tof index)
    back: dict[tuple[int, int], tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]] = {}
    start = np.full(n_t, neg)
    start[0] = 0.0
    arrive[(0, camp_i)] = start
    states = 0
    lattice_index = np.arange(n_t)
    shifted = np.empty((n_tof, n_t))
    # subsets in increasing popcount so every predecessor is final before it is expanded
    subsets = sorted(range(full + 1), key=lambda m: (bin(m).count("1"), m))
    best_final = neg
    best_end: tuple[int, int, int, int] | None = None  # (S, j, departure index, return tof index)

    def ready_of(value: FloatArray) -> tuple[FloatArray, NDArray[np.int64]]:
        """Camping: the best arrival value at or before each departure epoch, and its index."""

        ready = np.maximum.accumulate(value)
        src = np.maximum.accumulate(np.where(value == ready, lattice_index, 0))
        return ready, src

    for subset in subsets:
        # the empty set holds the start state and the "left the camp uncollected" states it
        # feeds, so the camp is expanded first
        locations = (
            [camp_i, *(j for j in range(k) if j != camp_i)] if subset == 0 else list(range(k))
        )
        for j in locations:
            if subset >> j & 1:
                continue
            key = (subset, j)
            value = arrive.get(key)
            if value is None:
                continue
            states += 1
            ready, _src = ready_of(value)
            if not np.isfinite(ready[-1]):
                continue
            collected_now = subset | (1 << j)
            depart_value = ready + mined[j]  # collect j on departure
            mass_hop = float(mass_by_subset[collected_now])
            aj = ids[j]
            # terminal: everything collected once j is, return to Earth
            if collected_now == full:
                ret = table.earth_return(aj)[t0:].astype(np.float64)
                cost = table.return_propellant(ret, mass_hop, table.return_tofs, asteroid=aj, t0=t0)
                cand = depart_value[:, None] - w * cost
                if np.any(np.isfinite(cand)):
                    flat = int(np.argmax(cand))
                    t_i, r_i = divmod(flat, cand.shape[1])
                    if cand[t_i, r_i] > best_final + 1e-9:
                        best_final = float(cand[t_i, r_i])
                        best_end = (subset, j, t_i, r_i)
                continue
            for l_i in range(k):
                if l_i == j or collected_now >> l_i & 1:
                    continue
                if (aj, ids[l_i]) in banned:
                    continue
                # the skip move: leave the camp without collecting (only from the start state)
                moves = [(collected_now, depart_value, mass_hop)]
                if subset == 0 and j == camp_i:
                    moves.append((0, ready, mass_after_deploys))
                penalty = None if phase_penalty is None else phase_penalty(j, l_i)
                for new_subset, base, mass in moves:
                    frac = fraction(j, l_i, mass)
                    cand = base[:, None] - (w * mass) * frac  # (n_t, n_tof)
                    if penalty is not None:
                        cand = cand - (w * penalty)[:, None]
                    shifted.fill(neg)
                    for k_i in range(n_tof):
                        step = int(tof_steps[k_i])
                        if step < n_t:
                            shifted[k_i, step:] = cand[: n_t - step, k_i]
                    best_k = np.argmax(shifted, axis=0)
                    best_val = shifted[best_k, lattice_index]
                    new_key = (new_subset, l_i)
                    target = arrive.get(new_key)
                    if target is None:
                        target = np.full(n_t, neg)
                        arrive[new_key] = target
                        # int32 back-pointers: the 2^k x k states x n_t lattice of a 10-asteroid
                        # tour is ~10k arrays; halving the pointer width takes the per-DP
                        # working set from ~32 to ~20 bytes per state-epoch
                        back[new_key] = (
                            np.full(n_t, -1, dtype=np.int32),
                            np.full(n_t, -1, dtype=np.int32),
                            np.full(n_t, -1, dtype=np.int32),
                        )
                    better = best_val > target + 1e-9
                    if not np.any(better):
                        continue
                    b_prev, b_dep, b_tof = back[new_key]
                    target[better] = best_val[better]
                    b_prev[better] = j if new_subset != subset else j + k
                    b_tof[better] = best_k[better]
                    b_dep[better] = lattice_index[better] - tof_steps[best_k[better]]
    if best_end is None:
        return None
    # -- reconstruct -----------------------------------------------------------------------
    subset, j, t_dep, r_i = best_end
    collect: dict[int, float] = {}
    hops: list[tuple[int, int, float, float, float]] = []
    hop_propellant: list[float] = []
    hop_phase: list[float] = []
    penalty_total = 0.0
    return_departure = float(epochs[t_dep])
    return_tof = float(table.return_tofs[r_i])
    return_dv = float(table.earth_return(ids[j])[t0 + t_dep, r_i])
    collect[ids[j]] = return_departure
    order = [ids[j]]
    reposition = False
    # walk back: at (subset, j) we departed at t_dep; find the arrival that fed it
    _ready, src = ready_of(arrive[(subset, j)])
    t_arr = int(src[t_dep])
    while (subset, j) != (0, camp_i):
        b_prev, b_dep, b_tof = back[(subset, j)]
        prev = int(b_prev[t_arr])
        dep_i = int(b_dep[t_arr])
        tof_i = int(b_tof[t_arr])
        skipped = prev >= k
        prev_j = prev - k if skipped else prev
        tof = float(table.tofs[tof_i])
        dv = float(table.hop(ids[prev_j], ids[j])[t0 + dep_i, tof_i])
        hops.append((ids[prev_j], ids[j], float(epochs[dep_i]), tof, dv))
        move_mass = mass_after_deploys if skipped else float(mass_by_subset[subset])
        hop_propellant.append(move_mass * float(fraction(prev_j, j, move_mass)[dep_i, tof_i]))
        if phase_penalty is not None:
            hop_phase.append(
                float(table.phase_deg(ids[prev_j], ids[j], epochs[dep_i : dep_i + 1])[0])
            )
            penalty_total += float(phase_penalty(prev_j, j)[dep_i])
        if skipped:
            reposition = True
            subset_prev = subset
        else:
            collect[ids[prev_j]] = float(epochs[dep_i])
            order.append(ids[prev_j])
            subset_prev = subset & ~(1 << prev_j)
        subset, j = subset_prev, prev_j
        _ready, src = ready_of(arrive[(subset, j)])
        t_arr = int(src[dep_i])
    hops.reverse()
    hop_propellant.reverse()
    hop_phase.reverse()
    order.reverse()
    collected_proxy = sum(
        C.maximum_collected_mass(collect[a] - deploy_epoch[a]) for a in ids if a in collect
    )
    weighted = sum(
        weights.get(a, 1.0) * C.maximum_collected_mass(collect[a] - deploy_epoch[a]) for a in ids
    )
    return CollectTour(
        tuple(order),
        collect,
        hops,
        reposition,
        best_final,
        collected_proxy,
        # objective = weighted - w x (propellant + phase penalty)
        (weighted - best_final) / w - penalty_total,
        return_departure,
        return_tof,
        return_dv,
        states,
        {
            "lattice_start": float(epochs[0]),
            "asteroids": k,
            "propellant_weight": w,
            "burn_per_hop_kg": burn_per_hop,
        },
        hop_propellant,
        hop_phase,
        penalty_total,
    )
