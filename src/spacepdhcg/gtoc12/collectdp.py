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

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .screening import (
    lambert_hops,
    low_thrust_inflation,
    propellant_for_delta_v,
    thrust_authority_km_s,
)

FloatArray = NDArray[np.float64]
EARTH_ID = 0


@dataclass(frozen=True, slots=True)
class CollectDPSettings:
    step_days: float = 30.0
    # collect hop TOFs (multiples of ``step_days``)
    tofs: tuple[float, ...] = (90.0, 120.0, 150.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 600.0)
    return_tofs: tuple[float, ...] = tuple(float(x) for x in range(300, 901, 60))
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
    cache_pairs: int = 20_000  # bounded pair cache (float32 (n_t, n_tof) per pair)

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
        self.lambert_evaluations = 0

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

    def hop_propellant(self, dv: FloatArray, mass: float, tofs: FloatArray) -> FloatArray:
        """Propellant (kg) of hops ``dv`` at ``mass`` over ``tofs``; ``inf`` when not flyable."""

        s = self.settings
        dv = np.asarray(dv, dtype=np.float64)
        authority = thrust_authority_km_s(mass, tofs, 1.0)
        ok = np.isfinite(dv) & (dv <= s.hop_authority_ratio * authority)
        if s.hop_inflation_slope is None:
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

    def return_propellant(self, dv: FloatArray, mass: float, tofs: FloatArray) -> FloatArray:
        s = self.settings
        dv = np.asarray(dv, dtype=np.float64)
        authority = thrust_authority_km_s(mass, tofs, 1.0)
        ok = np.isfinite(dv) & (dv <= s.return_authority_ratio * authority)
        propellant = propellant_for_delta_v(mass, np.where(ok, dv, 0.0) * s.return_inflation)
        return np.where(ok, propellant, np.inf)


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
) -> CollectTour | None:
    """Best collect tour over ``deployed`` for a ship sitting at ``camp`` at ``camp_epoch``.

    ``deployed`` are ``(asteroid, deploy epoch)`` pairs (the camp asteroid among them);
    ``mass_after_deploys`` the ship mass at the camp.  ``propellant_weight`` overrides the
    settings' objective weight.  Returns ``None`` when no tour closes.

    Propellant is priced as ``mass x f`` with ``f = 1 - exp(-inflation x ΔV / v_e)`` tabulated
    per pair once (feasibility and the inflation model evaluated at the heaviest mass the ship
    can reach, so a lighter ship is never priced optimistically); the mass of a hop is the camp
    mass plus the miners collected so far (mined to the window end).
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
    tof_steps = table.tof_steps
    n_tof = tof_steps.shape[0]
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
    mass_ref = mass_after_deploys + float(mined_end.sum())
    mass_by_subset = np.array(
        [
            mass_after_deploys + sum(mined_end[i] for i in range(k) if m >> i & 1)
            for m in range(full + 1)
        ]
    )

    fractions: dict[tuple[int, int], FloatArray] = {}

    def fraction(j: int, l_i: int) -> FloatArray:
        """Propellant per kg of ship mass for hop ``j -> l`` on the local lattice x TOF grid."""

        key = (j, l_i)
        cached = fractions.get(key)
        if cached is None:
            dv = table.hop(ids[j], ids[l_i])[t0:].astype(np.float64)
            cost = table.hop_propellant(dv, mass_ref, table.tofs)
            cached = cost / mass_ref
            fractions[key] = cached
        return cached

    # DP tables: value on *arrival* at location j with collected set S, per lattice epoch
    neg = -np.inf
    arrive: dict[tuple[int, int], FloatArray] = {}
    # back-pointers on arrival: (previous location [+k when the camp was left uncollected],
    # departure index, tof index)
    back: dict[tuple[int, int], tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = {}
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
                cost = table.return_propellant(ret, mass_hop, table.return_tofs)
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
                frac = fraction(j, l_i)
                # the skip move: leave the camp without collecting (only from the start state)
                moves = [(collected_now, depart_value, mass_hop)]
                if subset == 0 and j == camp_i:
                    moves.append((0, ready, mass_after_deploys))
                for new_subset, base, mass in moves:
                    cand = base[:, None] - (w * mass) * frac  # (n_t, n_tof)
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
                        back[new_key] = (
                            np.full(n_t, -1, dtype=np.int64),
                            np.full(n_t, -1, dtype=np.int64),
                            np.full(n_t, -1, dtype=np.int64),
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
        (weighted - best_final) / w,  # objective = weighted - w x propellant
        return_departure,
        return_tof,
        return_dv,
        states,
        {"lattice_start": float(epochs[0]), "asteroids": k, "propellant_weight": w},
    )
