"""Deterministic beam search for self-cleaning GTOC12 mining-ship routes.

A route is ``Earth -> A1 -> ... -> Ak (deploy at each) -> camp at Ak -> collection tour over
{A1..Ak} -> Earth``.  Deploy hops are expanded forwards from a launch-epoch grid with Lambert
rendezvous costs; the collection tour is scheduled *backwards* from the end of the window so every
miner works as long as possible.  Costs are impulsive proxies inflated for finite thrust; the
low-thrust refinement (``pipeline``) replaces them with certified arcs.

The candidate generator encodes what the archived JPL/Antipodes solutions do (see
``references.py`` and ``docs/GTOC12_TRACK.md``): the next asteroid is picked in *position space
at the departure epoch* — within a few hundredths of an AU in semi-major axis, a few degrees of
inclination, and a few degrees of phase — so that a 100-250 day, sub-revolution hop costs
50-100 kg.  Chains also keep a propellant reserve for the collection tour so the beam does not
fill up with deploy phases that can never be collected.

Everything is deterministic: candidate order is fixed, ties break on asteroid ID, and the only use
of ``seed`` is to shuffle nothing unless ``randomise`` is requested.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .clusters import ClusterBands, ComovingClusters
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .proxies import phasing_edelbaum_proxy
from .screening import (
    exhaust_velocity_km_s,
    lambert_hops,
    low_thrust_inflation,
    propellant_for_delta_v,
    screen_asteroid_hops,
    screen_earth_to_asteroids,
    thrust_authority_km_s,
)

FloatArray = NDArray[np.float64]
EARTH_ID = 0


@dataclass(frozen=True, slots=True)
class SearchSettings:
    beam_width: int = 24
    max_deploys: int = 10
    min_deploys: int = 1
    launch_epochs: tuple[float, ...] = tuple(C.MISSION_START_MJD + np.arange(0.0, 731.0, 30.0))
    earth_leg_tofs: tuple[float, ...] = tuple(np.arange(300.0, 901.0, 50.0))
    hop_tofs: tuple[float, ...] = (
        60.0,
        90.0,
        120.0,
        150.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
        480.0,
    )
    collect_hop_tofs: tuple[float, ...] = (
        90.0,
        120.0,
        150.0,
        180.0,
        240.0,
        300.0,
        360.0,
        420.0,
        480.0,
        600.0,
        720.0,
    )
    deploy_wait_days: tuple[float, ...] = (0.0, 30.0, 60.0, 120.0)
    max_per_first: int = 8  # beam diversity: variants sharing the first asteroid
    neighbours: int = 48
    # position-space candidate metric scales (reference-hop p95 values; see references.py)
    band_a_au: float = 0.04
    band_i_deg: float = 4.5  # relative inclination (vector difference), deg
    band_e: float = 0.06  # eccentricity-vector difference
    band_phase_deg: float = 3.3
    filter_scale: float = 1.5  # element-band filter = filter_scale x reference p95 bands
    # Leg model: propellant = rocket equation at ``inflation x`` the zero-revolution Lambert ΔV,
    # and a leg is admissible while Lambert ΔV / (T_max/m x TOF) <= the role's authority ratio.
    #
    # Earth out: the 111 archived Earth legs (all to a = 2.73-2.80 AU, 490-565 days) cost
    # 0.83x their Lambert ΔV and fly at 0.72 of the *full* authority, but that is measured on
    # *their* legs.  For the legs our beam picks the story is different: the nine certified
    # Earth legs of the fleet runs all had Lambert ratio <= 0.49 and cost 0.86-2.22x (median
    # 1.4x) the Lambert ΔV, while every ratio-0.71 leg the 0.85 limit admitted
    # (fleet6_coop_v1, E->6014/15614/26515) failed SCvx with virtual control left.  The
    # single-conic Lambert arc with a free 6 km/s asymptote is a loose proxy here, so the limit
    # stays at the certified envelope and the re-timer calibrates per pair.
    earth_out_inflation: float = 1.6
    earth_out_authority_ratio: float = 0.5
    # Earth return: certified returns cost ~1.0x Lambert at ratios <= 0.4; above that is
    # untested by SCvx so the limit stays at the proven envelope (the re-timer bans what does
    # not fly).
    earth_return_inflation: float = 1.6
    earth_return_authority_ratio: float = 0.5
    # hops: reference hops cost 1.16x Lambert (p90 1.34).  The beam keeps the 0.667 ratio
    # (1.2x ΔV within 0.8 duty) that found the 544-548 kg / 8-asteroid chains: tightening it
    # to the certified 0.49 envelope cut chains to depth 5-6 (376-446 kg) because collect hops
    # of the heavy ship no longer fit the window; the re-timer applies 0.45 to what it moves.
    hop_inflation: float = 1.2
    # ratio-dependent hop inflation ``floor + slope x (Lambert ΔV / full authority)``, fitted on
    # 1674 certified hops (``screening.low_thrust_inflation``); ``None`` keeps the flat factor.
    # Off in the beam by default: the beam's chains are seeds the re-timer re-prices anyway, and
    # with the model the beam closes fewer, shorter chains (6 asteroids / 401 kg vs 7 / 440 kg
    # on the 99-member family 0) because it prices its fast deploy hops out of the mass budget.
    hop_inflation_slope: float | None = None
    hop_inflation_floor: float = 1.05
    hop_authority_ratio: float = 0.667
    end_margin_days: float = 2.0
    return_window_days: float = 600.0
    collect_wait_window_days: float = 600.0
    max_per_deployed_set: int = 2
    first_level_limit: int = 4000
    # injected (certified) Earth legs unlock the Lambert grid for their target within this many
    # days of the certified launch and TOF, priced at the per-target measured/Lambert ratio
    first_level_window_days: float = 200.0
    earth_block: int = 1500  # asteroids per Earth-leg screening block (bounds memory)
    schedule_step_days: float = 15.0
    wait_penalty: float = 1.0  # kg propellant-equivalent per kg of mining mass forgone
    # a collect hop may take up to slack x the mean time left per remaining collect; 1.5 was
    # neutral on the full-catalogue probe once the hop ratio went back to 0.667, so it is off
    collect_span_slack: float = float("inf")
    reserve_fraction: float = 0.9  # collect-phase hop propellant ~ deploy-phase hop propellant
    return_reserve_kg: float = 250.0  # reference Earth returns cost 190-230 kg
    # beam heuristic weights; full-catalogue chains die on the 15-year window with 230-430 kg
    # of propellant unused, but pricing time (0.02-0.05 kg/day) steered the beam into 120-180
    # day hops that SCvx could not fly (full_catalogue_search4/5), so it is off by default
    propellant_weight: float = 0.15
    time_weight: float = 0.0  # kg of heuristic score per day of deploy-phase duration
    # cluster-first prior (clusters.py): Earth targets need at least ``cluster_min_density``
    # co-moving neighbours, and partials earn ``cluster_bonus_kg`` x (unvisited co-moving
    # neighbours of the current asteroid, capped at the remaining deploy slots) / max_deploys
    # Off by default: on the full-catalogue pool the prior (density >= 8, 150 kg) and the
    # co-moving-first expansion lost the 544 kg / 8-asteroid chain (446 kg at best) - the
    # densest co-moving families are not the ones a 0.5-ratio Earth leg reaches.  The
    # co-moving structure is exploited after the beam instead: insertion candidates, orphan
    # ranking and cooperative seeding all use the same element bands.
    cluster_min_density: int = 0
    cluster_bonus_kg: float = 0.0
    cluster_density_cap: int = 30  # Earth-target bonus saturates at this co-moving density
    cluster_radius: float = 1.5
    cluster_phase_band_deg: float = 8.0
    cluster_neighbours_first: bool = False  # expansions try co-moving neighbours first
    # cooperative pricing: first-level bonus for Earth targets co-moving with a seed (orphan)
    seed_bonus_kg: float = 120.0
    initial_mass: float = C.MAX_INITIAL_MASS_KG
    time_budget_seconds: float = float("inf")  # stop expanding (keep completed plans) past this
    seed: int = 0
    randomise: bool = False


@dataclass(frozen=True, slots=True)
class EarthLeg:
    """A pre-screened (typically SCvx-certified) Earth -> asteroid leg to seed the beam with.

    ``propellant_kg`` is what the leg really costs (the SCvx-measured value when certified), so
    the chain built on it starts from the right mass instead of the inflated Lambert estimate.
    """

    target: int
    launch_epoch: float
    tof_days: float
    delta_v_km_s: float  # zero-revolution Lambert proxy (kept for the plan record)
    propellant_kg: float
    certified: bool = True

    @property
    def arrival_epoch(self) -> float:
        return self.launch_epoch + self.tof_days


@dataclass(frozen=True, slots=True)
class PlannedLeg:
    from_id: int  # 0 = Earth
    to_id: int
    departure_epoch: float
    arrival_epoch: float
    delta_v_proxy_km_s: float
    inflation: float
    role: str  # "earth_out" | "deploy_hop" | "collect_hop" | "earth_return" | "camp"

    @property
    def tof_days(self) -> float:
        return self.arrival_epoch - self.departure_epoch


@dataclass(frozen=True, slots=True)
class RoutePlan:
    legs: tuple[PlannedLeg, ...]
    deploy_epochs: dict[int, float]
    collect_epochs: dict[int, float]
    collected_mass: dict[int, float]
    propellant_proxy_kg: float
    final_mass_proxy_kg: float
    # cooperative collection: deploy epochs of miners this ship collects but another ship
    # deployed (the collected mass is mined from that epoch); empty for self-cleaning plans
    foreign_deploy_epochs: dict[int, float] = field(default_factory=dict)

    @property
    def asteroids(self) -> tuple[int, ...]:
        """Every asteroid the ship touches: its deploys first, then foreign collects."""

        foreign = [a for a in self.collect_epochs if a not in self.deploy_epochs]
        return (*self.deploy_epochs, *foreign)

    @property
    def orphaned(self) -> tuple[int, ...]:
        """Asteroids this ship deploys on but leaves for another ship to collect."""

        return tuple(a for a in self.deploy_epochs if a not in self.collect_epochs)

    @property
    def self_cleaning(self) -> bool:
        return set(self.deploy_epochs) == set(self.collect_epochs)

    def deploy_epoch_of(self, asteroid: int) -> float:
        """Epoch the miner collected at ``asteroid`` was deployed (own or foreign)."""

        if asteroid in self.deploy_epochs:
            return self.deploy_epochs[asteroid]
        return self.foreign_deploy_epochs[asteroid]

    @property
    def total_collected_kg(self) -> float:
        return sum(self.collected_mass.values())

    @property
    def feasible(self) -> bool:
        return self.final_mass_proxy_kg >= C.DRY_MASS_KG + self.total_collected_kg

    @classmethod
    def from_summary(cls, data: dict[str, object]) -> RoutePlan:
        """Inverse of :meth:`summary` (used to re-time archived plans and in tests)."""

        defaults = SearchSettings()
        default_inflation = {
            "earth_out": defaults.earth_out_inflation,
            "earth_return": defaults.earth_return_inflation,
            "deploy_hop": defaults.hop_inflation,
            "collect_hop": defaults.hop_inflation,
            "camp": 1.0,
        }
        legs = tuple(
            PlannedLeg(
                int(item["from"]),
                int(item["to"]),
                float(item["t0"]),
                float(item["tf"]),
                float(item["dv_proxy_km_s"]),
                float(item.get("inflation", default_inflation[str(item["role"])])),
                str(item["role"]),
            )
            for item in data["legs"]  # type: ignore[union-attr]
        )
        return cls(
            legs,
            {int(k): float(v) for k, v in data["deploy_epochs"].items()},  # type: ignore[union-attr]
            {int(k): float(v) for k, v in data["collect_epochs"].items()},  # type: ignore[union-attr]
            {int(k): float(v) for k, v in data["collected_mass_kg"].items()},  # type: ignore[union-attr]
            float(data["propellant_proxy_kg"]),  # type: ignore[arg-type]
            float(data["final_mass_proxy_kg"]),  # type: ignore[arg-type]
            {
                int(k): float(v)
                for k, v in data.get("foreign_deploy_epochs", {}).items()  # type: ignore[union-attr]
            },
        )

    def summary(self) -> dict[str, object]:
        return {
            "asteroids": list(self.asteroids),
            "orphaned": list(self.orphaned),
            "self_cleaning": self.self_cleaning,
            "launch_epoch": self.legs[0].departure_epoch,
            "earth_return_epoch": self.legs[-1].arrival_epoch,
            "deploy_epochs": dict(self.deploy_epochs),
            "collect_epochs": dict(self.collect_epochs),
            "foreign_deploy_epochs": dict(self.foreign_deploy_epochs),
            "collected_mass_kg": dict(self.collected_mass),
            "total_collected_kg": self.total_collected_kg,
            "propellant_proxy_kg": self.propellant_proxy_kg,
            "final_mass_proxy_kg": self.final_mass_proxy_kg,
            "feasible_proxy": self.feasible,
            "legs": [
                {
                    "from": leg.from_id,
                    "to": leg.to_id,
                    "t0": leg.departure_epoch,
                    "tf": leg.arrival_epoch,
                    "dv_proxy_km_s": leg.delta_v_proxy_km_s,
                    "inflation": leg.inflation,
                    "role": leg.role,
                }
                for leg in self.legs
            ],
        }


@dataclass(slots=True)
class _Partial:
    legs: list[PlannedLeg]
    location: int
    epoch: float
    mass: float
    deployed: list[tuple[int, float]]
    hop_propellant: float = 0.0
    score: float = 0.0


@dataclass(slots=True)
class SearchResult:
    best: RoutePlan | None
    candidates: list[RoutePlan]
    expansions: int
    lambert_evaluations: int
    wall_seconds: float
    failures: list[dict[str, object]] = field(default_factory=list)
    depth_reached: int = 0
    best_by_depth: dict[int, float] = field(default_factory=dict)
    first_level: int = 0  # Earth-leg partials the beam started from


def element_deviations(
    catalogue: AsteroidCatalogue, source: int, pool: NDArray[np.int64]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """|Δa| (AU), |Δe-vector| and relative inclination (deg) between ``source`` and ``pool``.

    Vector forms are used because two orbits with equal e but different perihelion directions
    (or equal i but different nodes) are different ellipses that drift apart within years.
    """

    index = catalogue.index_of(pool)
    s_index = catalogue.index_of(source)
    da = (
        np.abs(catalogue.semi_major_axis_km[index] - catalogue.semi_major_axis_km[s_index])
        / C.AU_KM
    )
    varpi = catalogue.ascending_node_rad + catalogue.argument_of_perihelion_rad
    e_vec = catalogue.eccentricity[:, None] * np.stack([np.cos(varpi), np.sin(varpi)], axis=1)
    de = np.linalg.norm(e_vec[index] - e_vec[s_index], axis=1)
    inc = catalogue.inclination_rad
    node = catalogue.ascending_node_rad
    i_vec = inc[:, None] * np.stack([np.cos(node), np.sin(node)], axis=1)
    di = np.rad2deg(np.linalg.norm(i_vec[index] - i_vec[s_index], axis=1))
    return da, de, di


def positional_candidates(
    catalogue: AsteroidCatalogue,
    source: int,
    pool: NDArray[np.int64],
    epoch: float,
    settings: SearchSettings,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Rank ``pool`` by closeness to ``source`` in (a, e, i, phase) at ``epoch``.

    The metric is the sum of squared deviations scaled by the reference-hop p95 bands, i.e. the
    *cluster-first* generator: it favours asteroids that are physically near the ship at
    departure and on a nearly identical orbit, which is where the 50-100 kg hops live.
    """

    pool = pool[pool != source]
    r_pool, _ = asteroid_state(catalogue, pool, np.full(pool.shape[0], epoch))
    r_source, _ = asteroid_state(catalogue, source, epoch)
    da, de, di = element_deviations(catalogue, source, pool)
    cross_z = r_source[0] * r_pool[:, 1] - r_source[1] * r_pool[:, 0]
    phase = np.rad2deg(np.arctan2(cross_z, r_pool @ r_source))
    metric = (
        (da / settings.band_a_au) ** 2
        + (di / settings.band_i_deg) ** 2
        + (de / settings.band_e) ** 2
        + (phase / settings.band_phase_deg) ** 2
    )
    order = np.lexsort((pool, metric))
    return pool[order], metric[order]


def proxy_candidates(
    catalogue: AsteroidCatalogue,
    source: int,
    pool: NDArray[np.int64],
    epoch: float,
    settings: SearchSettings,
) -> tuple[NDArray[np.int64], FloatArray]:
    """Rank ``pool`` by the Lambert-free phasing/Edelbaum ΔV proxy over the hop TOF grid."""

    pool = pool[pool != source]
    proxy = phasing_edelbaum_proxy(catalogue, source, pool, epoch, np.asarray(settings.hop_tofs))
    order = np.lexsort((pool, proxy["best_delta_v"]))
    return pool[order], proxy["best_delta_v"][order]


class RouteSearch:
    def __init__(
        self,
        catalogue: AsteroidCatalogue,
        asteroid_ids: NDArray[np.int64],
        settings: SearchSettings | None = None,
        excluded: set[int] | frozenset[int] | None = None,
        weights: dict[int, float] | None = None,
        seeds: dict[int, float] | None = None,
        first_level: Sequence[EarthLeg] | None = None,
    ) -> None:
        self.catalogue = catalogue
        # cooperative pricing: uncollected miners of earlier ships (asteroid -> deploy epoch);
        # Earth targets co-moving with them earn ``seed_bonus_kg`` in the first level
        self.seeds: dict[int, float] = dict(seeds or {})
        # cluster pricing: when given, these (SCvx-certified) Earth legs seed the first level
        # (a calibrated grid around them) instead of the Lambert launch-grid screening
        self.first_level: tuple[EarthLeg, ...] | None = (
            None if first_level is None else tuple(first_level)
        )
        # legs SCvx refused: ``(from, to)`` asteroid pairs are never hopped again (deploy or
        # collect), ``(target, launch, tof)`` Earth legs never seed the first level again.  The
        # pricing loop fills these so a refused chain is not rebuilt for the next ship slot.
        self.banned_pairs: set[tuple[int, int]] = set()
        self.banned_earth: set[tuple[int, float, float]] = set()
        banned = set(excluded or ())
        self.excluded: frozenset[int] = frozenset(banned)
        self.ids = np.asarray(
            sorted(int(item) for item in asteroid_ids if int(item) not in banned), dtype=np.int64
        )
        self.settings = settings or SearchSettings()
        # per-asteroid score weights (the frozen bonus coefficients); 1.0 when absent
        self.weights = weights or {}
        self.lambert_evaluations = 0
        self._hop_cache: dict[tuple[int, float], dict[str, FloatArray]] = {}
        self._return_cache: dict[int, list[tuple[float, float, float]]] = {}
        self._collect_cache: dict[tuple[int, int, float], list[tuple[float, float, float]]] = {}
        self.last_failure = ""
        self._band_cache: dict[int, NDArray[np.int64]] = {}
        self._clusters: ComovingClusters | None = None

    @property
    def clusters(self) -> ComovingClusters | None:
        """Co-moving clusters of the pool (built lazily, only when the prior is enabled)."""

        s = self.settings
        if s.cluster_min_density <= 0 and s.cluster_bonus_kg <= 0.0:
            return None
        if self._clusters is None:
            self._clusters = ComovingClusters(
                self.catalogue,
                self.ids,
                ClusterBands(
                    a_au=s.band_a_au,
                    e=s.band_e,
                    i_deg=s.band_i_deg,
                    phase_deg=s.cluster_phase_band_deg,
                    radius=s.cluster_radius,
                ),
            )
        return self._clusters

    def seeded_mask(self, pool: NDArray[np.int64]) -> NDArray[np.float64] | None:
        """1.0 for pool asteroids co-moving (within the filter bands) with a seed asteroid."""

        s = self.settings
        if not self.seeds or s.seed_bonus_kg <= 0.0 or pool.shape[0] == 0:
            return None
        mask = np.zeros(pool.shape[0], dtype=bool)
        for seed in sorted(self.seeds):
            da, de, di = element_deviations(self.catalogue, seed, pool)
            mask |= (
                (da <= s.filter_scale * s.band_a_au)
                & (de <= s.filter_scale * s.band_e)
                & (di <= s.filter_scale * s.band_i_deg)
            )
        return mask.astype(np.float64)

    def _cluster_potential_kg(self, partial: _Partial) -> float:
        clusters = self.clusters
        if clusters is None or self.settings.cluster_bonus_kg <= 0.0:
            return 0.0
        visited = {item for item, _ in partial.deployed}
        remaining = max(self.settings.max_deploys - len(partial.deployed), 0)
        potential = min(clusters.unvisited_potential(partial.location, visited), remaining)
        return self.settings.cluster_bonus_kg * potential / max(self.settings.max_deploys, 1)

    def weighted(self, plan: RoutePlan) -> float:
        """Bonus-weighted collected mass (the fixed post-competition score of the plan)."""

        return sum(self.weights.get(a, 1.0) * m for a, m in plan.collected_mass.items())

    # -- proxies --

    def _propellant(self, mass: float, delta_v: float, inflation: float) -> float:
        return float(propellant_for_delta_v(mass, delta_v * inflation))

    def hop_inflation_for(self, delta_v: float, mass: float, tof: float) -> float:
        """Propellant inflation of an asteroid hop at this mass and TOF (ratio model or flat)."""

        s = self.settings
        if s.hop_inflation_slope is None:
            return s.hop_inflation
        return float(
            low_thrust_inflation(
                delta_v, mass, tof, floor=s.hop_inflation_floor, slope=s.hop_inflation_slope
            )
        )

    def limits(self, role: str) -> tuple[float, float]:
        """(propellant inflation, Lambert-ΔV / full-authority ratio limit) for a leg role."""

        s = self.settings
        if role == "earth_out":
            return s.earth_out_inflation, s.earth_out_authority_ratio
        if role == "earth_return":
            return s.earth_return_inflation, s.earth_return_authority_ratio
        return s.hop_inflation, s.hop_authority_ratio

    def _feasible(self, mass: float, delta_v: float, tof: float, role: str) -> bool:
        _, ratio = self.limits(role)
        return delta_v <= ratio * float(thrust_authority_km_s(mass, tof, 1.0))

    def band_pool(self, asteroid_id: int) -> NDArray[np.int64]:
        """Asteroids on orbits similar enough to ``asteroid_id`` to be collectable years later.

        Reference hops stay within |Δa| 0.04 AU, |Δe| 0.045, |Δi| 3° (p95); the filter uses 1.5x
        those bands and falls back to the nearest ``neighbours`` asteroids by scaled element
        distance when the pool is too sparse (e.g. the reduced instance).
        """

        if asteroid_id in self._band_cache:
            return self._band_cache[asteroid_id]
        s = self.settings
        pool = self.ids[self.ids != asteroid_id]
        da, de, di = element_deviations(self.catalogue, asteroid_id, pool)
        inside = (
            (da <= s.filter_scale * s.band_a_au)
            & (de <= s.filter_scale * s.band_e)
            & (di <= s.filter_scale * s.band_i_deg)
        )
        if inside.sum() >= s.neighbours:
            chosen = pool[inside]
        else:
            metric = (da / s.band_a_au) ** 2 + (de / s.band_e) ** 2 + (di / s.band_i_deg) ** 2
            chosen = pool[np.lexsort((pool, metric))[: s.neighbours]]
        self._band_cache[asteroid_id] = np.sort(chosen)
        return self._band_cache[asteroid_id]

    def candidates(self, asteroid_id: int, epoch: float) -> NDArray[np.int64]:
        """Union of the proxy-ΔV ranking and the positional (cluster) ranking, proxy first."""

        s = self.settings
        pool = self.band_pool(asteroid_id)
        by_proxy, _ = proxy_candidates(self.catalogue, asteroid_id, pool, epoch, s)
        by_position, _ = positional_candidates(self.catalogue, asteroid_id, pool, epoch, s)
        chosen: list[int] = []
        seen: set[int] = set()
        comoving: list[int] = []
        clusters = self.clusters
        if clusters is not None and s.cluster_neighbours_first and clusters.contains(asteroid_id):
            # (a source outside the pool - e.g. an excluded asteroid of an archived plan being
            # re-timed - simply gets no co-moving preference)
            allowed = set(self.ids.tolist())
            comoving = [int(a) for a in clusters.neighbours(asteroid_id) if int(a) in allowed][
                : s.neighbours
            ]
        for item in (
            comoving + list(by_proxy[: s.neighbours]) + list(by_position[: s.neighbours // 2])
        ):
            if int(item) not in seen:
                seen.add(int(item))
                chosen.append(int(item))
        return np.asarray(chosen, dtype=np.int64)

    def hops_from(self, asteroid_id: int, epoch: float) -> dict[str, FloatArray]:
        key = (asteroid_id, round(epoch, 6))
        if key not in self._hop_cache:
            targets = self.candidates(asteroid_id, epoch)
            result = screen_asteroid_hops(
                self.catalogue, asteroid_id, targets, epoch, np.asarray(self.settings.hop_tofs)
            )
            self.lambert_evaluations += 2 * targets.shape[0] * len(self.settings.hop_tofs)
            self._hop_cache[key] = result
        return self._hop_cache[key]

    def _reserve(self, partial: _Partial) -> float:
        """Propellant the collection tour and Earth return will need (reference-calibrated)."""

        s = self.settings
        return s.reserve_fraction * partial.hop_propellant + s.return_reserve_kg

    # -- search --

    def _first_level(self) -> list[_Partial]:
        """Earth -> A1 candidates, screened block-wise to bound memory at catalogue scale."""

        s = self.settings
        if s.max_deploys < 1 or self.ids.shape[0] == 0:
            return []
        if self.first_level is not None:
            return self._injected_first_level()
        epochs = np.asarray(s.launch_epochs)
        tofs = np.asarray(s.earth_leg_tofs)
        horizon = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS
        tof_grid = np.broadcast_to(tofs[None, None, :], (1, epochs.shape[0], tofs.shape[0]))
        out_inflation, out_ratio = self.limits("earth_out")
        authority = out_ratio * thrust_authority_km_s(s.initial_mass, tof_grid, 1.0)
        arrival_grid = epochs[None, :, None] + tofs[None, None, :]
        mined_grid = (
            C.MINING_RATE_KG_PER_YEAR * np.maximum(horizon - arrival_grid, 0.0) / C.YEAR_DAYS
        )
        kept: list[tuple[float, int, float, float, float, float]] = []
        clusters = self.clusters
        pool = self.ids
        if clusters is not None and s.cluster_min_density > 0:
            # cluster-first: only asteroids with enough co-moving neighbours can start a chain
            dense = np.asarray(
                [clusters.density_of(int(a)) >= s.cluster_min_density for a in pool], dtype=bool
            )
            # sparse pools (reduced instances, tests) fall back to the densest quartile so the
            # prior never empties the first level
            if int(dense.sum()) < s.beam_width:
                densities = np.asarray([clusters.density_of(int(a)) for a in pool])
                cutoff = np.quantile(densities, 0.75) if densities.size else 0
                dense = densities >= cutoff
            pool = pool[dense]
        seeded = self.seeded_mask(pool)
        for start in range(0, pool.shape[0], s.earth_block):
            block = pool[start : start + s.earth_block]
            grid = screen_earth_to_asteroids(self.catalogue, block, epochs, tofs)
            self.lambert_evaluations += 2 * grid["total_delta_v"].size
            dv_grid = np.where(grid["feasible"], grid["total_delta_v"], np.inf)
            ok = np.isfinite(dv_grid) & (dv_grid <= authority)
            propellant_grid = propellant_for_delta_v(s.initial_mass, dv_grid * out_inflation)
            weight = np.asarray([self.weights.get(int(a), 1.0) for a in block])[:, None, None]
            score_grid = np.where(
                ok,
                weight * mined_grid - s.propellant_weight * (propellant_grid + C.MINER_MASS_KG),
                -np.inf,
            )
            if clusters is not None and s.cluster_bonus_kg > 0.0:
                potential = np.asarray(
                    [
                        min(clusters.density_of(int(a)), s.cluster_density_cap)
                        / s.cluster_density_cap
                        for a in block
                    ]
                )[:, None, None]
                score_grid = np.where(ok, score_grid + s.cluster_bonus_kg * potential, -np.inf)
            if seeded is not None:
                # pricing seeded from clusters with uncollected miners: chains starting there can
                # pick the orphans up in their collect tour (cooperative collection)
                bonus = s.seed_bonus_kg * seeded[start : start + s.earth_block][:, None, None]
                score_grid = np.where(ok, score_grid + bonus, -np.inf)
            flat = np.argsort(-score_grid.ravel(), kind="stable")[: s.first_level_limit]
            for index in flat:
                a_index, e_index, t_index = np.unravel_index(int(index), score_grid.shape)
                if not ok[a_index, e_index, t_index]:
                    break
                kept.append(
                    (
                        float(score_grid[a_index, e_index, t_index]),
                        int(block[a_index]),
                        float(epochs[e_index]),
                        float(tofs[t_index]),
                        float(dv_grid[a_index, e_index, t_index]),
                        float(propellant_grid[a_index, e_index, t_index]),
                    )
                )
        kept.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        beam: list[_Partial] = []
        for _score, asteroid, launch, tof, dv, propellant in kept[: s.first_level_limit]:
            arrival = launch + tof
            leg = PlannedLeg(EARTH_ID, asteroid, launch, arrival, dv, out_inflation, "earth_out")
            mass = s.initial_mass - propellant - C.MINER_MASS_KG
            beam.append(_Partial([leg], asteroid, arrival, mass, [(asteroid, arrival)]))
        return beam

    def _injected_first_level(self) -> list[_Partial]:
        """First level from pre-certified Earth legs: a calibrated grid around each of them.

        The certified legs say which targets SCvx can really reach and what the Earth leg truly
        costs. Seeding the beam with those exact legs alone starves it (a handful of partials,
        one arrival epoch each), so each certified leg also unlocks the Lambert launch/TOF grid
        for its target within ``first_level_window_days`` of the certified launch and TOF, priced
        with the per-target inflation ``measured Delta-V / Lambert Delta-V`` instead of the global
        one. The certified legs themselves are kept at their measured propellant.
        """

        s = self.settings
        _out_inflation, out_ratio = self.limits("earth_out")
        horizon = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS
        allowed = set(self.ids.tolist())
        legs = [leg for leg in self.first_level or () if leg.target in allowed]
        if not legs:
            return []
        exhaust = exhaust_velocity_km_s()
        epochs = np.asarray(s.launch_epochs)
        tofs = np.asarray(s.earth_leg_tofs)
        window = s.first_level_window_days
        # per-target calibration: the smallest measured/Lambert ratio over that target's legs
        calibration: dict[int, float] = {}
        for leg in legs:
            true_dv = exhaust * math.log(s.initial_mass / (s.initial_mass - leg.propellant_kg))
            ratio = true_dv / max(leg.delta_v_km_s, 1e-9)
            calibration[leg.target] = min(calibration.get(leg.target, np.inf), ratio)
        # (score, target, launch, tof, lambert dv, propellant, inflation)
        kept: list[tuple[float, int, float, float, float, float, float]] = []
        seen: set[tuple[int, float, float]] = set()
        for leg in legs:
            key = (leg.target, leg.launch_epoch, leg.tof_days)
            if key in seen:
                continue
            seen.add(key)
            mined = C.MINING_RATE_KG_PER_YEAR * max(horizon - leg.arrival_epoch, 0.0) / C.YEAR_DAYS
            score = self.weights.get(leg.target, 1.0) * mined - s.propellant_weight * (
                leg.propellant_kg + C.MINER_MASS_KG
            )
            kept.append(
                (
                    score,
                    leg.target,
                    leg.launch_epoch,
                    leg.tof_days,
                    leg.delta_v_km_s,
                    leg.propellant_kg,
                    calibration[leg.target],
                )
            )
        if window > 0.0 and epochs.size and tofs.size:
            targets = np.asarray(sorted(calibration), dtype=np.int64)
            grid = screen_earth_to_asteroids(self.catalogue, targets, epochs, tofs)
            self.lambert_evaluations += 2 * grid["total_delta_v"].size
            dv_grid = np.where(grid["feasible"], grid["total_delta_v"], np.inf)
            tof_grid = np.broadcast_to(tofs[None, :], (epochs.shape[0], tofs.shape[0]))
            authority = out_ratio * thrust_authority_km_s(s.initial_mass, tof_grid, 1.0)
            arrival_grid = epochs[:, None] + tofs[None, :]
            mined_grid = (
                C.MINING_RATE_KG_PER_YEAR * np.maximum(horizon - arrival_grid, 0.0) / C.YEAR_DAYS
            )
            for t_index, target in enumerate(targets.tolist()):
                inflation = calibration[target]
                near = np.zeros((epochs.shape[0], tofs.shape[0]), dtype=bool)
                for leg in legs:
                    if leg.target != target:
                        continue
                    near |= (np.abs(epochs[:, None] - leg.launch_epoch) <= window) & (
                        np.abs(tofs[None, :] - leg.tof_days) <= window
                    )
                dv = dv_grid[t_index]
                ok = near & np.isfinite(dv) & (dv * inflation <= authority)
                propellant = propellant_for_delta_v(s.initial_mass, dv * inflation)
                weight = self.weights.get(target, 1.0)
                score_grid = np.where(
                    ok,
                    weight * mined_grid - s.propellant_weight * (propellant + C.MINER_MASS_KG),
                    -np.inf,
                )
                for e_index, t2_index in zip(*np.nonzero(ok), strict=True):
                    key = (target, float(epochs[e_index]), float(tofs[t2_index]))
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(
                        (
                            float(score_grid[e_index, t2_index]),
                            target,
                            key[1],
                            key[2],
                            float(dv[e_index, t2_index]),
                            float(propellant[e_index, t2_index]),
                            inflation,
                        )
                    )
        kept = [item for item in kept if (item[1], item[2], item[3]) not in self.banned_earth]
        kept.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        beam: list[_Partial] = []
        for _score, target, launch, tof, dv, propellant, inflation in kept[: s.first_level_limit]:
            arrival = launch + tof
            planned = PlannedLeg(EARTH_ID, target, launch, arrival, dv, inflation, "earth_out")
            mass = s.initial_mass - propellant - C.MINER_MASS_KG
            beam.append(_Partial([planned], target, arrival, mass, [(target, arrival)]))
        return beam

    def _expand(self, partial: _Partial) -> list[_Partial]:
        s = self.settings
        visited = {item for item, _ in partial.deployed}
        children: list[_Partial] = []
        for wait in s.deploy_wait_days:
            departure = partial.epoch + float(wait)
            hops = self.hops_from(partial.location, departure)
            for t_index, target in enumerate(hops["target_ids"]):
                target = int(target)
                if target in visited or (partial.location, target) in self.banned_pairs:
                    continue
                for f_index, tof in enumerate(hops["tofs_days"]):
                    if not hops["feasible"][t_index, f_index]:
                        continue
                    dv = float(hops["total_delta_v"][t_index, f_index])
                    if not self._feasible(partial.mass, dv, float(tof), "deploy_hop"):
                        continue
                    inflation = self.hop_inflation_for(dv, partial.mass, float(tof))
                    propellant = self._propellant(partial.mass, dv, inflation)
                    arrival = departure + float(tof)
                    if arrival > C.MISSION_END_MJD - 3.0 * C.YEAR_DAYS:
                        continue
                    legs = list(partial.legs)
                    if wait > 0.0:
                        legs.append(
                            PlannedLeg(
                                partial.location,
                                partial.location,
                                partial.epoch,
                                departure,
                                0.0,
                                1.0,
                                "camp",
                            )
                        )
                    legs.append(
                        PlannedLeg(
                            partial.location,
                            target,
                            departure,
                            arrival,
                            dv,
                            inflation,
                            "deploy_hop",
                        )
                    )
                    children.append(
                        _Partial(
                            legs,
                            target,
                            arrival,
                            partial.mass - propellant - C.MINER_MASS_KG,
                            [*partial.deployed, (target, arrival)],
                            partial.hop_propellant + propellant,
                        )
                    )
        return children

    def run(self) -> SearchResult:
        started = time.perf_counter()
        s = self.settings
        beam = self._first_level()
        if not beam:
            return SearchResult(None, [], 0, self.lambert_evaluations, 0.0, [], 0, {}, 0)
        expansions = 0
        completed: list[RoutePlan] = []
        failures: list[dict[str, object]] = []
        best_by_depth: dict[int, float] = {}
        current = self._select(beam)
        depth = 1
        for partial in current:
            plan = self._complete(partial)
            if plan is not None:
                completed.append(plan)
                best_by_depth[1] = max(best_by_depth.get(1, 0.0), plan.total_collected_kg)
        for depth in range(2, s.max_deploys + 1):
            if time.perf_counter() - started > s.time_budget_seconds:
                failures.append({"reason": "time budget exhausted", "depth": depth - 1})
                depth -= 1
                break
            next_beam: list[_Partial] = []
            for partial in current:
                expansions += 1
                next_beam.extend(self._expand(partial))
            current = self._select(next_beam)
            if not current:
                depth -= 1
                break
            for partial in current:
                if len(partial.deployed) >= s.min_deploys:
                    plan = self._complete(partial)
                    if plan is not None:
                        completed.append(plan)
                        best_by_depth[depth] = max(
                            best_by_depth.get(depth, 0.0), plan.total_collected_kg
                        )
                    else:
                        failures.append(
                            {
                                "asteroids": [item for item, _ in partial.deployed],
                                "reason": f"no feasible collection tour ({self.last_failure})",
                                "mass_after_deploys_kg": partial.mass,
                                "deploy_end_epoch": partial.epoch,
                            }
                        )
        completed.sort(
            key=lambda item: (-self.weighted(item), item.propellant_proxy_kg, item.asteroids)
        )
        best = next((item for item in completed if item.feasible), None)
        return SearchResult(
            best,
            completed,
            expansions,
            self.lambert_evaluations,
            time.perf_counter() - started,
            failures,
            depth,
            best_by_depth,
            len(beam),
        )

    def _return_feasible(self, asteroid: int, mass_guess: float) -> bool:
        """Cached test that *some* Earth return from ``asteroid`` fits inside the final window."""

        if asteroid not in self._return_cache:
            end = C.MISSION_END_MJD - self.settings.end_margin_days
            self._return_cache[asteroid] = self._return_options(asteroid, end)
        return any(
            self._feasible(mass_guess, dv, tof, "earth_return")
            for dv, _departure, tof in self._return_cache[asteroid]
        )

    def _select(self, partials: list[_Partial]) -> list[_Partial]:
        """Stable top-``beam_width`` by heuristic, with diversity, reserve and return pruning.

        Score = expected mined mass minus a propellant penalty.  At most
        ``max_per_deployed_set`` variants of one deployed set survive; chains whose mass after the
        deploy phase cannot cover the dry mass plus the collect-phase reserve are dropped, as are
        chains whose first asteroid (the last one collected before the Earth return) has no
        feasible return.
        """

        end = C.MISSION_END_MJD - 2.0 * C.YEAR_DAYS  # rough collection horizon
        for partial in partials:
            mined = sum(
                self.weights.get(asteroid, 1.0)
                * C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
                for asteroid, deploy_epoch in partial.deployed
            )
            spent = self.settings.initial_mass - partial.mass
            elapsed = partial.epoch - partial.legs[0].departure_epoch
            partial.score = (
                mined
                - self.settings.propellant_weight * spent
                - self.settings.time_weight * elapsed
                + self._cluster_potential_kg(partial)
            )
        ordered = sorted(
            partials,
            key=lambda item: (-item.score, item.epoch, tuple(a for a, _ in item.deployed)),
        )
        selected: list[_Partial] = []
        per_set: dict[tuple[int, ...], int] = {}
        per_first: dict[int, int] = {}
        for partial in ordered:
            if len(selected) >= self.settings.beam_width:
                break
            if partial.mass < C.DRY_MASS_KG + self._reserve(partial):
                continue
            key = tuple(sorted(a for a, _ in partial.deployed))
            if per_set.get(key, 0) >= self.settings.max_per_deployed_set:
                continue
            first = partial.deployed[0][0]
            if per_first.get(first, 0) >= self.settings.max_per_first:
                continue
            # mass at the Earth-return departure: the collect tour has burnt the deploy-phase
            # surplus, so the ship is dry mass + cargo + the return propellant, not the
            # post-deploy mass plus cargo (that guess made every ratio-0.35 return look 0.7 and
            # pruned whole families whose returns SCvx flies without trouble)
            mined = sum(C.maximum_collected_mass(max(end - d, 0.0)) for _, d in partial.deployed)
            guess = min(
                partial.mass + mined, C.DRY_MASS_KG + mined + self.settings.return_reserve_kg
            )
            if not self._return_feasible(first, guess):
                continue
            per_set[key] = per_set.get(key, 0) + 1
            per_first[first] = per_first.get(first, 0) + 1
            selected.append(partial)
        return selected

    def _return_options(self, asteroid: int, end: float) -> list[tuple[float, float, float]]:
        """Candidate ``(dv, departure, tof)`` Earth returns arriving inside the final window."""

        s = self.settings
        arrivals = np.arange(end - s.return_window_days, end + 1e-9, s.schedule_step_days)
        tofs = np.asarray(s.earth_leg_tofs)
        a_idx, t_idx = np.meshgrid(
            np.arange(arrivals.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
        )
        a_idx, t_idx = a_idx.ravel(), t_idx.ravel()
        departures = arrivals[a_idx] - tofs[t_idx]
        r_s, v_s = asteroid_state(
            self.catalogue, np.full(departures.shape[0], asteroid), departures
        )
        r_e, v_e = earth_state(arrivals[a_idx])
        hop = lambert_hops(
            r_s,
            v_s,
            r_e,
            v_e,
            departures,
            tofs[t_idx],
            arrival_allowance_km_s=C.MAX_VINF_EARTH_KM_S,
        )
        self.lambert_evaluations += 2 * departures.shape[0]
        options = [
            (float(hop.total_delta_v[k]), float(departures[k]), float(tofs[t_idx[k]]))
            for k in range(departures.shape[0])
            if hop.feasible[k] and np.isfinite(hop.total_delta_v[k])
        ]
        options.sort(key=lambda item: (item[0], -item[1]))
        return options

    def _collect_hop_options(
        self, source: int, target: int, latest_arrival: float
    ) -> list[tuple[float, float, float]]:
        """Candidate ``(dv, departure, tof)`` hops ``source -> target`` arriving by the deadline.

        The ship may arrive early and camp at ``target`` until the scheduled collection, and it
        collects at ``source`` when it departs; later departures therefore mine more.
        """

        key = (source, target, round(latest_arrival, 6))
        if key in self._collect_cache:
            return self._collect_cache[key]
        s = self.settings
        tofs = np.asarray(s.collect_hop_tofs)
        waits = np.arange(0.0, s.collect_wait_window_days + 1e-9, s.schedule_step_days)
        w_idx, t_idx = np.meshgrid(
            np.arange(waits.shape[0]), np.arange(tofs.shape[0]), indexing="ij"
        )
        w_idx, t_idx = w_idx.ravel(), t_idx.ravel()
        arrivals = latest_arrival - waits[w_idx]
        departures = arrivals - tofs[t_idx]
        r_s, v_s = asteroid_state(self.catalogue, np.full(departures.shape[0], source), departures)
        r_t, v_t = asteroid_state(self.catalogue, np.full(departures.shape[0], target), arrivals)
        hop = lambert_hops(r_s, v_s, r_t, v_t, departures, tofs[t_idx])
        self.lambert_evaluations += 2 * departures.shape[0]
        options = [
            (float(hop.total_delta_v[k]), float(departures[k]), float(tofs[t_idx[k]]))
            for k in range(departures.shape[0])
            if hop.feasible[k] and np.isfinite(hop.total_delta_v[k])
        ]
        self._collect_cache[key] = options
        return options

    def _best_collect_hop(
        self,
        source: int,
        target: int,
        epoch: float,
        mass_guess: float,
        penalty_scale: float = 1.0,
        max_span_days: float = np.inf,
    ) -> tuple[float, tuple[float, float, float] | None]:
        s = self.settings
        best_hop = None
        best_cost = np.inf
        if (source, target) in self.banned_pairs:
            return best_cost, None
        for dv, departure, tof in self._collect_hop_options(source, target, epoch):
            if epoch - departure > max_span_days:
                continue  # hop + camp would not leave time for the remaining collections
            if not self._feasible(mass_guess, dv, tof, "collect_hop"):
                continue
            # propellant proxy plus the mining mass lost by collecting ``source`` earlier (the
            # whole hop duration counts: the miner at ``source`` stops when the ship leaves)
            lost = C.maximum_collected_mass(epoch - departure)
            cost = self._propellant(mass_guess, dv, self.hop_inflation_for(dv, mass_guess, tof))
            cost += s.wait_penalty * penalty_scale * lost
            if cost < best_cost - 1e-12 or (
                abs(cost - best_cost) <= 1e-12 and best_hop is not None and departure > best_hop[1]
            ):
                best_cost = cost
                best_hop = (dv, departure, tof)
        return best_cost, best_hop

    TOUR_MODES = ("greedy", "reverse", "forward", "forward_revisit")

    def _complete(self, partial: _Partial) -> RoutePlan | None:
        """Schedule the collection tour: best of the greedy-backward, reverse and forward orders.

        The forward tours (collect in deployment order after one repositioning hop back to the
        first asteroid) are what make the collect hops as cheap as the deploy hops: the deploy
        chain follows the family's phase drift, and traversing it backwards fights that drift
        (measured on family 0: reverse collect hops 2.3-3.3 km/s where the same pairs cost
        1.2-2.0 km/s on the way out).
        """

        plans: list[RoutePlan] = []
        reasons: list[str] = []
        for penalty_scale in (1.0, 4.0, 16.0):
            for mode in self.TOUR_MODES:
                plan = self._schedule(partial, mode, penalty_scale)
                if plan is not None:
                    plans.append(plan)
                else:
                    reasons.append(f"{mode}x{penalty_scale:g}:{self.last_failure}")
            if plans:
                break
        if not plans:
            self.last_failure = ",".join(reasons)
            return None
        plans.sort(key=lambda item: (-self.weighted(item), item.propellant_proxy_kg))
        return plans[0]

    def _schedule(
        self, partial: _Partial, mode: str | bool, penalty_scale: float = 1.0
    ) -> RoutePlan | None:
        """Schedule the collection tour and Earth return backwards from the window end.

        ``mode`` is one of :attr:`TOUR_MODES` (``True``/``False`` are accepted for the legacy
        greedy/reverse flags):

        * ``"greedy"`` - the first deployed asteroid is collected last (the return departs from
          it); the remaining order is chosen greedily backwards by proxy cost, with the camp
          asteroid (last deployed) forced to be the first collected because the ship is there;
        * ``"reverse"`` - strict reverse of deployment (camp first, first-deployed last);
        * ``"forward"`` - the camp asteroid is collected first (on departure, as usual), then the
          others in deployment order; the return departs from the second-to-last deployed one;
        * ``"forward_revisit"`` - the ship leaves the camp *without collecting*, collects in
          deployment order and returns from the camp asteroid after collecting it last on the
          revisit.  The repositioning hop is charged like a collect hop but triggers no collection.
        """

        if isinstance(mode, bool):
            mode = "greedy" if mode else "reverse"
        if mode not in self.TOUR_MODES:
            raise ValueError(f"unknown tour mode {mode!r}")
        s = self.settings
        deploy = dict(partial.deployed)
        deployed_order = [asteroid for asteroid, _ in partial.deployed]
        remaining = list(deployed_order)
        end = C.MISSION_END_MJD - s.end_margin_days
        camp_asteroid = partial.location
        # explicit collect order (first collected -> last collected) for the ordered modes
        order: list[int] | None
        if mode == "reverse":
            order = list(reversed(deployed_order))
        elif mode == "forward":
            order = [camp_asteroid, *deployed_order[:-1]]
        elif mode == "forward_revisit":
            order = list(deployed_order)
        else:
            order = None
        if order is not None and len(order) < 2 and mode != "reverse":
            self.last_failure = "forward_needs_two"
            return None
        # the last asteroid collected before returning to Earth
        first = order[-1] if order is not None else remaining[0]
        mass_guess = partial.mass + sum(
            C.maximum_collected_mass(max(end - deploy_epoch, 0.0))
            for _, deploy_epoch in partial.deployed
        )
        best_return = None
        return_inflation, _ = self.limits("earth_return")
        for dv, departure, tof in self._return_options(first, end):
            if self._feasible(mass_guess, dv, tof, "earth_return"):
                best_return = (dv, departure, tof)
                break
        if best_return is None:
            self.last_failure = "no_return"
            return None
        legs_backward: list[PlannedLeg] = [
            PlannedLeg(
                first,
                EARTH_ID,
                best_return[1],
                best_return[1] + best_return[2],
                best_return[0],
                return_inflation,
                "earth_return",
            )
        ]
        collect: dict[int, float] = {first: best_return[1]}
        epoch = best_return[1]  # collection (=departure) epoch at the current asteroid
        location = first
        remaining.remove(first)
        sequence = order[:-1] if order is not None else None  # still to place, collect order
        # forward_revisit: after the collections (built backwards) one more hop brings the ship
        # from its camp to the first-deployed asteroid without collecting anything
        reposition = mode == "forward_revisit"
        while remaining or reposition:
            if sequence is not None:
                # ordered modes: the asteroid collected just before ``location`` is the last of
                # the sequence still to place; then (revisit) the repositioning hop from the camp
                choices = [sequence[-1]] if sequence else [camp_asteroid]
            elif len(remaining) == 1:
                choices = list(remaining)
            else:
                choices = [a for a in remaining if a != camp_asteroid] or list(remaining)
            # time-aware: the remaining hops (each hop + camp) must fit between the end of the
            # deploy phase and the current collection epoch; the slack lets one hop run long
            # when others are short.  Without this the propellant-first choice picks 480-720
            # day hops and 8-10 asteroid chains die with ``camp_negative``.
            time_left = epoch - partial.epoch
            hops_left = len(remaining) + (1 if reposition else 0)
            max_span = s.collect_span_slack * time_left / hops_left
            best_choice = None
            for previous in choices:
                cost, hop = self._best_collect_hop(
                    previous, location, epoch, mass_guess, penalty_scale, max_span
                )
                if hop is not None and (best_choice is None or cost < best_choice[0] - 1e-12):
                    best_choice = (cost, previous, hop)
            if best_choice is None:
                self.last_failure = "no_collect_hop" if remaining else "no_reposition_hop"
                return None
            _cost, previous, best_hop = best_choice
            arrival = best_hop[1] + best_hop[2]
            if arrival < epoch:
                legs_backward.append(
                    PlannedLeg(location, location, arrival, epoch, 0.0, 1.0, "camp")
                )
            legs_backward.append(
                PlannedLeg(
                    previous,
                    location,
                    best_hop[1],
                    arrival,
                    best_hop[0],
                    self.hop_inflation_for(best_hop[0], mass_guess, best_hop[2]),
                    "collect_hop",
                )
            )
            epoch = best_hop[1]
            location = previous
            if remaining:
                collect[previous] = best_hop[1]
                remaining.remove(previous)
                if sequence is not None:
                    sequence.pop()
            else:
                reposition = False  # the camp -> first-deployed hop collects nothing
        if location != camp_asteroid:
            self.last_failure = "tour_not_ending_at_camp"
            return None
        # camp at the last deployed asteroid between its deploy and its collection
        camp_start = partial.epoch
        camp_end = epoch
        if camp_end - camp_start < 0.0:
            self.last_failure = "camp_negative"
            return None
        for asteroid in deploy:
            if collect[asteroid] - deploy[asteroid] < C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS:
                self.last_failure = "stay_too_short"
                return None
        legs = list(partial.legs)
        if camp_end > camp_start:
            legs.append(
                PlannedLeg(
                    partial.location, partial.location, camp_start, camp_end, 0.0, 1.0, "camp"
                )
            )
        legs.extend(reversed(legs_backward))
        # mass proxy forward through the collection tour (heavier ship after each collection)
        mass = partial.mass
        collected: dict[int, float] = {}
        propellant_total = s.initial_mass - partial.mass - C.MINER_MASS_KG * len(deploy)
        for leg in legs[len(partial.legs) :]:
            if leg.role == "camp":
                continue
            if leg.role == "collect_hop" or leg.role == "earth_return":
                # collection happens at departure of the leg (not on the forward tour's
                # repositioning hop, which leaves the camp asteroid for a later revisit)
                asteroid = leg.from_id
                if abs(collect[asteroid] - leg.departure_epoch) < 1e-6:
                    gained = C.maximum_collected_mass(collect[asteroid] - deploy[asteroid])
                    collected[asteroid] = gained
                    mass += gained
            if not self._feasible(mass, leg.delta_v_proxy_km_s, leg.tof_days, leg.role):
                self.last_failure = "leg_authority"
                return None
            inflation = leg.inflation
            if leg.role == "collect_hop":  # priced at the guessed mass above; use the actual one
                inflation = self.hop_inflation_for(leg.delta_v_proxy_km_s, mass, leg.tof_days)
            propellant = self._propellant(mass, leg.delta_v_proxy_km_s, inflation)
            propellant_total += propellant
            mass -= propellant
        if mass < C.DRY_MASS_KG + sum(collected.values()):
            self.last_failure = "mass_below_dry_plus_collected"
            return None
        return RoutePlan(tuple(legs), deploy, collect, collected, propellant_total, mass)
