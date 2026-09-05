"""Reference-chain prior: what a good deploy chain looks like, measured on the archived solutions.

The eighth iteration showed that our collect hops (85 kg / 210 d median) are dearer than the
references' (66 kg / 181 d) because of the *deploy chain*, not the tour: the references pay
~70 kg more per ship on deploy hops to put consecutive miners on cheap harvest pairs.  Neither a
per-pair deploy-time surcharge nor a post-beam substitution moves that, so the ninth iteration
scores partial chains in the beam by their actual collect tour (``search.RouteSearch._select``
with ``chain_tour_scoring``) and steers the expansion order with a *prior* extracted here.

:func:`extract_chain_prior` decodes the archived JPL/Antipodes files with the shared itinerary
decoder (``references.decode_itineraries``) and records, per ship, the propellant split by role
(Earth-out, deploy hops, collect hops, Earth return), the per-hop propellant and TOF, and the
geometry of every hop - the pair's mean-longitude difference at departure (phase alignment)
and its semi-major-axis gap - plus the compactness of the deployed loop.  The quantiles are
stored as data (``benchmarks/gtoc12/chain_prior_v1.json``) with the source files' hashes so the
extraction is reproducible; :class:`ChainPrior` reads the targets back and prices a partial
chain's deviation from them (:meth:`ChainPrior.penalty`), which the beam subtracts from the chain
score.  Nothing in the penalty is a hand-set threshold: every cap and floor is a quantile of
the reference distribution named in the JSON, and the only free parameter is the weight the
beam applies to the excess kilograms (``SearchSettings.chain_prior_weight``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import constants as C
from .clusters import mean_longitude
from .data import AsteroidCatalogue
from .references import LegInfo, ShipItinerary, decode_itineraries
from .solution import Solution

__all__ = [
    "ChainPrior",
    "extract_chain_prior",
    "load_chain_prior",
    "reference_solution_files",
]

SCHEMA_VERSION = "1.0.0"
QUANTILES = ("p10", "p25", "median", "p75", "p90")
# the archived reference solutions (pinned in benchmarks/gtoc12/pins.json; not redistributed)
REFERENCE_FILES = (
    "GTOC12_JPL_merged_solution_36sc.txt",
    "37_mass_optimal_self_cleaning.txt",
    "39_mass_optimal.txt",
)


def reference_solution_files(data_dir: Path) -> list[Path]:
    """The archived reference files present under ``data_dir`` (may be empty)."""

    return [data_dir / name for name in REFERENCE_FILES if (data_dir / name).is_file()]


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    out: dict[str, float] = {"n": int(arr.shape[0]), "mean": float(arr.mean())}
    for name, q in zip(QUANTILES, (10, 25, 50, 75, 90), strict=True):
        out[name] = float(np.percentile(arr, q))
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hop_geometry(catalogue: AsteroidCatalogue, leg: LegInfo) -> tuple[float, float, float] | None:
    """``(|Δλ| at departure in deg, |Δλ| at arrival in deg, Δa in AU)`` of an asteroid pair."""

    if leg.from_body <= 0 or leg.to_body <= 0:
        return None
    index = catalogue.index_of(np.asarray([leg.from_body, leg.to_body], dtype=np.int64))
    delta_a = float(
        (catalogue.semi_major_axis_km[index[1]] - catalogue.semi_major_axis_km[index[0]]) / C.AU_KM
    )
    out: list[float] = []
    for epoch in (leg.departure_epoch, leg.arrival_epoch):
        lon = mean_longitude(catalogue, index, float(epoch))
        delta = float((lon[1] - lon[0] + math.pi) % (2.0 * math.pi) - math.pi)
        out.append(abs(math.degrees(delta)))
    return out[0], out[1], delta_a


def _ship_record(catalogue: AsteroidCatalogue, ship: ShipItinerary) -> dict[str, Any] | None:
    """Per-ship split and geometry; ``None`` for ships that do not deploy and collect."""

    by_role: dict[str, list[LegInfo]] = {}
    for leg in ship.legs:
        by_role.setdefault(leg.role, []).append(leg)
    deploy_hops = by_role.get("deploy_hop", [])
    collect_hops = by_role.get("collect_hop", [])
    if not ship.deploys or not collect_hops:
        return None
    total = {role: sum(leg.propellant_kg for leg in legs) for role, legs in by_role.items()}
    deploy_kg = total.get("deploy_hop", 0.0)
    collect_kg = total.get("collect_hop", 0.0)
    deploy_geometry = [g for g in (_hop_geometry(catalogue, leg) for leg in deploy_hops) if g]
    collect_geometry = [g for g in (_hop_geometry(catalogue, leg) for leg in collect_hops) if g]
    # loop compactness: the spread of the deployed asteroids' semi-major axes and the largest
    # mean-longitude gap between consecutive collect stops when the tour flies them
    deployed = np.asarray([a for a, _ in ship.deploys], dtype=np.int64)
    index = catalogue.index_of(deployed)
    a_au = catalogue.semi_major_axis_km[index] / C.AU_KM
    return {
        "ship_id": ship.ship_id,
        "asteroids": len(ship.deploys),
        "collected_kg": float(ship.collected_mass_kg),
        "earth_out_kg": float(total.get("earth_out", 0.0)),
        "deploy_hops_kg": float(deploy_kg),
        "collect_hops_kg": float(collect_kg),
        "earth_return_kg": float(total.get("earth_return", 0.0)),
        "collect_share": float(collect_kg / (deploy_kg + collect_kg))
        if deploy_kg + collect_kg > 0.0
        else None,
        "deploy_hop_kg": [float(leg.propellant_kg) for leg in deploy_hops],
        "collect_hop_kg": [float(leg.propellant_kg) for leg in collect_hops],
        "deploy_hop_tof_days": [float(leg.tof_days) for leg in deploy_hops],
        "collect_hop_tof_days": [float(leg.tof_days) for leg in collect_hops],
        "deploy_phase_deg": [g[0] for g in deploy_geometry],
        "collect_phase_deg": [g[0] for g in collect_geometry],
        "collect_phase_arrival_deg": [g[1] for g in collect_geometry],
        "collect_delta_a_au": [abs(g[2]) for g in collect_geometry],
        "loop_a_spread_au": float(a_au.max() - a_au.min()) if a_au.size else 0.0,
        "loop_max_phase_deg": max((g[0] for g in collect_geometry), default=0.0),
    }


def extract_chain_prior(
    catalogue: AsteroidCatalogue,
    solution_paths: list[Path] | tuple[Path, ...],
    *,
    commit: str = "",
) -> dict[str, Any]:
    """Decode the reference solutions and return the prior document (JSON-serialisable).

    ``targets`` holds the quantities the beam uses; ``distributions`` the full quantile tables
    they come from; ``ships`` the per-ship records (for the report); ``sources`` the files'
    SHA-256 so a re-extraction on the same pins reproduces the document bit for bit.
    """

    ships: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in solution_paths:
        path = Path(path)
        itineraries = decode_itineraries(Solution.read(path), catalogue)
        count = 0
        for itinerary in itineraries:
            record = _ship_record(catalogue, itinerary)
            if record is not None:
                record["source"] = path.name
                ships.append(record)
                count += 1
        sources.append({"file": path.name, "sha256": _sha256(path), "ships": count})
    if not ships:
        raise ValueError("no reference ship deploys and collects: nothing to extract")

    def gather(key: str) -> list[float]:
        values: list[float] = []
        for ship in ships:
            item = ship[key]
            if isinstance(item, list):
                values.extend(float(v) for v in item)
            elif item is not None:
                values.append(float(item))
        return values

    distributions = {
        key: _quantiles(gather(key))
        for key in (
            "asteroids",
            "collected_kg",
            "earth_out_kg",
            "deploy_hops_kg",
            "collect_hops_kg",
            "earth_return_kg",
            "collect_share",
            "deploy_hop_kg",
            "collect_hop_kg",
            "deploy_hop_tof_days",
            "collect_hop_tof_days",
            "deploy_phase_deg",
            "collect_phase_deg",
            "collect_phase_arrival_deg",
            "collect_delta_a_au",
            "loop_a_spread_au",
            "loop_max_phase_deg",
        )
    }
    targets = {
        # per-hop caps: a chain whose projected collect hops run above the reference p75 per
        # hop is off the reference manifold; its deploy hops below the reference p25 with a dear
        # collect tour is the "cheap deploy, dear harvest" signature the eighth iteration found
        "collect_hop_kg_p75": distributions["collect_hop_kg"]["p75"],
        "collect_hop_kg_median": distributions["collect_hop_kg"]["median"],
        "deploy_hop_kg_p25": distributions["deploy_hop_kg"]["p25"],
        "deploy_hop_kg_median": distributions["deploy_hop_kg"]["median"],
        # per-ship totals (report and sanity checks; the penalty works per hop)
        "collect_per_ship_kg_p90": distributions["collect_hops_kg"]["p90"],
        "deploy_per_ship_kg_p10": distributions["deploy_hops_kg"]["p10"],
        "collect_share_p90": distributions["collect_share"]["p90"],
        "asteroids_per_ship_median": distributions["asteroids"]["median"],
        "collect_hop_tof_days_median": distributions["collect_hop_tof_days"]["median"],
        "collect_phase_deg_p75": distributions["collect_phase_deg"]["p75"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "sources": sources,
        "ships_decoded": len(ships),
        "targets": targets,
        "distributions": distributions,
        "ships": ships,
    }


@dataclass(frozen=True, slots=True)
class ChainPrior:
    """Reference targets a partial chain is priced against (all kg per hop unless noted)."""

    collect_hop_kg_p75: float
    collect_hop_kg_median: float
    deploy_hop_kg_p25: float
    deploy_hop_kg_median: float
    collect_share_p90: float
    source: str = ""

    @classmethod
    def from_document(cls, document: dict[str, Any], *, source: str = "") -> ChainPrior:
        targets = document["targets"]
        return cls(
            float(targets["collect_hop_kg_p75"]),
            float(targets["collect_hop_kg_median"]),
            float(targets["deploy_hop_kg_p25"]),
            float(targets["deploy_hop_kg_median"]),
            float(targets["collect_share_p90"]),
            source,
        )

    def penalty(
        self, *, deploy_hops_kg: float, deploy_hops: int, collect_hops_kg: float, collect_hops: int
    ) -> float:
        """Kilograms a partial chain lies off the reference manifold (0 on it).

        Two terms, both per hop so partial chains of any depth are comparable:

        * *dear harvest*: the projected collect propellant above ``collect_hops x p75`` of the
          reference collect hop;
        * *cheap deploy, dear harvest*: when the collect hops run above the reference median,
          the deploy propellant *below* ``deploy_hops x p25`` of the reference deploy hop - the
          references buy their cheap harvest with dearer deploy hops, and a chain that has not
          paid that is charged the shortfall.

        Monotone: non-decreasing in ``collect_hops_kg``, non-increasing in ``deploy_hops_kg``.
        """

        if collect_hops <= 0:
            return 0.0
        excess = max(0.0, collect_hops_kg - collect_hops * self.collect_hop_kg_p75)
        shortfall = 0.0
        if deploy_hops > 0 and collect_hops_kg > collect_hops * self.collect_hop_kg_median:
            shortfall = max(0.0, deploy_hops * self.deploy_hop_kg_p25 - deploy_hops_kg)
        return excess + shortfall

    def summary(self) -> dict[str, Any]:
        return {
            "collect_hop_kg_p75": self.collect_hop_kg_p75,
            "collect_hop_kg_median": self.collect_hop_kg_median,
            "deploy_hop_kg_p25": self.deploy_hop_kg_p25,
            "deploy_hop_kg_median": self.deploy_hop_kg_median,
            "collect_share_p90": self.collect_share_p90,
            "source": self.source,
        }


def load_chain_prior(path: str | Path) -> ChainPrior:
    """Read a prior document written by :func:`extract_chain_prior`."""

    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version", "").split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"unsupported chain prior schema in {path}")
    return ChainPrior.from_document(document, source=str(path))
