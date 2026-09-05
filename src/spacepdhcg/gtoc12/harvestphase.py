"""Harvest-phase prior: the phase alignment of consecutive collect stops, measured on the
archived references, and what a misaligned harvest hop costs the beam and the collect DP.

The ninth iteration closed the deploy spend and the Earth return against the references and
left one geometric difference: the references' consecutive miners sit within |Δλ| 2.7 deg
(p75 4.8 deg) of each other at the collect *departure* (``chain_prior_v1.json``), ours do not,
and the collect DP then flies 210-day hops where the references fly 180-day ones - a phase our
chains present at harvest that nothing in the beam scores (the chain prior prices propellant,
not phase; the DP's calibrated pair table prices every cell but does not prefer the aligned
ones when the surrogate propellant is comparable).

:func:`extract_harvest_phase` decodes the reference solutions with the shared itinerary
decoder and records, per collect hop, ``|Δλ|`` of the pair at the departure epoch (the mean
longitude difference, wrapped), the hop's propellant, its TOF and the pair's semi-major-axis
gap.  The document (``benchmarks/gtoc12/harvest_phase_v1.json``) stores the ``|Δλ|`` quantiles
and histogram, the per-bin median propellant and TOF, and two least-squares slopes over the
reference hops - kilograms per degree and days per degree of misalignment - with the source
files' SHA-256 so the extraction is reproducible bit for bit.  :class:`HarvestPhasePrior` reads
the targets back and prices a hop's misalignment in kilograms (:meth:`HarvestPhasePrior.
penalty_kg`): zero at or below the reference p75, then ``(kg/deg + days/deg x the fleet's
exchange rate) x (|Δλ| - p75)`` - the propellant the references pay per degree plus the
mining time they lose per degree, the latter converted at ``asteroids per ship x 10 kg/yr``.
Every number in it is a named quantity of the JSON; the beam's and the DP's weights are the
only free parameters.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .clusters import mean_longitude
from .data import AsteroidCatalogue
from .references import LegInfo, ShipItinerary, decode_itineraries
from .solution import Solution

__all__ = [
    "HarvestPhasePrior",
    "extract_harvest_phase",
    "load_harvest_phase",
    "phase_deg_at",
]

SCHEMA_VERSION = "1.0.0"
QUANTILES = ("p10", "p25", "median", "p75", "p90")
# |Δλ| histogram edges (deg); the last bin is open-ended to 180
PHASE_BINS_DEG = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 60.0, 180.0)
# the slopes are fitted on harvest hops: inside this misalignment and TOF (the far tail is a
# handful of repositioning hops of 900-2000 days between loops, not harvest hops; the p90 TOF
# of the reference collect hops is 904 d because of them)
FIT_MAX_PHASE_DEG = 30.0
FIT_MAX_TOF_DAYS = 400.0
FloatArray = NDArray[np.float64]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: FloatArray) -> dict[str, float]:
    if values.shape[0] == 0:
        return {"n": 0}
    out: dict[str, float] = {"n": int(values.shape[0]), "mean": float(values.mean())}
    for name, q in zip(QUANTILES, (10, 25, 50, 75, 90), strict=True):
        out[name] = float(np.percentile(values, q))
    return out


def phase_deg_at(catalogue: AsteroidCatalogue, source: int, target: int, epoch: float) -> float:
    """``|Δλ|`` (deg, in [0, 180]) of the pair's mean longitudes at ``epoch``."""

    index = catalogue.index_of(np.asarray([source, target], dtype=np.int64))
    lon = mean_longitude(catalogue, index, float(epoch))
    delta = float((lon[1] - lon[0] + math.pi) % (2.0 * math.pi) - math.pi)
    return abs(math.degrees(delta))


def _hop_record(catalogue: AsteroidCatalogue, ship: ShipItinerary, leg: LegInfo) -> dict | None:
    if leg.from_body <= 0 or leg.to_body <= 0:
        return None
    index = catalogue.index_of(np.asarray([leg.from_body, leg.to_body], dtype=np.int64))
    delta_a = float(
        (catalogue.semi_major_axis_km[index[1]] - catalogue.semi_major_axis_km[index[0]]) / C.AU_KM
    )
    return {
        "ship_id": ship.ship_id,
        "from": int(leg.from_body),
        "to": int(leg.to_body),
        "departure_epoch": float(leg.departure_epoch),
        "phase_deg": phase_deg_at(catalogue, leg.from_body, leg.to_body, leg.departure_epoch),
        "propellant_kg": float(leg.propellant_kg),
        "tof_days": float(leg.tof_days),
        "delta_a_au": abs(delta_a),
    }


def _slope(x: FloatArray, y: FloatArray) -> dict[str, float]:
    """Least-squares ``y = intercept + slope x`` (``n`` points); NaN slope below two points."""

    if x.shape[0] < 2 or float(np.ptp(x)) <= 0.0:
        return {"n": int(x.shape[0]), "slope": math.nan, "intercept": math.nan}
    design = np.column_stack([np.ones_like(x), x])
    coef, *_rest = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    return {
        "n": int(x.shape[0]),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
    }


def extract_harvest_phase(
    catalogue: AsteroidCatalogue,
    solution_paths: list[Path] | tuple[Path, ...],
    *,
    commit: str = "",
) -> dict[str, Any]:
    """Decode the reference solutions and return the harvest-phase document (JSON-serialisable).

    ``targets`` holds what the beam and the DP read back; ``distributions`` the quantiles,
    histogram and per-bin medians they come from; ``fits`` the slopes; ``hops`` the per-hop
    records (for the report); ``sources`` the files' SHA-256.
    """

    hops: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    asteroids_per_ship: list[int] = []
    for path in solution_paths:
        path = Path(path)
        count = 0
        for itinerary in decode_itineraries(Solution.read(path), catalogue):
            collect = [leg for leg in itinerary.legs if leg.role == "collect_hop"]
            if not itinerary.deploys or not collect:
                continue
            asteroids_per_ship.append(len(itinerary.deploys))
            for leg in collect:
                record = _hop_record(catalogue, itinerary, leg)
                if record is not None:
                    record["source"] = path.name
                    hops.append(record)
                    count += 1
        sources.append({"file": path.name, "sha256": _sha256(path), "hops": count})
    if not hops:
        raise ValueError("no reference collect hop decoded: nothing to extract")
    phase = np.asarray([h["phase_deg"] for h in hops], dtype=np.float64)
    propellant = np.asarray([h["propellant_kg"] for h in hops], dtype=np.float64)
    tof = np.asarray([h["tof_days"] for h in hops], dtype=np.float64)
    edges = np.asarray(PHASE_BINS_DEG, dtype=np.float64)
    which = np.clip(np.searchsorted(edges, phase, side="right") - 1, 0, edges.shape[0] - 2)
    bins = []
    for b in range(edges.shape[0] - 1):
        mask = which == b
        bins.append(
            {
                "lo_deg": float(edges[b]),
                "hi_deg": float(edges[b + 1]),
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "propellant_kg_median": float(np.median(propellant[mask])) if mask.any() else None,
                "tof_days_median": float(np.median(tof[mask])) if mask.any() else None,
            }
        )
    fit_mask = (phase <= FIT_MAX_PHASE_DEG) & (tof <= FIT_MAX_TOF_DAYS)
    fits = {
        "max_phase_deg": FIT_MAX_PHASE_DEG,
        "max_tof_days": FIT_MAX_TOF_DAYS,
        "hops_fitted": int(fit_mask.sum()),
        "propellant_kg_per_deg": _slope(phase[fit_mask], propellant[fit_mask]),
        "tof_days_per_deg": _slope(phase[fit_mask], tof[fit_mask]),
    }
    per_ship = float(np.median(np.asarray(asteroids_per_ship, dtype=np.float64)))
    exchange = per_ship * C.MINING_RATE_KG_PER_YEAR / C.YEAR_DAYS
    distributions = {
        "phase_deg": _quantiles(phase),
        "propellant_kg": _quantiles(propellant),
        "tof_days": _quantiles(tof),
        "delta_a_au": _quantiles(np.asarray([h["delta_a_au"] for h in hops])),
        "histogram": bins,
    }
    targets = {
        "phase_deg_median": distributions["phase_deg"]["median"],
        "phase_deg_p75": distributions["phase_deg"]["p75"],
        "phase_deg_p90": distributions["phase_deg"]["p90"],
        # kilograms of collect-hop propellant the references pay per degree of misalignment
        # and the days of hop TOF they spend per degree (both floored at zero: a prior never
        # rewards misalignment)
        "kg_per_deg": max(0.0, fits["propellant_kg_per_deg"]["slope"]),
        "days_per_deg": max(0.0, fits["tof_days_per_deg"]["slope"]),
        # a day of hop TOF delays every later collect: the fleet's exchange rate, asteroids
        # per reference ship x the mining rate
        "asteroids_per_ship_median": per_ship,
        "exchange_kg_per_day": exchange,
        "collect_hop_tof_days_median": distributions["tof_days"]["median"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "sources": sources,
        "hops_decoded": len(hops),
        "targets": targets,
        "distributions": distributions,
        "fits": fits,
        "hops": hops,
    }


@dataclass(frozen=True, slots=True)
class HarvestPhasePrior:
    """Reference phase targets a harvest hop is priced against."""

    p75_deg: float
    median_deg: float
    kg_per_deg: float
    days_per_deg: float
    exchange_kg_per_day: float
    source: str = ""

    @classmethod
    def from_document(cls, document: dict[str, Any], *, source: str = "") -> HarvestPhasePrior:
        targets = document["targets"]
        return cls(
            float(targets["phase_deg_p75"]),
            float(targets["phase_deg_median"]),
            float(targets["kg_per_deg"]),
            float(targets["days_per_deg"]),
            float(targets["exchange_kg_per_day"]),
            source,
        )

    @property
    def kg_per_deg_total(self) -> float:
        """Propellant plus mining time per degree of misalignment, in kg."""

        return self.kg_per_deg + self.days_per_deg * self.exchange_kg_per_day

    def penalty_kg(self, phase_deg: FloatArray | float) -> FloatArray | float:
        """Kilograms a harvest hop lies off the reference phase manifold: zero at or below
        the reference p75 of ``|Δλ|``, then :attr:`kg_per_deg_total` per degree above it.
        Non-decreasing in ``|Δλ|``; vectorised."""

        phase = np.abs(np.asarray(phase_deg, dtype=np.float64))
        excess = np.maximum(phase - self.p75_deg, 0.0)
        return self.kg_per_deg_total * excess

    def summary(self) -> dict[str, Any]:
        return {
            "p75_deg": self.p75_deg,
            "median_deg": self.median_deg,
            "kg_per_deg": self.kg_per_deg,
            "days_per_deg": self.days_per_deg,
            "exchange_kg_per_day": self.exchange_kg_per_day,
            "kg_per_deg_total": self.kg_per_deg_total,
            "source": self.source,
        }


def load_harvest_phase(path: str | Path) -> HarvestPhasePrior:
    """Read a document written by :func:`extract_harvest_phase`."""

    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version", "").split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(f"unsupported harvest phase schema in {path}")
    return HarvestPhasePrior.from_document(document, source=str(path))
