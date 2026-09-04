"""Per-role leg cost distributions of GTOC12 solution files (ours and the references).

Every solution - ours or an archived JPL/Antipodes file - is decoded with the same itinerary
decoder (``references.decode_itineraries``), so Earth legs, deploy hops, collect hops and
returns are classified identically and the distributions are directly comparable.  This is
the report behind "Earth-leg propellant vs the references" and "hop cost before/after".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .data import AsteroidCatalogue
from .references import LegInfo, decode_itineraries
from .solution import Solution

ROLES = ("earth_out", "deploy_hop", "collect_hop", "earth_return")
HOP_BINS_KG = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 300.0, np.inf)
EARTH_BINS_KG = (0.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 700.0, np.inf)


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.shape[0]),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _histogram(values: list[float], edges: tuple[float, ...]) -> list[dict[str, float]]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=np.asarray(edges))
    return [
        {"lo": float(lo), "hi": float(hi), "count": int(n), "fraction": float(n / arr.shape[0])}
        for lo, hi, n in zip(edges[:-1], edges[1:], counts, strict=True)
    ]


@dataclass(slots=True)
class LegCostReport:
    name: str
    ships: int
    legs: dict[str, list[LegInfo]] = field(default_factory=dict)

    def propellant(self, role: str) -> list[float]:
        return [leg.propellant_kg for leg in self.legs.get(role, [])]

    def tof(self, role: str) -> list[float]:
        return [leg.tof_days for leg in self.legs.get(role, [])]

    def summary(self, *, cheap_hop_kg: float = 75.0) -> dict[str, Any]:
        hops = self.propellant("deploy_hop") + self.propellant("collect_hop")
        out: dict[str, Any] = {"name": self.name, "ships": self.ships, "roles": {}}
        for role in ROLES:
            prop = self.propellant(role)
            edges = EARTH_BINS_KG if role in ("earth_out", "earth_return") else HOP_BINS_KG
            out["roles"][role] = {
                "propellant_kg": _percentiles(prop),
                "tof_days": _percentiles(self.tof(role)),
                "histogram_kg": _histogram(prop, edges),
            }
        out["hops_at_or_under_cheap_kg"] = (
            float(np.mean(np.asarray(hops) <= cheap_hop_kg)) if hops else None
        )
        out["cheap_hop_kg"] = cheap_hop_kg
        out["per_ship_propellant_kg"] = {
            role: (sum(self.propellant(role)) / self.ships if self.ships else None)
            for role in ROLES
        }
        return out


def leg_costs(name: str, path: str | Path, catalogue: AsteroidCatalogue) -> LegCostReport:
    """Decode one solution file into per-role leg lists."""

    itineraries = decode_itineraries(Solution.read(path), catalogue)
    legs: dict[str, list[LegInfo]] = {role: [] for role in ROLES}
    for itinerary in itineraries:
        for leg in itinerary.legs:
            if leg.role in legs:
                legs[leg.role].append(leg)
    return LegCostReport(name, len(itineraries), legs)


def compare(
    solutions: dict[str, str | Path], catalogue: AsteroidCatalogue, *, cheap_hop_kg: float = 75.0
) -> dict[str, Any]:
    """Side-by-side per-role distributions for several named solution files."""

    reports = {name: leg_costs(name, path, catalogue) for name, path in solutions.items()}
    return {
        "cheap_hop_kg": cheap_hop_kg,
        "solutions": {
            name: report.summary(cheap_hop_kg=cheap_hop_kg) for name, report in reports.items()
        },
    }


def format_table(comparison: dict[str, Any]) -> str:
    """Compact text table: per role, median/mean propellant and median TOF per solution."""

    lines = []
    names = list(comparison["solutions"])
    header = f"{'role':13s}{'stat':22s}" + "".join(f"{n[:18]:>20s}" for n in names)
    lines.append(header)
    for role in ROLES:
        for stat, key, unit in (
            ("propellant median", "median", "kg"),
            ("propellant mean", "mean", "kg"),
            ("propellant p90", "p90", "kg"),
            ("tof median", "median", "d"),
        ):
            row = f"{role:13s}{stat + ' [' + unit + ']':22s}"
            for name in names:
                block = comparison["solutions"][name]["roles"][role]
                source = block["propellant_kg"] if unit == "kg" else block["tof_days"]
                value = source.get(key)
                row += f"{'-' if value is None else f'{value:.1f}':>20s}"
            lines.append(row)
    row = f"{'hops':13s}{'<= cheap fraction':22s}"
    for name in names:
        value = comparison["solutions"][name]["hops_at_or_under_cheap_kg"]
        row += f"{'-' if value is None else f'{value:.2f}':>20s}"
    lines.append(row)
    row = f"{'per ship':13s}{'earth_out [kg]':22s}"
    for name in names:
        value = comparison["solutions"][name]["per_ship_propellant_kg"]["earth_out"]
        row += f"{'-' if value is None else f'{value:.1f}':>20s}"
    lines.append(row)
    row = f"{'per ship':13s}{'hops [kg]':22s}"
    for name in names:
        per = comparison["solutions"][name]["per_ship_propellant_kg"]
        value = None
        if per["deploy_hop"] is not None and per["collect_hop"] is not None:
            value = per["deploy_hop"] + per["collect_hop"]
        row += f"{'-' if value is None else f'{value:.1f}':>20s}"
    lines.append(row)
    return "\n".join(lines)
