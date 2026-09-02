"""TOPS (Trajectory Optimization Problem Space, ESA zero-order-hold release) ingestion.

The pinned revision ``24fe8849`` of https://gitlab.com/EuropeanSpaceAgency/zero-order-hold ships
four JSON databases (``zoh/dbs/_tops_{twobody,mee,cr3bp,ss}.json``).  Every problem is ingested
with its metadata; the campaign selection (one easy two-body, one multi-revolution two-body, one
inclination/eccentricity change, one CR3BP) is made from metadata alone and frozen in
``benchmarks/literature/tops_selection.json``.  Only the Cartesian two-body family is supported
by the current dynamics; MEE, CR3BP, and solar-sail problems are recorded as unsupported.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from spacepdhcg.literature import external_sources
from spacepdhcg.literature.low_thrust import LowThrustProblem, solve_low_thrust

FAMILY_FILES = {
    "two_body_cartesian": "tops.twobody",
    "modified_equinoctial": "tops.mee",
    "cr3bp": "tops.cr3bp",
    "solar_sail": "tops.solar_sail",
}

SUPPORT = {
    "two_body_cartesian": ("supported", None),
    "modified_equinoctial": (
        "unsupported",
        "modified-equinoctial-element dynamics are not implemented; boundary states would need "
        "MEE->Cartesian conversion and the reference solutions are given in MEE decision vectors",
    ),
    "cr3bp": ("unsupported", "circular restricted three-body dynamics are not implemented"),
    "solar_sail": ("unsupported", "solar-sail control model is not implemented"),
}


@dataclass(frozen=True, slots=True)
class TopsProblem:
    key: str
    family: str
    info: str
    fixed_time: bool
    tof_bounds: tuple[float, float]
    max_thrust: float | None
    exhaust_velocity: float | None
    initial_mass: float | None
    mu: float | None
    period_ratio: float | None
    revolutions_estimate: float | None
    raw: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw")
        return payload


def ingest() -> list[TopsProblem]:
    problems: list[TopsProblem] = []
    for family, artifact in FAMILY_FILES.items():
        path = external_sources.fetch(artifact)
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, raw in data.items():
            tof = tuple(float(v) for v in raw["tof_bounds"])
            period_s = raw.get("period_s")
            period_f = raw.get("period_f")
            ratio = float(period_f) / float(period_s) if period_s and period_f else None
            revolutions = None
            if period_s and period_f:
                mean_period = 0.5 * (float(period_s) + float(period_f))
                revolutions = tof[1] / mean_period
            problems.append(
                TopsProblem(
                    key=f"{family}:{key}",
                    family=family,
                    info=str(raw.get("info", "")),
                    fixed_time=math.isclose(tof[0], tof[1]),
                    tof_bounds=tof,
                    max_thrust=raw.get("max_thrust"),
                    exhaust_velocity=raw.get("veff"),
                    initial_mass=raw.get("m_s"),
                    mu=raw.get("mu", raw.get("mu_cr3bp")),
                    period_ratio=ratio,
                    revolutions_estimate=revolutions,
                    raw=raw,
                )
            )
    return problems


def select_by_metadata(problems: list[TopsProblem]) -> dict[str, Any]:
    """Deterministic metadata rules (documented in the returned record)."""

    twobody = [p for p in problems if p.family == "two_body_cartesian"]
    cr3bp = [p for p in problems if p.family == "cr3bp"]
    # Easy: fixed-time two-body with the fewest estimated revolutions.
    easy = min(
        (p for p in twobody if p.fixed_time), key=lambda p: (p.revolutions_estimate or 0, p.key)
    )
    # Multi-revolution: two-body whose metadata text declares a multi-rev arc.
    multirev = min((p for p in twobody if "multirev" in p.info.lower()), key=lambda p: p.key)
    # Inclination/eccentricity change: two-body whose info mentions inclination or highly-elliptic.
    inclination = min(
        (
            p
            for p in twobody
            if "inclination" in p.info.lower() or "highly-elliptic" in p.info.lower()
        ),
        key=lambda p: p.key,
    )
    # CR3BP: fixed-time case with the fewest estimated revolutions.
    cr3 = min(
        (p for p in cr3bp if p.fixed_time), key=lambda p: (p.revolutions_estimate or 0, p.key)
    )
    return {
        "rules": {
            "easy_two_body": (
                "fixed-time Cartesian two-body problem with the smallest tof/mean-period ratio"
            ),
            "multi_revolution_two_body": (
                "Cartesian two-body problem whose info string declares a multirev arc (lowest key)"
            ),
            "inclination_or_eccentricity_change": (
                "Cartesian two-body problem whose info string mentions inclination or "
                "highly-elliptic (lowest key)"
            ),
            "cr3bp": "fixed-time CR3BP problem with the smallest tof/mean-period ratio",
        },
        "selected": {
            "easy_two_body": easy.key,
            "multi_revolution_two_body": multirev.key,
            "inclination_or_eccentricity_change": inclination.key,
            "cr3bp": cr3.key,
        },
    }


def to_low_thrust_problem(problem: TopsProblem) -> LowThrustProblem:
    if problem.family != "two_body_cartesian":
        raise ValueError(f"{problem.key} is not a Cartesian two-body problem")
    raw = problem.raw
    return LowThrustProblem(
        initial_state=tuple(float(v) for v in raw["state_s"]),
        final_state=tuple(float(v) for v in raw["state_f"]),
        initial_mass=float(raw["m_s"]),
        max_thrust=float(raw["max_thrust"]),
        exhaust_velocity=float(raw["veff"]),
        tof_bounds=problem.tof_bounds,
        mu=float(raw["mu"]),
        label=problem.key,
    )


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    try:
        problems = ingest()
    except external_sources.ArtifactUnavailable as error:
        return {
            "target_id": document["id"],
            "status": "blocked",
            "published": {},
            "measured": {},
            "gap": {},
            "labels": {},
            "envelope": {},
            "commands": [f"spacepdhcg literature run {document['id']}"],
            "notes": [f"blocked: {error}"],
        }
    by_key = {p.key: p for p in problems}
    frozen = document["frozen_selection"]
    nodes = int(options.get("nodes", document.get("nodes", 120)))
    max_iterations = int(options.get("max_iterations", document.get("max_iterations", 40)))
    runs: dict[str, Any] = {}
    statuses: dict[str, str] = {}
    for role, key in frozen.items():
        problem = by_key[key]
        support, reason = SUPPORT[problem.family]
        if support != "supported":
            runs[role] = {"problem": key, "status": "unsupported", "reason": reason}
            statuses[role] = "unsupported"
            continue
        lt_problem = to_low_thrust_problem(problem)
        result = solve_low_thrust(
            lt_problem,
            nodes=nodes,
            max_iterations=max_iterations,
            revolutions=round(problem.revolutions_estimate)
            if problem.revolutions_estimate and problem.revolutions_estimate > 1.5
            else None,
            hard_trust_radius=0.5,
        )
        runs[role] = {"problem": key, **result.as_dict()}
        statuses[role] = result.outcome.status
    supported_ok = all(
        statuses[role] == "converged"
        for role in statuses
        if runs[role].get("status") != "unsupported"
    )
    return {
        "target_id": document["id"],
        "status": "reproduced" if supported_ok else "gap",
        "published": {
            "reference_objectives": (
                "none published for the Cartesian two-body problems at the pinned revision"
            )
        },
        "measured": {
            role: {
                "problem": run["problem"],
                "status": run["status"],
                "final_mass": run.get("final_mass"),
                "time_of_flight": run.get("time_of_flight"),
                "reason": run.get("reason"),
            }
            for role, run in runs.items()
        },
        "gap": {},
        "labels": {f"measured.{role}": "measured-local" for role in runs},
        "envelope": {"nodes": nodes, "problem_count_at_pinned_revision": len(problems)},
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            "the ISSFD paper describes 28 problems; the pinned repository revision contains "
            f"{len(problems)} (database is actively expanding); selection made from metadata only",
        ],
        "details": {"runs": runs, "problems": [p.as_dict() for p in problems]},
    }
