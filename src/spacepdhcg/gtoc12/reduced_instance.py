"""Deterministic preregistered reduced GTOC12 instance.

The rule in ``benchmarks/gtoc12/reduced_instance_v1.json`` was committed before any route search
ran.  Asteroids are chosen purely from catalogue metadata (an element-range eligibility filter
followed by a salted SHA-256 rank), so the selection is reproducible from the rule file and the
pinned catalogue alone.  Official dynamics, ship rules and verifier semantics are untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg import resources

from .constants import AU_KM
from .data import AsteroidCatalogue

DEFAULT_RULE_ASSET = "benchmarks/gtoc12/reduced_instance_v1.json"


def default_rule_path() -> Path:
    """Location of the preregistered rule (override, checkout, or wheel copy)."""

    return resources.asset_path(DEFAULT_RULE_ASSET)


def canonical_sha256(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ReducedInstance:
    instance_id: str
    rule: dict[str, Any]
    rule_sha256: str
    asteroid_ids: NDArray[np.int64]
    eligible_count: int
    ships: int
    start_mjd: float
    end_mjd: float

    @property
    def selection_sha256(self) -> str:
        return hashlib.sha256(",".join(map(str, self.asteroid_ids.tolist())).encode()).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "rule_sha256": self.rule_sha256,
            "eligible_count": self.eligible_count,
            "selected_count": int(self.asteroid_ids.shape[0]),
            "selection_sha256": self.selection_sha256,
            "first_ids": self.asteroid_ids[:10].tolist(),
            "ships": self.ships,
            "window_mjd": [self.start_mjd, self.end_mjd],
        }


def load_rule(path: Path | None = None) -> tuple[dict[str, Any], str]:
    rule_path = path or default_rule_path()
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    return payload, canonical_sha256(payload)


def rank_keys(instance_id: str, asteroid_ids: NDArray[np.int64]) -> list[int]:
    return [
        int.from_bytes(hashlib.sha256(f"{instance_id}:{int(item)}".encode()).digest(), "big")
        for item in asteroid_ids
    ]


def build_reduced_instance(
    catalogue: AsteroidCatalogue, rule_path: Path | None = None
) -> ReducedInstance:
    rule, digest = load_rule(rule_path)
    if rule["catalogue"]["sha256"] != catalogue.source_sha256 and catalogue.source_sha256:
        raise ValueError("reduced-instance rule pins a different catalogue")
    eligibility = rule["selection"]["eligibility"]
    a_au = catalogue.semi_major_axis_km / AU_KM
    eligible = (
        (np.rad2deg(catalogue.inclination_rad) <= float(eligibility["max_inclination_deg"]))
        & (catalogue.eccentricity <= float(eligibility["max_eccentricity"]))
        & (a_au >= float(eligibility["semi_major_axis_au"][0]))
        & (a_au <= float(eligibility["semi_major_axis_au"][1]))
    )
    eligible_ids = catalogue.ids[eligible]
    keys = rank_keys(rule["instance_id"], eligible_ids)
    order = sorted(range(len(eligible_ids)), key=lambda k: (keys[k], int(eligible_ids[k])))
    count = int(rule["selection"]["count"])
    selected = np.asarray([int(eligible_ids[k]) for k in order[:count]], dtype=np.int64)
    return ReducedInstance(
        instance_id=rule["instance_id"],
        rule=rule,
        rule_sha256=digest,
        asteroid_ids=selected,
        eligible_count=int(eligible.sum()),
        ships=int(rule["fleet"]["ships"]),
        start_mjd=float(rule["window"]["start_mjd"]),
        end_mjd=float(rule["window"]["end_mjd"]),
    )
