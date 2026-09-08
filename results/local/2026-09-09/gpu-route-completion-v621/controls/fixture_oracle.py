"""CPU-only completion fixture arithmetic and a frozen _finish cross-check.

This is an evidence adapter, not a production search implementation. Model
metadata is already resolved by the builder; no geometry or native call occurs.
"""

# Frozen-source path must be selected before importing the project.
# ruff: noqa: E402

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

KIT = Path(__file__).resolve().parent
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
sys.path.insert(0, str(KIT / "source/src"))

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.collectdp import CollectDPSettings, CollectPairTable
from spacepdhcg.gtoc12.hopcalib import InflationFit
from spacepdhcg.gtoc12.screening import (
    exhaust_velocity_km_s,
    low_thrust_inflation,
    propellant_for_delta_v,
    return_inflation_model,
    thrust_authority_km_s,
)
from spacepdhcg.gtoc12.search import PlannedLeg, RouteSearch, SearchSettings, _Partial

ROLES = {"camp": 0, "earth_out": 1, "deploy_hop": 2, "collect_hop": 3, "earth_return": 4}
MODELS = {"flat": 0, "ratio": 1, "fit5": 2, "return": 3, "certified_flat": 4, "table_return": 5}
FAILURES = {
    "ok": 0,
    "uncollected": 1,
    "stay_too_short": 2,
    "leg_authority": 3,
    "invalid_inflation": 4,
    "mass_below_dry_plus_collected": 5,
    "invalid_mining_stay": 6,
}
STAGES = {"unvisited": 0, "camp": 1, "authority": 2, "inflation": 3, "costed": 4, "mining": 5}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"float64": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, dict):
        if set(value) == {"float64"}:
            return float(value["float64"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def policy():
    return {
        "abi_version": 1,
        "sum_mode": 1,
        "reserved0": 0,
        "reserved1": 0,
        "initial_mass": C.MAX_INITIAL_MASS_KG,
        "dry_mass": C.DRY_MASS_KG,
        "miner_mass": C.MINER_MASS_KG,
        "thrust": C.THRUST_MAX_N,
        "exhaust": exhaust_velocity_km_s(),
        "mining_rate": C.MINING_RATE_KG_PER_YEAR,
        "year_days": C.YEAR_DAYS,
        "minimum_stay": C.MIN_MINING_STAY_YEARS * C.YEAR_DAYS,
    }


def leg(role="collect_hop", model="flat", source_deploy=0, **changes):
    return {
        "role": ROLES[role],
        "model": MODELS[model],
        "source_deploy": source_deploy,
        "reserved": 0,
        "departure": 1000.0,
        "arrival": 1200.0,
        "dv": 0.0,
        "flat": 1.2,
        "floor": 1.0,
        "slope": 0.65,
        "fit": [1.0, 0.0, 0.0, 0.0, 0.0],
        "delta_a_au": 0.0,
        "delta_longitude_rad": 0.0,
        "authority_ratio": 0.667,
        **changes,
    }


def inflation(row, mass):
    tof = row["arrival"] - row["departure"]
    model = row["model"]
    if model in (MODELS["flat"], MODELS["certified_flat"]):
        return row["flat"]
    if model == MODELS["ratio"]:
        return float(
            low_thrust_inflation(row["dv"], mass, tof, floor=row["floor"], slope=row["slope"])
        )
    authority = float(thrust_authority_km_s(mass, tof, 1.0))
    if model == MODELS["fit5"]:
        model_fit = InflationFit(tuple(row["fit"]), quantile=0.65, floor=row["floor"])
        return float(
            model_fit.inflation(row["dv"], mass, tof, row["delta_a_au"], row["delta_longitude_rad"])
        )
    if model == MODELS["return"]:
        return float(return_inflation_model(tof, row["dv"] / max(authority, 1e-12)))
    if model == MODELS["table_return"]:
        return float(
            CollectPairTable(None, CollectDPSettings()).return_inflation(row["dv"], mass, tof)[()]
        )
    raise AssertionError(model)


def empty_leg():
    return {
        "stage": 0,
        "pickup": 0,
        "mass_before": 0.0,
        "gained": 0.0,
        "departure_mass": 0.0,
        "mass_after": 0.0,
        "authority": 0.0,
        "inflation": 0.0,
        "propellant": 0.0,
        "tof": 0.0,
    }


def evaluate(case):
    """Full ABI detail oracle using pinned scalar cost functions, no native loading."""
    p = policy()
    mass = case["partial_mass"]
    deploys, legs = case["deploys"], case["legs"]
    fuel = (p["initial_mass"] - mass) - p["miner_mass"] * len(deploys)
    collected = {}
    details = [empty_leg() for _ in legs]
    processed = 0

    def result(reason="ok", failed_leg=-1, failed_deploy=-1):
        cargo = sum(collected.values())
        return {
            "result": {
                "failure": FAILURES[reason],
                "failed_leg": failed_leg,
                "failed_deploy": failed_deploy,
                "processed_legs": processed,
                "propellant": fuel,
                "final_mass": mass,
                "collected": cargo,
                "margin": mass - (p["dry_mass"] + cargo),
            },
            "failure_name": reason,
            "leg_results": details,
            "collected_by_deploy": [collected.get(i, 0.0) for i in range(len(deploys))],
            "pickup_insertion_order": list(collected),
            "source_return_kind": "ValueError"
            if reason == "invalid_mining_stay"
            else "RoutePlan"
            if reason == "ok"
            else "None",
        }

    for i, row in enumerate(deploys):
        if not row["has_collect"]:
            return result("uncollected", failed_deploy=i)
        if row["collect_epoch"] - row["deploy_epoch"] < p["minimum_stay"] - 1e-6:
            return result("stay_too_short", failed_deploy=i)
    for i, row in enumerate(legs):
        d = details[i]
        tof = row["arrival"] - row["departure"]
        d.update(mass_before=mass, departure_mass=mass, mass_after=mass, tof=tof)
        if row["role"] == ROLES["camp"]:
            d.update(stage=STAGES["camp"], inflation=row["flat"])
            processed += 1
            continue
        if row["role"] in (ROLES["collect_hop"], ROLES["earth_return"]):
            index = row["source_deploy"]
            dep = deploys[index]
            if abs(dep["collect_epoch"] - row["departure"]) < 1e-6:
                d["pickup"] = 1
                try:
                    gained = C.maximum_collected_mass(dep["collect_epoch"] - dep["deploy_epoch"])
                except ValueError:
                    d["stage"] = STAGES["mining"]
                    return result("invalid_mining_stay", i, index)
                collected[index] = gained
                mass += gained
                d.update(gained=gained, departure_mass=mass, mass_after=mass)
        authority = float(thrust_authority_km_s(mass, tof, 1.0))
        d["authority"] = authority
        if not row["dv"] <= row["authority_ratio"] * authority:
            d["stage"] = STAGES["authority"]
            return result("leg_authority", i)
        factor = inflation(row, mass)
        d["inflation"] = factor
        if not math.isfinite(factor) or factor < 0.0:
            d["stage"] = STAGES["inflation"]
            return result("invalid_inflation", i)
        spent = float(propellant_for_delta_v(mass, row["dv"] * factor))
        fuel += spent
        mass -= spent
        d.update(stage=STAGES["costed"], propellant=spent, mass_after=mass)
        processed += 1
    if mass < p["dry_mass"] + sum(collected.values()):
        return result("mass_below_dry_plus_collected")
    return result()


def source_check(case, expected):
    """Run the exact frozen _finish gate flow; resolved model metadata replaces lookup only."""
    dep = {i + 1: row["deploy_epoch"] for i, row in enumerate(case["deploys"])}
    collect = {
        i + 1: row["collect_epoch"] for i, row in enumerate(case["deploys"]) if row["has_collect"]
    }
    forward = [
        PlannedLeg(
            row["source_deploy"] + 1 if row["source_deploy"] >= 0 else 999,
            0 if row["role"] == ROLES["earth_return"] else 998,
            row["departure"],
            row["arrival"],
            row["dv"],
            row["flat"],
            next(k for k, v in ROLES.items() if v == row["role"]),
        )
        for row in case["legs"]
    ]
    flights = [row for row in case["legs"] if row["role"] != ROLES["camp"]]

    class Search(RouteSearch):
        def __init__(self):
            super().__init__(None, np.asarray(list(dep), dtype=np.int64), SearchSettings())
            self.flight = -1
            self.calls = []
            self._collect_table = self

        def _feasible(self, mass, dv, tof, role):
            self.flight += 1
            self.calls.append((mass, dv, tof))
            return dv <= flights[self.flight]["authority_ratio"] * float(
                thrust_authority_km_s(mass, tof, 1.0)
            )

        def _dp_hop_inflation(self, source, target, departure, dv, mass, tof):
            return inflation(flights[self.flight], mass)

        def return_inflation_at(self, source, departure, tof, dv, mass):
            return inflation(flights[self.flight], mass)

    with patch.object(ctypes, "CDLL", side_effect=AssertionError("Native loading forbidden")):
        search = Search()
        partial = _Partial([], 1, 0.0, case["partial_mass"], list(dep.items()))
        try:
            plan = search._finish(partial, dep, collect, forward, use_collect_table=True)
        except ValueError:
            assert expected["failure_name"] == "invalid_mining_stay"
            return {"source_return_kind": "ValueError", "authority_calls": len(search.calls)}
        kind = "None" if plan is None else "RoutePlan"
        assert kind == expected["source_return_kind"], (case["id"], kind, expected)
        if plan is None:
            assert search.last_failure == expected["failure_name"]
        else:
            assert plan.final_mass_proxy_kg == expected["result"]["final_mass"]
            assert plan.propellant_proxy_kg == expected["result"]["propellant"]
            assert sum(plan.collected_mass.values()) == expected["result"]["collected"]
            assert list(plan.collected_mass) == [i + 1 for i in expected["pickup_insertion_order"]]
            assert len(plan.legs) == expected["result"]["processed_legs"]
        return {"source_return_kind": kind, "authority_calls": len(search.calls)}
