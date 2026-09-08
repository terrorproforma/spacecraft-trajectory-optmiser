"""Construct immutable historical requests without Lambert/geometry/cost evaluation."""

# Select the frozen source before importing the project.
# ruff: noqa: E402
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import sys
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
sys.path.insert(0, str(KIT / "source/src"))

import numpy as np
from control_comparator import compare, decode, encode

from spacepdhcg.gtoc12.search import PlannedLeg, RoutePlan, RouteSearch, SearchSettings, _Partial


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(encode(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def construct(group, catalogue):
    started = time.perf_counter()
    doc = decode(read(KIT / "inputs/fixtures.json"))
    all_cases = doc["historical"]
    cases = [all_cases[i] for i in group["fixture_indices"]]
    defaults = {f.name: f.default for f in dataclasses.fields(SearchSettings)}
    settings = read(KIT / "inputs/v616-report.json")["settings"]
    settings = {
        k: defaults[k]
        if v is None and isinstance(defaults[k], float) and not math.isfinite(defaults[k])
        else v
        for k, v in settings.items()
    }
    if group["model"] == "existing_fit_no_refit":
        settings["collect_dp_inflation_fit"] = str(KIT / "inputs/hop_inflation_fit.json")
    search = RouteSearch(catalogue, np.asarray([], dtype=np.int64), SearchSettings(**settings))
    prototypes = {}
    for i in sorted(set(group["fixture_indices"])):
        case = all_cases[i]
        archive = read(KIT / "inputs" / f"ship-{case['provenance']['ship']:02}.json")
        assert (
            sha(KIT / "inputs" / f"ship-{case['provenance']['ship']:02}.json")
            == case["provenance"]["route_sha256"]
        )
        plan = RoutePlan.from_summary(archive["plan"])
        split = max(j for j, row in enumerate(plan.legs) if row.role == "deploy_hop") + 1
        ids = case["provenance"]["deploy_asteroid_ids"]
        deploy = {a: d["deploy_epoch"] for a, d in zip(ids, case["deploys"], strict=True)}
        collect = {a: d["collect_epoch"] for a, d in zip(ids, case["deploys"], strict=True)}
        assert deploy == plan.deploy_epochs and collect == plan.collect_epochs
        partial = _Partial(
            list(plan.legs[:split]),
            plan.legs[split - 1].to_id,
            plan.legs[split - 1].arrival_epoch,
            case["partial_mass"],
            list(deploy.items()),
        )
        forward = []
        for row, provenance in zip(case["legs"], case["provenance"]["legs"], strict=True):
            if provenance["kind"] == "builder_inserted_camp":
                source = target = provenance["at_asteroid"]
                role = "camp"
            else:
                source, target = provenance["from_id"], provenance["to_id"]
                role = next(key for key, value in doc["roles"].items() if value == row["role"])
            forward.append(
                PlannedLeg(source, target, row["departure"], row["arrival"], row["dv"], 1.0, role)
            )
        prototypes[i] = (partial, deploy, collect, forward, True)
    requests = [copy.deepcopy(prototypes[i]) for i in group["fixture_indices"]]
    return search, requests, cases, time.perf_counter() - started


def signature(requests):
    data = [
        {
            "partial": dataclasses.asdict(p),
            "deploy": d,
            "collect": c,
            "legs": [dataclasses.asdict(leg) for leg in legs],
            "use_table": use_table,
        }
        for p, d, c, legs, use_table in requests
    ]
    return hashlib.sha256(
        json.dumps(encode(data), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_plans(rows, requests, cases):
    assert len(rows) == len(requests) == len(cases)
    failures = []
    accepted = 0
    for (plan, reason), request, case in zip(rows, requests, cases, strict=True):
        expected = case["expected"]
        wanted_reason = "" if expected["failure_name"] == "ok" else expected["failure_name"]
        if reason != wanted_reason or (plan is not None) != (wanted_reason == ""):
            failures.append({"id": case["id"], "reason": reason, "expected_reason": wanted_reason})
            continue
        if plan is None:
            continue
        accepted += 1
        partial, deploy, collect, forward, _ = request
        assert plan.deploy_epochs == deploy and plan.collect_epochs == collect
        assert tuple(plan.legs[: len(partial.legs)]) == tuple(partial.legs)
        assert len(plan.legs) == len(partial.legs) + len(forward)
        ids = case["provenance"]["deploy_asteroid_ids"]
        cargo = {
            ids[i]: expected["collected_by_deploy"][i] for i in expected["pickup_insertion_order"]
        }
        assert plan.collected_mass == cargo and list(plan.collected_mass) == list(cargo)
        for a, b, detail in zip(
            plan.legs[len(partial.legs) :], forward, expected["leg_results"], strict=True
        ):
            assert dataclasses.replace(a, inflation=b.inflation) == b
            assert math.isclose(a.inflation, detail["inflation"], rel_tol=2e-13, abs_tol=2e-10)
        for observed, value in (
            (plan.final_mass_proxy_kg, expected["result"]["final_mass"]),
            (plan.propellant_proxy_kg, expected["result"]["propellant"]),
        ):
            assert math.isclose(observed, value, rel_tol=2e-13, abs_tol=2e-10)
    assert not failures, failures
    return {"passed": True, "accepted": accepted, "rejected": len(rows) - accepted}


def raw_workspace(workspace, requests):
    n = len(requests)
    nd, nl = sum(len(row[1]) for row in requests), sum(len(row[3]) for row in requests)
    arrays = {
        "results": workspace.results[:n].copy(),
        "leg_results": workspace.leg_results[:nl].copy(),
        "collected_by_deploy": workspace.collected[:nd].copy(),
        "stats": workspace.stats.copy(),
    }
    raw = {
        key: [{name: row[name].item() for name in array.dtype.names} for row in array]
        for key, array in arrays.items()
        if array.dtype.names
    }
    raw["collected_by_deploy"] = arrays["collected_by_deploy"].tolist()
    return arrays, raw


def validate_native(workspace, requests, cases):
    arrays, raw = raw_workspace(workspace, requests)
    comparison = read(KIT / "inputs/gpu-profile.json")["comparison"]
    result = compare(cases, raw, comparison)
    return arrays, result
