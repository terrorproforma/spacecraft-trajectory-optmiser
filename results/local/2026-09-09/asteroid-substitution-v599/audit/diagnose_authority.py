"""CPU replay of recorded v599 rejections; inspect failures without changing arithmetic."""

import ctypes
import dataclasses
import importlib.util
import inspect
import json
import os
from collections import Counter
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v599_authority_driver", root / "run.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.activate_source()

from spacepdhcg.gtoc12.bundles import (  # noqa: E402
    ClusterPricingSettings, cluster_retime_settings, cluster_search_settings,
)
from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue  # noqa: E402
from spacepdhcg.gtoc12.jointopt import JointItinerary, JointSettings, route_from_summary  # noqa: E402
from spacepdhcg.gtoc12.lambert import using_lambert_backend  # noqa: E402
from spacepdhcg.gtoc12.retiming import Retimer, visits_of  # noqa: E402

os.environ.update(
    CUDA_VISIBLE_DEVICES="", SPACEPDHCG_TEST_GTOC12_JOINT_BATCH="0",
    SPACEPDHCG_GTOC12_DATA="/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data",
)
audit_dir = root / "audit"
audit_dir.mkdir(exist_ok=True)
output = audit_dir / "authority-replay.json"
if output.exists():
    raise FileExistsError(output)
report = json.loads((root / "output/report.json").read_text())
plan = json.loads((root / "plan/plan.json").read_text())
assert report["complete"]
assert driver.sha256(root / "run.py") == report["driver_sha256"]
started = time.perf_counter()
native_attempts = []


def forbidden(*args, **kwargs):
    native_attempts.append(str(args))
    raise AssertionError("CPU authority audit must not load a native library")


def evaluator(ship, summaries, catalogue, weights):
    route = route_from_summary(summaries[ship])
    policy = ClusterPricingSettings(
        collect_dp_inflation_fit=str(root / "source/results/gtoc12/hop_inflation_fit.json")
    )
    retimer = Retimer(catalogue, cluster_search_settings(policy, len(catalogue.ids)),
                      dataclasses.replace(cluster_retime_settings(policy, last=True), step_days=15),
                      weights=weights)
    retimer.protect_earth_leg(route.plan)
    joint = JointItinerary(catalogue, retimer, weights=weights,
                          settings=JointSettings(insert=False, max_moves_per_mesh=2,
                                                 time_budget_seconds=15))
    joint.learn(route)
    trace = []
    original_fail = joint._fail

    def traced_fail(reason):
        # Read the real evaluator frame only; all return values and decisions are
        # those of the unmodified published _fail/_forward implementation.
        frame = inspect.currentframe().f_back
        if reason == "leg_authority" and frame.f_code is JointItinerary._forward.__code__:
            values = frame.f_locals
            measured = values["measured"]
            trace.append({
                "leg_index": values["j"], "from": values["visit"].body,
                "to": values["nxt"].body, "role": values["role"],
                "departure": float(values["dep"][values["j"]]),
                "arrival": float(values["arr"][values["j"] + 1]),
                "mass_kg": float(values["mass"]), "lambert_km_s": values["lambert"],
                "authority_ratio": Retimer.authority_ratio(values["lambert"], values["mass"], values["tof"]),
                "ratio_limit": float(values["ratio_limit"]),
                "measured_key_exists": measured is not None,
                "measured_mass_kg": None if measured is None else measured.mass_before_kg,
                "measured_mass_delta_kg": None if measured is None else values["mass"] - measured.mass_before_kg,
                "measured_mass_tolerance_kg": values["tolerance"],
            })
        del frame
        return original_fail(reason)

    joint._fail = traced_fail
    return route, joint, trace


with patch.object(ctypes, "CDLL", side_effect=forbidden), using_lambert_backend("numpy"):
    catalogue, bonus = load_catalogue(), load_bonus_table()
    summaries, _, _ = driver.load_inputs()
    weights = {int(a): bonus.for_asteroid(int(a)) for a in catalogue.ids}
    states = {}
    originals = []
    for selected in plan["selected_ships"]:
        ship = selected["ship"]
        route, joint, trace = evaluator(ship, summaries, catalogue, weights)
        visits, arr, dep = visits_of(route.plan)
        value = joint.evaluate(visits, np.asarray(arr), np.asarray(dep))
        originals.append({"ship": ship, "feasible": value.feasible, "failure": value.failure,
                          "weighted_kg": value.weighted_kg, "raw_kg": value.collected_kg,
                          "measured_legs": value.measured_legs,
                          "flight_legs": len(visits) - 1, "trace": list(trace)})
        states[ship] = (joint, trace, visits, arr, dep)
    expected = {row["case"]: row for row in report["screened_cases"]}
    rows = []
    for case in plan["replacement_cases"]:
        joint, trace, visits, arr, dep = states[case["ship"]]
        changed = driver.replacement_visits(visits, case["old"], case["new"], set(plan["occupied_asteroids"]))
        gpu_samples = {sample["mode"]: sample for sample in expected[case["case"]]["samples"]}
        for mode, a, d in driver.epoch_seeds(changed, arr, dep, case["new"]):
            trace.clear()
            value = joint.evaluate(changed, a, d)
            gpu = gpu_samples[mode]
            item = {"case": case["case"], "ship": case["ship"], "old": case["old"], "new": case["new"],
                    "mode": mode, "CPU_failure": value.failure, "GPU_failure": gpu["failure"],
                    "same_outcome": value.failure == gpu["failure"] and value.feasible == gpu["feasible_surrogate"],
                    "feasible": value.feasible}
            if trace:
                item["first_failure"] = trace[0]
                item["failure_on_changed_incident_leg"] = case["new"] in (trace[0]["from"], trace[0]["to"])
            rows.append(item)
        if len(rows) % 150 == 0:
            print(json.dumps({"CPU_replayed_samples": len(rows), "seconds": time.perf_counter() - started}), flush=True)
    native_calls = sum(joint.evaluations for joint, *_ in states.values())

categories = Counter()
for row in rows:
    if row["CPU_failure"] != "leg_authority":
        categories[row["CPU_failure"] or "feasible"] += 1
    elif row["failure_on_changed_incident_leg"]:
        categories["authority_on_changed_incident_leg"] += 1
    else:
        categories["authority_on_unchanged_leg"] += 1
record = {
    "kind": "CPU_only_replay_of_frozen_surrogate_no_GPU_or_SCvx",
    "arithmetic_and_thresholds_unchanged": True, "source_commit": driver.COMMIT,
    "driver_sha256": driver.sha256(root / "run.py"), "auditor_sha256": driver.sha256(__file__),
    "GPU_report_sha256": driver.sha256(root / "output/report.json"),
    "originals": originals, "samples": rows, "counts": dict(categories),
    "all_GPU_CPU_outcomes_agree": all(row["same_outcome"] for row in rows),
    "CPU_evaluations_including_originals": native_calls,
    "native_library_load_attempts": native_attempts, "GPU_calls": 0,
    "seconds": time.perf_counter() - started,
}
driver.write_json(output, record)
print(json.dumps({key: value for key, value in record.items() if key != "samples"}), flush=True)
