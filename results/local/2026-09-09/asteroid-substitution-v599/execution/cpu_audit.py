"""Rebuild the deterministic inventory and all seeds without native calls."""

import argparse
import ctypes
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v599_driver_audit", root / "run.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.activate_source()
# Imports deliberately follow validation/activation of the frozen source tree.
from spacepdhcg.gtoc12.data import load_bonus_table, load_catalogue  # noqa: E402
from spacepdhcg.gtoc12.jointopt import JointItinerary, route_from_summary  # noqa: E402
from spacepdhcg.gtoc12.retiming import visits_of  # noqa: E402
from spacepdhcg.gtoc12.solution import Solution  # noqa: E402

os.environ["SPACEPDHCG_GTOC12_DATA"] = (
    "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data"
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / "cpu-audit.json")
args = parser.parse_args()
started = time.perf_counter()
native_attempts = []


def forbidden(*args, **kwargs):
    native_attempts.append(str(args))
    raise AssertionError("CPU preparation attempted a native library load")


with patch.object(ctypes, "CDLL", side_effect=forbidden):
    catalogue, bonus = load_catalogue(), load_bonus_table()
    weights = {int(a): bonus.for_asteroid(int(a)) for a in catalogue.ids}
    summaries, verified, inputs = driver.load_inputs()
    solution = Solution.read(root / "inputs/Result.txt")
    fresh = driver.make_plan(summaries, verified, weights, catalogue, solution)
    saved = json.loads((root / "plan/plan.json").read_text())
    # Tuples (the TOF grid) normalize to their JSON list representation.
    if json.loads(json.dumps(fresh)) != saved:
        raise AssertionError("CPU candidate plan is not reproducible")
    occupied = set(saved["occupied_asteroids"])
    inventories, seed_count, camp_cases, maximum_polish = {}, 0, 0, 0
    for selected in saved["selected_ships"]:
        ship = selected["ship"]
        route = route_from_summary(summaries[ship])
        visits, arr, dep = visits_of(route.plan)
        inventories[ship] = (visits, arr, dep)
        maximum_polish = max(
            maximum_polish, 1 + 4 * len(list(JointItinerary.moves(len(visits), 8)))
        )
    for case in saved["replacement_cases"]:
        visits, arr, dep = inventories[case["ship"]]
        before = [(v.body, v.deploy, v.collect, v.role_out) for v in visits]
        replacement = driver.replacement_visits(visits, case["old"], case["new"], occupied)
        assert [(v.body, v.deploy, v.collect, v.role_out) for v in visits] == before
        assert case["new"] not in occupied
        assert all(v.body != case["old"] for v in replacement)
        assert (
            Counter(v.body for v in replacement)[case["new"]]
            == Counter(v.body for v in visits)[case["old"]]
        )
        camp_cases += int(any(v.body == case["old"] and v.deploy and v.collect for v in visits))
        for mode, a, d in driver.epoch_seeds(replacement, arr, dep, case["new"]):
            seed_count += 1
            deploy = {v.body: a[i] for i, v in enumerate(replacement) if v.deploy}
            collect = {v.body: d[i] for i, v in enumerate(replacement) if v.collect}
            assert all(collect[b] - deploy[b] >= 365.25 - 1e-6 for b in collect)
            assert set(deploy) - set(collect) == set(
                int(b) for b in summaries[case["ship"]]["plan"]["orphaned"]
            )
            assert a[0] == arr[0] and d[-1] == dep[-1]
            if mode == "unchanged_epochs":
                assert list(a) == list(arr) and list(d) == list(dep)
    maximum_joint_work = seed_count + 4 * maximum_polish
    assert maximum_joint_work <= 6000
    assert len(saved["replacement_cases"]) <= 768
    assert len(saved["selected_ships"]) == 3
    assert not native_attempts
record = {
    "kind": "CPU_inventory_and_construction_audit_no_native_execution",
    "passed": True,
    "source_commit": driver.COMMIT,
    "driver_sha256": driver.sha256(root / "run.py"),
    "source_manifest_sha256": driver.sha256(root / "source-sha256.json"),
    "plan_sha256": driver.sha256(root / "plan/plan.json"),
    "input_sha256": inputs,
    "cases": len(saved["replacement_cases"]),
    "epoch_seeds": seed_count,
    "camp_cases": camp_cases,
    "maximum_joint_candidates_including_polish": maximum_joint_work,
    "selected_ships": [r["ship"] for r in saved["selected_ships"]],
    "raw_feasibility_slack_kg": saved["raw_feasibility_slack_kg"],
    "native_library_load_attempts": native_attempts,
    "gpu_calls": 0,
    "seconds": time.perf_counter() - started,
}
driver.write_json(args.output, record)
print(json.dumps(record))
