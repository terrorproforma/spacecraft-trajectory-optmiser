"""Compose exact existing ship sections; no new dynamics or verification."""

import hashlib
import json
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
BASE = ROOT / "results/lambda/2026-09-09/gpu-regeneration-v799/h100-best"
ROUTE = ROOT / "build/performance/seeded-candidate-boundary-merit-v627/campaign"
OLD = ROOT / "build/performance/refinement-queue-v623/inputs/incumbent.txt"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def parts(data):
    lines = data.splitlines(keepends=True)
    found = [i for i, line in enumerate(lines) if line.split() and line.split()[0] == b"8"]
    assert found == list(range(found[0], found[-1] + 1))
    return b"".join(lines[:found[0]]), b"".join(lines[found[0]:found[-1] + 1]), b"".join(lines[found[-1] + 1:])


def inventory(data):
    ships = {}
    for line in data.splitlines():
        row = line.split()
        if row:
            ship, event = int(row[0]), int(row[1])
            ships.setdefault(ship, set())
            if event > 0:
                ships[ship].add(event)
    return ships


def main():
    current, replacement = BASE / "Result.txt", ROUTE / "output/fleet/Result.txt"
    assert sha(current) == "97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48"
    assert sha(replacement) == "931defc362c9299d28d089aa46130c05f92bf9ee38ea03d870b4dc2a83915946"
    assert sha(OLD) == "d368babcf0656ab55b9a21bd8ccde4eef22629a37f9bc8f8a1877eb15a3e7efa"
    baseline = read(BASE / "campaign-report.json")
    candidate = read(ROUTE / "output/report.json")
    binding = read(ROUTE / "output/fleet/checker-binding.json")
    assert baseline["solution_sha256"] == sha(current) and baseline["qualified"]
    assert baseline["independent"]["ok"] and baseline["official"]["ok"]
    assert binding["result_sha256"] == sha(replacement)
    assert candidate["acceptance"]["accepted_improvement"] and candidate["acceptance"]["both_checkers_passed"]
    before, old_section, after = parts(current.read_bytes())
    assert old_section == parts(OLD.read_bytes())[1]
    new_section = parts(replacement.read_bytes())[1]
    composed = before + new_section + after
    assert parts(composed) == (before, new_section, after)
    old_footprints, new_footprints = inventory(current.read_bytes()), inventory(composed)
    assert old_footprints == new_footprints
    assert len(new_footprints) == 23 and len(set.union(*new_footprints.values())) == 200
    assert all(not new_footprints[8] & values for ship, values in new_footprints.items() if ship != 8)
    inputs = KIT / "inputs"
    inputs.mkdir()
    path = inputs / "Result.txt"
    with path.open("xb") as stream:
        stream.write(composed)
    assert parts(path.read_bytes())[0] == before and parts(path.read_bytes())[2] == after
    plan = {
        "stage": "prepared_only",
        "changed_ship": 8,
        "source_result_sha256": sha(current),
        "certified_route_result_sha256": sha(replacement),
        "composed_result_sha256": sha(path),
        "all_other_22_ship_sections_byte_exact": True,
        "source_ship8_matches_historical_v767_exactly": True,
        "unchanged_footprints": True,
        "required_ships": 23,
        "required_asteroids": 200,
        "baseline": baseline["independent"],
        "certified_delta": candidate["acceptance"],
        "expected_raw_kg": baseline["independent"]["total_mass_kg"] + candidate["acceptance"]["raw_delta_from_retained_kg"],
        "expected_weighted_kg": baseline["independent"]["weighted_score_fixed_bonus_kg"] + candidate["acceptance"]["weighted_delta_from_retained_kg"],
        "reporting_bound_kg": 1e-8,
        "hard_deadline_seconds": 120,
        "TERM_grace_seconds": 10,
        "KILL_reap_seconds": 30,
        "calls": {"independent_fullfleet": 1, "official_fullfleet": 1, "GPU": 0, "SCvx": 0, "search": 0, "extra_leg_certificates": 0},
        "source_sha256": {},
    }
    sources = [Path(__file__), KIT / "run.py", current, BASE / "campaign-report.json", replacement, ROUTE / "output/report.json", ROUTE / "output/fleet/checker-binding.json", ROUTE / "ready.json", OLD]
    plan["source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    with (KIT / "plan.json").open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plan_sha256": sha(KIT / "plan.json"), "composed_result_sha256": sha(path), "expected_raw_kg": plan["expected_raw_kg"], "expected_weighted_kg": plan["expected_weighted_kg"]}))


if __name__ == "__main__":
    main()
