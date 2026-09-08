"""Audit saved H100 results and reconstruct return boundaries; CPU-only preparation."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=REPO)
    args = p.parse_args()
    source = REPO / "build/performance/retrieved-joint-v632/evidence/campaign-v636"
    names = ("baseline0", "candidate0", "candidate1", "baseline1")
    routes, campaigns = {}, {}
    inputs = {}
    for name in names:
        path = source / name / "candidates/ship_15_attempt_02/refinement.json"
        campaign = source / name / "report.json"
        routes[name] = json.loads(path.read_text())
        campaigns[name] = json.loads(campaign.read_text())
        inputs[path.relative_to(REPO).as_posix()] = digest(path)
        inputs[campaign.relative_to(REPO).as_posix()] = digest(campaign)
    assert len({object_digest(r["plan"]) for r in routes.values()}) == 1
    assert len({object_digest(r["scvx_settings"]) for r in campaigns.values()}) == 1
    assert len({object_digest(r["native_sha256"]) for r in campaigns.values()}) == 1
    assert len({object_digest(r["source_sha256"]) for r in campaigns.values()}) == 1
    assert all(r["passes"] == 1 and len(r["legs"]) == 18 for r in routes.values())
    report = {"kind": "saved_evidence_audit_and_cpu_boundary_reconstruction_no_gpu_execution",
              "source_sha256": inputs, "plan_sha256": object_digest(routes["candidate1"]["plan"]),
              "all_plans_equal": True, "all_scvx_settings_equal": True,
              "all_native_library_hashes_equal": True, "all_python_source_hashes_equal": True,
              "legs": [], "runs": {}, "limitations": [
                  "Historical raw native tensors and seed/workspace state were not captured.",
                  "Recorded return initial masses differ; this is not a bit-identical-input experiment.",
                  "A local RTX replay cannot establish H100 repeatability if it does not reproduce the failure.",
                  "No root cause is inferred solely from tiny upstream mass differences or the device-selection flag."]}
    for index in range(18):
        legs = {name: routes[name]["legs"][index] for name in names}
        assert len({(r["from"], r["to"], r["t0"], r["tf"]) for r in legs.values()}) == 1
        before = [r["mass_before"] for r in legs.values()]
        after = [r["mass_after"] for r in legs.values() if math.isfinite(r["mass_after"])]
        report["legs"].append({"index": index,
            **{k: legs["candidate1"][k] for k in ("from", "to", "t0", "tf")},
            "initial_mass_spread_kg": max(before) - min(before),
            "certified_final_mass_spread_kg": max(after) - min(after),
            "runs": {n: {k: row[k] for k in ("mass_before", "mass_after", "status", "certified", "scvx_iterations")}
                     for n, row in legs.items()}})
    for name, route in routes.items():
        leg = route["legs"][-1]
        record = campaigns[name]["refinements"][-1]
        report["runs"][name] = {
            "mass_before_kg": leg["mass_before"], "mass_before_hex": float(leg["mass_before"]).hex(),
            "status": leg["status"], "iterations": leg["scvx_iterations"], "solve_seconds": leg["solve_seconds"],
            "qualified_iterations": [r["outer_iteration"] for r in leg["conic_reports"] if r["qualified"]],
            "return_conic_reports": leg["conic_reports"], "failures": route["failures"],
            "raw_route_kg": route["total_collected_kg"],
            "residual_propellant_kg": route["final_mass_kg"] - 500 if route["certified"] else None,
            "full_fleet_weighted_score_kg": record.get("verification", {}).get("score_kg"),
            "promotion_status": record["status"]}

    sys.meta_path = [f for f in sys.meta_path if f.__class__.__module__ != "_editable_skbc_spacepdhcg"]
    sys.path.insert(0, str(args.repo.resolve() / "src"))
    from spacepdhcg.gtoc12 import constants as C
    from spacepdhcg.gtoc12.data import load_catalogue
    from spacepdhcg.gtoc12.low_thrust import ScvxSettings
    from spacepdhcg.gtoc12.pipeline import body_state
    catalogue = load_catalogue()
    settings = ScvxSettings(**campaigns["candidate1"]["scvx_settings"])
    plan = routes["candidate1"]["plan"]
    leg = routes["candidate1"]["legs"][-1]
    # Follow refine_route's insertion order and maximum-collected-mass calculation.
    # Failed RefinedRoute.collected_mass_kg is all zero as a failure sentinel and
    # must never be used to reconstruct the physical carried mass on this leg.
    collected = {int(a): C.maximum_collected_mass(float(t) - float(plan["deploy_epochs"][a]))
                 for a, t in plan["collect_epochs"].items()}
    carried = sum(collected[int(a)] for a, t in plan["collect_epochs"].items() if float(t) <= leg["t0"])
    assert carried == routes["candidate0"]["total_collected_kg"]
    r0, v0 = body_state(catalogue, leg["from"], leg["t0"])
    rf, vf = body_state(catalogue, leg["to"], leg["tf"])
    boundary = {"departure_epoch": leg["t0"], "arrival_epoch": leg["tf"],
                "departure_position": r0.tolist(), "departure_velocity": v0.tolist(),
                "arrival_position": rf.tolist(), "arrival_velocity": vf.tolist(),
                "free_departure_vinf": False, "free_arrival_vinf": True,
                "minimum_final_mass": C.DRY_MASS_KG + carried}
    fixture = {"kind": "exact_recorded_return_masses_with_reconstructed_pinned_ephemeris",
               "case": "ship_15_attempt_02_return_leg_17", "from": 13077, "to": 0,
               "source_sha256": inputs, "catalogue_sha256": catalogue.source_sha256,
               "settings": dataclasses.asdict(settings), "boundary_shared": boundary,
               "collected_mass_kg": collected, "carried_mass_kg": carried,
               "variants": {name: {"initial_mass": routes[name]["legs"][-1]["mass_before"],
                                    "initial_mass_hex": float(routes[name]["legs"][-1]["mass_before"]).hex(),
                                    "historical_status": routes[name]["legs"][-1]["status"]}
                            for name in ("candidate1", "candidate0")},
               "repeat_order": ["candidate1", "candidate0"] * 3,
               "physics_tolerances_changed": False}
    report["return_boundary_shared"] = boundary
    report["carried_mass_kg"] = carried
    report["catalogue_sha256"] = catalogue.source_sha256
    report["relative_return_initial_mass_spread"] = report["legs"][-1]["initial_mass_spread_kg"] / leg["mass_before"]
    for name, value in (("audit.json", clean(report)), ("fixture.json", fixture)):
        (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fixture_sha256": digest(HERE / "fixture.json"),
                      "minimum_final_mass": boundary["minimum_final_mass"], "carried_mass_kg": carried,
                      "initial_mass_spread_kg": report["legs"][-1]["initial_mass_spread_kg"],
                      "relative_initial_mass_spread": report["relative_return_initial_mass_spread"]}, indent=2))


if __name__ == "__main__":
    main()
