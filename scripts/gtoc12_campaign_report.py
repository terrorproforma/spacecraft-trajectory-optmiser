#!/usr/bin/env python
"""GTOC12 campaign report: leg roles, |Δλ| at harvest, chain masses, paired families, masters.

Reads archived ``route_summary.json`` files (cluster-fleet, joint-itinerary and fleet-master runs),
the fleet-master run reports and the committed reference documents
(``benchmarks/gtoc12/chain_prior_v1.json``, ``benchmarks/gtoc12/harvest_phase_v1.json``) and
writes one JSON with, per run: the per-ship propellant split (Earth-out, deploy hops, collect
hops, Earth return), the collect-hop and Earth-out TOFs, the ``|Δλ|`` of every collect hop at its
departure, the chain-mass distribution (>= 550/600/650/700 kg over unique asteroid sets), the
paired-family comparison of two cluster-fleet arms (best ship per family label), the joint runs'
Earth-leg stage totals and the fleet masters' score / ships / average / LP gap.

Usage (from the repository root, ``PYTHONPATH=src``)::

    python scripts/gtoc12_campaign_report.py --output results/gtoc12/leg_stats/v10_report.json \
        --run cluster_fleet_v10 --run cluster_fleet_v10_control --run cluster_fleet_v9 \
        --pair cluster_fleet_v10:cluster_fleet_v10_control \
        --pair cluster_fleet_v10:cluster_fleet_v9 \
        --joint joint_itinerary_v4 --joint joint_itinerary_v5 --joint joint_itinerary_v10 \
        --master fleet_master_v10 --master fleet_master_v9 --master fleet_master_h100_v2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.harvestphase import phase_deg_at

RUNS = Path("results/gtoc12/runs")
ROLES = ("earth_out", "deploy_hop", "collect_hop", "earth_return")
THRESHOLDS = (550.0, 587.8, 599.5, 600.0, 610.6, 650.0, 700.0)


def _quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def ship_record(catalogue, path: Path) -> dict[str, Any] | None:
    """Roles, TOFs and harvest phases of one archived ship (``None`` for unusable files)."""

    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    plan = summary.get("plan") or {}
    plan_legs = [leg for leg in plan.get("legs", []) if leg.get("role") != "camp"]
    legs = summary.get("legs") or []
    if not plan_legs or len(plan_legs) != len(legs):
        return None
    by_role: dict[str, float] = dict.fromkeys(ROLES, 0.0)
    collect_tof: list[float] = []
    collect_phase: list[float] = []
    collect_kg: list[float] = []
    deploy_tof: list[float] = []
    deploy_kg: list[float] = []
    earth_out = None
    earth_return = None
    for planned, flown in zip(plan_legs, legs, strict=True):
        role = planned.get("role")
        t0 = float(flown.get("t0", planned.get("t0")))
        tf = float(flown.get("tf", planned.get("tf")))
        propellant = flown.get("propellant_kg")
        if propellant is None:
            before, after = flown.get("mass_before"), flown.get("mass_after")
            propellant = (before - after) if before is not None and after is not None else 0.0
        propellant = float(propellant)
        if role in by_role:
            by_role[role] += propellant
        if role == "collect_hop":
            collect_tof.append(tf - t0)
            collect_kg.append(propellant)
            collect_phase.append(
                phase_deg_at(catalogue, int(planned["from"]), int(planned["to"]), t0)
            )
        elif role == "deploy_hop":
            deploy_tof.append(tf - t0)
            deploy_kg.append(propellant)
        elif role == "earth_out":
            earth_out = {
                "launch": t0,
                "arrival": tf,
                "tof_days": tf - t0,
                "propellant_kg": propellant,
            }
        elif role == "earth_return":
            earth_return = {"departure": t0, "tof_days": tf - t0, "propellant_kg": propellant}
    match = re.search(r"family_(\d+)", str(path))
    return {
        "path": str(path),
        "family": int(match.group(1)) if match else None,
        "asteroids": tuple(
            sorted(int(a) for a in plan.get("asteroids", summary.get("asteroids", [])))
        ),
        "collected_kg": float(summary.get("total_collected_kg", 0.0)),
        "certified": bool(summary.get("certified", False)),
        "final_mass_kg": summary.get("final_mass_kg"),
        "roles_kg": by_role,
        "earth_out": earth_out,
        "earth_return": earth_return,
        "collect_hop_tof_days": collect_tof,
        "collect_hop_kg": collect_kg,
        "collect_phase_deg": collect_phase,
        "deploy_hop_tof_days": deploy_tof,
        "deploy_hop_kg": deploy_kg,
    }


def load_run(catalogue, run: str, *, fleet_only: bool = False) -> list[dict[str, Any]]:
    root = RUNS / run
    pattern = "columns/**/route_summary.json" if fleet_only else "**/route_summary.json"
    records = []
    for path in sorted(root.glob(pattern)):
        record = ship_record(catalogue, path)
        if record is not None and record["certified"]:
            records.append(record)
    return records


def unique_best(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, ...], dict[str, Any]] = {}
    for record in records:
        key = record["asteroids"]
        if key not in best or record["collected_kg"] > best[key]["collected_kg"]:
            best[key] = record
    return sorted(best.values(), key=lambda r: -r["collected_kg"])


def run_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ships = unique_best(records)
    per_ship = {role: [r["roles_kg"][role] for r in ships] for role in ROLES}
    masses = [r["collected_kg"] for r in ships]
    collect_tof = [t for r in ships for t in r["collect_hop_tof_days"]]
    collect_kg = [k for r in ships for k in r["collect_hop_kg"]]
    phase = [p for r in ships for p in r["collect_phase_deg"]]
    return {
        "ships": len(ships),
        "archives": len(records),
        "collected_kg": _quantiles(masses),
        "asteroids_per_ship": _quantiles([float(len(r["asteroids"])) for r in ships]),
        "chain_mass_counts": {f"ge_{t:g}": int(sum(m >= t for m in masses)) for t in THRESHOLDS},
        "top10_kg": [round(m, 1) for m in masses[:10]],
        "per_ship_propellant_kg": {role: _quantiles(v) for role, v in per_ship.items()},
        "earth_out_tof_days": _quantiles(
            [r["earth_out"]["tof_days"] for r in ships if r["earth_out"]]
        ),
        "earth_out_launch_offset_days": _quantiles(
            [r["earth_out"]["launch"] - 64328.0 for r in ships if r["earth_out"]]
        ),
        "earth_return_tof_days": _quantiles(
            [r["earth_return"]["tof_days"] for r in ships if r["earth_return"]]
        ),
        "collect_hop_kg": _quantiles(collect_kg),
        "collect_hop_tof_days": _quantiles(collect_tof),
        "collect_hop_share_le75": (sum(k <= 75.0 for k in collect_kg) / len(collect_kg))
        if collect_kg
        else None,
        "deploy_hop_kg": _quantiles([k for r in ships for k in r["deploy_hop_kg"]]),
        "deploy_hop_tof_days": _quantiles([t for r in ships for t in r["deploy_hop_tof_days"]]),
        "collect_phase_deg": _quantiles(phase),
        "collect_phase_share_le_p75_ref": None,  # filled by the caller with the reference p75
        "phase_values": phase,
    }


def paired_families(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    def best_per_family(records):
        best: dict[int, float] = {}
        for r in records:
            if r["family"] is None:
                continue
            best[r["family"]] = max(best.get(r["family"], -math.inf), r["collected_kg"])
        return best

    fa, fb = best_per_family(a), best_per_family(b)
    common = sorted(set(fa) & set(fb))
    deltas = [fa[f] - fb[f] for f in common]
    return {
        "common": len(common),
        "a_only": sorted(set(fa) - set(fb)),
        "b_only": sorted(set(fb) - set(fa)),
        "up": int(sum(d > 0.05 for d in deltas)),
        "down": int(sum(d < -0.05 for d in deltas)),
        "mean_delta": statistics.fmean(deltas) if deltas else None,
        "median_delta": statistics.median(deltas) if deltas else None,
        "pairs": [
            {
                "family": f,
                "a": round(fa[f], 1),
                "b": round(fb[f], 1),
                "delta": round(fa[f] - fb[f], 1),
            }
            for f in common
        ],
    }


def joint_summary(run: str) -> dict[str, Any] | None:
    report_path = RUNS / run / "run_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    out = {
        k: report.get(k)
        for k in (
            "tasks",
            "attempted",
            "improved",
            "inserted",
            "gain_kg_total",
            "wall_seconds",
            "worker_peak_rss_mb",
            "earth_leg",
        )
    }
    out["settings"] = {
        k: report.get("settings", {}).get(k)
        for k in ("workers", "top", "min_collected_kg", "earth_leg", "earth_leg_certifications")
    }
    ships_path = RUNS / run / "ships.jsonl"
    if ships_path.exists():
        before_after = []
        earth_tof_before, earth_tof_after, earth_kg_before, earth_kg_after = [], [], [], []
        for line in ships_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("status") != "improved":
                continue
            before_after.append((record["archived_kg"], record.get("after_kg")))
            lb = record.get("legs_before") or []
            la = record.get("legs_after") or []
            eb = next((leg for leg in lb if leg["role"] == "earth_out"), None)
            ea = next((leg for leg in la if leg["role"] == "earth_out"), None)
            if eb and ea:
                earth_tof_before.append(eb["tof_days"])
                earth_tof_after.append(ea["tof_days"])
                if eb.get("propellant_kg") is not None and ea.get("propellant_kg") is not None:
                    earth_kg_before.append(eb["propellant_kg"])
                    earth_kg_after.append(ea["propellant_kg"])
        out["improved_ships"] = {
            "gain_kg": _quantiles([a - b for b, a in before_after if a is not None]),
            "earth_out_tof_before": _quantiles(earth_tof_before),
            "earth_out_tof_after": _quantiles(earth_tof_after),
            "earth_out_kg_before": _quantiles(earth_kg_before),
            "earth_out_kg_after": _quantiles(earth_kg_after),
            "earth_out_shortened": int(
                sum(a < b - 1e-6 for a, b in zip(earth_tof_after, earth_tof_before, strict=True))
            ),
        }
    return out


def master_summary(run: str) -> dict[str, Any] | None:
    report_path = RUNS / run / "run_report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    master = report.get("master") or {}
    fleet_path = RUNS / run / "fleet" / "fleet.json"
    fleet = json.loads(fleet_path.read_text(encoding="utf-8")) if fleet_path.exists() else {}
    per_ship = (fleet.get("fleet") or {}).get("collected_kg_per_ship")
    if per_ship:  # the verified fleet's per-ship masses (bundle columns hold several ships)
        masses = sorted((float(m) for m in per_ship), reverse=True)
    else:
        masses = sorted(
            (float(c["collected_kg"]) for c in master.get("selected", []) if c.get("collected_kg")),
            reverse=True,
        )
    ships = int(master.get("ships") or len(masses))
    average = (sum(masses) / ships) if masses and ships else None
    threshold_next = 250.0 * math.log((ships + 1) / 2.0) if ships else None
    return {
        "run": run,
        "ships": ships,
        "score_kg": fleet.get("score_kg", master.get("collected_kg")),
        "average_kg": average,
        "masses": [round(m, 1) for m in masses],
        "objective_kg": master.get("objective_kg"),
        "lp_bound_kg": master.get("lp_bound_kg"),
        "lp_gap_kg": master.get("lp_gap_kg"),
        "proven_optimal": master.get("proven_optimal"),
        "ship_limit": master.get("ship_limit"),
        "columns": master.get("columns"),
        "recertified_routes": report.get("recertified_routes"),
        "sources": len(report.get("sources") or []),
        "wall_seconds_total": report.get("wall_seconds_total"),
        "official_ok": (fleet.get("official") or {}).get("ok"),
        "independent_ok": (fleet.get("independent") or {}).get("ok"),
        "asteroids": len((fleet.get("fleet") or {}).get("asteroids", [])),
        "next_ship_threshold_avg_kg": threshold_next,
        "next_ship_needs_kg": (
            (ships + 1) * threshold_next - sum(masses) if threshold_next and masses else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run", action="append", default=[], help="cluster-fleet run id (repeatable)"
    )
    parser.add_argument(
        "--pair", action="append", default=[], help="A:B run ids for the paired families"
    )
    parser.add_argument("--joint", action="append", default=[], help="joint-itinerary run id")
    parser.add_argument("--master", action="append", default=[], help="fleet-master run id")
    parser.add_argument(
        "--fleet-ships",
        action="append",
        default=[],
        help="fleet-master run id whose fleet's ships are summarised (columns of the selected fleet)",
    )
    args = parser.parse_args()

    catalogue = load_catalogue()
    references = {}
    prior_path = Path("benchmarks/gtoc12/chain_prior_v1.json")
    phase_path = Path("benchmarks/gtoc12/harvest_phase_v1.json")
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        ships = prior.get("ships", [])
        references["per_ship_propellant_kg"] = {
            "earth_out": _quantiles([s["earth_out_kg"] for s in ships]),
            "deploy_hop": _quantiles([s["deploy_hops_kg"] for s in ships]),
            "collect_hop": _quantiles([s["collect_hops_kg"] for s in ships]),
            "earth_return": _quantiles([s["earth_return_kg"] for s in ships]),
        }
        references["collected_kg"] = _quantiles([s["collected_kg"] for s in ships])
        references["asteroids_per_ship"] = _quantiles([float(s["asteroids"]) for s in ships])
    if phase_path.exists():
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        references["collect_phase_deg"] = phase["distributions"]["phase_deg"]
        references["collect_hop_tof_days"] = phase["distributions"]["tof_days"]
        references["collect_hop_kg"] = phase["distributions"]["propellant_kg"]
        references["targets"] = phase["targets"]
    p75_ref = (references.get("targets") or {}).get("phase_deg_p75")

    report: dict[str, Any] = {
        "references": references,
        "runs": {},
        "pairs": {},
        "joint": {},
        "masters": {},
        "fleets": {},
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for run in args.run:
        if not (RUNS / run).exists():
            report["runs"][run] = None
            continue
        records = load_run(catalogue, run)
        loaded[run] = records
        summary = run_summary(records)
        phases = summary.pop("phase_values")
        if p75_ref is not None and phases:
            summary["collect_phase_share_le_p75_ref"] = float(
                sum(p <= p75_ref for p in phases) / len(phases)
            )
        report["runs"][run] = summary
    for pair in args.pair:
        a, b = pair.split(":")
        if a in loaded and b in loaded:
            report["pairs"][pair] = paired_families(loaded[a], loaded[b])
    for run in args.joint:
        report["joint"][run] = joint_summary(run)
    for run in args.master:
        report["masters"][run] = master_summary(run)
    for run in args.fleet_ships:
        if (RUNS / run / "columns").exists():
            report_path = RUNS / run / "run_report.json"
            data = json.loads(report_path.read_text(encoding="utf-8"))
            labels = {c["label"] for c in (data.get("master") or {}).get("selected", [])}
            records = load_run(catalogue, run, fleet_only=True)
            # the fleet's columns are the selected labels; match by asteroid set through ``columns``
            selected_sets = {
                tuple(sorted(int(x) for x in c.get("deploys", [])))
                for c in (data.get("master") or {}).get("selected", [])
            }
            fleet_records = [r for r in records if r["asteroids"] in selected_sets]
            summary = run_summary(fleet_records)
            phases = summary.pop("phase_values")
            if p75_ref is not None and phases:
                summary["collect_phase_share_le_p75_ref"] = float(
                    sum(p <= p75_ref for p in phases) / len(phases)
                )
            summary["labels"] = len(labels)
            report["fleets"][run] = summary
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "runs": list(report["runs"]),
                "masters": {k: (v or {}).get("score_kg") for k, v in report["masters"].items()},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
