"""Audit saved v622 readbacks and accounting; no model/geometry/solver evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

KIT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def main():
    report = read(KIT / "output/report.json")
    selection = read(KIT / "selection.json")
    assert report["selection_sha256"] == sha(KIT / "selection.json")
    paths = sorted((KIT / "output").glob("ship*.json"))
    assert len(paths) == 54
    rows = [read(path) for path in paths]
    summaries = {}
    arithmetic_checks = 0
    for row in rows:
        observed = row["proxy_readback"]
        assert hashlib.sha256(canonical(observed)).hexdigest() == row["readback_sha256"]
        ship = row["ship"]
        route_path = KIT / "inputs" / f"ship-{ship:02}.json"
        assert sha(route_path) == row["input_route_sha256"]
        archive = read(route_path)
        assert archive["certified"] and archive["master_certified"] and not archive["failures"]
        assert all(x["certified"] for x in archive["legs"])
        plan = archive["plan"]
        assert plan["collected_mass_kg"] == archive["collected_mass_kg"]
        raw = sum(archive["collected_mass_kg"].values())
        assert math.isclose(raw, row["raw_cargo_kg"], abs_tol=2e-10)
        assert row["archived_final_dry_margin_kg"] > 0
        if row["provenance"]["kind"].startswith("saved"):
            assert ship in selection["development_ships"]
            continue
        assert ship in selection["queue_held_out_ships"]
        split = max(i for i, leg in enumerate(plan["legs"]) if leg["role"] == "deploy_hop") + 1
        prefix_end = plan["legs"][split - 1]
        measured = {(x["from"], x["to"], x["t0"], x["tf"]): x for x in archive["legs"]}
        key = lambda x: (x["from"], x["to"], x["t0"], x["tf"])
        running = measured[key(prefix_end)]["mass_after"] - 40.0
        assert running == observed["prefix_mass_kg"]
        flights = [x for x in plan["legs"][split:] if x["role"] != "camp"]
        assert len(flights) == len(observed["forward_trace"])
        picked = {}
        for leg, trace in zip(flights, observed["forward_trace"], strict=True):
            source = str(leg["from"])
            if abs(plan["collect_epochs"][source] - leg["t0"]) < 1e-6:
                picked[source] = archive["collected_mass_kg"][source]
                running += picked[source]
            assert trace["authority_pass"]
            assert math.isclose(running, trace["mass_before_kg"], abs_tol=2e-10)
            running -= trace["propellant_kg"]
            assert math.isclose(running, trace["mass_after_kg"], abs_tol=2e-10)
            assert trace["inflation_spent"] >= 0
            arithmetic_checks += 2
        assert picked == archive["collected_mass_kg"]
        assert math.isclose(running, observed["result"]["final_mass"], abs_tol=2e-10)
        assert (running < 500.0 + raw) == bool(row["proxy_failure"])
        predicted = sum(x["propellant_kg"] for x in observed["forward_trace"])
        actual = sum(measured[key(x)]["propellant_kg"] for x in flights)
        bias = predicted - actual
        proxy_margin = running - (500.0 + raw)
        assert math.isclose(row["archived_final_dry_margin_kg"] - proxy_margin, bias, abs_tol=1e-7)
        summaries[row["id"]] = {
            "predicted_collection_and_return_fuel_kg": predicted,
            "archived_measured_collection_and_return_fuel_kg": actual,
            "sequential_completion_fuel_error_kg": bias,
            "archived_dry_margin_kg": row["archived_final_dry_margin_kg"],
            "proxy_dry_plus_cargo_margin_kg": proxy_margin,
        }
    groups = []
    for group in report["queue_groups"]:
        ids = {x["id"] for x in group["dispositions"]}
        group_rows = [r for r in rows if r["id"] in ids]
        eligible = [r for r in group_rows if not r["admission_blocker"]]
        ranked = sorted(eligible, key=lambda x: (x["proxy_deficit_kg"], x["request_sha256"]))
        uncertain = [r for r in ranked if r["proxy_failure"]][:2]
        regular = [r for r in ranked if not r["proxy_failure"]][: 4 - len(uncertain)]
        chosen = [r["id"] for r in regular + uncertain]
        assert set(chosen) == set(group["selected"])
        assert len(chosen) <= 4 and len(uncertain) <= 2
        groups.append(
            {
                "group": group["group"],
                "model": group["model"],
                "prefix": group["prefix"],
                "priority_order": chosen,
                "selected_uncertain": len(uncertain),
            }
        )
    assert len(summaries) == 34
    findings = {
        "passed": True,
        "report_sha256": sha(KIT / "output/report.json"),
        "source_hashes_checked": True,
        "raw_readback_identities": 54,
        "saved_scalar_bookkeeping_checks": arithmetic_checks,
        "fresh_model_evaluations_in_audit": 0,
        "producer_counts": dict(Counter(r["provenance"]["kind"] for r in rows)),
        "ordered_shortlists": groups,
        "held_out_sequential_fuel_residuals": summaries,
        "scope": "Bookkeeping and saved-readback audit; no model re-evaluation or certification.",
    }
    with (KIT / "readback-audit.json").open("x") as stream:
        json.dump(findings, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"passed": True, "readbacks": len(rows), "bookkeeping_checks": arithmetic_checks}
        )
    )


if __name__ == "__main__":
    main()
