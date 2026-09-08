"""Pin policy, route-disjoint split and existing evidence before new queue outcomes."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
CONTROLS = ROOT / "build/performance/completion-native-controls-v621"
HOST = ROOT / "build/performance/completion-adapter-cpu-v621b"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert sha(HOST / "report.json") == (
        "2e7175e8f818d11987fc240b5511c54b498f5485798501703c03e0906359bc8a"
    )
    assert sha(CONTROLS / "fixtures.json") == (
        "bd2581b1a7034961adce09f063a5bf57610c3636d4cd95591dcdbd36dab97713"
    )
    (KIT / "inputs").mkdir(exist_ok=False)
    (KIT / "source").mkdir(exist_ok=False)
    source_index = {}
    for name, digest in read(HOST / "report.json")["source_sha256"].items():
        if not name.startswith("src/"):
            continue
        origin = HOST / "source" / name
        assert sha(origin) == digest
        dest = KIT / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, dest)
        source_index[name] = digest
    for name in (
        "src/spacepdhcg/gtoc12/refinement_admission.py",
        "tests/test_gtoc12_refinement_admission.py",
    ):
        dest = KIT / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
        source_index[name] = sha(dest)
    files = [
        (CONTROLS / name, name)
        for name in (
            "fixtures.json",
            "features.json",
            "residuals.json",
            "residual-audit.json",
            "input-provenance.json",
        )
    ]
    inventory = read(CONTROLS / "inputs/incumbent-inventory.json")["routes"]
    for name in (
        "incumbent-inventory.json",
        "hop_inflation_fit.json",
        "v616-report.json",
        "selection-v619.json",
        "completion-oracle-v619.json",
    ):
        files.append((CONTROLS / "inputs" / name, name))
    for row in inventory:
        origin = CONTROLS / "inputs" / f"ship-{row['ship']:02}.json"
        assert sha(origin) == row["sha256"]
        files.append((origin, origin.name))
    for name in ("Result.txt", "fresh-verification.json", "provenance.json"):
        files.append(
            (
                ROOT / "build/performance/incident-window-substitution-v604/inputs" / name,
                "v604-" + name,
            )
        )
    files.append(
        (ROOT / "build/performance/truth-set-v606/fixed_refine.py", "fixed_refine-v606.py")
    )
    provenance = {}
    for origin, name in files:
        shutil.copyfile(origin, KIT / "inputs" / name)
        provenance[name] = {"source": origin.relative_to(ROOT).as_posix(), "sha256": sha(origin)}
    verifier = read(KIT / "inputs/v604-fresh-verification.json")
    assert sha(KIT / "inputs/v604-Result.txt") == verifier["solution_sha256"]
    assert verifier["independent"]["ok"] and verifier["official"]["ok"]
    footprints = {
        row["ship"]: set(read(KIT / "inputs" / f"ship-{row['ship']:02}.json")["asteroids"])
        for row in inventory
    }
    overlap = [
        (a, b) for a in footprints for b in footprints if a < b and footprints[a] & footprints[b]
    ]
    assert not overlap
    development = [1, 4, 7, 10, 23]
    held_out = [
        x["ship"] for x in inventory if x["independent_inventory"] and x["ship"] not in development
    ]
    assert len(held_out) == 17
    write(
        KIT / "selection.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "development_ships": development,
            "queue_held_out_ships": held_out,
            "excluded": [{"ship": 3, "reason": "independent_inventory=false"}],
            "cross_route_asteroid_overlaps": overlap,
            "route_source_sha256": {str(x["ship"]): x["sha256"] for x in inventory},
            "split_scope": (
                "All historical outcomes have been exposed in prior audits. Held-out means "
                "excluded from queue policy development; not unseen data or held out from "
                "the old inflation fit."
            ),
            "development_saved_traces": 20,
            "held_out_scalar_finish_cpu_calls": 34,
            "held_out_prefix": "archived measured deployment prefix",
            "models": ["v616_no_fit", "existing_fit_no_refit"],
            "policy": {"retained_requests": 48, "refinement_budget": 4, "uncertainty_budget": 2},
            "priority": (
                "ascending final proxy deficit, descending weighted gain, "
                "exact prescription SHA; no learned deficit cutoff"
            ),
            "replay_controls": (
                "equal-score immutable replacements may be inventoried; "
                "queue cannot claim/launch them"
            ),
            "actual_refinements": 0,
            "GPU_calls": 0,
            "Lambert_calls": 0,
            "new_geometry_calls": 0,
            "model_refits": 0,
            "fresh_certifications": 0,
        },
    )
    write(KIT / "source-sha256.json", source_index)
    write(KIT / "input-provenance.json", provenance)
    write(
        KIT / "preparation.json",
        {
            "source_base": (
                "frozen final-b v621 adapter source; isolated new admission helper/test overlay"
            ),
            "source_sha256": sha(KIT / "source-sha256.json"),
            "selection_sha256": sha(KIT / "selection.json"),
            "input_provenance_sha256": sha(KIT / "input-provenance.json"),
            "policy_source_sha256": source_index["src/spacepdhcg/gtoc12/refinement_admission.py"],
            "historical_full_fleet_Result_sha256": verifier["solution_sha256"],
            "archival_verification": (
                "Exact old Result/checker evidence retained; "
                "no new trajectory or fleet verification in v622"
            ),
        },
    )
    print(
        json.dumps(
            {
                "source_files": len(source_index),
                "inputs": len(files),
                "development": development,
                "held_out": held_out,
                "selection_sha256": sha(KIT / "selection.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
