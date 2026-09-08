"""Freeze the matched historical completion benchmark; do not execute costing/GPU."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
CONTROL = ROOT / "build/performance/completion-native-controls-v621"
HOST = ROOT / "build/performance/completion-adapter-cpu-v621b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert (
        sha(HOST / "report.json")
        == "2e7175e8f818d11987fc240b5511c54b498f5485798501703c03e0906359bc8a"
    )
    assert (
        sha(CONTROL / "fixtures.json")
        == "bd2581b1a7034961adce09f063a5bf57610c3636d4cd95591dcdbd36dab97713"
    )
    report = read(HOST / "report.json")
    (KIT / "source").mkdir(exist_ok=False)
    for name, digest in report["source_sha256"].items():
        source = HOST / "source" / name
        assert sha(source) == digest
        target = KIT / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    inputs = KIT / "inputs"
    inputs.mkdir(exist_ok=False)
    files = [
        (CONTROL / name, name) for name in ("fixtures.json", "features.json", "gpu-profile.json")
    ]
    files += [
        (CONTROL / "inputs" / name, name)
        for name in (
            "ship-01.json",
            "ship-04.json",
            "ship-07.json",
            "ship-10.json",
            "ship-23.json",
            "v616-report.json",
            "hop_inflation_fit.json",
        )
    ]
    files += [
        (HOST / "report.json", "host-validation.json"),
        (ROOT / "build/performance/completion-fullcore-v621/manifest.json", "native-manifest.json"),
    ]
    for source, name in files:
        shutil.copyfile(source, inputs / name)
    shutil.copyfile(CONTROL / "gpu_batch.py", KIT / "control_comparator.py")
    write(KIT / "source-sha256.json", report["source_sha256"])
    write(KIT / "input-sha256.json", {p.name: sha(p) for p in sorted(inputs.iterdir())})
    historical = read(inputs / "fixtures.json")["historical"]
    groups = []
    sizes = [4, 24, 48, 64, 256, 1024]
    for model in ("v616_no_fit", "existing_fit_no_refit"):
        pool = [i for i, row in enumerate(historical) if row["provenance"]["model"] == model]
        pool.sort(key=lambda i: (historical[i]["expected"]["failure_name"] != "ok", i))
        assert (
            len(pool) == 10
            and sum(historical[i]["expected"]["failure_name"] == "ok" for i in pool) == 1
        )
        for size in sizes:
            indices = [pool[i % len(pool)] for i in range(size)]
            groups.append(
                {
                    "id": f"{model}-n{size}",
                    "model": model,
                    "size": size,
                    "fixture_indices": indices,
                    "unique_control_count": len(set(indices)),
                    "expected_accepted": sum(
                        historical[i]["expected"]["failure_name"] == "ok" for i in indices
                    ),
                    "role": "observed_v616_shortlist"
                    if size == 24
                    else "source_default_shortlist"
                    if size == 48
                    else "schedule_batch_size_only"
                    if size == 4
                    else "prospective_scaling_batch",
                }
            )
    total = sum(g["size"] for g in groups) * 5
    assert total == 14200
    write(
        KIT / "plan.json",
        {
            "scope": (
                "Matched production completion costing on repeated historical requests, "
                "no mission speedup claim"
            ),
            "historical_fleet": "v595/v616, not the current fleet",
            "methods": ["RouteSearch._finish_cpu", "RouteSearch._finish_many"],
            "model_provenance": (
                "Both use CollectPairTable metadata, cached saved pair features; "
                "no generated geometry"
            ),
            "groups": groups,
            "source_size_evidence": {
                "four": (
                    "search.py TOUR_MODES has4 entries; benchmark uses archival DP requests "
                    "at this size, not a measured _complete invocation"
                ),
                "twenty_four": "v616 savedsettings beam_width24/chain_tour_candidates24",
                "forty_eight": (
                    "frozen SearchSettings.chain_tour_candidates default48; "
                    "_filter can yield fewer requests"
                ),
                "large": (
                    "64/256/1024 are prospective scaling probes, not observed live shortlist sizes"
                ),
            },
            "first_call": (
                "One call per backend/group; GPU fresh workspace, "
                "only first group may include process CUDA initialization"
            ),
            "warm_order": ["cpu", "gpu", "gpu", "cpu", "gpu", "cpu", "cpu", "gpu"],
            "gpu_evaluate_calls": 60,
            "gpu_candidate_evaluations": total,
            "cpu_candidate_evaluations": total,
            "unique_historical_controls": 20,
            "maximum_worker_seconds": 180,
            "automatic_retries": 0,
            "construction_and_validation": "Measured separately, excluded from method timers",
            "throughput_unit": (
                "proxy candidate evaluations/second; "
                "repeated controls are not distinct trajectories/solutions"
            ),
            "new_geometry_calls": 0,
            "Lambert_calls": 0,
            "mission_searches": 0,
            "refinements": 0,
            "new_physics_certifications": 0,
            "promotions": 0,
        },
    )
    print(
        json.dumps(
            {
                "groups": len(groups),
                "gpu_calls_planned": 60,
                "candidate_evaluations_per_backend": total,
                "GPU_calls_executed": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
