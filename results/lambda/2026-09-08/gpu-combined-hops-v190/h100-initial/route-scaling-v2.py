"""Compare a bounded deterministic route search with NumPy/CUDA Lambert screening."""

import argparse
import json
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("small", "reduced"), default="small")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("output must be new and repeats must be positive")
    catalogue = load_catalogue()
    ids = build_reduced_instance(catalogue).asteroid_ids[: 60 if args.profile == "small" else 1000]
    settings = SearchSettings(
        beam_width=4,
        max_deploys=2,
        neighbours=8,
        launch_epochs=tuple(float(x) for x in C.MISSION_START_MJD + np.arange(0, 731, 90)),
        earth_leg_tofs=(600.0, 750.0, 900.0),
        hop_tofs=(90.0, 180.0),
        harvest_substitution=False,
    )
    if args.profile == "reduced":
        settings = replace(
            settings,
            beam_width=8,
            max_deploys=4,
            neighbours=24,
            earth_leg_tofs=(450.0, 600.0, 750.0, 900.0),
            hop_tofs=(90.0, 180.0, 270.0, 360.0),
        )
    report = dict(
        catalogue_sha256=catalogue.source_sha256,
        asteroid_ids=ids.tolist(),
        profile=args.profile,
        settings={
            k: (str(v) if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in asdict(settings).items()
        },
        runs={},
    )
    for backend in ["numpy", "cuda"]:
        start = time.perf_counter()
        with using_lambert_backend(backend) as gpu:
            result = RouteSearch(catalogue, ids, settings).run()
            cold = time.perf_counter() - start
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = RouteSearch(catalogue, ids, settings).run()
                times.append(time.perf_counter() - start)
            record = dict(
                seconds=statistics.median(times),
                cold_seconds=cold,
                warm_seconds=times,
                expansions=result.expansions,
                reported_lambert_evaluations=result.lambert_evaluations,
                actual_cuda_branches=gpu.evaluations if gpu else None,
                gpu_batches=gpu.batches if gpu else None,
                candidates=[candidate.summary() for candidate in result.candidates],
            )
            report["runs"][backend] = record
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            backend,
            {k: v for k, v in record.items() if k != "candidates"},
            len(result.candidates),
            flush=True,
        )
    cpu, gpu = report["runs"]["numpy"], report["runs"]["cuda"]
    assert cpu["candidates"] and len(cpu["candidates"]) == len(gpu["candidates"])
    assert cpu["expansions"] == gpu["expansions"]
    for a, b in zip(cpu["candidates"], gpu["candidates"], strict=True):
        for key in [
            "asteroids",
            "deploy_epochs",
            "collect_epochs",
            "collected_mass_kg",
            "launch_epoch",
            "earth_return_epoch",
        ]:
            assert a[key] == b[key], (key, a[key], b[key])
        for key in ["propellant_proxy_kg", "final_mass_proxy_kg"]:
            assert abs(a[key] - b[key]) < 1e-6, (key, a[key], b[key])
    report.update(
        complete=True,
        speedup=cpu["seconds"] / gpu["seconds"],
        fleet_score_changed=False,
        scope=(
            "Lambert proxy candidate search only; candidates still require "
            "low-thrust refinement and independent certification."
        ),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("speedup", report["speedup"], flush=True)


if __name__ == "__main__":
    main()
