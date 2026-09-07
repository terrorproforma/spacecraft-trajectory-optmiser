"""Paired route search: GPU transfer screening with CPU versus GPU collection option selection."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12 import lambert
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.reduced_instance import build_reduced_instance
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("new output and positive repeats required")
    catalogue = load_catalogue()
    pool = build_reduced_instance(catalogue).asteroid_ids
    settings = SearchSettings(
        beam_width=8,
        max_deploys=4,
        neighbours=24,
        launch_epochs=tuple(float(x) for x in C.MISSION_START_MJD + np.arange(0, 731, 90)),
        earth_leg_tofs=(450.0, 600.0, 750.0, 900.0),
        hop_tofs=(90.0, 180.0, 270.0, 360.0),
        harvest_substitution=False,
    )
    modes = {"cpu_collection": [], "gpu_collection": []}
    outputs = {}
    queries = {}
    original = lambert.cuda_select_collection
    with lambert.using_lambert_backend("cuda") as gpu:
        try:
            for repeat in range(args.repeats + 1):
                for mode in list(modes) if repeat % 2 else list(reversed(modes)):
                    lambert.cuda_select_collection = (
                        original if mode == "gpu_collection" else lambda *a, **k: None
                    )
                    before = gpu.telemetry.get("completed_collection_options", 0)
                    start = time.perf_counter()
                    result = RouteSearch(catalogue, pool, settings).run()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        modes[mode].append(elapsed)
                    outputs[mode] = result
                    queries[mode] = gpu.telemetry.get("completed_collection_options", 0) - before
        finally:
            lambert.cuda_select_collection = original
    a, b = outputs.values()
    assert a.lambert_evaluations == b.lambert_evaluations
    assert len(a.candidates) == len(b.candidates) and a.candidates
    for x, y in zip(a.candidates, b.candidates, strict=True):
        assert x.summary() == y.summary()
    medians = {k: statistics.median(v) for k, v in modes.items()}
    report = dict(
        complete=True,
        seconds=modes,
        median_seconds=medians,
        speedup=medians["cpu_collection"] / medians["gpu_collection"],
        collection_options=queries,
        branches_per_search=b.lambert_evaluations,
        candidates=len(b.candidates),
        catalogue_sha256=catalogue.source_sha256,
        runtime_sha256=hashlib.sha256(
            Path(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]).read_bytes()
        ).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
