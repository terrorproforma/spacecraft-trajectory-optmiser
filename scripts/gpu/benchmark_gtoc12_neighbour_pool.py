"""Verify and time neighbour selection over the full 60,000-asteroid catalogue."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend
from spacepdhcg.gtoc12.search import RouteSearch, SearchSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    catalogue = load_catalogue()
    settings = SearchSettings(neighbours=48, hop_tofs=(60.0, 90.0, 180.0, 360.0, 480.0))
    cases = [(1, 64328.0), (12000, 65500.0), (30000, 66800.0), (50000, 68000.0)]
    report = dict(
        catalogue_sha256=catalogue.source_sha256,
        pool_size=len(catalogue),
        cases=cases,
        runs={},
        runtime_sha256=hashlib.sha256(
            Path(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]).read_bytes()
        ).hexdigest(),
    )
    outputs = {}
    for backend in ["numpy", "cuda"]:
        with using_lambert_backend(backend):
            search = RouteSearch(catalogue, catalogue.ids, settings)
            times = []
            for _ in range(4):
                start = time.perf_counter()
                outputs[backend] = [search.candidates(source, epoch) for source, epoch in cases]
                times.append(time.perf_counter() - start)
            report["runs"][backend] = dict(
                cold_seconds=times[0],
                warm_seconds=times[1:],
                median_seconds=statistics.median(times[1:]),
            )
    for a, b in zip(outputs["numpy"], outputs["cuda"], strict=True):
        np.testing.assert_array_equal(a, b)
    report.update(
        complete=True,
        selected_ids=[a.tolist() for a in outputs["cuda"]],
        speedup=report["runs"]["numpy"]["median_seconds"]
        / report["runs"]["cuda"]["median_seconds"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "selected_ids"}, indent=2))


if __name__ == "__main__":
    main()
