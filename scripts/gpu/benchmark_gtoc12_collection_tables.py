"""Paired cold collection table builds with CPU or CUDA ephemeris preparation."""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend


def run(source, destination):
    fixtures = json.loads(Path(source).read_text())["fixtures"]
    catalogue = load_catalogue()
    native = collectdp.cuda_leg_table
    rows = []
    with using_lambert_backend("cuda"):
        try:
            for size in [2, 4, 8]:
                fixture = fixtures[str(size)]
                ids = [int(a) for a, _ in fixture["deployed"]]
                for repeat in range(6):
                    results = {}
                    for mode in ["cpu", "cuda"] if repeat % 2 == 0 else ["cuda", "cpu"]:
                        collectdp.cuda_leg_table = native if mode == "cuda" else lambda *args: None
                        start = time.perf_counter()
                        table = collectdp.CollectPairTable(
                            catalogue, collectdp.CollectDPSettings(**fixture["settings"])
                        )
                        values = [table.earth_return(a) for a in ids]
                        values.extend(table.hop(a, b) for a in ids for b in ids if a != b)
                        elapsed = time.perf_counter() - start
                        results[mode] = values
                        rows.append(dict(asteroids=size, repeat=repeat, mode=mode, seconds=elapsed))
                    for cpu, cuda in zip(results["cpu"], results["cuda"], strict=True):
                        np.testing.assert_array_equal(np.isfinite(cpu), np.isfinite(cuda))
                        np.testing.assert_allclose(cpu, cuda, rtol=2e-7, atol=2e-7)
        finally:
            collectdp.cuda_leg_table = native
    medians = {
        str(size): {
            mode: statistics.median(
                r["seconds"]
                for r in rows
                if r["asteroids"] == size and r["mode"] == mode and r["repeat"] > 0
            )
            for mode in ["cpu", "cuda"]
        }
        for size in [2, 4, 8]
    }
    Path(destination).write_text(
        json.dumps(
            dict(
                medians=medians,
                rows=rows,
                scope=(
                    "Cold tables, warm CUDA runtime. Six alternating pairs; discard first pair. "
                    "Both modes use CUDA Lambert. Not certified mission throughput."
                ),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    run(*sys.argv[1:])
