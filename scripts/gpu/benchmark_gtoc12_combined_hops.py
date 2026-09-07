"""Paired comparison of GPU roots + CPU costs against combined GPU screening."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import constants as C
from spacepdhcg.gtoc12 import screening
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.ephemeris import asteroid_state, earth_state
from spacepdhcg.gtoc12.lambert import using_lambert_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    catalogue = load_catalogue()
    n = 8192
    ids = np.linspace(1, 60000, n, dtype=int)
    epochs = C.MISSION_START_MJD + np.linspace(0, 730, n)
    tofs = 300 + (np.arange(n) * 71 % 601)
    r1, v1 = earth_state(epochs)
    r2, v2 = asteroid_state(catalogue, ids, epochs + tofs)
    original = screening.cuda_screen_hops
    times = {"gpu_roots_cpu_costs": [], "combined_gpu": []}
    outputs = {}
    counters = {}
    with using_lambert_backend("cuda") as gpu:
        try:
            # Alternate order, retain the same workspace and exclude first-use warmup.
            for repeat in range(7):
                for mode in list(times) if repeat % 2 else list(reversed(times)):
                    screening.cuda_screen_hops = (
                        original if mode == "combined_gpu" else (lambda *args, **kwargs: None)
                    )
                    before = gpu.batches
                    start = time.perf_counter()
                    result = screening.lambert_hops(
                        r1, v1, r2, v2, epochs, tofs, departure_allowance_km_s=6
                    )
                    elapsed = time.perf_counter() - start
                    if repeat:
                        times[mode].append(elapsed)
                    outputs[mode] = result
                    counters[mode] = gpu.batches - before
        finally:
            screening.cuda_screen_hops = original
    a, b = outputs.values()
    np.testing.assert_array_equal(a.feasible, b.feasible)
    for name in ["departure_delta_v", "arrival_delta_v", "departure_velocity", "arrival_velocity"]:
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), atol=1e-10, rtol=1e-12)
    medians = {mode: statistics.median(values) for mode, values in times.items()}
    report = dict(
        complete=True,
        transfers=n,
        seconds=times,
        median_seconds=medians,
        batches_per_call=counters,
        speedup=medians["gpu_roots_cpu_costs"] / medians["combined_gpu"],
        runtime_sha256=hashlib.sha256(
            Path(os.environ["SPACEPDHCG_GTOC12_CUDA_LIBRARY"]).read_bytes()
        ).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
