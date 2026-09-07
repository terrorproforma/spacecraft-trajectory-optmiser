"""Paired CPU/CUDA complete collection tours with retained pair tables."""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12.collectdp import CollectDPSettings, CollectPairTable, plan_collect_tour
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend


def run(source, output):
    fixtures = json.loads(source.read_text())["fixtures"]
    cat = load_catalogue()
    rows = []
    with using_lambert_backend("cuda") as gpu:
        for size in ["3", "6", "9"]:
            f = fixtures[size]
            assert not f["settings"]["inflation_fit"] and not f["settings"]["harvest_phase"]
            assert not f["return_sweeps"]
            table = CollectPairTable(cat, CollectDPSettings(**f["settings"]))
            options = dict(f["kwargs"])
            options["weights"] = {int(k): v for k, v in (options.get("weights") or {}).items()}
            options["banned_pairs"] = {tuple(pair) for pair in options.get("banned_pairs", [])}
            reference = None
            for repeat in range(6):
                for mode in ["cpu", "cuda"] if repeat % 2 == 0 else ["cuda", "cpu"]:
                    gpu.collect_dp_cuda = mode == "cuda"
                    start = time.perf_counter()
                    result = plan_collect_tour(
                        table,
                        f["deployed"],
                        f["camp"],
                        f["camp_epoch"],
                        f["mass_after_deploys"],
                        **options,
                    )
                    seconds = time.perf_counter() - start
                    record = None if result is None else dataclasses.asdict(result)
                    if reference is None:
                        reference = (record,)
                    expected = reference[0]
                    assert (record is None) == (expected is None)
                    if record is not None:
                        for field in [
                            "order",
                            "collect_epochs",
                            "hops",
                            "reposition",
                            "return_departure",
                            "return_tof",
                            "dp_states",
                        ]:
                            assert record[field] == expected[field], (size, field)
                        for field in ["objective_kg", "propellant_proxy_kg", "hop_propellant_kg"]:
                            np.testing.assert_allclose(
                                record[field], expected[field], rtol=0, atol=1e-7
                            )
                    rows.append(
                        dict(
                            asteroids=int(size),
                            repeat=repeat,
                            mode=mode,
                            seconds=seconds,
                            objective_kg=None if result is None else result.objective_kg,
                        )
                    )
            print(size, "asteroids paired comparison passed", flush=True)
    medians = {
        size: {
            mode: float(
                np.median(
                    [
                        r["seconds"]
                        for r in rows
                        if r["asteroids"] == int(size) and r["mode"] == mode and r["repeat"] > 0
                    ]
                )
            )
            for mode in ["cpu", "cuda"]
        }
        for size in ["3", "6", "9"]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            dict(
                medians=medians,
                rows=rows,
                scope=("Six alternating pairs per size, first pair discarded; complete tours "
                       "including both mass passes, retained pair tables. "
                       "No CPU state-transition fallback in CUDA mode."),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
