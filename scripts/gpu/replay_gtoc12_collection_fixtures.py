"""Replay real collection-DP inputs and audit mass-dependent fraction caching.

Requires the pinned catalogue and an explicit CUDA Lambert library. These are
proxy collection tours, not independently certified low-thrust missions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from spacepdhcg.gtoc12 import collectdp
from spacepdhcg.gtoc12.data import load_catalogue
from spacepdhcg.gtoc12.lambert import using_lambert_backend


def run(source: Path, output: Path, reference: str, mass_key: str) -> None:
    fixtures = json.loads(source.read_text())["fixtures"]
    catalogue = load_catalogue()
    original = collectdp._solve_collect_dp
    audit = {"fraction_calls": 0, "different_mass_hits": 0, "max_mass_difference_kg": 0.0}

    def instrument(*args, **kwargs):
        fraction = args[14]
        cells = dict(zip(fraction.__code__.co_freevars, fraction.__closure__, strict=True))
        cache = cells["fractions"].cell_contents
        masses = {}

        def observed(j, successor, mass):
            key = (j, successor, round(mass) if mass_key == "rounded" else mass)
            audit["fraction_calls"] += 1
            if key not in cache:
                masses[key] = mass
            elif key in masses and masses[key] != mass:
                audit["different_mass_hits"] += 1
                audit["max_mass_difference_kg"] = max(
                    audit["max_mass_difference_kg"], abs(masses[key] - mass)
                )
            return fraction(j, successor, mass)

        return original(*args[:14], observed, *args[15:], **kwargs)

    rows = []
    collectdp._solve_collect_dp = instrument
    try:
        with using_lambert_backend("cuda") as gpu:
            for size, fixture in sorted(fixtures.items(), key=lambda item: int(item[0])):
                settings = fixture["settings"]
                # Refuse to silently drop calibrated models or certified return overrides.
                if (
                    settings["inflation_fit"]
                    or settings["harvest_phase"]
                    or fixture["return_sweeps"]
                ):
                    raise ValueError("This fixture needs its calibrated model/sweep payload")
                table = collectdp.CollectPairTable(
                    catalogue, collectdp.CollectDPSettings(**settings)
                )
                options = dict(fixture["kwargs"])
                if options.get("weights"):
                    options["weights"] = {int(k): v for k, v in options["weights"].items()}
                if options.get("banned_pairs"):
                    options["banned_pairs"] = {tuple(pair) for pair in options["banned_pairs"]}
                start = time.perf_counter()
                result = collectdp.plan_collect_tour(
                    table,
                    fixture["deployed"],
                    fixture["camp"],
                    fixture["camp_epoch"],
                    fixture["mass_after_deploys"],
                    **options,
                )
                seconds = time.perf_counter() - start
                archived = fixture["expected"]
                if reference == "uncached":
                    table.settings = dataclasses.replace(table.settings, fraction_cache_entries=0)
                    uncached = collectdp.plan_collect_tour(
                        table,
                        fixture["deployed"],
                        fixture["camp"],
                        fixture["camp_epoch"],
                        fixture["mass_after_deploys"],
                        **options,
                    )
                    # Normalise keys/tuples just as the archived JSON does.
                    expected = (
                        None
                        if uncached is None
                        else json.loads(json.dumps(dataclasses.asdict(uncached)))
                    )
                else:
                    expected = archived
                assert (result is None) == (expected is None), size
                if result is not None:
                    assert list(result.order) == expected["order"], size
                    assert result.collect_epochs == {
                        int(k): v for k, v in expected["collect_epochs"].items()
                    }, size
                    assert result.reposition == expected["reposition"], size
                    assert result.return_departure == expected["return_departure"], size
                    assert result.return_tof == expected["return_tof"], size
                    for field in ["objective_kg", "collected_proxy_kg", "propellant_proxy_kg"]:
                        np.testing.assert_allclose(
                            getattr(result, field), expected[field], rtol=0, atol=1e-6
                        )
                    np.testing.assert_allclose(result.hops, expected["hops"], rtol=2e-9, atol=2e-8)
                rows.append(
                    {
                        "asteroids": int(size),
                        "seconds": seconds,
                        "archived_objective_delta_kg": (
                            result.objective_kg - archived["objective_kg"]
                            if result is not None and archived is not None
                            else None
                        ),
                        "result": None if result is None else dataclasses.asdict(result),
                    }
                )
                print(
                    size,
                    "asteroids: matching",
                    "infeasible" if result is None else "tour",
                    flush=True,
                )
            telemetry = dict(gpu.telemetry)
    finally:
        collectdp._solve_collect_dp = original
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "rows": rows,
                "cache_audit": audit,
                "screening": telemetry,
                "reference": reference,
                "mass_key": mass_key,
                "scope": (
                    "Single replay per captured input; includes cold pair-table builds and "
                    "cache instrumentation. Not a throughput benchmark."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", choices=["uncached", "archived"], default="uncached")
    parser.add_argument("--mass-key", choices=["exact", "rounded"], default="exact")
    args = parser.parse_args()
    run(args.source, args.output, args.reference, args.mass_key)
