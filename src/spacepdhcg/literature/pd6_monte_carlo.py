"""P1-D-MC: independent 6-DoF powered-descent batch (Chari et al., 2024).

Source: Chari et al., "Fast Monte Carlo Analysis for 6-DoF Powered-Descent Guidance via
GPU-Accelerated Sequential Convex Programming", AIAA SciTech 2024, DOI 10.2514/6.2024-1762
(arXiv:2404.18034).  The published dispersion samples the initial position components from
``U(6, 9)``, ``U(3, 6)``, ``U(1, 2)`` (non-dimensional).  This module

* draws and commits the sampled initial positions with fixed seeds (one seed per batch size,
  so the batch of 16 is not a prefix of the batch of 256 unless the seeds coincide);
* runs every sample through the independent CPU free-final-time 6-DoF SCvx of
  :mod:`spacepdhcg.literature.pd6_szmuk_2018` (the same vehicle model family as the paper's
  Szmuk-style formulation) and reports convergence probability, objective/violation
  distributions, and accepted trajectories per second.

This is an *independent batch*: every trajectory is optimised on its own.  It is not a coupled
robust-scenario optimisation and must not be described as one.
"""

from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from spacepdhcg.literature.pd6_szmuk_2018 import Szmuk2018Parameters, reproduce

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_PATH = REPOSITORY_ROOT / "benchmarks" / "literature" / "chari_2024_initial_positions.json"

PUBLISHED_DISTRIBUTION = ((6.0, 9.0), (3.0, 6.0), (1.0, 2.0))
BATCH_SEEDS = {
    1: 20240101,
    16: 20240116,
    64: 20240164,
    256: 20240256,
    1024: 20241024,
    2048: 20242048,
}


def sample_initial_positions(batch_size: int, seed: int | None = None) -> np.ndarray:
    seed = BATCH_SEEDS[batch_size] if seed is None else seed
    rng = np.random.default_rng(seed)
    columns = [rng.uniform(low, high, size=batch_size) for low, high in PUBLISHED_DISTRIBUTION]
    return np.column_stack(columns)


def build_sample_file(batch_sizes: tuple[int, ...] = (1, 16, 64, 256)) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "source": (
            "Chari et al. 2024, DOI 10.2514/6.2024-1762; "
            "initial position ~ [U(6,9), U(3,6), U(1,2)]"
        ),
        "generator": (
            "numpy.random.default_rng(seed).uniform per component, column order (up, east, north)"
        ),
        "batches": {
            str(size): {
                "seed": BATCH_SEEDS[size],
                "positions": sample_initial_positions(size).tolist(),
            }
            for size in batch_sizes
        },
    }


def load_samples(path: Path = SAMPLES_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _solve_one(arguments: tuple[list[float], dict[str, Any]]) -> dict[str, Any]:
    position, overrides = arguments
    base = Szmuk2018Parameters()
    payload = {f: getattr(base, f) for f in base.__dataclass_fields__}
    payload.update(overrides)
    payload["initial_position"] = tuple(float(v) for v in position)
    parameters = Szmuk2018Parameters(**payload)
    start = perf_counter()
    result = reproduce(parameters, tf_guess=float(overrides.get("_tf_guess", 5.0)), substeps=4)
    wall = perf_counter() - start
    return {
        "initial_position": list(position),
        "status": result.outcome.status,
        "time_of_flight": result.time_of_flight,
        "fuel_used": result.fuel_used,
        "iterations": result.iterations_to_converge,
        "replay_defect_inf": result.outcome.replay_defect_inf,
        "max_path_violation": result.max_path_violation,
        "wall_seconds": wall,
    }


@dataclass(slots=True)
class BatchSummary:
    batch_size: int
    seed: int
    solved: int
    converged: int
    convergence_probability: float
    accepted: int
    accepted_probability: float
    accepted_trajectories_per_second: float
    wall_seconds: float
    workers: int
    tof_median: float | None
    tof_iqr: tuple[float, float] | None
    fuel_median: float | None
    violation_max: float | None
    defect_max: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_batch(
    batch_size: int,
    *,
    max_iterations: int = 25,
    workers: int | None = None,
    accept_defect: float = 1.0e-5,
    accept_violation: float = 1.0e-6,
    samples: dict[str, Any] | None = None,
) -> tuple[BatchSummary, list[dict[str, Any]]]:
    document = samples or load_samples()
    batch = document["batches"][str(batch_size)]
    positions = batch["positions"]
    overrides = {"max_iterations": max_iterations, "virtual_tolerance": 1.0e-8}
    worker_count = workers or max(1, min(len(positions), (os.cpu_count() or 2) - 1))
    start = perf_counter()
    if worker_count == 1 or len(positions) == 1:
        results = [_solve_one((p, overrides)) for p in positions]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            results = list(pool.map(_solve_one, [(p, overrides) for p in positions]))
    wall = perf_counter() - start
    converged = [r for r in results if r["status"] == "converged"]
    accepted = [
        r
        for r in results
        if r["replay_defect_inf"] <= accept_defect and r["max_path_violation"] <= accept_violation
    ]
    tofs = sorted(r["time_of_flight"] for r in accepted)
    fuels = [r["fuel_used"] for r in accepted]
    summary = BatchSummary(
        batch_size=batch_size,
        seed=batch["seed"],
        solved=len(results),
        converged=len(converged),
        convergence_probability=len(converged) / len(results),
        accepted=len(accepted),
        accepted_probability=len(accepted) / len(results),
        accepted_trajectories_per_second=len(accepted) / wall if wall > 0 else float("nan"),
        wall_seconds=wall,
        workers=worker_count,
        tof_median=statistics.median(tofs) if tofs else None,
        tof_iqr=(float(np.percentile(tofs, 25)), float(np.percentile(tofs, 75))) if tofs else None,
        fuel_median=statistics.median(fuels) if fuels else None,
        violation_max=max(r["max_path_violation"] for r in results) if results else None,
        defect_max=max(r["replay_defect_inf"] for r in results) if results else None,
    )
    return summary, results


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    batch_sizes = tuple(
        int(v) for v in options.get("batch_sizes", document.get("batch_sizes", [1, 16, 64]))
    )
    max_iterations = int(options.get("max_iterations", document.get("max_outer_iterations", 25)))
    workers = options.get("workers")
    summaries: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for size in batch_sizes:
        summary, results = run_batch(size, max_iterations=max_iterations, workers=workers)
        summaries[str(size)] = summary.as_dict()
        details[str(size)] = results
    # There is no published objective to reproduce; the batch counts as reproduced only when the
    # independent CPU solver accepts (replay defect <= 1e-5, violation <= 1e-6) at least 90 % of
    # every committed batch within the published 25-iteration budget.
    acceptable = all(s["accepted_probability"] >= 0.9 for s in summaries.values())
    return {
        "target_id": document["id"],
        "status": "reproduced" if acceptable else "gap",
        "published": document.get("published", {}),
        "measured": {
            "cpu_independent_batch": summaries,
            "gpu_persistent_batch": {
                "status": "blocked",
                "reason": document.get("gpu_blocker"),
            },
        },
        "gap": {},
        "labels": {
            "published.distribution": "published-reference",
            "published.batch_size_256": "published-reference",
            "measured.cpu_independent_batch": "measured-local",
        },
        "envelope": {
            "solver": (
                "independent CPU free-final-time 6-DoF SCvx (Szmuk 2018 vehicle model), "
                "one process per trajectory"
            ),
            "acceptance": "replay defect <= 1e-5 and path violation <= 1e-6",
        },
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": [
            (
                "independent batch: no shared controls, no non-anticipativity constraints; not "
                "robust optimisation"
            ),
            (
                "the paper's vehicle parameter table is not reproduced here; the Szmuk 2018 Table "
                "1 vehicle is used with the published position dispersion"
            ),
        ],
        "details": details,
    }
