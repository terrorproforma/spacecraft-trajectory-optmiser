"""Exactly one bounded CUDA-operator generation; no solver or fleet mutation."""

import argparse
import dataclasses
import json
import os
import time
import traceback
from pathlib import Path

import common


def persist_pool(directory, plans, weights, baseline, beam_score, certified_keys):
    """Preserve every ordered generated plan before the real shortlist is evaluated."""
    from domain import attrition, row_for

    rows = [row_for(p, i, weights, baseline, beam_score(p)) for i, p in enumerate(plans)]
    common.write(directory / "candidate-pool.json", rows)
    pool_hash = common.sha(directory / "candidate-pool.json")
    report = attrition(plans, certified_keys, rows)
    report["immutable_pool_sha256"] = pool_hash
    common.write(directory / "shortlist-attrition.json", report)
    return report


def execute(args):
    import fcntl

    common.ready_check()
    common.activate()
    profile = common.runtime_check()
    if args.lock.resolve() != Path(profile["lock"]).resolve():
        raise ValueError("Exact shared GPU lock required")
    if args.lock_fd is None:
        raise ValueError("Run via the single foreground supervisor")
    observed, expected = os.fstat(args.lock_fd), args.lock.stat()
    if (observed.st_dev, observed.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Wrong inherited lock descriptor")
    with os.fdopen(os.dup(args.lock_fd), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return locked_run(args)


def locked_run(args):
    import numpy as np
    from domain import LIMITS, load_inputs, settings
    from recording import GenerationBudget, Journal, native_guards, recorded_class

    from spacepdhcg.gtoc12.lambert import using_lambert_backend
    from spacepdhcg.gtoc12.search import RouteSearch

    cfg = settings(args.wall_seconds)
    catalogue, weights, seed, allowed, excluded, baseline = load_inputs()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    journal = Journal(out, args.wall_seconds)
    report = {
        "status": "initialising",
        "complete": False,
        "counts_complete": False,
        "limits": LIMITS,
        "wall_budget_seconds": args.wall_seconds,
        "settings": dataclasses.asdict(cfg),
        "baseline": baseline,
        "ready_sha256": common.sha(common.ROOT / "ready-manifest.json"),
        "driver_sha256": common.sha(__file__),
        "pid": os.getpid(),
        "generation_input_sha256": common.sha(common.ROOT / "generation-input.json"),
        "native_low_thrust_solves": 0,
        "full_route_refinements": 0,
        "fullfleet_verifications": 0,
        "promotions": 0,
        "incumbent_retained": True,
        "source_commit": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7",
        "scope": (
            "CUDA Lambert/geometry/collection DP; Python beam, input construction "
            "and forward mass orchestration"
        ),
        "precision": (
            "Unchanged frozen implementation: FP64 numerical inputs/arithmetic "
            "with FP32 resident collection delta-v cache"
        ),
    }
    common.write(out / "report.json", report)
    search = recorded_class(RouteSearch)(
        catalogue,
        np.asarray(allowed),
        cfg,
        excluded=excluded,
        weights=weights,
        seeds={},
        first_level=[seed],
    )
    search.attach(journal)
    result = None
    code = 0
    try:
        with using_lambert_backend(
            "cuda", maximum_batch_size=LIMITS["native_batch_capacity"]
        ) as gpu:
            try:
                if not gpu.collect_dp_cuda:
                    raise RuntimeError("Native collection DP required")
                report["status"] = "generating"
                common.write(out / "report.json", report)
                with native_guards(journal):
                    result = search.run()
                if not gpu.telemetry["gpu_used"]:
                    raise RuntimeError("No CUDA numerical work observed")
                report.update(
                    status="complete",
                    complete=True,
                    counts_complete=True,
                    result={
                        "expansions": result.expansions,
                        "wall_seconds": result.wall_seconds,
                        "lambert_evaluations": result.lambert_evaluations,
                        "depth_reached": result.depth_reached,
                        "first_level": result.first_level,
                        "best_by_depth_raw_proxy_kg": result.best_by_depth,
                        "depth_limit_reached": result.depth_reached == cfg.max_deploys,
                        "stopped_by_internal_wall_budget": any(
                            failure.get("reason") == "time budget exhausted"
                            for failure in result.failures
                        ),
                    },
                )
                common.write(out / "search-failures.json", result.failures)
            finally:
                report["screening_telemetry"] = dict(gpu.telemetry)
                search.release_caches()
    except GenerationBudget as error:
        report.update(
            status="bounded_partial_pool", error=str(error), complete=False, counts_complete=True
        )
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        code = 1
    finally:
        # On a caught budget boundary retain every finished completion in final beam order.
        # An outer kill leaves the append-only completion journal and its raw counts intact.
        plans = (
            result.candidates
            if result is not None
            else sorted(
                journal.successes,
                key=lambda p: (-search.plan_score(p), p.propellant_proxy_kg, p.asteroids),
            )
        )
        key = (seed.target, seed.launch_epoch, seed.tof_days)
        report["pool_attrition"] = persist_pool(
            out, plans, weights, baseline, search.plan_score, {key}
        )
        report.update(
            counts=journal.counts,
            generated_candidates=len(plans),
            collect_dp_stats=search.collect_dp_stats,
            chain_tour_stats=search.chain_tour_stats,
            substitution_stats=search.substitution_stats,
            wall_seconds=time.perf_counter() - journal.started,
        )
        common.write(out / "report.json", report)
        common.write(
            out / "output-sha256.json",
            {
                str(path.relative_to(out)): common.sha(path)
                for path in sorted(out.iterdir())
                if path.is_file() and path.name != "output-sha256.json"
            },
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "generated_candidates": len(plans),
                "counts": journal.counts,
                "output": str(out),
            }
        ),
        flush=True,
    )
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=common.ROOT / "output")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--wall-seconds", type=float, default=120)
    args = parser.parse_args()
    if not args.execute:
        print("Preparation only. Review ready-manifest.json, then use launch.py --execute.")
    else:
        if args.lock is None:
            parser.error("Explicit supervisor lock required")
        raise SystemExit(execute(args))
