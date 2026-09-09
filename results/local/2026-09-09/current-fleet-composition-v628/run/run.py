"""One CPU-only two-checker pass, under a120-second foreground supervisor."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
ROUTE = ROOT / "build/performance/seeded-candidate-boundary-merit-v627/campaign"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify_pins():
    plan = read(KIT / "plan.json")
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROUTE / "ready.json") == "fa26177cd9ec3b452f68f7ce37407a00125ca7885a06b963d2525b7df5d3d56c"
    for name, digest in read(ROUTE / "ready.json")["files"].items():
        assert sha(ROUTE / name) == digest, name
    assert sha(KIT / "inputs/Result.txt") == plan["composed_result_sha256"]
    return plan


def worker():
    assert os.environ.get("SPACEPDHCG_COMPOSITION_PARENT") == str(os.getppid())
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    plan = verify_pins()
    output = KIT / "output"
    path = KIT / "inputs/Result.txt"
    started = time.perf_counter()
    report = {"complete": False, "status": "running", "independent_calls": 0, "official_calls": 0, "GPU_calls": 0, "native_solves": 0, "search_calls": 0, "extra_leg_certificates": 0, "result_sha256": sha(path), "incumbent_promoted": False}

    def save():
        temporary = output / "partial-report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, output / "partial-report.json")

    try:
        sys.path.insert(0, str(ROUTE))
        import common as C

        C.activate()
        from spacepdhcg.gtoc12.official import run_official_verifier
        from spacepdhcg.gtoc12.verifier import Gtoc12Verifier

        catalogue, bonus = C.catalogue_and_bonus()
        report["catalogue_sha256"] = catalogue.source_sha256
        report["bonus_sha256"] = bonus.source_sha256
        report["independent_calls"] += 1
        save()
        stamp = time.perf_counter()
        independent = Gtoc12Verifier(catalogue, bonus=bonus).verify_file(path)
        report["independent_seconds"] = time.perf_counter() - stamp
        write(output / "independent-full.json", dataclasses.asdict(independent))
        independent_summary = independent.summary() | {"result_sha256": sha(path)}
        write(output / "independent.json", independent_summary)
        assert sha(path) == plan["composed_result_sha256"]
        remaining = 120 - (time.perf_counter() - started)
        assert remaining > 0
        report["official_calls"] += 1
        save()
        official = run_official_verifier(path, timeout=remaining, keep_directory=output / "official")
        write(output / "official-full.json", dataclasses.asdict(official))
        official_summary = official.summary() | {"result_sha256": sha(path)}
        write(output / "official.json", official_summary)
        assert sha(path) == sha(output / "official/Result.txt") == plan["composed_result_sha256"]
        write(output / "checker-binding.json", {"result_sha256": sha(path), "independent": independent_summary, "official": official_summary})
        raw = independent_summary["total_mass_kg"]
        weighted = independent_summary["weighted_score_fixed_bonus_kg"]
        finite = all(isinstance(x, (float, int)) and math.isfinite(x) for x in (raw, weighted))
        qualified = bool(
            finite and independent_summary["ok"] and official_summary["ok"]
            and independent_summary["ships"] == official_summary["ships"] == 23
            and independent_summary["mined_asteroids"] == official_summary["mined_asteroids"] == 200
            and independent_summary["ship_limit"] >= 23
            and abs(raw - plan["expected_raw_kg"]) <= 1e-8
            and abs(weighted - plan["expected_weighted_kg"]) <= 1e-8
            and raw >= plan["baseline"]["total_mass_kg"]
            and weighted > plan["baseline"]["weighted_score_fixed_bonus_kg"] + 1e-8
        )
        report.update(status="verified_current_fleet_improvement" if qualified else "full_fleet_rejected", qualified=qualified, independent=independent_summary, official=official_summary, raw_delta_kg=raw - plan["baseline"]["total_mass_kg"], weighted_delta_kg=weighted - plan["baseline"]["weighted_score_fixed_bonus_kg"], physics_tolerances_unchanged=True, reporting_bound_kg=1e-8)
    except BaseException as error:
        report.update(status="execution_failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(complete=True, seconds=time.perf_counter() - started)
        write(output / "report.json", report)
        save()


def reap(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=30)


def supervise():
    plan = verify_pins()
    write(KIT / "launch-marker.json", {"pid": os.getpid(), "plan_sha256": sha(KIT / "plan.json"), "run_sha256": sha(__file__)})
    output = KIT / "output"
    output.mkdir()
    report = {"complete": False, "supervisor_pid": os.getpid(), "result_sha256": plan["composed_result_sha256"], "hard_deadline_seconds": 120, "TERM_grace_seconds": 10, "KILL_reap_seconds": 30}
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f"Signal{signum}")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SPACEPDHCG_", "QOCO_", "PDHCG_"))}
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", SPACEPDHCG_COMPOSITION_PARENT=str(os.getpid()))
        with (output / "worker.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), "--worker"], cwd=KIT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            report["child_pid"] = child.pid
            print(json.dumps(report), flush=True)
            report["exit_code"] = child.wait(timeout=120)
        assert report["exit_code"] == 0
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        reap(child)
        report.update(complete=True, child_exit_after_cleanup=None if child is None else child.poll())
        write(output / "launch-report.json", report)


if __name__ == "__main__":
    assert __debug__
    worker() if sys.argv[1:] == ["--worker"] else supervise()
