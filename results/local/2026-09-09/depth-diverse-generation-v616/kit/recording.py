"""Transparent generation journaling and hard work caps; no selection policy change."""

import json
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from domain import LIMITS


class GenerationBudget(RuntimeError):
    pass


class Journal:
    def __init__(self, directory, wall_seconds):
        self.directory = directory
        self.started = time.perf_counter()
        self.deadline = self.started + wall_seconds
        self.counts = {}
        self.successes = []

    def event(self, item):
        with (self.directory / "generation-events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, allow_nan=False) + "\n")

    def begin(self, stage, detail=None):
        if time.perf_counter() >= self.deadline:
            raise GenerationBudget("Wall budget reached before " + stage)
        started = self.counts.get(stage + "_started", 0)
        if started >= LIMITS[stage]:
            raise GenerationBudget("Hard stage budget reached: " + stage)
        self.counts[stage + "_started"] = started + 1
        self.event(
            {
                "event": "started",
                "stage": stage,
                "number": started + 1,
                "detail": detail,
                "elapsed_seconds": time.perf_counter() - self.started,
            }
        )

    def end(self, stage, detail=None):
        self.counts[stage + "_finished"] = self.counts.get(stage + "_finished", 0) + 1
        self.event(
            {
                "event": "finished",
                "stage": stage,
                "number": self.counts[stage + "_finished"],
                "detail": detail,
                "elapsed_seconds": time.perf_counter() - self.started,
            }
        )


@contextmanager
def native_guards(journal):
    from spacepdhcg.gtoc12 import bundles, earthleg, gpu_collect_dp, gpu_scvx, low_thrust, pipeline

    original = gpu_collect_dp.cuda_collect_dp

    def gpu_dp(*args, **kwargs):
        journal.begin("collection_dp_passes")
        value = original(*args, **kwargs)
        if value is NotImplemented:
            raise RuntimeError("CPU collection DP fallback forbidden in this generation")
        journal.end("collection_dp_passes", {"feasible": value is not None})
        return value

    def forbidden(*args, **kwargs):
        raise RuntimeError("This generation-only kit forbids all low-thrust/refinement calls")

    with ExitStack() as stack:
        stack.enter_context(patch.object(gpu_collect_dp, "cuda_collect_dp", gpu_dp))
        for module, name in [
            (gpu_scvx, "solve_native"),
            (pipeline, "refine_route"),
            (bundles, "refine_route"),
            (bundles, "certify_earth_legs"),
            (earthleg, "refine_leg_scvx"),
            (low_thrust, "solve_leg"),
            (pipeline, "solve_leg"),
        ]:
            # Fail preparation if this frozen API changes, rather than silently losing a guard.
            stack.enter_context(patch.object(module, name, forbidden))
        yield


def recorded_class(base):
    class RecordedSearch(base):
        def attach(self, journal):
            self.journal = journal

        def run(self):
            self.journal.begin("generation_calls")
            result = super().run()
            self.journal.end("generation_calls", {"candidates": len(result.candidates)})
            return result

        def _expand(self, partial):
            self.journal.begin("expansions", {"depth": len(partial.deployed)})
            value = super()._expand(partial)
            self.journal.end("expansions", {"children": len(value)})
            return value

        def _chain_tour(self, partial):
            self.journal.begin("chain_tour_calls", {"depth": len(partial.deployed)})
            value = super()._chain_tour(partial)
            self.journal.end("chain_tour_calls", {"feasible": value is not None})
            return value

        def _complete(self, partial):
            detail = {"depth": len(partial.deployed), "deployed": list(partial.deployed)}
            self.journal.begin("completion_attempts", detail)
            value = super()._complete(partial)
            if value is not None:
                self.journal.successes.append(value)
                # Every generated completion survives an interruption of a later native call.
                with (self.journal.directory / "completed-plans.jsonl").open("a") as stream:
                    stream.write(json.dumps(value.summary(), allow_nan=False) + "\n")
            self.journal.end(
                "completion_attempts",
                {
                    **detail,
                    "feasible": value is not None,
                    "failure": self.last_failure if value is None else None,
                },
            )
            return value

    return RecordedSearch
