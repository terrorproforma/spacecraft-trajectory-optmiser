"""CPU-only differential review of frozen RouteSearch orchestration methods.

Only standard-library AST-extracted methods execute. Route construction and
completion outcomes are deterministic stubs; no project imports, native loads,
numerical solver calls or GPU work occur. Input methods retain their exact
source segments and the hashes of the complete source files they came from.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace as S


def digest(data):
    return hashlib.sha256(data).hexdigest()


def compare(inputs):
    package = types.ModuleType("reviewpkg")
    package.__path__ = []
    sys.modules[package.__name__] = package
    backend = types.ModuleType("reviewpkg.gpu_completion")
    backend.enabled = lambda: True
    sys.modules[backend.__name__] = backend

    def make_class(variant, name):
        method = inputs[variant]["methods"][name]
        scope = {
            "__name__": "reviewpkg.logic",
            "__package__": "reviewpkg",
            "np": S(inf=math.inf, mean=lambda x: sum(x) / len(x), isfinite=math.isfinite),
        }
        exec(compile("from __future__ import annotations\n" + method, name, "exec"), scope)
        owner = type("RouteSearch", (), {name: scope[name]})
        scope["RouteSearch"] = owner
        return owner, scope

    heuristic = []
    for first in itertools.product(range(4), repeat=4):
        pattern = first + (2, 0, 1, 3)
        outputs = []
        for variant in ("baseline", "candidate"):
            owner, _ = make_class(variant, "_complete")
            search = owner()
            search.TOUR_MODES = ("a", "b", "c", "d")
            search.settings = S(collect_dp=False)
            search.last_failure = "initial"
            search.plan_score = lambda plan: plan.score
            search.calls = []

            def build(this, partial, mode, weight):
                this.calls.append((mode, weight))
                value = pattern[(int(weight) != 1) * 4 + this.TOUR_MODES.index(mode)]
                if value == 0:
                    this.last_failure = "builder_" + mode
                    return None
                return ({}, {}, (mode, weight, value))

            def finish(this, requests):
                out = []
                for partial, deploy, collect, leg, table in requests:
                    mode, weight, value = leg
                    if value == 1:
                        this.last_failure = "native_" + mode
                        out.append((None, this.last_failure))
                    else:
                        out.append((S(id=(mode, weight), score=value, propellant_proxy_kg=4-value), ""))
                return out

            def schedule(this, partial, mode, weight):
                request = build(this, partial, mode, weight)
                return None if request is None else finish(this, [(partial, *request, False)])[0][0]

            owner._schedule = schedule
            owner._schedule_forward = build
            owner._finish_many = finish
            plan = search._complete(S(deployed=[]))
            outputs.append((None if plan is None else plan.id, search.last_failure, search.calls))
        assert outputs[0] == outputs[1], (pattern, outputs)
        heuristic.append({"pattern": pattern, "output": outputs[0]})

    dp = []
    for pair in itertools.product(range(4), repeat=2):
        outputs = []
        for variant in ("baseline", "candidate"):
            owner, scope = make_class(variant, "_schedule_dp")
            search = owner()
            search.settings = S(collect_dp_propellant_weight=1.0, propellant_weight=0.1)
            search.collect_table = None
            search.weights = {}
            search.banned_pairs = set()
            search.last_failure = "initial"
            search.calls = []
            counter = [0]

            def tour_fn(*args, **kwargs):
                index = counter[0]
                counter[0] += 1
                search.calls.append(kwargs.copy())
                value = pair[index]
                return None if value == 0 else S(i=index, v=value, hop_propellant_kg=[4.0, 8.0] if index == 0 else [2.0])

            scope["plan_collect_tour"] = tour_fn

            def finish_one(this, partial, tour):
                if tour.v in (1, 2):
                    this.last_failure = ("builder_" if tour.v == 1 else "native_") + str(tour.i)
                    return None
                return tour.i

            def finish_many(this, rows):
                out = []
                for partial, tour in rows:
                    plan = finish_one(this, partial, tour)
                    out.append((plan, this.last_failure if plan is None else ""))
                return out

            owner._plan_from_tour = finish_one
            owner._plans_from_tours = finish_many
            result = search._schedule_dp(S(deployed=[], location=1, epoch=0.0, mass=900.0))
            outputs.append((result, search.last_failure, search.calls))
        assert outputs[0] == outputs[1], (pair, outputs)
        dp.append({"pattern": pair, "output": outputs[0]})

    def function(variant, name):
        return ast.parse(inputs[variant]["methods"][name]).body[0]

    old_finish = function("baseline", "_finish")
    new_finish = function("candidate", "_finish_cpu")
    new_finish.name = old_finish.name
    finish_equal = ast.dump(old_finish) == ast.dump(new_finish)
    old_chain = function("baseline", "_chain_score")
    new_chain = function("candidate", "_score_chain_plan")
    start = next(i for i, node in enumerate(old_chain.body) if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp))
    chain_equal = [ast.dump(n) for n in old_chain.body[start:]] == [ast.dump(n) for n in new_chain.body[2:]]
    assert finish_equal and chain_equal
    return {
        "passed": True,
        "GPU_calls": 0,
        "native_loads": 0,
        "heuristic_patterns": len(heuristic),
        "DP_two_weight_patterns": len(dp),
        "finish_cpu_AST_unchanged": finish_equal,
        "chain_score_math_AST_unchanged": chain_equal,
        "compared": ["selected plan/order", "last_failure", "builder mode/penalty call order", "first-tour burn supplied to second DP"],
        "heuristic_encoding": "0=builder failure;1=completion failure;2,3=successful plan scores; a second penalty round mixes all four outcomes",
        "DP_encoding": "0=no tour;1=builder failure;2=completion failure;3=successful plan",
        "limitations": "Stubbed route builders and completion outcomes test orchestration only. Native math, exception side effects, custom overrides, and GPU execution are not exercised.",
        "heuristic": heuristic,
        "DP": dp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=Path(__file__).with_name("inputs.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("findings.json"))
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    result = compare(json.loads(raw))
    result["inputs_sha256"] = digest(raw)
    result["review_script_sha256"] = digest(Path(__file__).read_bytes())
    args.output.write_text(json.dumps(result, indent=2, default=lambda value: sorted(value) if isinstance(value, set) else vars(value)) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("heuristic", "DP")}, indent=2))


if __name__ == "__main__":
    main()
