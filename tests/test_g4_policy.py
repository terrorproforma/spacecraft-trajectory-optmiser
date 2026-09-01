from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from spacepdhcg.experiments.g4 import (
    G4ContractError,
    load_policy,
    runtime_configuration,
    verify_runtime_behavior,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_g4_policy_covers_frozen_matrix_and_modes() -> None:
    policy = _load("benchmarks/g4_policy.json")
    paper1 = _load("benchmarks/paper1_matrix.json")
    required_modes = {
        "fixed-tight",
        "fixed-loose",
        "adaptive",
        "adaptive+polish",
        "pure-gpu-ipm",
        "hybrid-pdhcg-ipm",
    }
    assert policy["gate"] == "G4"
    assert set(policy["policies"]) == required_modes
    assert policy["warm_start_modes"] == paper1["warm_start_modes"]
    assert policy["matrix"]["measured_repeats"] == paper1["measured_repeats"]
    assert (
        policy["matrix"]["randomised_instances_per_coordinate"]
        == paper1["randomised_instances_per_coordinate"]
    )
    registered = {family["id"]: family for family in paper1["families"]}
    for family, values in policy["matrix"]["families"].items():
        assert values["intervals"] == registered[family]["intervals"]


def test_g4_split_is_disjoint_and_preserves_registered_seeds() -> None:
    policy = _load("benchmarks/g4_policy.json")
    paper1 = _load("benchmarks/paper1_matrix.json")
    split = policy["tuning_evaluation_split"]
    tuning = set(split["tuning_seeds"])
    evaluation = set(split["evaluation_seeds"])
    assert tuning.isdisjoint(evaluation)
    registered = next(
        family["seeds"] for family in paper1["families"] if family["id"] == "P1-A-banded"
    )
    assert tuning | evaluation == set(registered)
    assert len(evaluation) == 20
    assert split["evaluation_may_not_change_policy"] is True


def test_g4_decision_thresholds_match_preregistered_claims() -> None:
    policy = _load("benchmarks/g4_policy.json")
    h5 = policy["decision_thresholds"]["H5"]
    h6 = policy["decision_thresholds"]["H6"]
    assert h5["supported_minimum_time_reduction"] == 0.15
    assert h5["rejected_minimum_slowdown"] == 0.10
    assert h5["maximum_failure_rate_increase"] == 0.02
    assert h6["supported_minimum_time_reduction"] == 0.10
    assert h6["maximum_hybrid_to_ipm_residual_factor"] == 2.0
    assert h6["minimum_unpolished_residual_decades"] == 1.0
    assert policy["statistics"]["bootstrap_samples"] == 10_000
    assert policy["statistics"]["bootstrap_seed"] == 20_260_901


def test_qoco_lock_is_gpu_ipm_and_records_warm_start_limit() -> None:
    lock = _load("third_party/qoco_gpu.lock.json")
    assert len(lock["commit"]) == 40
    assert len(lock["tree"]) == 40
    assert lock["cuda"]["toolkit"] == "12.8.93"
    assert lock["cuda"]["architecture"] == 120
    assert lock["build"]["algebra"] == "cuda"
    assert lock["build"]["precision"] == "float64"
    assert lock["warm_start"]["primal"] is True
    assert lock["warm_start"]["dual"] is False
    assert "second_order" in lock["mathematical_contract"]["cones"]


def test_policy_lock_and_generated_cpp_header_are_current() -> None:
    lock = (ROOT / "benchmarks/g4_policy.sha256").read_text(encoding="utf-8").split()[0]
    loaded = load_policy(ROOT / "benchmarks/g4_policy.json", expected_sha256=lock)
    assert loaded.sha256 in (
        ROOT / "cpp/include/spacepdhcg/scvx/g4_policy.generated.hpp"
    ).read_text(encoding="utf-8")
    subprocess.run(
        [
            "python3",
            "scripts/generate_g4_policy_header.py",
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


def test_policy_hash_drift_is_fatal(tmp_path: Path) -> None:
    source = ROOT / "benchmarks/g4_policy.json"
    altered = tmp_path / "g4_policy.json"
    altered.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(G4ContractError, match="policy hash drift"):
        load_policy(altered, expected_sha256="0" * 64)


def test_runtime_plan_parameterises_every_frozen_mode() -> None:
    loaded = load_policy(ROOT / "benchmarks/g4_policy.json")
    tolerances: set[float] = set()
    for tier in ("coarse", "medium", "tight", "ipm"):
        for scaling in ("always_refresh", "reuse", "refresh_if_needed"):
            for warm in ("cold", "primal", "primal_dual"):
                config = runtime_configuration(
                    loaded,
                    family="P1-C-pd3",
                    policy_name="fixed-tight",
                    quality_tier=tier,
                    scaling_mode=scaling,
                    warm_mode=warm,
                )
                tolerances.add(config["solver"]["inner_tolerance"])
                assert config["scaling_mode"] == scaling
                assert config["warm_start_mode"] == warm
    assert tolerances == {1e-3, 1e-4, 1e-6, 1e-8}


def test_runtime_actual_behavior_must_match_request() -> None:
    loaded = load_policy(ROOT / "benchmarks/g4_policy.json")
    requested = runtime_configuration(
        loaded,
        family="P1-C-pd3",
        policy_name="adaptive",
        quality_tier="tight",
        scaling_mode="reuse",
        warm_mode="primal",
    )
    reported = {
        "policy_sha256": loaded.sha256,
        "requested": {
            key: requested[key]
            for key in (
                "policy",
                "quality_tier",
                "quality_tolerance",
                "scaling_mode",
                "warm_start_mode",
            )
        },
        "actual": {
            **{
                key: requested[key]
                for key in (
                    "policy",
                    "quality_tier",
                    "quality_tolerance",
                    "scaling_mode",
                    "warm_start_mode",
                )
            },
            "solver": requested["solver"],
            "resolve": requested["resolve"],
        },
    }
    assert verify_runtime_behavior(requested, reported) == []
    reported["actual"]["scaling_mode"] = "always_refresh"
    assert "actual scaling_mode differs from request" in verify_runtime_behavior(
        requested, reported
    )
