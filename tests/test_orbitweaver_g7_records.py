from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from spacepdhcg.orbitweaver import (
    PAPER2_MATRIX_SHA256,
    SINGLE_GPU_G7_STAGES,
    CertificationChecks,
    CertificationRecord,
    Checkpoint,
    ResultRecord,
    RunManifest,
    single_gpu_completion_record,
)
from spacepdhcg.orbitweaver.contracts import (
    FAILURE_STATUSES,
    G7_CHECKPOINT_SCHEMA,
    G7_CONFIG_SCHEMA,
    G7_MANIFEST_SCHEMA,
    G7_RESULT_SCHEMA,
    SCHEMAS,
    ContractError,
    validate_named,
)

ROOT = Path(__file__).resolve().parents[1]
HEX_A = "a" * 64
HEX_B = "b" * 64


def manifest() -> RunManifest:
    return RunManifest(
        schema_version=1,
        run_id="paper2-test",
        repository_commit="1" * 40,
        seed=42,
        backend="cpu-reference",
        ownership="single_gpu",
        device_ids=(0,),
        config_sha256=HEX_A,
        paper2_matrix_sha256=PAPER2_MATRIX_SHA256,
        repeat_count=7,
        toolchain={
            "python": "3.12.13",
            "compiler": "gcc 11.4",
            "cmake": "cmake 4.1",
            "cuda": None,
        },
        hardware={"os": "Linux", "cpu": "test-cpu", "gpus": []},
        evidence_level="cpu_correctness_tested",
    )


def telemetry() -> dict[str, Any]:
    return {
        "submitted": 2,
        "completed": 2,
        "feasible": 1,
        "failed": 1,
        "cancelled": 0,
        "batches": 1,
        "maximum_observed_batch": 2,
        "estimated_peak_buffer_bytes": 128,
        "group_batches": {"11:coarse_convex:8:1": 1},
        "ownership_batches": {"0:0": 1},
    }


def result(
    run_manifest: RunManifest,
    *,
    status: str = "converged",
    certified: bool = True,
) -> ResultRecord:
    certification = (
        CertificationRecord(
            True,
            CertificationChecks(1e-7, 2e-7, 3e-7, 4e-7, 5e-7),
            "independent-rk4",
            "accepted",
        )
        if certified
        else None
    )
    terminal = status in FAILURE_STATUSES
    no_incumbent = status in {"infeasible", "unsupported"}
    return ResultRecord(
        schema_version=1,
        run_id=run_manifest.run_id,
        manifest_sha256=run_manifest.sha256(),
        paper2_matrix_sha256=PAPER2_MATRIX_SHA256,
        seed=run_manifest.seed,
        repeat_index=0,
        status=status,
        incumbent=None if no_incumbent else 10.0,
        lower_bound=None if no_incumbent else 8.0,
        optimality_gap=None if no_incumbent else 2.0,
        certified=certified,
        certification=certification,
        telemetry=telemetry(),
        failures=(
            ({"status": status, "diagnostic": f"{status} test", "deterministic_id": 7},)
            if terminal
            else ()
        ),
    )


def test_authoritative_schemas_match_materialised_json() -> None:
    files = {
        "config": "orbitweaver_g7_config.schema.json",
        "manifest": "orbitweaver_g7_manifest.schema.json",
        "checkpoint": "orbitweaver_g7_checkpoint.schema.json",
        "result": "orbitweaver_g7_result.schema.json",
    }
    for name, filename in files.items():
        materialised = json.loads(
            (ROOT / "experiments" / "schema" / filename).read_text(encoding="utf-8")
        )
        assert materialised == SCHEMAS[name]
        jsonschema.Draft202012Validator.check_schema(materialised)


def test_manifest_round_trip_requires_exact_matrix_hash(tmp_path: Path) -> None:
    value = manifest()
    path = tmp_path / "manifest.json"
    value.write(path)
    assert RunManifest.read(path) == value
    payload = value.to_dict()
    payload.pop("paper2_matrix_sha256")
    with pytest.raises((ValueError, ContractError)):
        RunManifest.from_dict(payload)
    payload["paper2_matrix_sha256"] = "0" * 64
    with pytest.raises((ValueError, ContractError)):
        RunManifest.from_dict(payload)
    payload = value.to_dict()
    payload["unknown"] = True
    with pytest.raises((ValueError, ContractError)):
        RunManifest.from_dict(payload)


def test_scoped_manifest_and_single_gpu_completion_semantics() -> None:
    legacy = manifest()
    scoped = replace(
        legacy,
        schema_version=2,
        campaign_scope_id="single-gpu-v1",
        evidence_level="one_gpu_correctness_tested",
    )
    scoped.validate()
    completion = single_gpu_completion_record(
        scoped,
        [result(scoped)],
        SINGLE_GPU_G7_STAGES,
    )
    assert completion["status"] == "complete-in-scope"
    assert completion["campaign_scope_id"] == "single-gpu-v1"
    assert "physical multi-GPU scaling" in completion["deferred_claims"]

    with pytest.raises(ValueError, match="cross-scope"):
        replace(scoped, ownership="g5_distributed", device_ids=(0, 1)).validate()
    with pytest.raises(ValueError, match="cross-scope"):
        replace(scoped, evidence_level="physical_multi_gpu_tested").validate()
    with pytest.raises(ValueError, match="stage inventory"):
        single_gpu_completion_record(scoped, [result(scoped)], {"coarse_convex"})
    with pytest.raises(ValueError, match="independently certified"):
        single_gpu_completion_record(
            scoped,
            [result(scoped, status="timeout", certified=False)],
            SINGLE_GPU_G7_STAGES,
        )


def test_checkpoint_round_trip_and_manifest_pin_mismatches(tmp_path: Path) -> None:
    run_manifest = manifest()
    checkpoint = Checkpoint(
        schema_version=1,
        run_id=run_manifest.run_id,
        manifest_sha256=run_manifest.sha256(),
        paper2_matrix_sha256=PAPER2_MATRIX_SHA256,
        seed=run_manifest.seed,
        repeat_index=2,
        completed_batches=3,
        incumbent=10.0,
        lower_bound=8.0,
        completed_arc_ids=(1, 2),
        warm_tokens=(10, 20),
    )
    path = tmp_path / "checkpoint.json"
    checkpoint.write(path)
    assert Checkpoint.read(path, run_manifest) == checkpoint
    with pytest.raises(ValueError):
        replace(checkpoint, completed_arc_ids=(2, 1)).validate()
    with pytest.raises((ValueError, ContractError)):
        replace(checkpoint, paper2_matrix_sha256="0" * 64).validate()
    for changed in [
        replace(checkpoint, seed=43),
        replace(checkpoint, repeat_index=run_manifest.repeat_count),
        replace(checkpoint, manifest_sha256=HEX_B),
    ]:
        with pytest.raises(ValueError):
            changed.validate(run_manifest)


@pytest.mark.parametrize(
    "status",
    ["failed", "censored", "unsupported", "oom", "timeout", "infeasible", "cancelled"],
)
def test_terminal_result_records_are_explicit_and_valid(status: str) -> None:
    run_manifest = manifest()
    value = result(run_manifest, status=status, certified=False)
    value.validate(run_manifest)
    assert ResultRecord.from_dict(value.to_dict(), run_manifest) == value


def test_result_round_trip_and_certification_constraints(tmp_path: Path) -> None:
    run_manifest = manifest()
    value = result(run_manifest)
    path = tmp_path / "result.json"
    value.write(path, run_manifest)
    assert ResultRecord.read(path, run_manifest) == value
    with pytest.raises((ValueError, ContractError)):
        replace(value, certification=None).validate(run_manifest)
    rejected = CertificationRecord(
        False,
        CertificationChecks(0.0, 0.0, 1.0, 0.0, 0.0),
        "independent-rk4",
        "rejected",
    )
    with pytest.raises((ValueError, ContractError)):
        replace(value, certification=rejected).validate(run_manifest)
    with pytest.raises(ValueError):
        replace(value, lower_bound=11.0, optimality_gap=0.0).validate(run_manifest)
    with pytest.raises(ValueError):
        replace(value, optimality_gap=1.0).validate(run_manifest)


@pytest.mark.parametrize("field", ["incumbent", "lower_bound", "optimality_gap"])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -1.0])
def test_result_rejects_nonfinite_and_negative_numbers(field: str, bad: float) -> None:
    value = result(manifest())
    with pytest.raises((ValueError, ContractError)):
        replace(value, **{field: bad}).validate()


def test_result_rejects_status_unknown_fields_failures_and_counter_drift() -> None:
    run_manifest = manifest()
    payload = result(run_manifest).to_dict()
    payload["status"] = "optimal"
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)
    payload = result(run_manifest).to_dict()
    payload["surprise"] = 1
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)
    for section in ["certification", "telemetry"]:
        payload = result(run_manifest).to_dict()
        payload[section]["unknown"] = 1
        with pytest.raises((ValueError, ContractError)):
            ResultRecord.from_dict(payload)
    payload = result(run_manifest, status="failed", certified=False).to_dict()
    payload["failures"][0]["unknown"] = 1
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)
    terminal = result(run_manifest, status="timeout", certified=False)
    with pytest.raises(ValueError):
        replace(terminal, failures=()).validate()
    broken = copy.deepcopy(result(run_manifest).telemetry)
    broken["completed"] = 1
    with pytest.raises(ValueError):
        replace(result(run_manifest), telemetry=broken).validate()


def test_nested_numeric_constraints_are_enforced() -> None:
    run_manifest = manifest()
    payload = result(run_manifest).to_dict()
    payload["certification"]["checks"]["integration_error"] = math.inf
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)
    payload = result(run_manifest).to_dict()
    payload["telemetry"]["batches"] = -1
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)
    payload = result(run_manifest, status="oom", certified=False).to_dict()
    payload["failures"][0]["deterministic_id"] = -1
    with pytest.raises((ValueError, ContractError)):
        ResultRecord.from_dict(payload)


def test_result_rejects_seed_repeat_and_manifest_mismatches() -> None:
    run_manifest = manifest()
    value = result(run_manifest)
    for changed in [
        replace(value, seed=43),
        replace(value, repeat_index=run_manifest.repeat_count),
        replace(value, manifest_sha256=HEX_B),
        replace(value, paper2_matrix_sha256="0" * 64),
    ]:
        with pytest.raises((ValueError, ContractError)):
            changed.validate(run_manifest)


def test_read_rejects_nonstandard_json_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"incumbent":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        ResultRecord.read(path)


def test_schema_and_internal_validator_differential_property() -> None:
    rng = random.Random(981723)
    bases = {
        "manifest": manifest().to_dict(),
        "result": result(manifest()).to_dict(),
        "checkpoint": Checkpoint(
            1,
            "paper2-test",
            HEX_A,
            PAPER2_MATRIX_SHA256,
            42,
            0,
            0,
            None,
            None,
            (),
            (),
        ).to_dict(),
        "config": {
            "schema_version": 1,
            "seed": 42,
            "repeat_count": 7,
            "maximum_batch_size": 2,
            "maximum_buffered_arcs": 4,
            "maximum_workspace_bytes": 1024,
            "top_k": 2,
            "risk_measure": "expected",
            "certification_tolerance": 1e-5,
        },
    }
    bad_values: list[Any] = [None, True, -1, "", [], {}, "wrong"]
    for name, base in bases.items():
        validator = jsonschema.Draft202012Validator(SCHEMAS[name])
        candidates = [base]
        keys = list(base)
        for _ in range(200):
            candidate = copy.deepcopy(base)
            action = rng.randrange(3)
            key = rng.choice(keys)
            if action == 0:
                candidate.pop(key, None)
            elif action == 1:
                candidate[key] = copy.deepcopy(rng.choice(bad_values))
            else:
                candidate[f"unknown_{rng.randrange(5)}"] = 1
            candidates.append(candidate)
        for candidate in candidates:
            schema_valid = validator.is_valid(candidate)
            try:
                validate_named(candidate, name)
            except ContractError:
                internal_valid = False
            else:
                internal_valid = True
            assert internal_valid == schema_valid, (name, candidate)


def test_schema_constants_are_the_expected_objects() -> None:
    assert SCHEMAS == {
        "config": G7_CONFIG_SCHEMA,
        "manifest": G7_MANIFEST_SCHEMA,
        "checkpoint": G7_CHECKPOINT_SCHEMA,
        "result": G7_RESULT_SCHEMA,
    }
