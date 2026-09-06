from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jsonschema
import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts/gpu/run_g3_h1.py"
    spec = importlib.util.spec_from_file_location("run_g3_h1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample(intervals: int, omega: float = 0.0) -> dict:
    return {
        "intervals": intervals,
        "status": "qualified",
        "record": {
            "omega_persist": omega,
            "canonical_residual": 1.0e-8,
            "nonlinear_residual": 1.0e-10,
            "cpu_gpu_trajectory": 0.0,
            "topology_allocations_after_create": 0,
            "topology_copies_after_create": 0,
            "scvx_total_seconds": 1.0,
        },
    }


def test_h1_supported_requires_five_matched_repeats() -> None:
    module = _module()
    supported = module._coordinate_summary([_sample(20) for _ in range(5)])
    insufficient = module._coordinate_summary([_sample(20) for _ in range(4)])
    assert supported["status"] == "supported"
    assert supported["omega_bootstrap_95"] == [0.0, 0.0]
    assert insufficient["status"] == "unresolved"


def test_h1_preserves_censored_and_topology_failures() -> None:
    module = _module()
    samples = [_sample(50) for _ in range(5)]
    samples[0]["record"]["topology_copies_after_create"] = 1
    samples.append({"intervals": 50, "status": "timeout"})
    summary = module._coordinate_summary(samples)
    assert summary["status"] == "unresolved"
    assert summary["topology_clean"] is False
    assert summary["censored_count"] == 1


def test_h1_sustained_boundary_uses_three_coordinates() -> None:
    module = _module()
    coordinates = [
        {"intervals": 20, "summary": {"status": "unresolved"}},
        {"intervals": 50, "summary": {"status": "supported"}},
        {"intervals": 100, "summary": {"status": "supported"}},
        {"intervals": 500, "summary": {"status": "supported"}},
    ]
    assert module._sustained_boundary(coordinates) == 50


def test_h1_parser_ignores_non_json_numeric_sentinels_in_other_records() -> None:
    module = _module()
    stdout = "\n".join(
        (
            '{"case":"production_outer","ratio":-inf}',
            '{"case":"h1_hcw","intervals":20,"canonical_residual":1e-9}',
        )
    )
    record = module._parse_record(stdout)
    assert record["case"] == "h1_hcw"
    assert record["intervals"] == 20


def _archived_sample():
    # Verbatim last record of the archived 9e75b47 H100 H1 sweep. Keep the
    # regression self-contained when tests are included in a source distribution.
    path = Path(__file__).parent / "fixtures/g3_h1_h100_10000.json"
    return json.loads(path.read_text())


def test_h1_compact_matches_actual_h100_work_and_schema() -> None:
    module = _module()
    sample = _archived_sample()
    result = module._compact_result(sample, "a" * 40, "test", Path("raw.jsonl"), "b" * 64, 41)
    assert result["work"]["inner_iterations"] == 3
    assert result["work"]["outer_iterations"] == 3
    assert result["work"]["accepted_steps"] == 0
    assert result["work"]["rejected_steps"] == 0
    assert result["work"]["polish_used"] is None
    assert result["resources"]["peak_device_bytes"] is None
    assert result["resources"]["reserved_device_bytes"] is None
    assert result["aggregation"]["measured_repeats"] == 1
    assert result["aggregation"]["warmup_repeats"] == 0
    assert result["timing"]["scvx_total_seconds"] == 15.2584705
    schema = json.loads(
        (Path(__file__).parents[1] / "experiments/schema/paper1_result.schema.json").read_text()
    )
    jsonschema.validate(result, schema)


def test_h1_missing_work_is_unknown_even_when_requested_repeats_are_known() -> None:
    sample = _archived_sample()
    sample.pop("stdout")
    result = _module()._compact_result(sample, "a" * 40, "test", Path("raw"), "b" * 64, 0)
    assert result["work"]["inner_iterations"] is None
    assert result["work"]["outer_iterations"] is None
    assert result["work"]["accepted_steps"] is None


def test_h1_rejects_conflicting_counters() -> None:
    sample = _archived_sample()
    h1 = dict(sample["record"], inner_iterations=0)
    stdout = sample["stdout"].splitlines()[0] + "\n" + json.dumps(h1)
    with pytest.raises(RuntimeError, match="conflicting HCW counter"):
        _module()._parse_record(stdout)


@pytest.mark.parametrize("status", ["timeout", "failed"])
def test_h1_censored_work_is_not_reported_as_zero(status) -> None:
    result = _module()._compact_result(
        {"intervals": 10000, "status": status}, "a" * 40, "test", Path("raw"), "b" * 64, 0
    )
    assert all(value is None for value in result["work"].values())
    assert result["resources"]["peak_device_bytes"] is None
    assert result["aggregation"]["censored_count"] == 1
