from __future__ import annotations

import importlib.util
from pathlib import Path


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
