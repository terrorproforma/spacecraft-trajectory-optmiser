from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from spacepdhcg.benchmarks.cw_repeat import run_benchmark as run_hcw_box
from spacepdhcg.benchmarks.powered_descent_scvx import run as run_pd3


def _module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "cpu" / "finalize_reference_campaign.py"
    specification = importlib.util.spec_from_file_location("cpu_reference_campaign", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _matrix_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "cpu" / "run_supported_matrix.py"
    specification = importlib.util.spec_from_file_location("run_supported_matrix", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _supported_finalizer_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "cpu" / "finalize_supported_matrix.py"
    specification = importlib.util.spec_from_file_location("finalize_supported_matrix", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_frozen_coordinate_ledger_is_complete_and_fail_closed() -> None:
    module = _module()
    root = Path(__file__).resolve().parents[1]
    paper1_path = root / "benchmarks" / "paper1_matrix.json"
    paper2_path = root / "benchmarks" / "paper2_matrix.json"
    paper1 = module._matrix_coordinates(
        "paper1",
        module._load(paper1_path),
        module.sha256_path(paper1_path),
    )
    paper2 = module._matrix_coordinates(
        "paper2",
        module._load(paper2_path),
        module.sha256_path(paper2_path),
    )

    assert len(paper1) == 13_676
    assert len(paper2) == 2_648
    assert len({item["coordinate_id"] for item in (*paper1, *paper2)}) == 16_324
    assert {item["classification"] for item in paper1} == {"unrun"}
    assert {item["classification"] for item in paper2} == {"unrun", "unsupported"}
    assert all(item["reason"] for item in (*paper1, *paper2))


def test_chart_source_contains_required_provenance() -> None:
    module = _module()
    environment = {
        "time_range_utc": {
            "start": "2026-09-01T00:00:00Z",
            "end": "2026-09-01T01:00:00Z",
        }
    }
    source = module._chart_metadata(
        "D00",
        "Diagnostic",
        [{"name": "x", "unit": "count"}],
        ["series"],
        "No transformation.",
        environment,
        [],
    )

    assert source["source_commit"] == module.FROZEN_COMMIT
    assert source["axes"][0]["unit"] == "count"
    assert source["series_legend"] == ["series"]
    assert source["transformation_aggregation_caption"]


def test_supported_matrix_driver_expands_same_frozen_inventory(tmp_path: Path) -> None:
    module = _matrix_module()
    root = Path(__file__).resolve().parents[1]
    paper1 = json.loads((root / "benchmarks" / "paper1_matrix.json").read_text())
    paper2 = json.loads((root / "benchmarks" / "paper2_matrix.json").read_text())
    coordinates = module._coordinates(paper1, "paper1") + module._coordinates(paper2, "paper2")
    assert len(coordinates) == 16_324
    assert len({item["coordinate_id"] for item in coordinates}) == len(coordinates)

    module._OUTPUT = tmp_path
    module._ENVIRONMENT_SHA256 = "0" * 64
    module._SCHEMA = json.loads(
        (root / "experiments/schema/cpu_reference_result.schema.json").read_text()
    )
    coordinate = next(
        item
        for item in coordinates
        if item["family"] == "P1-A-banded"
        and item["parameters"]
        == {
            "intervals": 10,
            "state_dimensions": 4,
            "control_dimensions": 3,
            "control_sets": "box",
            "weight_log10_spans": 0.0,
            "seeds": 17,
        }
    )
    result = module._banded(coordinate)
    assert result["disposition"] == "executed"
    assert result["quality"]["qualified"]
    assert result["quality"]["canonical_natural_residual"] <= 1.0e-8


def test_semantic_reproducibility_excludes_only_observation_fields() -> None:
    module = _supported_finalizer_module()
    record = {
        "coordinate_id": "a" * 24,
        "family": "P1-A-banded",
        "quality": {"qualified": True},
        "timing": {"wall_seconds": 1.0},
        "resources": {"peak_host_bytes": 100},
        "artifacts": {"result": "result.json"},
    }
    semantic = module._semantic(record)
    assert semantic == {
        "coordinate_id": "a" * 24,
        "family": "P1-A-banded",
        "quality": {"qualified": True},
    }
    assert module._distribution([1.0, 2.0, 3.0]) == {
        "count": 3,
        "minimum": 1.0,
        "q1": 1.5,
        "median": 2.0,
        "q3": 2.5,
        "maximum": 3.0,
    }


def test_coordinate_specific_benchmark_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="update_magnitude"):
        run_hcw_box(repeats=2, intervals=20, update_magnitude=-1.0)
    with pytest.raises(ValueError, match="initial_dispersion_scale"):
        run_pd3(
            intervals=20,
            step_seconds=2.0,
            max_iterations=1,
            tolerance=1.0e-3,
            initial_dispersion_scale=-1.0,
        )
