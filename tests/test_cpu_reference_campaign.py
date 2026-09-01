from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "cpu" / "finalize_reference_campaign.py"
    specification = importlib.util.spec_from_file_location("cpu_reference_campaign", path)
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
