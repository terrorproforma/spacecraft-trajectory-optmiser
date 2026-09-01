from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/gpu/run_g4_qualification.py"
    spec = importlib.util.spec_from_file_location("run_g4_qualification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_extracts_exactly_one_sample() -> None:
    module = _module()
    records = module.parse_json_lines(
        'noise\n{"case":"g4_iteration","outer":0}\n'
        '{"case":"g4_sample","qualified":false}\n'
    )
    assert module.sample_record(records)["qualified"] is False


def test_coverage_retains_unsupported_and_censored_coordinates() -> None:
    module = _module()
    policy = json.loads(
        (ROOT / "benchmarks/g4_policy.json").read_text(encoding="utf-8")
    )
    executed = [
        {
            "family": "P1-C-pd3",
            "intervals": 20,
            "policy": "adaptive",
            "warm_start": "cold",
            "quality_tier": "tight",
            "status": "unqualified",
        }
    ]
    rows = module.coverage(policy, executed)
    dispositions = {row["disposition"] for row in rows}
    assert {"unqualified", "unsupported", "censored"} <= dispositions
    assert len(rows) == 3 * 5 * 6 * 3 * 4


def test_power_integration_reports_large_gaps(tmp_path: Path) -> None:
    module = _module()
    trace = tmp_path / "power.csv"
    trace.write_text(
        "2026/09/01 12:00:00.000, 100\n"
        "2026/09/01 12:00:00.050, 120\n"
        "2026/09/01 12:00:00.300, 140\n",
        encoding="utf-8",
    )
    result = module.integrate_power(trace)
    assert result["samples"] == 3
    assert abs(result["energy_joules"] - 38.0) < 1.0e-3
    assert result["sampling_gap"] is True
