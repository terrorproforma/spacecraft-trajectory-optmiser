from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, object]:
    with (ROOT / "benchmarks" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_repository_and_families(manifest: dict[str, object]) -> None:
    assert manifest["repository"] == "terrorproforma/spacecraft-trajectory-optmiser"
    families = manifest["families"]
    assert isinstance(families, list) and families
    identifiers = [family["id"] for family in families]
    assert len(identifiers) == len(set(identifiers))


def test_paper1_manifest_is_complete_and_gpu_explicit() -> None:
    manifest = _load("paper1_matrix.json")
    _validate_repository_and_families(manifest)
    assert manifest["schema_version"] == "2.0.0"
    assert manifest["status"] == "preregistered pre-GPU matrix"
    assert manifest["upstream_pdhcg_commit"] == ("167c8b72b4b96d2f94d405b8763e485514192b81")
    assert manifest["result_schema"] == "experiments/schema/paper1_result.schema.json"
    assert manifest["decision_rules"] == "papers/paper1/CLAIMS_AND_DECISION_RULES.md"
    assert manifest["figure_schema"] == "papers/paper1/FIGURE_SCHEMA.md"
    assert manifest["notation"] == "papers/paper1/NOTATION.md"

    backends = manifest["solver_backends"]
    assert isinstance(backends, list)
    backend_names = {backend["id"] for backend in backends}
    assert {
        "clarabel-cpu",
        "osqp-cpu",
        "pdhcg-upstream-one-shot",
        "spacepdhcg-persistent",
        "qoco-gpu",
        "cuclarabel",
        "structured-pipg",
        "hybrid-pdhcg-ipm",
    } == backend_names
    requires_gpu = {backend["id"]: backend["requires_gpu"] for backend in backends}
    assert requires_gpu["spacepdhcg-persistent"] is True
    assert requires_gpu["clarabel-cpu"] is False

    assert {family["id"] for family in manifest["families"]} == {
        "P1-A-banded",
        "P1-B-hcw",
        "P1-C-pd3",
        "P1-D-pd6",
        "P1-E-low-thrust",
        "P1-F-robust-pd",
    }
    campaigns = manifest["campaigns"]
    assert isinstance(campaigns, list)
    assert [campaign["id"] for campaign in campaigns] == [
        "gate-0-cpu",
        "gate-1-one-shot",
        "gate-2-persistence",
        "gate-3-deterministic-scvx",
        "gate-4-adaptive-hybrid",
        "gate-5-multigpu",
    ]
    for field_group in (
        "required_quality_fields",
        "required_timing_fields",
        "required_resource_fields",
    ):
        fields = manifest[field_group]
        assert isinstance(fields, list) and fields
        assert len(fields) == len(set(fields))


def test_paper2_manifest_covers_exact_and_robust_routes() -> None:
    manifest = _load("paper2_matrix.json")
    _validate_repository_and_families(manifest)
    assert manifest["schema_version"] == 1
    metrics = manifest["required_metrics"]
    assert isinstance(metrics, list) and metrics
    assert len(metrics) == len(set(metrics))
    assert "exact_elementary_labels" in manifest["route_methods"]
    assert "robust_scvx" in manifest["arc_fidelities"]
    assert {family["id"] for family in manifest["families"]} == {
        "P2-A",
        "P2-B",
        "P2-C",
        "P2-D",
        "P2-E",
    }
