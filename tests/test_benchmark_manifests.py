from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict[str, object]:
    with (ROOT / "benchmarks" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _validate_common(manifest: dict[str, object]) -> None:
    assert manifest["schema_version"] == 1
    assert manifest["repository"] == "terrorproforma/spacecraft-trajectory-optmiser"
    families = manifest["families"]
    assert isinstance(families, list) and families
    identifiers = [family["id"] for family in families]
    assert len(identifiers) == len(set(identifiers))
    metrics = manifest["required_metrics"]
    assert isinstance(metrics, list) and metrics
    assert len(metrics) == len(set(metrics))


def test_paper1_manifest_is_complete_and_gpu_explicit() -> None:
    manifest = _load("paper1_matrix.json")
    _validate_common(manifest)
    assert manifest["upstream_pdhcg_commit"] == (
        "167c8b72b4b96d2f94d405b8763e485514192b81"
    )
    backends = manifest["solver_backends"]
    assert isinstance(backends, list)
    backend_names = {backend["name"] for backend in backends}
    assert {
        "clarabel-cpu",
        "osqp-cpu",
        "pdhcg-one-shot",
        "spacepdhcg-persistent",
        "qoco-gpu",
        "cuclarabel",
    } <= backend_names
    requires_gpu = {backend["name"]: backend["requires_gpu"] for backend in backends}
    assert requires_gpu["spacepdhcg-persistent"] is True
    assert requires_gpu["clarabel-cpu"] is False
    assert {family["id"] for family in manifest["families"]} == {
        "P1-A",
        "P1-B",
        "P1-C",
        "P1-D",
        "P1-E",
        "P1-F",
    }


def test_paper2_manifest_covers_exact_and_robust_routes() -> None:
    manifest = _load("paper2_matrix.json")
    _validate_common(manifest)
    assert "exact_elementary_labels" in manifest["route_methods"]
    assert "robust_scvx" in manifest["arc_fidelities"]
    assert {family["id"] for family in manifest["families"]} == {
        "P2-A",
        "P2-B",
        "P2-C",
        "P2-D",
        "P2-E",
    }
