from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.paper1 import (
    EvidenceError,
    FreezeError,
    build_campaign,
    build_decisions,
    freeze_campaign,
    generate_synthetic_campaign,
    load_campaign,
    verify_reproducible_build,
)
from spacepdhcg.paper1.aggregate import FIGURES, TABLES
from spacepdhcg.paper1.decisions import BOOTSTRAP_SAMPLES, OUTCOMES
from spacepdhcg.paper1.evidence import load_archived_run


def _campaign(tmp_path: Path) -> Path:
    return generate_synthetic_campaign(tmp_path / "campaign")


def test_synthetic_campaign_retains_all_terminal_classes(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    runs = load_campaign(campaign)
    statuses = {run.status for run in runs}
    assert {
        "qualified",
        "oom",
        "timeout",
        "numerical",
        "unsupported",
        "infeasible",
        "failed",
        "unrun",
    } <= statuses
    assert len(runs) == 14
    assert all("SYNTHETIC FIXTURE ONLY" in run.result["notes"][0] for run in runs)


def test_evidence_fails_closed_on_hash_and_run_id_mismatch(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    run = campaign / "syn-positive-001"
    envelope_path = run / "evidence-record.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["manifest_sha256"] = "0" * 64
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(EvidenceError, match="manifest content hash mismatch"):
        load_archived_run(run)

    generate_synthetic_campaign(campaign)
    result_path = run / "paper1-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["identity"]["run_id"] = "mismatched"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(EvidenceError, match=r"run-id traceability mismatch|content hash mismatch"):
        load_archived_run(run)


def test_local_only_and_nonindependent_evidence_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    run = campaign / "syn-positive-001"
    envelope_path = run / "evidence-record.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["residual_evidence"]["uri"] = "file:///tmp/residual.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(EvidenceError, match="not recognisably immutable"):
        load_archived_run(run)

    generate_synthetic_campaign(campaign)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    envelope["replay_evidence"] = copy.deepcopy(envelope["residual_evidence"])
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(EvidenceError, match="must be independent"):
        load_archived_run(run)


def test_build_emits_only_frozen_products_and_maps_failures(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    manifest = build_campaign(campaign, output, synthetic=True)
    expected = [product.product_id for product in (*FIGURES, *TABLES)]
    assert manifest["product_manifest"]["product_ids"] == expected
    assert (output / "products/fig04_horizon_crossover.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "products/fig04_horizon_crossover.png").read_bytes().startswith(b"\x89PNG")
    mapped = {
        run_id
        for product in manifest["product_manifest"]["products"]
        for run_id in product["run_ids"]
    }
    assert {run.run_id for run in load_campaign(campaign)} <= mapped


def test_decisions_are_complete_but_do_not_invent_resolution(tmp_path: Path) -> None:
    runs = load_campaign(_campaign(tmp_path))
    index = build_decisions(runs, tmp_path / "decisions")
    assert {item["hypothesis"] for item in index["decisions"]} == {
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
    }
    for item in index["decisions"]:
        record = json.loads((tmp_path / "decisions" / item["file"]).read_text(encoding="utf-8"))
        assert record["outcome"] in OUTCOMES
        assert record["bootstrap_samples"] == BOOTSTRAP_SAMPLES
    assert {item["outcome"] for item in index["decisions"]} == {
        "supported",
        "rejected",
        "mixed",
        "unresolved",
    }


def test_synthetic_campaign_can_never_freeze(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(FreezeError, match=r"synthetic campaigns.*never be frozen"):
        freeze_campaign(
            Path.cwd(),
            campaign,
            campaign / "campaign-config.synthetic.json",
            tmp_path / "frozen",
        )


def test_incomplete_campaign_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    (campaign / "syn-timeout-001/evidence-record.json").unlink()
    runs = load_campaign(campaign)
    assert "syn-timeout-001" not in {run.run_id for run in runs}
    config_path = campaign / "campaign-config.synthetic.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["synthetic"] = False
    config["repository_commit"] = "0" * 40
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FreezeError):
        freeze_campaign(Path.cwd(), campaign, config_path, tmp_path / "freeze")


def test_build_is_byte_reproducible(tmp_path: Path) -> None:
    result = verify_reproducible_build(_campaign(tmp_path), synthetic=True)
    assert result["reproducible"] is True
    assert result["file_count"] > 40


def test_g6_schemas_and_generated_decisions_validate(tmp_path: Path) -> None:
    schema_root = Path(__file__).parents[1] / "experiments/schema"
    schemas = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in schema_root.glob("paper1_*.schema.json")
    }
    for schema in schemas.values():
        Draft202012Validator.check_schema(schema)

    campaign = _campaign(tmp_path)
    envelope = json.loads(
        (campaign / "syn-positive-001/evidence-record.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schemas["paper1_evidence.schema"]).validate(envelope)
    config = json.loads((campaign / "campaign-config.synthetic.json").read_text(encoding="utf-8"))
    Draft202012Validator(schemas["paper1_campaign.schema"]).validate(config)
    decisions = tmp_path / "decisions"
    index = build_decisions(load_campaign(campaign), decisions)
    validator = Draft202012Validator(schemas["paper1_decision.schema"])
    for item in index["decisions"]:
        validator.validate(json.loads((decisions / item["file"]).read_text(encoding="utf-8")))
    output = tmp_path / "products"
    manifest = build_campaign(campaign, output, synthetic=True)
    product_validator = Draft202012Validator(schemas["paper1_product_source.schema"])
    for item in manifest["product_manifest"]["products"]:
        product_validator.validate(
            json.loads((output / "products" / item["source"]).read_text(encoding="utf-8"))
        )
