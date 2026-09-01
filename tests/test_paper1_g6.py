from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.campaign_scope import ACTIVE_SINGLE_GPU_SCOPE_ID, scope_definition
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
from spacepdhcg.paper1.aggregate import (
    FIGURES,
    TABLES,
    AggregationError,
    build_products,
)
from spacepdhcg.paper1.decisions import BOOTSTRAP_SAMPLES, OUTCOMES
from spacepdhcg.paper1.evidence import ArchivedRun, load_archived_run
from spacepdhcg.paper1.freeze import CLAIM_PRODUCTS


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
    expected = [
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "F07",
        "F08",
        "F09",
        "F10",
        "F11",
        "F12",
        "T01",
        "T02",
        "T03",
        "T04",
        "T05",
        "T06",
        "T07",
        "T08",
    ]
    assert [product.product_id for product in (*FIGURES, *TABLES)] == expected
    assert manifest["product_manifest"]["product_ids"] == expected
    assert (output / "products/fig04_horizon_crossover.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "products/fig04_horizon_crossover.png").read_bytes().startswith(b"\x89PNG")
    for number, slug in (
        ("09", "accuracy_time_pareto"),
        ("10", "solver_regime_map"),
        ("11", "variational_validation"),
        ("12", "robust_residuals"),
    ):
        assert (output / f"products/fig{number}_{slug}.pdf").read_bytes().startswith(b"%PDF")
        assert (output / f"products/fig{number}_{slug}.png").read_bytes().startswith(b"\x89PNG")
    for number, slug in (("07", "regime_crossover"), ("08", "negative_mixed")):
        assert (output / f"products/tab{number}_{slug}.csv").is_file()
        assert (output / f"products/tab{number}_{slug}.tex").is_file()
    mapped = {
        run_id
        for product in manifest["product_manifest"]["products"]
        for run_id in product["run_ids"]
    }
    assert {run.run_id for run in load_campaign(campaign)} <= mapped


def test_broader_sources_are_traceable_and_never_manual(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    manifest = build_campaign(campaign, output, synthetic=True)
    known = {run.run_id for run in load_campaign(campaign)}
    for item in manifest["product_manifest"]["products"]:
        source = json.loads((output / "products" / item["source"]).read_text(encoding="utf-8"))
        assert source["manual_coordinates"] is False
        assert source["coordinate_origin"] == (
            "validated archived run and referenced evidence fields"
        )
        assert set(source["run_ids"]) <= known
        if item["product_id"] == "F10":
            assert all(set(cell["run_ids"]) <= known for cell in source["data"])
        elif item["product_id"] in {"F11", "F12"}:
            assert all(point["run_id"] in known for point in source["data"])
    pareto = json.loads(
        (output / "products/fig09_accuracy_time_pareto.json").read_text(encoding="utf-8")
    )
    assert all(
        not point["canonical_residual_pareto"] and not point["nonlinear_residual_pareto"]
        for point in pareto["data"]
        if point["status"] != "qualified"
    )
    negative = json.loads(
        (output / "products/tab08_negative_mixed.json").read_text(encoding="utf-8")
    )
    assert {
        row[-1] for row in negative["rows"] if row[1] in {f"H{index}" for index in range(1, 7)}
    } == {"rejected", "mixed", "unresolved"}
    assert all(row[0] and row[6] for row in negative["rows"])
    retained_negative_ids = {run_id for row in negative["rows"] for run_id in row[0].split("|")}
    assert {
        run.run_id for run in load_campaign(campaign) if run.status != "qualified"
    } <= retained_negative_ids


def _decision_records(runs: tuple[ArchivedRun, ...], output: Path) -> dict[str, dict[str, object]]:
    build_decisions(runs, output)
    return {
        hypothesis: json.loads(
            (output / f"{hypothesis.lower()}-decision.json").read_text(encoding="utf-8")
        )
        for hypothesis in (f"H{index}" for index in range(1, 7))
    }


def test_f11_and_f12_missing_diagnostics_fail_closed(tmp_path: Path) -> None:
    runs = load_campaign(_campaign(tmp_path))
    decisions = _decision_records(runs, tmp_path / "decisions")
    for run in runs:
        if run.result["identity"]["family"] == "P1-C-pd3":
            run.manifest.experiment.pop("variational_trials", None)
    with pytest.raises(AggregationError, match="F11 requires trials"):
        build_products(runs, tmp_path / "missing-f11", decisions=decisions, synthetic=True)

    runs = load_campaign(_campaign(tmp_path))
    for run in runs:
        if run.result["identity"]["family"] == "P1-F-robust-pd":
            run.manifest.experiment["robust_iterations"] = [
                item
                for item in run.manifest.experiment["robust_iterations"]
                if item["risk_mode"] != "CVaR"
            ]
    with pytest.raises(AggregationError, match="F12 requires expected"):
        build_products(runs, tmp_path / "missing-f12", decisions=decisions, synthetic=True)


def test_regime_map_refuses_unsupported_unique_winner(tmp_path: Path) -> None:
    runs = load_campaign(_campaign(tmp_path))
    decisions = _decision_records(runs, tmp_path / "decisions")
    for run in runs:
        run.manifest.experiment.pop("measured_repeat_seconds", None)
    output = tmp_path / "no-paired-repeats"
    build_products(runs, output, decisions=decisions, synthetic=True)
    regime = json.loads((output / "fig10_solver_regime_map.json").read_text(encoding="utf-8"))
    assert all(cell["winner"] in {"tie", "no qualified solver"} for cell in regime["data"])
    assert all(cell["paired_confidence_interval_95"] == [None, None] for cell in regime["data"])


def test_reconciliation_document_preserves_authoritative_inventory() -> None:
    document = (
        Path(__file__).parents[1] / "papers/paper1/PRODUCT_CONTRACT_RECONCILIATION.md"
    ).read_text(encoding="utf-8")
    assert "Reconciliation version: **1.0.0**" in document
    assert "F01-F12 and T01-T08" in document
    assert "F11 placement" in document
    assert "F12 diagnostic status" in document
    assert {"F09", "F10", "T07", "T08"} <= {
        product_id for product_ids in CLAIM_PRODUCTS.values() for product_id in product_ids
    }
    assert "F12" in CLAIM_PRODUCTS["H4"]


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


def test_single_gpu_scope_defers_h4_and_excludes_physical_products(tmp_path: Path) -> None:
    runs = tuple(
        run
        for run in load_campaign(_campaign(tmp_path))
        if run.result["dimensions"]["gpus"] == 1
        and not run.result["identity"]["family"].startswith("P1-F")
    )
    decisions_path = tmp_path / "scoped-decisions"
    index = build_decisions(
        runs,
        decisions_path,
        campaign_scope_id=ACTIVE_SINGLE_GPU_SCOPE_ID,
    )
    records = {
        item["hypothesis"]: json.loads((decisions_path / item["file"]).read_text(encoding="utf-8"))
        for item in index["decisions"]
    }
    decision_schema = json.loads(
        (Path(__file__).parents[1] / "experiments/schema/paper1_decision.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for record in records.values():
        Draft202012Validator(decision_schema).validate(record)
    assert records["H4"]["outcome"] == "deferred-not-in-scope"
    assert records["H4"]["input_run_ids"] == []
    assert all(
        records[hypothesis]["outcome"] != "deferred-not-in-scope"
        for hypothesis in {"H1", "H2", "H3", "H5", "H6"}
    )

    scope = scope_definition(ACTIVE_SINGLE_GPU_SCOPE_ID)
    output = tmp_path / "scoped-products"
    manifest = build_products(
        runs,
        output,
        decisions=records,
        synthetic=True,
        campaign_scope_id=ACTIVE_SINGLE_GPU_SCOPE_ID,
        included_product_ids=scope["included_products"],
        deferred_product_ids=scope["deferred_products"],
    )
    assert manifest["campaign_scope_id"] == ACTIVE_SINGLE_GPU_SCOPE_ID
    assert set(manifest["product_ids"]) == set(scope["included_products"])
    assert {item["product_id"] for item in manifest["deferred_products"]} == set(
        scope["deferred_products"]
    )
    product_schema = json.loads(
        (
            Path(__file__).parents[1] / "experiments/schema/paper1_product_source.schema.json"
        ).read_text(encoding="utf-8")
    )
    for item in manifest["products"]:
        source = json.loads((output / item["source"]).read_text(encoding="utf-8"))
        assert source["campaign_scope_id"] == ACTIVE_SINGLE_GPU_SCOPE_ID
        Draft202012Validator(product_schema).validate(source)
    assert not (output / "fig07_multigpu_scaling.json").exists()
    assert not (output / "fig12_robust_residuals.json").exists()
    assert not (output / "tab06_robust_scaling.json").exists()


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
