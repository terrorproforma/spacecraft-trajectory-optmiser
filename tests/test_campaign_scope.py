from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.campaign_scope import (
    ACTIVE_SINGLE_GPU_REQUIREMENTS,
    ACTIVE_SINGLE_GPU_SCOPE_ID,
    DEFERRED_MULTI_GPU_REQUIREMENTS,
    HISTORICAL_FULL_SCOPE_ID,
    SCOPE_DEFINITIONS,
    CampaignScopeError,
    effective_scope_id,
    validate_claims_for_scope,
    validate_run_scope,
)
from spacepdhcg.paper1.evidence import load_campaign
from spacepdhcg.paper1.freeze import FreezeError, _check_scope_evidence, validate_campaign_config
from spacepdhcg.paper1.synthetic import generate_synthetic_campaign

ROOT = Path(__file__).resolve().parents[1]


def test_versioned_scope_records_match_code_and_schema() -> None:
    schema = json.loads(
        (ROOT / "experiments/schema/campaign_scope.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for scope_id, expected in SCOPE_DEFINITIONS.items():
        record = json.loads(
            (ROOT / f"benchmarks/campaign_scopes/{scope_id}.json").read_text(encoding="utf-8")
        )
        validator.validate(record)
        for key in (
            "scope_id",
            "active_hypotheses",
            "deferred_hypotheses",
            "included_products",
            "deferred_products",
            "allowed_gpu_counts",
            "requires_physical_g5",
        ):
            assert record[key] == expected[key]
        assert record.get("amendments", []) == expected.get("amendments", [])
        for amendment in record.get("amendments", []):
            for key in ("path", "lock", "document"):
                assert (ROOT / amendment[key]).is_file(), amendment[key]
            digest, name = (ROOT / amendment["lock"]).read_text().split()
            assert name == Path(amendment["path"]).name
            assert hashlib.sha256((ROOT / amendment["path"]).read_bytes()).hexdigest() == digest
    active = json.loads(
        (ROOT / f"benchmarks/campaign_scopes/{ACTIVE_SINGLE_GPU_SCOPE_ID}.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(active["active_requirements"]) == ACTIVE_SINGLE_GPU_REQUIREMENTS
    assert tuple(active["deferred_requirements"]) == DEFERRED_MULTI_GPU_REQUIREMENTS


def test_historical_config_is_readable_and_scoped_config_is_explicit(tmp_path: Path) -> None:
    campaign = generate_synthetic_campaign(tmp_path / "campaign")
    config_path = campaign / "campaign-config.synthetic.json"
    legacy = json.loads(config_path.read_text(encoding="utf-8"))
    validate_campaign_config(legacy)
    assert effective_scope_id(legacy) == HISTORICAL_FULL_SCOPE_ID

    scoped = dict(legacy)
    scoped["schema_version"] = "1.1.0"
    scoped["campaign_scope_id"] = ACTIVE_SINGLE_GPU_SCOPE_ID
    scoped["claims"] = {**legacy["claims"], "H4": []}
    validate_campaign_config(scoped)
    assert effective_scope_id(scoped) == ACTIVE_SINGLE_GPU_SCOPE_ID


def test_cross_scope_claims_and_evidence_fail_closed(tmp_path: Path) -> None:
    claims = {f"H{index}": [f"claim-h{index}"] for index in range(1, 7)}
    with pytest.raises(CampaignScopeError, match="cross-scope claims"):
        validate_claims_for_scope(ACTIVE_SINGLE_GPU_SCOPE_ID, claims)
    claims["H4"] = []
    validate_claims_for_scope(ACTIVE_SINGLE_GPU_SCOPE_ID, claims)

    validate_run_scope(ACTIVE_SINGLE_GPU_SCOPE_ID, family="P1-C-pd3", gpus=1)
    with pytest.raises(CampaignScopeError, match="2-GPU"):
        validate_run_scope(ACTIVE_SINGLE_GPU_SCOPE_ID, family="P1-C-pd3", gpus=2)
    with pytest.raises(CampaignScopeError, match="P1-F"):
        validate_run_scope(ACTIVE_SINGLE_GPU_SCOPE_ID, family="P1-F-robust-pd", gpus=1)

    campaign = generate_synthetic_campaign(tmp_path / "campaign")
    config = json.loads((campaign / "campaign-config.synthetic.json").read_text(encoding="utf-8"))
    config.update(schema_version="1.1.0", campaign_scope_id=ACTIVE_SINGLE_GPU_SCOPE_ID)
    with pytest.raises(FreezeError, match="cross-scope claims"):
        validate_campaign_config(config)


def test_freeze_scope_boundary_requires_physical_g5_only_for_full_scope(
    tmp_path: Path,
) -> None:
    runs = load_campaign(generate_synthetic_campaign(tmp_path / "campaign"))
    with pytest.raises(FreezeError, match="physical G5 evidence"):
        _check_scope_evidence(runs, HISTORICAL_FULL_SCOPE_ID)
    in_scope = tuple(
        run
        for run in runs
        if run.result["dimensions"]["gpus"] == 1
        and not run.result["identity"]["family"].startswith("P1-F")
    )
    _check_scope_evidence(in_scope, ACTIVE_SINGLE_GPU_SCOPE_ID)
