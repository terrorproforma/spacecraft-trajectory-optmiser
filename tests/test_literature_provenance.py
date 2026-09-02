"""Provenance store: schema, semantic rules, and freeze against regeneration."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from spacepdhcg.literature.provenance import (
    EVIDENCE_LABELS,
    STORE_PATH,
    ProvenanceError,
    load_provenance_store,
    validate_provenance_document,
)
from spacepdhcg.literature.registry import load_target_registry

ROOT = Path(__file__).resolve().parents[1]


def _document() -> dict:
    return json.loads(STORE_PATH.read_text(encoding="utf-8"))


def test_committed_store_validates_against_schema_and_registry() -> None:
    registry = load_target_registry()
    store = load_provenance_store(known_profiles=registry.ids())
    assert len(store.records) >= 100
    assert set(store.labels()) == set(EVIDENCE_LABELS)
    assert store.labels()["published-reference"] > 50
    assert store.labels()["descriptive-only"] >= 2
    # every profile in the store is a registered target
    assert set(store.profiles()) <= set(registry.ids())


def test_committed_store_matches_regeneration() -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "literature"))
    import build_provenance  # type: ignore[import-not-found]

    regenerated = build_provenance.build_store().as_dict()
    assert regenerated == _document(), "run scripts/literature/build_provenance.py"


def test_every_record_names_source_and_digits() -> None:
    store = load_provenance_store()
    for record in store.records:
        assert record.source.url.startswith(("http", "file:"))
        assert record.value_text.strip(), record.id
        if record.evidence_label == "descriptive-only":
            assert record.unrecoverable_reason, record.id
        if record.extraction_method == "secondary-citation":
            assert record.secondary_source is not None, record.id


def test_digit_preservation_is_enforced() -> None:
    document = _document()
    broken = copy.deepcopy(document)
    target = next(r for r in broken["records"] if isinstance(r["value"], float))
    target["value"] = target["value"] * 1.0001
    with pytest.raises(ProvenanceError, match="preserve all source digits"):
        validate_provenance_document(broken)


def test_descriptive_only_requires_reason_and_no_value() -> None:
    document = _document()
    broken = copy.deepcopy(document)
    target = next(r for r in broken["records"] if r["evidence_label"] == "descriptive-only")
    target.pop("unrecoverable_reason")
    with pytest.raises(ProvenanceError):
        validate_provenance_document(broken)


def test_unknown_label_and_duplicate_ids_rejected() -> None:
    document = _document()
    broken = copy.deepcopy(document)
    broken["records"][0]["evidence_label"] = "guessed"
    with pytest.raises(ProvenanceError, match="schema violation"):
        validate_provenance_document(broken)
    duplicate = copy.deepcopy(document)
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    with pytest.raises(ProvenanceError, match="duplicate"):
        validate_provenance_document(duplicate)


def test_figure_digitised_must_be_approximate() -> None:
    document = _document()
    broken = copy.deepcopy(document)
    record = broken["records"][0]
    record["extraction_method"] = "figure-digitised"
    record["approximate"] = False
    with pytest.raises(ProvenanceError):
        validate_provenance_document(broken)


def test_unknown_profile_rejected_when_registry_supplied() -> None:
    document = _document()
    broken = copy.deepcopy(document)
    broken["records"][0]["profile"] = "not-a-target"
    with pytest.raises(ProvenanceError, match="not a registered literature target"):
        validate_provenance_document(broken, known_profiles=load_target_registry().ids())


def test_key_pinned_values_present_with_source_digits() -> None:
    store = load_provenance_store()
    assert store["tafazzol-taheri-earth-mars.final_mass"].value_text == "603.935"
    assert store["tafazzol-taheri-earth-dionysus.final_mass"].value_text == "2718.33"
    assert store["acikmese-ploen-2007-pd3.fuel_used_glide_slope"].value_text == "399.5"
    assert store["blackmore-2010-pd3-case1.fuel_used"].value_text == "399.4"
    assert store["szmuk-acikmese-2018-pd6-2d.alpha_mdot"].evidence_label == "descriptive-only"
    assert store["gtopx-2021.best_known.gtoc1"].value_text == "-1581950.131840605288744"
