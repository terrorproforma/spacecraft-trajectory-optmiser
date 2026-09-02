#!/usr/bin/env python3
"""Regenerate ``benchmarks/literature/provenance.json`` from the curated records and the
imported ``benchmarks/literature_baselines.json`` reference data.

The committed store must equal the regenerated one (``tests/test_literature_provenance.py``).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from spacepdhcg.literature.pinned_values import ACCESSED, curated_records
from spacepdhcg.literature.provenance import (
    ProvenanceStore,
    ingest_literature_baselines,
    store_path,
    validate_provenance_document,
    write_provenance_store,
)
from spacepdhcg.literature.registry import load_target_registry

# literature_baselines profile id -> registry target id (where they differ)
PROFILE_ALIASES = {
    "acikmese-ploen-2007-pd3": "acikmese-ploen-2007-pd3",
    "szmuk-acikmese-2018-pd6": "szmuk-acikmese-2018-pd6-2d",
    "chari-et-al-2024-pd6-monte-carlo": "chari-2024-pd6-monte-carlo",
    "tafazzol-taheri-earth-mars": "tafazzol-taheri-earth-mars",
    "tafazzol-taheri-earth-dionysus": "tafazzol-taheri-earth-dionysus",
    "esa-tops-2026": "esa-tops-2026",
    "gtopx-2021": "gtopx-2021",
}


def build_store() -> ProvenanceStore:
    curated = curated_records()
    curated_keys = {(record.profile, record.quantity) for record in curated}
    imported = []
    for record in ingest_literature_baselines(accessed=ACCESSED):
        profile = PROFILE_ALIASES.get(record.profile)
        if profile is None:
            continue  # profiles without a registered target keep their manifest-only status
        if (profile, record.quantity) in curated_keys:
            continue
        imported.append(
            replace(record, profile=profile, id=f"{profile}.baseline.{record.quantity}")
        )
    records = curated + imported
    registry = load_target_registry()
    document = ProvenanceStore(
        schema_version="1.0.0",
        generated_by="scripts/literature/build_provenance.py",
        records=tuple(records),
    ).as_dict()
    return validate_provenance_document(document, known_profiles=registry.ids())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destination (default: the committed benchmarks/literature/provenance.json)",
    )
    parser.add_argument("--check", action="store_true", help="fail if the committed store differs")
    arguments = parser.parse_args()
    if arguments.output is None:
        arguments.output = store_path()
    store = build_store()
    if arguments.check:
        import json

        committed = json.loads(arguments.output.read_text(encoding="utf-8"))
        if committed != store.as_dict():
            raise SystemExit(
                "provenance store is stale; rerun scripts/literature/build_provenance.py"
            )
        print(f"provenance store up to date: {len(store.records)} records")
        return
    write_provenance_store(store, arguments.output)
    print(f"wrote {len(store.records)} records to {arguments.output}")
    for label, count in store.labels().items():
        print(f"  {label:22s} {count}")


if __name__ == "__main__":
    main()
