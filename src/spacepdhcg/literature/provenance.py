"""Provenance store for literature-derived values.

Every external datum used by the comparative campaign is recorded with its source, the
exact digits printed by the source, the extraction method, and one evidence label from
``docs/COMPARATIVE_SOLVER_CAMPAIGN.md``:

* ``analytic`` - mathematically known optimum or invariant;
* ``published-reference`` - extracted from a paper or official release, not rerun here;
* ``reproduced-external`` - obtained by running pinned external code locally;
* ``measured-local`` - obtained from SpacePDHCG (or an independent local implementation);
* ``descriptive-only`` - the exact data is unrecoverable; the record documents why.

The JSON schema lives in ``experiments/schema/literature_provenance.schema.json``.  This module
adds the semantic checks the schema cannot express (digit preservation, unique ids, profile
cross-references).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = REPOSITORY_ROOT / "experiments" / "schema" / "literature_provenance.schema.json"
STORE_PATH = REPOSITORY_ROOT / "benchmarks" / "literature" / "provenance.json"
BASELINES_PATH = REPOSITORY_ROOT / "benchmarks" / "literature_baselines.json"

EVIDENCE_LABELS: tuple[str, ...] = (
    "analytic",
    "published-reference",
    "reproduced-external",
    "measured-local",
    "descriptive-only",
)

_NUMBER_PATTERN = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")


class ProvenanceError(ValueError):
    """Raised when a provenance document violates the schema or semantic rules."""


@dataclass(frozen=True, slots=True)
class ProvenanceSource:
    title: str
    url: str
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    version: str | None = None
    revision: str | None = None
    sha256: str | None = None
    location: str | None = None
    licence: str | None = None
    accessed: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"title": self.title, "url": self.url}
        for name in (
            "authors",
            "year",
            "doi",
            "version",
            "revision",
            "sha256",
            "location",
            "licence",
            "accessed",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvenanceSource:
        return cls(**{key: payload.get(key) for key in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    id: str
    profile: str
    quantity: str
    value: Any
    value_text: str
    units: str
    evidence_label: str
    extraction_method: str
    source: ProvenanceSource
    approximate: bool = False
    secondary_source: ProvenanceSource | None = None
    objective_convention: str | None = None
    unrecoverable_reason: str | None = None
    verification_status: str = "source-verified"
    notes: str = ""
    _extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "profile": self.profile,
            "quantity": self.quantity,
            "value": self.value,
            "value_text": self.value_text,
            "units": self.units,
            "evidence_label": self.evidence_label,
            "extraction_method": self.extraction_method,
            "approximate": self.approximate,
            "source": self.source.as_dict(),
        }
        if self.secondary_source is not None:
            payload["secondary_source"] = self.secondary_source.as_dict()
        if self.objective_convention is not None:
            payload["objective_convention"] = self.objective_convention
        if self.unrecoverable_reason is not None:
            payload["unrecoverable_reason"] = self.unrecoverable_reason
        payload["verification_status"] = self.verification_status
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProvenanceRecord:
        secondary = payload.get("secondary_source")
        return cls(
            id=payload["id"],
            profile=payload["profile"],
            quantity=payload["quantity"],
            value=payload.get("value"),
            value_text=payload["value_text"],
            units=payload["units"],
            evidence_label=payload["evidence_label"],
            extraction_method=payload["extraction_method"],
            source=ProvenanceSource.from_dict(payload["source"]),
            approximate=bool(payload.get("approximate", False)),
            secondary_source=ProvenanceSource.from_dict(secondary) if secondary else None,
            objective_convention=payload.get("objective_convention"),
            unrecoverable_reason=payload.get("unrecoverable_reason"),
            verification_status=payload.get("verification_status", "source-verified"),
            notes=payload.get("notes", ""),
        )


@dataclass(frozen=True, slots=True)
class ProvenanceStore:
    schema_version: str
    generated_by: str
    records: tuple[ProvenanceRecord, ...]

    def __getitem__(self, record_id: str) -> ProvenanceRecord:
        for record in self.records:
            if record.id == record_id:
                return record
        raise KeyError(record_id)

    def for_profile(self, profile: str) -> tuple[ProvenanceRecord, ...]:
        return tuple(record for record in self.records if record.profile == profile)

    def profiles(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for record in self.records:
            seen.setdefault(record.profile, None)
        return tuple(seen)

    def labels(self) -> dict[str, int]:
        counts = dict.fromkeys(EVIDENCE_LABELS, 0)
        for record in self.records:
            counts[record.evidence_label] += 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_document": "docs/COMPARATIVE_SOLVER_CAMPAIGN.md",
            "generated_by": self.generated_by,
            "records": [record.as_dict() for record in self.records],
        }


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _digits_preserved(value: Any, value_text: str) -> bool:
    """Check that the numeric ``value`` is exactly the number printed in ``value_text``."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, list):
        # Vectors may be printed in basis-vector shorthand ("-e_1", "1e-2 . I_3x3"), so the
        # rule is: every distinct non-zero magnitude in the value must appear verbatim as a
        # numeric token of the text, and the text must contain at least one numeric token.
        numbers = {abs(float(match.group(0))) for match in _NUMBER_PATTERN.finditer(value_text)}
        if not numbers:
            return False
        flat: list[float] = []

        def _flatten(item: Any) -> None:
            if isinstance(item, list):
                for sub in item:
                    _flatten(sub)
            else:
                flat.append(float(item))

        _flatten(value)
        return all(abs(entry) in numbers for entry in flat if entry != 0.0)
    numbers = [match.group(0) for match in _NUMBER_PATTERN.finditer(value_text)]
    if not numbers:
        return False
    # A scalar may be printed with a unit prefix or as an expression; the first number
    # must equal the stored value exactly.
    return float(numbers[0]) == float(value)


def validate_provenance_document(
    document: Mapping[str, Any],
    *,
    known_profiles: Iterable[str] | None = None,
    schema: Mapping[str, Any] | None = None,
) -> ProvenanceStore:
    """Validate against the JSON schema plus semantic rules and return the typed store."""

    schema_document = dict(schema) if schema is not None else load_schema()
    validator = jsonschema.Draft202012Validator(schema_document)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        raise ProvenanceError(f"schema violation at {location}: {first.message}")

    records = [ProvenanceRecord.from_dict(item) for item in document["records"]]
    seen: set[str] = set()
    profiles = set(known_profiles) if known_profiles is not None else None
    for record in records:
        if record.id in seen:
            raise ProvenanceError(f"duplicate provenance id {record.id!r}")
        seen.add(record.id)
        if record.evidence_label not in EVIDENCE_LABELS:
            raise ProvenanceError(f"{record.id}: unknown evidence label {record.evidence_label!r}")
        if record.evidence_label == "descriptive-only":
            if record.value is not None and record.extraction_method != "unrecoverable":
                raise ProvenanceError(
                    f"{record.id}: descriptive-only records must not carry a recovered value"
                )
        elif not _digits_preserved(record.value, record.value_text):
            raise ProvenanceError(
                f"{record.id}: value {record.value!r} is not the number printed in "
                f"value_text {record.value_text!r}; preserve all source digits"
            )
        if record.extraction_method == "figure-digitised" and not record.approximate:
            raise ProvenanceError(f"{record.id}: digitised plot values must be approximate")
        if record.evidence_label in {"measured-local", "reproduced-external"}:
            if record.extraction_method not in {"local-measurement", "code", "data-file"}:
                raise ProvenanceError(
                    f"{record.id}: {record.evidence_label} records must come from a local "
                    "measurement, pinned code, or a data file"
                )
        if profiles is not None and record.profile not in profiles:
            raise ProvenanceError(
                f"{record.id}: profile {record.profile!r} is not a registered literature target"
            )
    return ProvenanceStore(
        schema_version=document["schema_version"],
        generated_by=document["generated_by"],
        records=tuple(records),
    )


def load_provenance_store(
    path: Path = STORE_PATH,
    *,
    known_profiles: Iterable[str] | None = None,
) -> ProvenanceStore:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_provenance_document(document, known_profiles=known_profiles)


def write_provenance_store(store: ProvenanceStore, path: Path = STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(store.as_dict(), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def format_value_text(value: Any) -> str:
    """Render a value the way ``value_text`` should look when we are the source."""

    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_value_text(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def ingest_literature_baselines(
    path: Path = BASELINES_PATH,
    *,
    accessed: str,
) -> list[ProvenanceRecord]:
    """Turn ``benchmarks/literature_baselines.json`` reference data into provenance records.

    The baselines manifest was written before source-level verification, so every imported
    scalar is labelled ``published-reference`` with ``verification_status`` set to
    ``requires-source-verification`` unless a curated record in ``pinned_values`` supersedes it.
    """

    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    records: list[ProvenanceRecord] = []
    for profile in manifest["profiles"]:
        reference = profile.get("reference_data")
        if not isinstance(reference, dict):
            continue
        source = ProvenanceSource(
            title=profile.get("purpose", profile["id"]),
            url=profile.get("source_url") or f"https://doi.org/{profile['doi']}",
            doi=profile.get("doi"),
            version=profile.get("source_revision"),
            location="benchmarks/literature_baselines.json reference_data",
            accessed=accessed,
        )
        for key, value in reference.items():
            if isinstance(value, (list, dict)):
                # Structured sub-tables (GTOPX instances, GTOC editions) are pinned by the
                # curated records; skip their bulk import here.
                continue
            records.append(
                ProvenanceRecord(
                    id=f"{profile['id']}.baseline.{key}",
                    profile=profile["id"],
                    quantity=key,
                    value=value,
                    value_text=format_value_text(value),
                    units=_units_from_key(key),
                    evidence_label=profile.get("evidence_label", "published-reference"),
                    extraction_method="manifest-import",
                    approximate=False,
                    source=source,
                    verification_status="requires-source-verification",
                    notes="imported from the user's literature_baselines manifest; digits as "
                    "typed there, verified against the primary source only where a "
                    "curated record with the same quantity exists",
                )
            )
    return records


def _units_from_key(key: str) -> str:
    for suffix, units in (
        ("_days", "day"),
        ("_kg", "kg"),
        ("_seconds", "s"),
        ("_newtons", "N"),
        ("_nondimensional", "nondimensional"),
        ("_years", "year"),
    ):
        if key.endswith(suffix):
            return units
    return "dimensionless"
