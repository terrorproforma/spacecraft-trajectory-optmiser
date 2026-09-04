"""Fail-closed loading and indexing of archived Paper 1 evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from spacepdhcg.experiments import RunManifest, read_paper1_result

EVIDENCE_SCHEMA_VERSION: Final = "1.0.0"
TERMINAL_STATUSES: Final = frozenset(
    {
        "qualified",
        "unqualified",
        "unsupported",
        "oom",
        "timeout",
        "numerical",
        "infeasible",
        "unrun",
        "failed",
    }
)
FAILURE_STATUSES: Final = TERMINAL_STATUSES - {"qualified"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMBEDDED_SHA256 = re.compile(r"(?:^|[^0-9a-f])[0-9a-f]{64}(?:[^0-9a-f]|$)")
_CONTENT_ADDRESSED_SCHEMES = frozenset({"ipfs", "sha256"})
_VERSIONABLE_SCHEMES = frozenset({"artifact", "doi", "gs", "s3"})


class EvidenceError(ValueError):
    """Raised when archived evidence is incomplete, mutable, or untraceable."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the canonical UTF-8 representation used by all G6 hashes."""

    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)
    return path


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object")
    return value


def _require_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise EvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _immutable_uri(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{name} must be a non-empty immutable URI")
    parsed = urlparse(value)
    if parsed.scheme in _CONTENT_ADDRESSED_SCHEMES:
        return value
    versioned = _EMBEDDED_SHA256.search(parsed.path) is not None or any(
        token in parsed.query for token in ("versionId=", "generation=", "sha256=")
    )
    if parsed.scheme in _VERSIONABLE_SCHEMES and versioned:
        return value
    if parsed.scheme == "https" and (
        versioned or "/releases/download/" in parsed.path or "/artifacts/" in parsed.path
    ):
        return value
    raise EvidenceError(f"{name} is not recognisably immutable: {value!r}")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    uri: str
    sha256: str
    internal_index_sha256: str
    media_type: str
    local_path: str | None = None

    @classmethod
    def from_mapping(cls, value: Any, name: str) -> ArtifactReference:
        raw = _require_mapping(value, name)
        required = {"uri", "sha256", "internal_index_sha256", "media_type"}
        missing = sorted(required - set(raw))
        unknown = sorted(set(raw) - required - {"local_path"})
        if missing:
            raise EvidenceError(f"{name} is missing: {', '.join(missing)}")
        if unknown:
            raise EvidenceError(f"{name} has unknown fields: {', '.join(unknown)}")
        media_type = raw["media_type"]
        if not isinstance(media_type, str) or not media_type:
            raise EvidenceError(f"{name}.media_type must be non-empty")
        local_path = raw.get("local_path")
        if local_path is not None and (not isinstance(local_path, str) or not local_path):
            raise EvidenceError(f"{name}.local_path must be a non-empty string or null")
        return cls(
            uri=_immutable_uri(raw["uri"], f"{name}.uri"),
            sha256=_require_digest(raw["sha256"], f"{name}.sha256"),
            internal_index_sha256=_require_digest(
                raw["internal_index_sha256"], f"{name}.internal_index_sha256"
            ),
            media_type=media_type,
            local_path=local_path,
        )

    def verify_local(self, run_directory: Path) -> None:
        if self.local_path is None:
            return
        candidate = (run_directory / self.local_path).resolve()
        try:
            candidate.relative_to(run_directory.resolve())
        except ValueError as error:
            raise EvidenceError(
                f"artifact local path escapes run directory: {self.local_path}"
            ) from error
        if not candidate.is_file():
            raise EvidenceError(f"artifact payload is missing: {candidate}")
        actual = sha256_path(candidate)
        if actual != self.sha256:
            raise EvidenceError(
                f"artifact hash mismatch for {candidate}: {actual} != {self.sha256}"
            )


@dataclass(frozen=True, slots=True)
class ArchivedRun:
    run_directory: Path
    manifest_path: Path
    result_path: Path
    evidence_path: Path
    manifest: RunManifest
    result: Mapping[str, Any]
    residual_evidence: ArtifactReference
    replay_evidence: ArtifactReference
    archive: ArtifactReference

    @property
    def run_id(self) -> str:
        return str(self.result["identity"]["run_id"])

    @property
    def status(self) -> str:
        return str(self.result["identity"]["status"])

    @property
    def coordinate(self) -> tuple[Any, ...]:
        identity = self.result["identity"]
        dimensions = self.result["dimensions"]
        return (
            identity["family"],
            identity["instance_id"],
            dimensions["intervals"],
            dimensions["scenarios"],
            dimensions["gpus"],
            identity["solver"],
            identity["policy"],
            identity.get("quality_tier"),
            identity.get("warm_mode"),
        )


def load_archived_run(run_directory: str | Path, *, verify_payloads: bool = True) -> ArchivedRun:
    """Load one archived run and verify every G6 traceability edge."""

    directory = Path(run_directory)
    manifest_path = directory / "run-manifest.json"
    result_path = directory / "paper1-result.json"
    evidence_path = directory / "evidence-record.json"
    for path in (manifest_path, result_path, evidence_path):
        if not path.is_file():
            raise EvidenceError(f"archived run is missing {path.name}: {directory}")

    manifest = RunManifest.read(manifest_path)
    result = read_paper1_result(result_path)
    envelope = _require_mapping(json.loads(evidence_path.read_text(encoding="utf-8")), "evidence")
    required = {
        "schema_version",
        "run_id",
        "manifest_sha256",
        "result_sha256",
        "residual_evidence",
        "replay_evidence",
        "archive",
    }
    missing = sorted(required - set(envelope))
    unknown = sorted(set(envelope) - required)
    if missing or unknown:
        raise EvidenceError(
            f"evidence envelope fields invalid; missing={missing}, unknown={unknown}"
        )
    if envelope["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceError("unsupported evidence schema version")
    run_ids = {
        str(envelope["run_id"]),
        manifest.run_id,
        str(result["identity"]["run_id"]),
    }
    if len(run_ids) != 1:
        raise EvidenceError(f"run-id traceability mismatch: {sorted(run_ids)}")
    if manifest.repository["commit"] != result["identity"]["repository_commit"]:
        raise EvidenceError("manifest/result repository commit mismatch")
    if manifest.problem["family"] != result["identity"]["family"]:
        raise EvidenceError("manifest/result family mismatch")
    if manifest.problem["instance_id"] != result["identity"]["instance_id"]:
        raise EvidenceError("manifest/result instance mismatch")
    if manifest.solver["name"] != result["identity"]["solver"]:
        raise EvidenceError("manifest/result solver mismatch")
    if result["identity"]["status"] not in TERMINAL_STATUSES:
        raise EvidenceError("result does not retain a recognised terminal status")
    expected_manifest = _require_digest(envelope["manifest_sha256"], "manifest_sha256")
    expected_result = _require_digest(envelope["result_sha256"], "result_sha256")
    if sha256_path(manifest_path) != expected_manifest:
        raise EvidenceError("run manifest content hash mismatch")
    if sha256_path(result_path) != expected_result:
        raise EvidenceError("compact result content hash mismatch")

    residual = ArtifactReference.from_mapping(envelope["residual_evidence"], "residual_evidence")
    replay = ArtifactReference.from_mapping(envelope["replay_evidence"], "replay_evidence")
    archive = ArtifactReference.from_mapping(envelope["archive"], "archive")
    if residual.uri == replay.uri or residual.sha256 == replay.sha256:
        raise EvidenceError("residual and replay evidence must be independent artifacts")
    if verify_payloads:
        for artifact in (residual, replay, archive):
            artifact.verify_local(directory)
    return ArchivedRun(
        run_directory=directory,
        manifest_path=manifest_path,
        result_path=result_path,
        evidence_path=evidence_path,
        manifest=manifest,
        result=result,
        residual_evidence=residual,
        replay_evidence=replay,
        archive=archive,
    )


def load_campaign(root: str | Path, *, verify_payloads: bool = True) -> tuple[ArchivedRun, ...]:
    """Load all runs in stable run-id order without dropping failures."""

    campaign = Path(root)
    directories = sorted({path.parent for path in campaign.rglob("evidence-record.json")})
    if not directories:
        raise EvidenceError(f"campaign contains no archived evidence records: {campaign}")
    runs = tuple(load_archived_run(path, verify_payloads=verify_payloads) for path in directories)
    counts = Counter(run.run_id for run in runs)
    duplicates = sorted(run_id for run_id, count in counts.items() if count > 1)
    if duplicates:
        raise EvidenceError(f"duplicate run IDs: {', '.join(duplicates)}")
    return tuple(sorted(runs, key=lambda run: run.run_id))


def evidence_index(runs: Iterable[ArchivedRun]) -> dict[str, Any]:
    """Build the deterministic run-to-artifact index used by freeze checks."""

    entries = []
    status_counts: Counter[str] = Counter()
    for run in sorted(runs, key=lambda item: item.run_id):
        status_counts[run.status] += 1
        entries.append(
            {
                "run_id": run.run_id,
                "status": run.status,
                "coordinate": list(run.coordinate),
                "manifest_sha256": sha256_path(run.manifest_path),
                "result_sha256": sha256_path(run.result_path),
                "residual_uri": run.residual_evidence.uri,
                "residual_sha256": run.residual_evidence.sha256,
                "replay_uri": run.replay_evidence.uri,
                "replay_sha256": run.replay_evidence.sha256,
                "archive_uri": run.archive.uri,
                "archive_sha256": run.archive.sha256,
            }
        )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_count": len(entries),
        "status_counts": dict(sorted(status_counts.items())),
        "runs": entries,
    }
