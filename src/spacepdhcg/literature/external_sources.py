"""Pinned external artifacts (TOPS, GTOPX, GTOC data) with checksum verification.

Large downloads are never committed.  ``benchmarks/literature/external_sources.json`` pins every
URL with its SHA-256, size, licence note, and revision; :func:`fetch` returns the cached file
(``$SPACEPDHCG_LITERATURE_CACHE`` or ``~/.cache/spacepdhcg-literature``) after verifying the
digest, downloading only when ``online=True``.  Tests call :func:`require` and skip with an
explicit reason when an artifact is absent and downloads are disabled.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = REPOSITORY_ROOT / "benchmarks" / "literature" / "external_sources.json"
CACHE_ENV = "SPACEPDHCG_LITERATURE_CACHE"
ONLINE_ENV = "SPACEPDHCG_LITERATURE_ONLINE"
USER_AGENT = "spacepdhcg-literature-fetch/0.1"


class ArtifactUnavailable(RuntimeError):
    """Raised when a pinned artifact is not cached and downloads are disabled or fail."""


class ChecksumMismatch(RuntimeError):
    """Raised when a cached or downloaded artifact does not match its pinned digest."""


@dataclass(frozen=True, slots=True)
class ExternalArtifact:
    id: str
    family: str
    url: str
    relative_path: str
    sha256: str
    size_bytes: int
    licence: str
    revision: str | None
    description: str
    accessed: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExternalArtifact:
        return cls(
            id=payload["id"],
            family=payload["family"],
            url=payload["url"],
            relative_path=payload["relative_path"],
            sha256=payload["sha256"],
            size_bytes=int(payload["size_bytes"]),
            licence=payload["licence"],
            revision=payload.get("revision"),
            description=payload.get("description", ""),
            accessed=payload["accessed"],
        )


def cache_root() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "spacepdhcg-literature"


def online_allowed() -> bool:
    return os.environ.get(ONLINE_ENV, "").lower() in {"1", "true", "yes"}


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, ExternalArtifact]:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    artifacts = {}
    for item in document["artifacts"]:
        artifact = ExternalArtifact.from_dict(item)
        if artifact.id in artifacts:
            raise ValueError(f"duplicate artifact id {artifact.id}")
        artifacts[artifact.id] = artifact
    return artifacts


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_path(artifact: ExternalArtifact) -> Path:
    return cache_root() / artifact.relative_path


def verify(artifact: ExternalArtifact, path: Path) -> None:
    if not path.is_file():
        raise ArtifactUnavailable(f"{artifact.id}: {path} is missing")
    actual = sha256_of(path)
    if actual != artifact.sha256:
        raise ChecksumMismatch(
            f"{artifact.id}: sha256 {actual} does not match pinned {artifact.sha256}"
        )
    size = path.stat().st_size
    if size != artifact.size_bytes:
        raise ChecksumMismatch(
            f"{artifact.id}: size {size} differs from pinned {artifact.size_bytes}"
        )


def fetch(artifact_id: str, *, online: bool | None = None, manifest: dict | None = None) -> Path:
    artifacts = manifest or load_manifest()
    artifact = artifacts[artifact_id]
    path = cached_path(artifact)
    if path.is_file():
        verify(artifact, path)
        return path
    allowed = online_allowed() if online is None else online
    if not allowed:
        raise ArtifactUnavailable(
            f"{artifact.id}: not cached at {path}; set {ONLINE_ENV}=1 or run "
            "'spacepdhcg literature fetch' to download it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(artifact.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as handle:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            handle.write(chunk)
    verify(artifact, path)
    return path


def require(artifact_ids: Iterable[str]) -> dict[str, Path]:
    """Return cached paths for all ids or raise :class:`ArtifactUnavailable` for the first gap."""

    manifest = load_manifest()
    return {artifact_id: fetch(artifact_id, manifest=manifest) for artifact_id in artifact_ids}


def status() -> list[dict[str, Any]]:
    rows = []
    for artifact in load_manifest().values():
        path = cached_path(artifact)
        state = "missing"
        if path.is_file():
            try:
                verify(artifact, path)
                state = "verified"
            except ChecksumMismatch:
                state = "checksum-mismatch"
        rows.append(
            {"id": artifact.id, "family": artifact.family, "state": state, "path": str(path)}
        )
    return rows
