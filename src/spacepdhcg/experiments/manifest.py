"""Versioned, machine-readable evidence records for optimisation experiments.

The manifest deliberately uses only the Python standard library. Python is the
orchestration layer here; solver iterations, trajectory propagation and coefficient
updates remain in the C++/CUDA numerical core.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class HostMetadata:
    """Host and accelerator identity recorded independently of solver output."""

    hostname: str
    operating_system: str
    architecture: str
    processor: str
    logical_cpu_count: int | None
    memory_bytes: int | None
    accelerator_vendor: str | None = None
    accelerator_model: str | None = None
    accelerator_count: int | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    interconnect: str | None = None

    def validate(self) -> None:
        for name in ("hostname", "operating_system", "architecture"):
            if not getattr(self, name):
                raise ValueError(f"{name} may not be empty")
        for name in ("logical_cpu_count", "memory_bytes", "accelerator_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when supplied")


@dataclass(slots=True)
class RunManifest:
    """One immutable scientific run once written to an evidence directory."""

    run_id: str
    timestamp_utc: str
    repository: dict[str, Any]
    host: HostMetadata
    problem: dict[str, Any]
    solver: dict[str, Any]
    quality: dict[str, float | int | bool | None]
    timing: dict[str, float | int | None]
    status: str
    schema_version: str = SCHEMA_VERSION
    upstream: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema {self.schema_version!r} is not supported; "
                f"expected {SCHEMA_VERSION!r}"
            )
        if not self.run_id or not self.timestamp_utc:
            raise ValueError("run_id and timestamp_utc may not be empty")
        parsed = datetime.fromisoformat(self.timestamp_utc.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp_utc must include an explicit UTC offset")
        if not self.repository.get("commit"):
            raise ValueError("repository.commit is required")
        if not self.problem.get("family") or not self.problem.get("instance_id"):
            raise ValueError("problem family and instance_id are required")
        if not self.solver.get("name") or not self.solver.get("mode"):
            raise ValueError("solver name and mode are required")
        if self.status not in {
            "solved",
            "qualified",
            "iteration_limit",
            "infeasible",
            "failed",
            "skipped",
        }:
            raise ValueError(f"unsupported manifest status {self.status!r}")
        self.host.validate()
        self._validate_numeric_mapping(self.quality, "quality")
        self._validate_numeric_mapping(self.timing, "timing")
        for name, path in self.artifacts.items():
            if not name or not path:
                raise ValueError("artifact names and paths may not be empty")

    @staticmethod
    def _validate_numeric_mapping(
        values: Mapping[str, float | int | bool | None],
        name: str,
    ) -> None:
        for key, value in values.items():
            if not key:
                raise ValueError(f"{name} keys may not be empty")
            if isinstance(value, bool) or value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name}.{key} must be numeric, bool or null")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name}.{key} must be finite when supplied")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.as_dict(),
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"

    def write(self, path: str | Path) -> Path:
        """Write atomically so interrupted benchmark jobs cannot leave valid-looking JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(self.to_json(), encoding="utf-8")
        temporary.replace(destination)
        return destination

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RunManifest:
        data = dict(payload)
        host = data.get("host")
        if not isinstance(host, Mapping):
            raise TypeError("manifest host must be an object")
        data["host"] = HostMetadata(**dict(host))
        manifest = cls(**data)
        manifest.validate()
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("manifest root must be an object")
        return cls.from_dict(payload)


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(physical_pages, int):
        return None
    return page_size * physical_pages


def capture_host_metadata(
    *,
    accelerator_vendor: str | None = None,
    accelerator_model: str | None = None,
    accelerator_count: int | None = None,
    driver_version: str | None = None,
    runtime_version: str | None = None,
    interconnect: str | None = None,
) -> HostMetadata:
    """Capture stable host fields; accelerator fields are explicit, never guessed."""

    metadata = HostMetadata(
        hostname=socket.gethostname(),
        operating_system=platform.platform(),
        architecture=platform.machine(),
        processor=platform.processor() or platform.machine(),
        logical_cpu_count=os.cpu_count(),
        memory_bytes=_physical_memory_bytes(),
        accelerator_vendor=accelerator_vendor,
        accelerator_model=accelerator_model,
        accelerator_count=accelerator_count,
        driver_version=driver_version,
        runtime_version=runtime_version,
        interconnect=interconnect,
    )
    metadata.validate()
    return metadata


def make_run_manifest(
    *,
    repository_commit: str,
    problem: Mapping[str, Any],
    solver: Mapping[str, Any],
    status: str,
    quality: Mapping[str, float | int | bool | None],
    timing: Mapping[str, float | int | None],
    repository_url: str = "https://github.com/terrorproforma/spacecraft-trajectory-optmiser",
    branch: str | None = None,
    dirty: bool = False,
    host: HostMetadata | None = None,
    upstream: Mapping[str, Any] | None = None,
    experiment: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, str] | None = None,
    notes: list[str] | None = None,
) -> RunManifest:
    """Construct and validate a manifest with a UTC timestamp and opaque run UUID."""

    manifest = RunManifest(
        run_id=str(uuid4()),
        timestamp_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        repository={
            "url": repository_url,
            "commit": repository_commit,
            "branch": branch,
            "dirty": bool(dirty),
        },
        upstream=dict(upstream or {}),
        host=host or capture_host_metadata(),
        experiment=dict(experiment or {}),
        problem=dict(problem),
        solver=dict(solver),
        quality=dict(quality),
        timing=dict(timing),
        status=status,
        artifacts=dict(artifacts or {}),
        notes=list(notes or []),
    )
    manifest.validate()
    return manifest


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
