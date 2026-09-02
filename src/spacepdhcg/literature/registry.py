"""Machine-readable registry of literature reproduction targets.

``benchmarks/literature/targets.json`` lists every profile/target consumed by the tests, by the
``spacepdhcg literature`` CLI, and (once it lands) by the planner CLI.  Each target names a
runner ``module:function`` that accepts the profile document and an options mapping and returns a
JSON-serialisable record with at least ``target_id``, ``status``, and ``labels``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = REPOSITORY_ROOT / "benchmarks" / "literature" / "targets.json"

SUPPORT_LEVELS: tuple[str, ...] = ("supported", "partial", "unsupported", "descriptive-only")
RESULT_STATUSES: tuple[str, ...] = (
    "reproduced",
    "gap",
    "descriptive-only",
    "unsupported",
    "blocked",
)


class RegistryError(ValueError):
    """Raised when the target registry is malformed."""


@dataclass(frozen=True, slots=True)
class LiteratureTarget:
    id: str
    family: str | None
    title: str
    runner: str
    profile: str | None
    support: str
    unsupported_reason: str | None
    requires_artifacts: tuple[str, ...]
    requires_gpu: bool
    expected_labels: tuple[str, ...]
    published_objective_convention: str | None
    notes: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LiteratureTarget:
        required = {"id", "family", "title", "runner", "support", "expected_labels"}
        missing = required.difference(payload)
        if missing:
            raise RegistryError(f"target is missing keys {sorted(missing)}")
        support = payload["support"]
        if support not in SUPPORT_LEVELS:
            raise RegistryError(f"{payload['id']}: unknown support level {support!r}")
        if support in {"unsupported", "descriptive-only"} and not payload.get("unsupported_reason"):
            raise RegistryError(f"{payload['id']}: {support} targets need an unsupported_reason")
        runner = payload["runner"]
        if ":" not in runner:
            raise RegistryError(f"{payload['id']}: runner must be 'module:function'")
        return cls(
            id=payload["id"],
            family=payload["family"],
            title=payload["title"],
            runner=runner,
            profile=payload.get("profile"),
            support=support,
            unsupported_reason=payload.get("unsupported_reason"),
            requires_artifacts=tuple(payload.get("requires_artifacts", ())),
            requires_gpu=bool(payload.get("requires_gpu", False)),
            expected_labels=tuple(payload["expected_labels"]),
            published_objective_convention=payload.get("published_objective_convention"),
            notes=payload.get("notes", ""),
        )

    def load_profile(self, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
        if self.profile is None:
            return {}
        with (root / self.profile).open(encoding="utf-8") as handle:
            return json.load(handle)

    def resolve_runner(self) -> Callable[..., dict[str, Any]]:
        module_name, function_name = self.runner.split(":", 1)
        module = importlib.import_module(module_name)
        function = getattr(module, function_name, None)
        if function is None:
            raise RegistryError(f"{self.id}: runner {self.runner} does not exist")
        return function


@dataclass(frozen=True, slots=True)
class TargetRegistry:
    schema_version: str
    targets: tuple[LiteratureTarget, ...]

    def __getitem__(self, target_id: str) -> LiteratureTarget:
        for target in self.targets:
            if target.id == target_id:
                return target
        raise KeyError(target_id)

    def ids(self) -> tuple[str, ...]:
        return tuple(target.id for target in self.targets)

    def by_family(self, family: str) -> tuple[LiteratureTarget, ...]:
        return tuple(target for target in self.targets if target.family == family)


def load_target_registry(path: Path = REGISTRY_PATH) -> TargetRegistry:
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != "1.0.0":
        raise RegistryError("unsupported target registry schema version")
    targets = tuple(LiteratureTarget.from_dict(item) for item in document["targets"])
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise RegistryError("duplicate target ids")
    root = path.resolve().parents[2]
    for target in targets:
        if target.profile is not None and not (root / target.profile).is_file():
            raise RegistryError(f"{target.id}: profile {target.profile} is missing")
    return TargetRegistry(schema_version=document["schema_version"], targets=targets)


def run_target(
    target: LiteratureTarget,
    *,
    options: Mapping[str, Any] | None = None,
    root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Execute the registered runner and normalise the returned record."""

    runner = target.resolve_runner()
    record = runner(target.load_profile(root), options=dict(options or {}))
    if not isinstance(record, dict):
        raise RegistryError(f"{target.id}: runner must return a dict")
    record.setdefault("target_id", target.id)
    record.setdefault("family", target.family)
    if record.get("status") not in RESULT_STATUSES:
        raise RegistryError(
            f"{target.id}: runner returned status {record.get('status')!r}; "
            f"expected one of {RESULT_STATUSES}"
        )
    labels = record.setdefault("labels", {})
    if not isinstance(labels, dict):
        raise RegistryError(f"{target.id}: labels must map quantity -> evidence label")
    return record
