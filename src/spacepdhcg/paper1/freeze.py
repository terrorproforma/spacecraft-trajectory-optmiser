"""Campaign build, claim linkage, and fail-closed Paper 1 freeze checks."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from .aggregate import TIMING_COMPONENTS, build_products
from .decisions import build_decisions, validate_decision
from .evidence import (
    ArchivedRun,
    EvidenceError,
    evidence_index,
    load_campaign,
    sha256_path,
    write_canonical_json,
)

FREEZE_SCHEMA_VERSION: Final = "1.0.0"
SI_UNITS: Final = {
    "length": "metre",
    "time": "second",
    "mass": "kilogram",
    "angle": "radian",
    "force": "newton",
    "torque": "newton metre",
    "velocity": "metre per second",
    "acceleration": "metre per second squared",
    "energy": "joule",
    "memory": "byte",
}


class FreezeError(ValueError):
    """Raised when a campaign cannot be frozen without unsupported claims."""


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_json(path: Path, name: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FreezeError(f"missing {name}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FreezeError(f"{name} must be a JSON object")
    return payload


def _verify_pinned_file(repository: Path, item: Mapping[str, Any], category: str) -> None:
    path_value, expected = item.get("path"), item.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise FreezeError(f"{category} pin requires path")
    if not isinstance(expected, str) or len(expected) != 64:
        raise FreezeError(f"{category} pin requires SHA-256")
    path = (repository / path_value).resolve()
    try:
        path.relative_to(repository.resolve())
    except ValueError as error:
        raise FreezeError(f"{category} pin escapes repository: {path_value}") from error
    if not path.is_file() or sha256_path(path) != expected:
        raise FreezeError(f"{category} pin is missing or hash-mismatched: {path_value}")


def validate_campaign_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "repository_commit",
        "synthetic",
        "units",
        "required_coordinates",
        "hardware_manifests",
        "toolchain_manifests",
        "solver_locks",
        "claims",
    }
    missing, unknown = sorted(required - set(config)), sorted(set(config) - required)
    if missing or unknown:
        raise FreezeError(f"campaign config fields invalid; missing={missing}, unknown={unknown}")
    if config["schema_version"] != FREEZE_SCHEMA_VERSION:
        raise FreezeError("unsupported campaign config schema")
    commit = config["repository_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise FreezeError("campaign repository commit must be a full Git SHA")
    if not isinstance(config["synthetic"], bool):
        raise FreezeError("campaign synthetic flag must be boolean")
    if config["units"] != SI_UNITS:
        raise FreezeError("campaign units must exactly match the frozen SI unit manifest")
    if not isinstance(config["required_coordinates"], list) or not config["required_coordinates"]:
        raise FreezeError("campaign requires a non-empty coordinate inventory")
    for category in ("hardware_manifests", "toolchain_manifests", "solver_locks"):
        if not isinstance(config[category], list) or not config[category]:
            raise FreezeError(f"campaign requires pinned {category}")
    claims = config["claims"]
    if not isinstance(claims, Mapping) or set(claims) != {f"H{index}" for index in range(1, 7)}:
        raise FreezeError("claims must link exactly H1-H6")


def _matches(run: ArchivedRun, selector: Mapping[str, Any]) -> bool:
    identity, dimensions = run.result["identity"], run.result["dimensions"]
    values = {
        **identity,
        **dimensions,
        "run_id": run.run_id,
        "status": run.status,
    }
    return all(values.get(key) == value for key, value in selector.items())


def _check_coverage(runs: tuple[ArchivedRun, ...], config: Mapping[str, Any]) -> None:
    for index, raw in enumerate(config["required_coordinates"]):
        if not isinstance(raw, Mapping):
            raise FreezeError(f"required coordinate {index} must be an object")
        selector = raw.get("selector")
        if not isinstance(selector, Mapping) or not selector:
            raise FreezeError(f"required coordinate {index} lacks selector")
        matches = [run for run in runs if _matches(run, selector)]
        minimum_records = raw.get("minimum_records", 1)
        if not isinstance(minimum_records, int) or minimum_records < 1:
            raise FreezeError("minimum_records must be positive")
        if len(matches) < minimum_records:
            raise FreezeError(
                f"incomplete campaign coordinate {index}: {dict(selector)} "
                f"has {len(matches)}/{minimum_records} records"
            )
        minimum_instances = raw.get("minimum_instances", 1)
        instances = {run.result["identity"]["instance_id"] for run in matches}
        if len(instances) < minimum_instances:
            raise FreezeError(
                f"coordinate {index} has {len(instances)}/{minimum_instances} instances"
            )
        minimum_repeats = raw.get("minimum_measured_repeats", 5)
        for run in matches:
            measured = run.result["aggregation"]["measured_repeats"]
            if run.status == "qualified" and measured < minimum_repeats:
                raise FreezeError(
                    f"run {run.run_id} has {measured}/{minimum_repeats} measured repeats"
                )


def _check_matched_quality(runs: Iterable[ArchivedRun]) -> None:
    by_coordinate: dict[tuple[Any, ...], list[ArchivedRun]] = {}
    for run in runs:
        identity, dimensions = run.result["identity"], run.result["dimensions"]
        coordinate = (
            identity["family"],
            identity["instance_id"],
            dimensions["intervals"],
            dimensions["scenarios"],
            dimensions["gpus"],
            identity.get("quality_tier"),
        )
        by_coordinate.setdefault(coordinate, []).append(run)
    for coordinate, group in by_coordinate.items():
        qualified = [run for run in group if run.status == "qualified"]
        if len({run.result["identity"]["solver"] for run in qualified}) < 2:
            continue
        tiers = {run.result["identity"].get("quality_tier") for run in qualified}
        if len(tiers) != 1:
            raise FreezeError(f"matched coordinate has differing quality tiers: {coordinate}")
        for run in qualified:
            quality = run.result["quality"]
            if quality.get("matched_quality_state") not in {None, "matched"}:
                raise FreezeError(f"qualified comparison is not matched quality: {run.run_id}")
            if quality.get("independent_replay") is False:
                raise FreezeError(f"qualified result lacks independent replay: {run.run_id}")
            if quality.get("uses_solver_cached_residuals") is True:
                raise FreezeError(f"qualified result reuses solver residual buffers: {run.run_id}")


def _check_timing_and_classification(runs: Iterable[ArchivedRun]) -> None:
    for run in runs:
        identity = run.result["identity"]
        timing = run.result["timing"]
        resources = run.result["resources"]
        if run.manifest.experiment.get("units") != SI_UNITS:
            raise FreezeError(f"run lacks exact frozen SI unit manifest: {run.run_id}")
        if identity.get("warm_start") is None or identity.get("cold_start") is None:
            raise FreezeError(f"run lacks explicit warm/cold classification: {run.run_id}")
        if identity["warm_start"] and identity["cold_start"]:
            raise FreezeError(f"run is both warm and cold: {run.run_id}")
        if run.status != "qualified" and not identity.get("failure_reason"):
            raise FreezeError(f"non-qualified run lacks retained failure reason: {run.run_id}")
        if run.status != "qualified":
            continue
        if timing.get("accepted_trajectory_seconds") is None:
            raise FreezeError(f"qualified run lacks accepted-trajectory boundary: {run.run_id}")
        if not timing.get("accepted_timing_boundary"):
            raise FreezeError(f"qualified run lacks named timing boundary: {run.run_id}")
        components = [timing[name] for name in TIMING_COMPONENTS]
        if any(value is None for value in components):
            raise FreezeError(f"qualified run has incomplete timing components: {run.run_id}")
        total = sum(float(value) for value in components)
        expected = float(timing["scvx_total_seconds"])
        tolerance = max(1e-12, 1e-8 * expected)
        if abs(total - expected) > tolerance:
            raise FreezeError(f"qualified run timing identity fails: {run.run_id}")
        if timing.get("cuda_startup_included") is True:
            raise FreezeError(
                f"qualified run includes CUDA startup in measured boundary: {run.run_id}"
            )
        if resources.get("energy_joules") is not None and (
            resources.get("energy_scope") != "GPU-only"
            or resources.get("energy_valid") is not True
            or resources.get("energy_sampling_gaps") is not False
        ):
            raise FreezeError(f"qualified energy boundary is invalid: {run.run_id}")


def _checksums(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checksums.json":
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
        )
    return {"schema_version": FREEZE_SCHEMA_VERSION, "files": files}


def _claim_linkage(config: Mapping[str, Any], decisions: Path) -> dict[str, Any]:
    links = []
    for hypothesis in sorted(config["claims"]):
        record = _load_json(decisions / f"{hypothesis.lower()}-decision.json", hypothesis)
        validate_decision(record)
        if record["hypothesis"] != hypothesis:
            raise FreezeError(f"decision file mismatch for {hypothesis}")
        claims = config["claims"][hypothesis]
        if (
            not isinstance(claims, list)
            or not claims
            or not all(isinstance(claim, str) and claim for claim in claims)
        ):
            raise FreezeError(f"{hypothesis} requires non-empty manuscript claim IDs")
        links.append(
            {
                "hypothesis": hypothesis,
                "decision_outcome": record["outcome"],
                "decision_file": f"decisions/{hypothesis.lower()}-decision.json",
                "claim_ids": claims,
            }
        )
    return {"schema_version": FREEZE_SCHEMA_VERSION, "links": links}


def build_campaign(
    campaign_directory: str | Path,
    output_directory: str | Path,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    runs = load_campaign(campaign_directory)
    output = Path(output_directory)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    figures = build_products(runs, output / "products", synthetic=synthetic)
    decisions = build_decisions(runs, output / "decisions")
    index = evidence_index(runs)
    write_canonical_json(output / "evidence-index.json", index)
    manifest = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "synthetic": synthetic,
        "run_count": len(runs),
        "product_manifest": figures,
        "decision_index": decisions,
    }
    write_canonical_json(output / "campaign-build.json", manifest)
    write_canonical_json(output / "checksums.json", _checksums(output))
    return manifest


def freeze_campaign(
    repository: str | Path,
    campaign_directory: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Build and seal a complete real campaign, or refuse with explicit reasons."""

    repo = Path(repository).resolve()
    config = _load_json(Path(config_path), "campaign config")
    validate_campaign_config(config)
    if config["synthetic"]:
        raise FreezeError("synthetic campaigns can be built but can never be frozen")
    if _git(repo, "status", "--porcelain"):
        raise FreezeError("repository must be clean before campaign freeze")
    head = _git(repo, "rev-parse", "HEAD")
    if head != config["repository_commit"]:
        raise FreezeError(
            f"campaign commit {config['repository_commit']} != repository HEAD {head}"
        )
    for category in ("hardware_manifests", "toolchain_manifests", "solver_locks"):
        for item in config[category]:
            if not isinstance(item, Mapping):
                raise FreezeError(f"{category} entries must be objects")
            _verify_pinned_file(repo, item, category)
    try:
        runs = load_campaign(campaign_directory, verify_payloads=True)
    except EvidenceError as error:
        raise FreezeError(str(error)) from error
    _check_coverage(runs, config)
    _check_matched_quality(runs)
    _check_timing_and_classification(runs)
    output = Path(output_directory)
    build_campaign(campaign_directory, output, synthetic=False)
    product_manifest = _load_json(output / "products/build-manifest.json", "product manifest")
    mapped = Counter(
        run_id for product in product_manifest["products"] for run_id in product["run_ids"]
    )
    missing = sorted(run.run_id for run in runs if mapped[run.run_id] == 0)
    if missing:
        raise FreezeError(f"archived failures/results are silently excluded: {', '.join(missing)}")
    linkage = _claim_linkage(config, output / "decisions")
    write_canonical_json(output / "claim-decision-linkage.json", linkage)
    seal = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "campaign_id": config["campaign_id"],
        "repository_commit": head,
        "run_count": len(runs),
        "status_counts": dict(sorted(Counter(run.status for run in runs).items())),
        "decision_outcomes": {
            item["hypothesis"]: item["decision_outcome"] for item in linkage["links"]
        },
        "checksums_sha256": "",
        "statement": "Tool-generated completeness seal; not a scientific PASS claim.",
    }
    write_canonical_json(output / "checksums.json", _checksums(output))
    seal["checksums_sha256"] = sha256_path(output / "checksums.json")
    return write_canonical_json(output / "freeze-seal.json", seal)


def verify_reproducible_build(
    campaign_directory: str | Path,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    """Build twice in clean temporary directories and compare every output byte."""

    with (
        tempfile.TemporaryDirectory(prefix="spacepdhcg-g6-a-") as first_raw,
        tempfile.TemporaryDirectory(prefix="spacepdhcg-g6-b-") as second_raw,
    ):
        first, second = Path(first_raw), Path(second_raw)
        build_campaign(campaign_directory, first, synthetic=synthetic)
        build_campaign(campaign_directory, second, synthetic=synthetic)
        left, right = _checksums(first), _checksums(second)
        if left != right:
            left_map = {item["path"]: item["sha256"] for item in left["files"]}
            right_map = {item["path"]: item["sha256"] for item in right["files"]}
            differences = sorted(
                path
                for path in left_map.keys() | right_map.keys()
                if left_map.get(path) != right_map.get(path)
            )
            raise FreezeError(f"build is not byte reproducible: {', '.join(differences)}")
        return {
            "schema_version": FREEZE_SCHEMA_VERSION,
            "reproducible": True,
            "file_count": len(left["files"]),
            "aggregate_sha256": hashlib.sha256(
                json.dumps(left, sort_keys=True).encode()
            ).hexdigest(),
        }


def verify_clean_clone(
    repository: str | Path,
    campaign_relative_path: str,
    *,
    synthetic: bool,
) -> dict[str, Any]:
    """Verify the build from a Git-archive clean clone without local untracked inputs."""

    repo = Path(repository).resolve()
    if _git(repo, "status", "--porcelain"):
        raise FreezeError("clean-clone verification requires a clean repository")
    with tempfile.TemporaryDirectory(prefix="spacepdhcg-clean-clone-") as temporary:
        clone = Path(temporary) / "repository"
        subprocess.run(["git", "clone", "--no-local", str(repo), str(clone)], check=True)
        campaign = clone / campaign_relative_path
        result = verify_reproducible_build(campaign, synthetic=synthetic)
        result["repository_commit"] = _git(clone, "rev-parse", "HEAD")
        return result
