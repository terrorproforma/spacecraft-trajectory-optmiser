"""OrbitWeaver G7 configuration and frozen-matrix CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spacepdhcg import resources

from .contracts import validate_named
from .g7 import (
    Checkpoint,
    ResultRecord,
    RunManifest,
    load_frozen_paper2_matrix,
)

PAPER2_MATRIX_ASSET = "benchmarks/paper2_matrix.json"


def _object(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def validate_config(args: argparse.Namespace) -> int:
    value = _object(args.config)
    validate_named(value, "config")
    if value["maximum_batch_size"] > value["maximum_buffered_arcs"]:
        raise ValueError("maximum_batch_size exceeds maximum_buffered_arcs")
    print(
        json.dumps(
            {
                "valid": True,
                "seed": value["seed"],
                "repeat_count": value["repeat_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def validate_matrix(args: argparse.Namespace) -> int:
    # Without an explicit path the frozen matrix comes from the resolver (override, checkout,
    # or the copy packaged in the wheel) instead of a working-directory-relative guess.
    matrix = args.matrix or resources.asset_path(PAPER2_MATRIX_ASSET)
    value = load_frozen_paper2_matrix(matrix)
    print(
        json.dumps(
            {"valid": True, "families": [item["id"] for item in value["families"]]},
            sort_keys=True,
        )
    )
    return 0


def create_manifest(args: argparse.Namespace) -> int:
    manifest = RunManifest.capture(
        run_id=args.run_id,
        repository=args.repository,
        config_path=args.config,
        matrix_path=args.matrix,
        backend=args.backend,
        ownership=args.ownership,
        device_ids=tuple(args.device_id),
        evidence_level=args.evidence_level,
        campaign_scope_id=args.campaign_scope_id,
    )
    manifest.write(args.output)
    print(json.dumps({"valid": True, "manifest_sha256": manifest.sha256()}))
    return 0


def validate_manifest(args: argparse.Namespace) -> int:
    manifest = RunManifest.read(args.manifest)
    print(json.dumps({"valid": True, "manifest_sha256": manifest.sha256()}))
    return 0


def validate_checkpoint(args: argparse.Namespace) -> int:
    manifest = None if args.manifest is None else RunManifest.read(args.manifest)
    checkpoint = Checkpoint.read(args.checkpoint, manifest)
    print(json.dumps({"valid": True, "completed_batches": checkpoint.completed_batches}))
    return 0


def validate_result(args: argparse.Namespace) -> int:
    manifest = None if args.manifest is None else RunManifest.read(args.manifest)
    result = ResultRecord.read(args.result, manifest)
    print(json.dumps({"valid": True, "status": result.status}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="spacepdhcg-orbitweaver-g7",
        description="Validate bounded G7 correctness inputs; emits no performance claims.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("validate-config")
    config.add_argument("config")
    config.set_defaults(function=validate_config)
    matrix = commands.add_parser("validate-matrix")
    matrix.add_argument(
        "matrix",
        nargs="?",
        default=None,
        help=f"frozen Paper 2 matrix (default: resolved {PAPER2_MATRIX_ASSET})",
    )
    matrix.set_defaults(function=validate_matrix)
    create = commands.add_parser("create-manifest")
    create.add_argument("--run-id", required=True)
    create.add_argument("--repository", required=True)
    create.add_argument("--config", required=True)
    create.add_argument("--matrix", required=True)
    create.add_argument("--output", required=True)
    create.add_argument("--backend", required=True)
    create.add_argument(
        "--ownership",
        required=True,
        choices=["single_gpu", "logical_rank_mock", "g5_distributed"],
    )
    create.add_argument("--device-id", action="append", required=True, type=int)
    create.add_argument(
        "--campaign-scope-id",
        default="single-gpu-v1",
        choices=["single-gpu-v1", "full-multi-gpu-v1"],
    )
    create.add_argument(
        "--evidence-level",
        default="implemented_compiled",
        choices=[
            "implemented_compiled",
            "cpu_correctness_tested",
            "one_gpu_correctness_tested",
            "physical_multi_gpu_tested",
        ],
    )
    create.set_defaults(function=create_manifest)
    manifest = commands.add_parser("validate-manifest")
    manifest.add_argument("manifest")
    manifest.set_defaults(function=validate_manifest)
    checkpoint = commands.add_parser("validate-checkpoint")
    checkpoint.add_argument("checkpoint")
    checkpoint.add_argument("--manifest")
    checkpoint.set_defaults(function=validate_checkpoint)
    result = commands.add_parser("validate-result")
    result.add_argument("result")
    result.add_argument("--manifest")
    result.set_defaults(function=validate_result)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
