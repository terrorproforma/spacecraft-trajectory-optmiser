"""OrbitWeaver G7 configuration and frozen-matrix CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .g7 import load_frozen_paper2_matrix


def _object(path: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def validate_config(args: argparse.Namespace) -> int:
    value = _object(args.config)
    required = {
        "schema_version",
        "seed",
        "maximum_batch_size",
        "maximum_buffered_arcs",
        "maximum_workspace_bytes",
        "top_k",
        "risk_measure",
        "certification_tolerance",
    }
    missing = sorted(required - value.keys())
    if missing or value.get("schema_version") != 1:
        raise ValueError(f"invalid G7 config; missing={missing}")
    print(json.dumps({"valid": True, "seed": value["seed"]}, sort_keys=True))
    return 0


def validate_matrix(args: argparse.Namespace) -> int:
    value = load_frozen_paper2_matrix(args.matrix)
    print(
        json.dumps(
            {"valid": True, "families": [item["id"] for item in value["families"]]},
            sort_keys=True,
        )
    )
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
    matrix.add_argument("matrix", nargs="?", default="benchmarks/paper2_matrix.json")
    matrix.set_defaults(function=validate_matrix)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
