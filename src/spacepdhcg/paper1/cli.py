"""Command-line entry point for Paper 1 G6 tooling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .aggregate import AggregationError
from .decisions import DecisionError
from .evidence import EvidenceError, evidence_index, load_campaign, write_canonical_json
from .freeze import (
    FreezeError,
    build_campaign,
    freeze_campaign,
    verify_clean_clone,
    verify_reproducible_build,
)
from .synthetic import generate_synthetic_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spacepdhcg-paper1",
        description="Fail-closed Paper 1 evidence aggregation and freeze tooling.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate and index an archived campaign")
    validate.add_argument("campaign", type=Path)
    validate.add_argument("--output", type=Path)

    build = subcommands.add_parser("build", help="build frozen figures, tables, and decisions")
    build.add_argument("campaign", type=Path)
    build.add_argument("output", type=Path)
    build.add_argument("--synthetic", action="store_true")

    freeze = subcommands.add_parser("freeze", help="freeze a complete real campaign")
    freeze.add_argument("campaign", type=Path)
    freeze.add_argument("config", type=Path)
    freeze.add_argument("output", type=Path)
    freeze.add_argument("--repository", type=Path, default=Path.cwd())

    reproduce = subcommands.add_parser("verify-reproducible", help="compare two complete builds")
    reproduce.add_argument("campaign", type=Path)
    reproduce.add_argument("--synthetic", action="store_true")

    clean = subcommands.add_parser("verify-clean-clone", help="build from a clean Git clone")
    clean.add_argument("campaign_relative_path")
    clean.add_argument("--repository", type=Path, default=Path.cwd())
    clean.add_argument("--synthetic", action="store_true")

    demo = subcommands.add_parser("synthetic-demo", help="generate and build labelled fixtures")
    demo.add_argument("campaign", type=Path)
    demo.add_argument("output", type=Path)
    return parser


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate":
            index = evidence_index(load_campaign(arguments.campaign))
            if arguments.output:
                write_canonical_json(arguments.output, index)
            _print(index)
        elif arguments.command == "build":
            _print(
                build_campaign(arguments.campaign, arguments.output, synthetic=arguments.synthetic)
            )
        elif arguments.command == "freeze":
            path = freeze_campaign(
                arguments.repository,
                arguments.campaign,
                arguments.config,
                arguments.output,
            )
            _print({"freeze_seal": str(path)})
        elif arguments.command == "verify-reproducible":
            _print(verify_reproducible_build(arguments.campaign, synthetic=arguments.synthetic))
        elif arguments.command == "verify-clean-clone":
            _print(
                verify_clean_clone(
                    arguments.repository,
                    arguments.campaign_relative_path,
                    synthetic=arguments.synthetic,
                )
            )
        elif arguments.command == "synthetic-demo":
            generate_synthetic_campaign(arguments.campaign)
            _print(build_campaign(arguments.campaign, arguments.output, synthetic=True))
        else:  # pragma: no cover
            raise AssertionError(arguments.command)
    except (AggregationError, DecisionError, EvidenceError, FreezeError) as error:
        print(f"paper1 tooling refused operation: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
