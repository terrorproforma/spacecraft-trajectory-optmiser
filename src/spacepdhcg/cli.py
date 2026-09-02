"""Top-level ``spacepdhcg`` command line (``python -m spacepdhcg`` or the console script)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spacepdhcg", description="SpacePDHCG / OrbitWeaver tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    from spacepdhcg.gtoc12.cli import add_parser as add_gtoc12

    add_gtoc12(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
