"""Umbrella ``spacepdhcg`` command.

Currently exposes ``spacepdhcg literature ...``.  The planner CLI (``spacepdhcg plan``) is expected
to register itself here through :func:`register` once it lands on ``feat/planner-cli``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

_REGISTERED: dict[str, Callable[[argparse._SubParsersAction], None]] = {}


def register(name: str, adder: Callable[[argparse._SubParsersAction], None]) -> None:
    _REGISTERED[name] = adder


def build_parser() -> argparse.ArgumentParser:
    from spacepdhcg.literature import cli as literature_cli

    parser = argparse.ArgumentParser(prog="spacepdhcg", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    literature_cli.add_parser(subparsers)
    for adder in _REGISTERED.values():
        adder(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return int(arguments.func(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
