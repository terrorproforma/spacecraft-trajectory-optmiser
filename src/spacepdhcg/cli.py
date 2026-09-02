"""Umbrella ``spacepdhcg`` command (console script and ``python -m spacepdhcg``).

One dispatcher owns every track so the console entry point stays unique:

* planner commands at the top level - ``plan``, ``validate``, ``capabilities``, ``defaults``,
  ``summary`` (:mod:`spacepdhcg.planner.cli`);
* ``literature ...`` - literature reproduction targets (:mod:`spacepdhcg.literature.cli`).

Every leaf sub-parser stores its handler with ``set_defaults(func=...)`` (``function=`` is
accepted as an alias) and :func:`dispatch` calls it.  Additional groups can be attached at
import time through :func:`register`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

Adder = Callable[[argparse._SubParsersAction], None]

_REGISTERED: dict[str, Adder] = {}


def register(name: str, adder: Adder) -> None:
    """Register an extra sub-command group added on every :func:`build_parser` call."""

    _REGISTERED[name] = adder


def _core_adders() -> list[Adder]:
    # Imported lazily so ``spacepdhcg literature list`` does not pay for planner imports and
    # vice versa; each track module keeps its own heavy imports.
    from spacepdhcg.literature import cli as literature_cli
    from spacepdhcg.planner import cli as planner_cli

    return [planner_cli.add_commands, literature_cli.add_parser]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spacepdhcg",
        description=(
            "SpacePDHCG / OrbitWeaver tools: trajectory planner (plan, validate, capabilities, "
            "defaults, summary) and literature reproduction targets (literature ...)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for adder in _core_adders():
        adder(subparsers)
    for adder in _REGISTERED.values():
        adder(subparsers)
    return parser


def dispatch(arguments: argparse.Namespace) -> int:
    handler = getattr(arguments, "func", None) or getattr(arguments, "function", None)
    if handler is None:  # pragma: no cover - every registered leaf sets a handler
        raise SystemExit(f"spacepdhcg: no handler registered for {arguments.command!r}")
    return int(handler(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    return dispatch(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
