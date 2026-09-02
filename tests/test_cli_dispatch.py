"""The single ``spacepdhcg`` console entry point must serve every track's sub-commands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from spacepdhcg import cli
from spacepdhcg.planner.cli import PLANNER_COMMANDS
from spacepdhcg.planner.cli import main as planner_main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "planner" / "hcw_rendezvous.json"
GROUPS = ("literature", "gtoc12")


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("parser has no sub-commands")


def test_console_script_points_at_the_umbrella_dispatcher() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["spacepdhcg"] == "spacepdhcg.cli:main"
    assert list(scripts).count("spacepdhcg") == 1


def test_dispatcher_mounts_every_track() -> None:
    choices = _subcommands(cli.build_parser())
    assert set(PLANNER_COMMANDS) <= set(choices)
    assert set(GROUPS) <= set(choices)
    for name in PLANNER_COMMANDS:
        # every planner leaf carries its handler so dispatch() never falls back to a name table
        assert choices[name].get_default("func") is not None


def test_every_leaf_has_a_handler() -> None:
    parser = cli.build_parser()
    for group in GROUPS:
        leaves = _subcommands(_subcommands(parser)[group])
        assert leaves, group
        for name, leaf in leaves.items():
            handler = leaf.get_default("func") or leaf.get_default("function")
            assert callable(handler), f"{group} {name}"


def test_planner_validate_through_umbrella(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["validate", str(EXAMPLE), "--quiet"]) == 0
    assert capsys.readouterr().out.strip() == "valid"


def test_planner_standalone_main_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    assert planner_main(["validate", str(EXAMPLE), "--quiet"]) == 0
    assert capsys.readouterr().out.strip() == "valid"


def test_literature_list_through_umbrella(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["literature", "list"]) == 0
    assert "acikmese-ploen-2007-pd3" in capsys.readouterr().out


def test_gtoc12_group_parses_through_umbrella() -> None:
    # gtoc12 leaves store their handler as ``function``; dispatch() accepts both spellings
    from spacepdhcg.gtoc12.cli import cmd_reduced_instance

    arguments = cli.build_parser().parse_args(["gtoc12", "reduced-instance", "--list-ids"])
    assert arguments.command == "gtoc12"
    assert arguments.function is cmd_reduced_instance
    assert not hasattr(arguments, "func")


def test_register_adds_a_group() -> None:
    def adder(subparsers: argparse._SubParsersAction) -> None:
        leaf = subparsers.add_parser("probe-group")
        leaf.set_defaults(func=lambda arguments: 7)

    cli.register("probe-group", adder)
    try:
        assert cli.main(["probe-group"]) == 7
    finally:
        cli._REGISTERED.pop("probe-group")


def test_module_entry_point_lists_all_tracks() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT / "src"), env.get("PYTHONPATH", "")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-m", "spacepdhcg", "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    for name in (*PLANNER_COMMANDS, *GROUPS):
        assert name in completed.stdout
