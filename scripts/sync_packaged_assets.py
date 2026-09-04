#!/usr/bin/env python3
"""Mirror the frozen benchmark/spec assets into ``src/spacepdhcg/_data`` (or verify the mirror).

The repository copies under ``benchmarks/`` and ``experiments/schema/`` stay the source of truth.
The mirror is what an installed wheel reads (``spacepdhcg.resources`` resolves an override, then
the checkout, then the mirror), so it must be byte-identical: ``--check`` exits 1 and lists every
missing, differing, or stray file; without ``--check`` the mirror is rewritten from the originals
and stray files are removed.

Usage::

    python scripts/sync_packaged_assets.py [--check] [--repository DIR] [--packaged-dir DIR]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spacepdhcg.resources import (  # noqa: E402
    PACKAGE_DATA_DIRECTORY,
    PACKAGED_ASSETS,
    compare_packaged_assets,
    packaged_asset_files,
)


def sync(repository: Path, packaged: Path) -> list[str]:
    actions: list[str] = []
    expected = set(PACKAGED_ASSETS)
    for asset in PACKAGED_ASSETS:
        source = repository / asset
        if not source.is_file():
            raise SystemExit(f"cannot mirror {asset}: {source} does not exist")
        destination = packaged / asset
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.read_bytes() != source.read_bytes():
            shutil.copyfile(source, destination)
            actions.append(f"copied {asset}")
    for path in packaged_asset_files(packaged):
        relative = path.relative_to(packaged).as_posix()
        if relative not in expected:
            path.unlink()
            actions.append(f"removed stray {relative}")
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 on drift")
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument(
        "--packaged-dir",
        type=Path,
        default=None,
        help=f"mirror directory (default: <repository>/src/spacepdhcg/{PACKAGE_DATA_DIRECTORY})",
    )
    arguments = parser.parse_args(argv)
    repository = arguments.repository.resolve()
    packaged = arguments.packaged_dir or repository / "src" / "spacepdhcg" / PACKAGE_DATA_DIRECTORY
    packaged = packaged.resolve()

    if arguments.check:
        report = compare_packaged_assets(repository, packaged)
        problems = [f"{kind}: {item}" for kind, items in report.items() for item in items]
        if problems:
            print("\n".join(problems), file=sys.stderr)
            print(
                f"packaged assets in {packaged} drift from {repository}; run "
                "python scripts/sync_packaged_assets.py",
                file=sys.stderr,
            )
            return 1
        print(f"{len(PACKAGED_ASSETS)} packaged assets are byte-identical to {repository}")
        return 0

    actions = sync(repository, packaged)
    for action in actions:
        print(action)
    print(f"{len(PACKAGED_ASSETS)} assets mirrored into {packaged} ({len(actions)} changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
