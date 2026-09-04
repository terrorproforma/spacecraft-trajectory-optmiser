#!/usr/bin/env python3
"""Materialise the authoritative in-package OrbitWeaver G7 schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from spacepdhcg.orbitweaver.contracts import SCHEMAS

FILENAMES = {
    "config": "orbitweaver_g7_config.schema.json",
    "manifest": "orbitweaver_g7_manifest.schema.json",
    "checkpoint": "orbitweaver_g7_checkpoint.schema.json",
    "result": "orbitweaver_g7_result.schema.json",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/schema"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, filename in FILENAMES.items():
        encoded = json.dumps(SCHEMAS[name], indent=2, sort_keys=True) + "\n"
        destination = args.output / filename
        if args.check:
            if not destination.is_file() or destination.read_text(encoding="utf-8") != encoded:
                raise SystemExit(f"schema drift: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
