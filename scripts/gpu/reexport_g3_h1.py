#!/usr/bin/env python3
"""Create corrected H1 compact derivatives without changing archived evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema
from run_g3_h1 import _compact_result, _sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Archived H1 directory with raw and compact files"
    )
    parser.add_argument("output", type=Path, help="New directory (must not already exist)")
    args = parser.parse_args()
    raw = args.source.resolve() / "h1_raw.jsonl"
    raw_sha = _sha256(raw)
    repository = Path(__file__).resolve().parents[2]
    schema = json.loads((repository / "experiments/schema/paper1_result.schema.json").read_text())
    records, changes = [], []
    for index, line in enumerate(raw.read_text().splitlines()):
        sample = json.loads(line)
        original_path = args.source / "compact" / f"result-{index:03d}.json"
        original = json.loads(original_path.read_text())
        corrected = _compact_result(
            sample,
            original["identity"]["repository_commit"],
            original["identity"]["run_id"],
            raw,
            raw_sha,
            index,
        )
        corrected["identity"] = original["identity"].copy()
        corrected["notes"].append(
            f"Corrected derivative of {original_path.resolve()}; "
            f"original SHA256 {_sha256(original_path)}. Original evidence is unchanged."
        )
        jsonschema.validate(corrected, schema)
        records.append(corrected)
        changes.append(
            {
                "index": index,
                "work_before": original["work"],
                "work_after": corrected["work"],
                "original_compact_sha256": _sha256(original_path),
            }
        )
    args.output.mkdir(parents=True, exist_ok=False)
    for index, record in enumerate(records):
        (args.output / f"result-{index:03d}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
    audit = {
        "raw": str(raw),
        "raw_sha256": raw_sha,
        "sample_count": len(records),
        "exporter_sha256": _sha256(Path(__file__).with_name("run_g3_h1.py")),
        "changes": changes,
    }
    (args.output / "corrections.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Validated {len(records)} corrected records in {args.output}")


if __name__ == "__main__":
    main()
