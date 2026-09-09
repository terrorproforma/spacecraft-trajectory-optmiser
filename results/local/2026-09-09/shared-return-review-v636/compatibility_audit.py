"""Run the pinned v846 saved-data audit with its historical summation order.

No original package file is edited. Only the profile total expression changes
in memory; the original exact equality and all other assertions remain active.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import sys


AUDIT_SHA256 = "a13246091d565efea79199228b9e41220088988e54dc1594c75792c3942c3aa3"
MANIFEST_SHA256 = "5a9f44f529b9078eb1b15ea38d9cc84794ef91bff805fe19bc569f3e8e1ca609"
ORIGINAL_EXPRESSION = "sum(v[2] for v in stats.values())"
REPLACEMENT_EXPRESSION = "_sequential_total_seconds(stats.values())"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not __debug__:
        raise RuntimeError("The original audit requires assertions; do not use -O.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package", type=Path,
        default=Path(__file__).resolve().parents[4]
        / "results/lambda/2026-09-09/gpu-shared-return-options-v846",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    original_path = package / "audit_saved.py"
    assert sha256(original_path) == AUDIT_SHA256
    manifest_path = package / "sha256-manifest.json"
    assert sha256(manifest_path) == MANIFEST_SHA256
    manifest = json.loads(manifest_path.read_text())
    for relative, expected_hash in manifest.items():
        assert sha256(package / relative) == expected_hash, relative

    original = original_path.read_bytes().decode("utf-8")
    assert original.count(ORIGINAL_EXPRESSION) == 1
    modified = original.replace(ORIGINAL_EXPRESSION, REPLACEMENT_EXPRESSION)
    assert modified.replace(REPLACEMENT_EXPRESSION, ORIGINAL_EXPRESSION) == original
    observations = []
    scope = {"__file__": str(original_path)}

    def sequential_total_seconds(rows):
        terms = [row[2] for row in rows]
        total = 0.0
        for value in terms:
            total += value
        current_sum = sum(terms)
        observations.append({
            "side": scope["side"], "ship": scope["ship"],
            "field": "profile.total_seconds", "term_count": len(terms),
            "explicit_sequential_seconds": total,
            "current_builtin_sum_seconds": current_sum,
            "builtin_minus_sequential_seconds": current_sum - total,
            "builtin_minus_sequential_ulps": (current_sum - total) / math.ulp(total),
        })
        return total

    scope["_sequential_total_seconds"] = sequential_total_seconds
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(modified, str(original_path), "exec"), scope)
    audited = json.loads(captured.getvalue())
    assert len(observations) == 4
    assert audited == json.loads((package / "saved-audit.json").read_text())
    for item in observations:
        item["recorded_seconds"] = audited[item["side"]]["profile"][str(item["ship"])]["total_seconds"]
        assert item["explicit_sequential_seconds"] == item["recorded_seconds"]
    result = {
        "status": "passed", "scope": "saved bytes and scalar arithmetic only",
        "python": sys.version, "original_audit_sha256": AUDIT_SHA256,
        "original_manifest_sha256": MANIFEST_SHA256,
        "original_expression": ORIGINAL_EXPRESSION,
        "replacement_expression": REPLACEMENT_EXPRESSION,
        "original_assertions_preserved": True, "numeric_tolerances_added": 0,
        "matches_published_saved_audit_exactly": True,
        "profile_summation_observations": observations,
        "original_audit_result": audited,
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(text)
        print(json.dumps({"status": "passed", "output": str(args.output), "sha256": sha256(args.output)}))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
