#!/usr/bin/env python3
"""Preserve valid small-denominator divisions in QOCO's cone algebra."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(root: Path) -> dict:
    path = root / "include/definitions.h"
    original = path.read_text()
    before = "((qoco_abs(b) > (QOCOFloat)1e-15) ? ((a) / (b)) : QOCOFloat_MAX)"
    after = "(((b) != (QOCOFloat)0.0) ? ((a) / (b)) : QOCOFloat_MAX)"
    if original.count(before) != 1:
        raise RuntimeError("unexpected QOCO safe_div definition")
    # Interior-point cone determinants become small as the barrier decreases.
    # An absolute cutoff changes finite quotients into DBL_MAX and corrupts
    # the Newton system. Preserve the upstream exact-zero behavior only.
    updated = original.replace(before, after)
    path.write_text(updated)
    return {
        "path": str(path),
        "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(updated.encode()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
