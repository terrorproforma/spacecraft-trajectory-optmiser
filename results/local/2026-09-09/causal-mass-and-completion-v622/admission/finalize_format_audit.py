"""Reindex a finalizer-only formatting change, retaining the original manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

KIT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous_path = KIT / "validation/ready-before-finalizer-format.json"
    previous = json.loads(previous_path.read_text())
    # No production, frozen input, raw output, test or report changed after its
    # original index. The finalizer file itself only changed line wrapping.
    for name, info in previous["files"].items():
        if name != "finalize.py":
            assert sha(KIT / name) == info["sha256"], name
    index = {
        path.relative_to(KIT).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(KIT.rglob("*"))
        if path.is_file() and path != KIT / "ready.json"
    }
    revised = {
        **previous,
        "files": index,
        "file_count": len(index),
        "total_bytes": sum(x["bytes"] for x in index.values()),
        "finalizer_format_addendum": {
            "previous_ready_sha256": sha(previous_path),
            "old_worker": "validation/finalize-before-format.py",
            "changes": "Finalizer line wrapping only; all data/production/test hashes unchanged",
        },
    }
    (KIT / "ready.json").write_text(json.dumps(revised, indent=2, allow_nan=False) + "\n")
    for name, info in index.items():
        assert sha(KIT / name) == info["sha256"], name
    print(
        json.dumps(
            {
                "ready_sha256": sha(KIT / "ready.json"),
                "files": len(index),
                "bytes": revised["total_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
