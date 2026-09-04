#!/usr/bin/env python3
"""Fetch the pinned GTOC12 official data into the ignored data directory.

Thin wrapper around :mod:`spacepdhcg.gtoc12.fetch` (the same code behind
``spacepdhcg gtoc12 fetch``): every file listed in ``benchmarks/gtoc12/pins.json`` is downloaded
from the first reachable URL, its byte size and SHA-256 are checked against the pin, and the
verifier archive is extracted.  Nothing is kept when a digest disagrees.

Usage::

    python scripts/gtoc12/fetch_gtoc12_data.py [--data-dir DIR] [--only NAME ...] [--skip-optional]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "src" / "spacepdhcg").is_dir():  # pragma: no cover - checkout layout guard
    raise SystemExit("fetch_gtoc12_data.py must run from a source checkout")
sys.path.insert(0, str(ROOT / "src"))

from spacepdhcg.gtoc12.fetch import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
