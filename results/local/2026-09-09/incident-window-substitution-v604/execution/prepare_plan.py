"""Freeze the exact common candidate grid without loading native libraries."""

import argparse
import ctypes
import os
from pathlib import Path
from unittest.mock import patch

import run

root = Path(__file__).resolve().parent
os.environ["SPACEPDHCG_GTOC12_DATA"] = (
    "/home/angus/worktrees/spacepdhcg-release/benchmarks/gtoc12/data"
)
with patch.object(ctypes, "CDLL", side_effect=AssertionError("no native work during preparation")):
    run.main(argparse.Namespace(plan_only=True, execute=False, output=root / "plan"))
