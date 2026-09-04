"""Wrapper around the organisers' ``GTOC12_Verify`` Linux binary.

The binary expects ``GTOC12_Asteroids_Data.txt`` and ``Result.txt`` in its working directory (or as
two positional arguments) and writes ``ScoreData.txt`` (mined-asteroid count, then one
``asteroid_id mass`` row per mined asteroid) next to itself on success.  We run it in a scratch
directory so the pinned data directory is never mutated.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .data import official_verifier_binary, verified_path

_SHIPS = re.compile(r"The number of mining ships is\s+(\d+)")
_ASTEROIDS = re.compile(r"The number of mined asteroids is\s+(\d+)")
_MASS = re.compile(r"The total resource mass is\s+([0-9.eE+-]+)\s*kg")


@dataclass(slots=True)
class OfficialVerification:
    ok: bool
    ships: int | None
    mined_asteroids: int | None
    total_mass_kg: float | None
    stdout: str
    stderr: str
    return_code: int
    score_data: dict[int, float] = field(default_factory=dict)
    wall_seconds: float = 0.0

    def summary(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "ships": self.ships,
            "mined_asteroids": self.mined_asteroids,
            "total_mass_kg": self.total_mass_kg,
            "return_code": self.return_code,
            "wall_seconds": self.wall_seconds,
            "message": self.stdout.strip().splitlines()[0] if self.stdout.strip() else "",
        }


def official_verifier_available() -> bool:
    try:
        official_verifier_binary(extract=True)
        verified_path("GTOC12_Asteroids_Data.txt")
    except Exception:
        return False
    return True


def run_official_verifier(
    solution_path: str | Path,
    *,
    timeout: float = 600.0,
    keep_directory: Path | None = None,
) -> OfficialVerification:
    """Run the official Linux verifier on ``solution_path`` in an isolated scratch directory."""

    import time

    binary = official_verifier_binary()
    catalogue = verified_path("GTOC12_Asteroids_Data.txt")
    solution = Path(solution_path)
    if not solution.is_file():
        raise FileNotFoundError(solution)
    context = (
        tempfile.TemporaryDirectory(prefix="gtoc12-verify-") if keep_directory is None else None
    )
    workdir = Path(context.name) if context else keep_directory
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(binary, workdir / "GTOC12_Verify")
        (workdir / "GTOC12_Verify").chmod(0o755)
        shutil.copy2(catalogue, workdir / "GTOC12_Asteroids_Data.txt")
        shutil.copy2(solution, workdir / "Result.txt")
        started = time.perf_counter()
        completed = subprocess.run(
            ["./GTOC12_Verify"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - started
        stdout = completed.stdout
        ships = _SHIPS.search(stdout)
        asteroids = _ASTEROIDS.search(stdout)
        mass = _MASS.search(stdout)
        ok = "Check successfully" in stdout and completed.returncode == 0
        score: dict[int, float] = {}
        score_path = workdir / "ScoreData.txt"
        if ok and score_path.is_file():
            rows = score_path.read_text(encoding="utf-8").split()
            if rows:
                count = int(rows[0])
                values = rows[1:]
                for index in range(count):
                    score[int(values[2 * index])] = float(values[2 * index + 1])
        return OfficialVerification(
            ok=ok,
            ships=int(ships.group(1)) if ships else None,
            mined_asteroids=int(asteroids.group(1)) if asteroids else None,
            total_mass_kg=float(mass.group(1)) if mass else None,
            stdout=stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
            score_data=score,
            wall_seconds=elapsed,
        )
    finally:
        if context is not None:
            context.cleanup()
