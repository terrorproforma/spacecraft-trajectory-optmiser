"""GTOPX (Schlueter et al., SoftwareX 2021) evaluator wrapper and best-known-vector checks.

The pinned GTOPX 1.0 C source (GPL) is compiled locally into a shared library and called
through ctypes with the official signature ``void gtopx(int benchmark, double* f, double* g,
double* x)``.  The official best-known solution files are parsed and re-evaluated so the
published objectives are checked exactly (``published-reference`` versus ``reproduced-external``
with the pinned evaluator).  No global optimiser is run here.
"""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spacepdhcg.literature import external_sources

BENCHMARKS: dict[str, dict[str, Any]] = {
    "cassini1": {"number": 1, "objectives": 1, "variables": 6, "constraints": 4},
    "cassini2": {"number": 2, "objectives": 1, "variables": 22, "constraints": 0},
    "messenger_reduced": {"number": 3, "objectives": 1, "variables": 18, "constraints": 0},
    "messenger_full": {"number": 4, "objectives": 1, "variables": 26, "constraints": 0},
    "gtoc1": {"number": 5, "objectives": 1, "variables": 8, "constraints": 6},
    "rosetta": {"number": 6, "objectives": 1, "variables": 22, "constraints": 0},
    "sagas": {"number": 7, "objectives": 1, "variables": 12, "constraints": 2},
    "cassini1_minlp": {"number": 8, "objectives": 1, "variables": 10, "constraints": 4},
}

_X_PATTERN = re.compile(r"x\[\s*(\d+)\]\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")
_F_PATTERN = re.compile(r"f\[\s*0\]\s*=\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")


class GTOPXBuildError(RuntimeError):
    """Raised when the GTOPX shared library cannot be compiled."""


def library_path() -> Path:
    return external_sources.cache_root() / "gtopx" / "build" / "libgtopx.so"


def build_library(*, force: bool = False) -> Path:
    """Compile the pinned ``gtopx.cpp`` (C-linkage variant) into a shared library."""

    source = external_sources.fetch("gtopx.source")
    target = library_path()
    if target.is_file() and not force and target.stat().st_mtime >= source.stat().st_mtime:
        return target
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise GTOPXBuildError("no C++ compiler (g++/clang++) available to build GTOPX")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O2", "-fPIC", "-shared", "-w", str(source), "-o", str(target), "-lm"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise GTOPXBuildError(f"GTOPX build failed: {completed.stderr[-2000:]}")
    return target


class GTOPXEvaluator:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or build_library()
        self._library = ctypes.CDLL(str(self.path))
        self._library.gtopx.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self._library.gtopx.restype = None

    def evaluate(self, benchmark: str, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        spec = BENCHMARKS[benchmark]
        vector = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
        if vector.shape != (spec["variables"],):
            raise ValueError(
                f"{benchmark} expects {spec['variables']} variables, got {vector.shape}"
            )
        f = np.zeros(max(spec["objectives"], 1), dtype=np.float64)
        g = np.zeros(max(spec["constraints"], 1), dtype=np.float64)
        self._library.gtopx(
            int(spec["number"]),
            f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            g.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return f[: spec["objectives"]], g[: spec["constraints"]]


@dataclass(frozen=True, slots=True)
class BestKnownSolution:
    benchmark: str
    objective_text: str
    objective: float
    vector_text: tuple[str, ...]
    vector: tuple[float, ...]


def parse_solution_file(benchmark: str, path: Path) -> BestKnownSolution:
    text = path.read_text(encoding="utf-8", errors="replace")
    f_match = _F_PATTERN.search(text)
    if f_match is None:
        raise ValueError(f"{path}: no objective line")
    entries = {int(index): value for index, value in _X_PATTERN.findall(text)}
    expected = BENCHMARKS[benchmark]["variables"]
    if sorted(entries) != list(range(expected)):
        raise ValueError(f"{path}: expected {expected} x entries, found {sorted(entries)}")
    texts = tuple(entries[i] for i in range(expected))
    return BestKnownSolution(
        benchmark=benchmark,
        objective_text=f_match.group(1),
        objective=float(f_match.group(1)),
        vector_text=texts,
        vector=tuple(float(value) for value in texts),
    )


def load_best_known(benchmark: str) -> BestKnownSolution:
    path = external_sources.fetch(f"gtopx.solution.{benchmark}")
    return parse_solution_file(benchmark, path)


def verify_best_known(
    benchmarks: tuple[str, ...] = ("cassini1", "rosetta", "messenger_reduced", "gtoc1"),
    *,
    relative_tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    evaluator = GTOPXEvaluator()
    rows = {}
    for benchmark in benchmarks:
        solution = load_best_known(benchmark)
        f, g = evaluator.evaluate(benchmark, np.asarray(solution.vector))
        difference = float(f[0] - solution.objective)
        # Rule 2 of the evidence policy: compare at the precision the source printed.
        decimals = (
            len(solution.objective_text.split(".")[1]) if "." in solution.objective_text else 0
        )
        printed_half_unit = 0.5 * 10.0 ** (-decimals)
        rows[benchmark] = {
            "published_objective_text": solution.objective_text,
            "published_objective": solution.objective,
            "evaluated_objective": float(f[0]),
            "difference": difference,
            "relative_difference": difference / abs(solution.objective),
            "printed_decimals": decimals,
            "constraints": [float(value) for value in g],
            "constraints_satisfied": bool(np.all(np.asarray(g) >= -1.0e-12)) if len(g) else True,
            "reproduced_exactly": abs(difference) <= relative_tolerance * abs(solution.objective),
            "reproduced_to_printed_precision": abs(difference) <= printed_half_unit + 1.0e-12,
        }
    return rows


def run_target(document: dict[str, Any], *, options: dict[str, Any]) -> dict[str, Any]:
    benchmarks = tuple(
        document.get("benchmarks", ["cassini1", "rosetta", "messenger_reduced", "gtoc1"])
    )
    try:
        rows = verify_best_known(benchmarks)
    except external_sources.ArtifactUnavailable as error:
        return {
            "target_id": document["id"],
            "status": "blocked",
            "published": {},
            "measured": {},
            "gap": {},
            "labels": {},
            "envelope": {},
            "commands": [f"spacepdhcg literature run {document['id']}"],
            "notes": [f"blocked: {error}"],
        }
    all_exact = all(row["reproduced_to_printed_precision"] for row in rows.values())
    return {
        "target_id": document["id"],
        "status": "reproduced" if all_exact else "gap",
        "published": {b: rows[b]["published_objective_text"] for b in rows},
        "measured": {b: rows[b]["evaluated_objective"] for b in rows},
        "gap": {b: rows[b]["difference"] for b in rows},
        "labels": {
            **{f"published.{b}": "published-reference" for b in rows},
            **{f"measured.{b}": "reproduced-external" for b in rows},
        },
        "envelope": {
            "comparison": (
                "exact evaluator re-evaluation of the official vector, compared at the "
                "precision printed in the official solution file"
            ),
            "exact_relative_tolerance": 1.0e-9,
            "exact_count": sum(1 for row in rows.values() if row["reproduced_exactly"]),
        },
        "commands": [f"spacepdhcg literature run {document['id']}"],
        "notes": ["no global optimiser run; evaluator provided as a test target"],
        "details": rows,
    }
