#!/usr/bin/env python3
"""Compare exact G3 PD3 KKT states block-by-block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from diagnose_g3_pd3 import _project_cones, load_dump, load_persistent_solution

from spacepdhcg.backends import PersistentClarabel


def _maximum(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def analyze(dump: Path, persistent_output: Path) -> dict[str, Any]:
    canonical = load_dump(dump)
    persistent = load_persistent_solution(persistent_output)
    cpu = PersistentClarabel(
        canonical,
        tolerance=1.0e-10,
        iteration_limit=2_000,
    ).solve()
    if not cpu.solved:
        raise RuntimeError(f"Clarabel oracle failed: {cpu.status}")
    cpu_dual = cpu.dual.copy()
    cpu_dual[canonical.structure.n_constraints :] *= -1.0

    structure = canonical.structure
    values = canonical.values
    quadratic = structure.quadratic.matrix(values.quadratic)
    scalar = structure.constraint.matrix(values.constraint)
    affine = structure.affine_cone.matrix(values.affine_cone)
    scalar_blocks = {
        "initial": slice(0, 7),
        "dynamics": slice(7, 21),
        "terminal": slice(21, 27),
        "virtual_epigraph": slice(27, 55),
        "tilt": slice(55, 57),
    }
    affine_blocks = {
        "thrust": slice(0, 8),
        "glide": slice(8, 17),
        "stage_trust": slice(17, 41),
        "terminal_trust": slice(41, 49),
    }
    variable_blocks = {
        "states": slice(0, 21),
        "controls": slice(21, 29),
        "virtual_control": slice(29, 43),
        "virtual_epigraph": slice(43, 57),
    }

    def state_record(
        primal: np.ndarray,
        dual: np.ndarray,
    ) -> dict[str, Any]:
        scalar_value = np.asarray(scalar @ primal)
        affine_value = np.asarray(affine @ primal + values.affine_offset)
        scalar_dual = dual[: structure.n_constraints]
        affine_dual = dual[structure.n_constraints :]
        scalar_natural = scalar_value - np.clip(
            scalar_value + scalar_dual,
            values.lower,
            values.upper,
        )
        affine_natural = affine_value - _project_cones(
            affine_value + affine_dual,
            structure.affine_cones,
        )
        objective_gradient = np.asarray(quadratic @ primal + values.linear)
        scalar_adjoint = np.asarray(scalar.T @ scalar_dual)
        affine_adjoint = np.asarray(affine.T @ affine_dual)
        reduced_gradient = objective_gradient + scalar_adjoint + affine_adjoint
        stationarity = primal - np.clip(
            primal - reduced_gradient,
            values.variable_lower,
            values.variable_upper,
        )
        scalar_records: dict[str, Any] = {}
        for name, block in scalar_blocks.items():
            lower = values.lower[block]
            upper = values.upper[block]
            equality = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
            active_upper = np.isfinite(upper) & (np.abs(scalar_value[block] - upper) <= 1.0e-6)
            scalar_records[name] = {
                "dual_inf": _maximum(scalar_dual[block]),
                "dual_error_from_cpu_inf": _maximum(scalar_dual[block] - cpu_dual[block]),
                "natural_inf": _maximum(scalar_natural[block]),
                "active_upper_count": int(np.count_nonzero(active_upper)),
                "invalid_sign_count": int(
                    np.count_nonzero((~equality) & (scalar_dual[block] < -1.0e-12))
                ),
            }
        affine_records: dict[str, Any] = {}
        for name, block in affine_blocks.items():
            dual_block = slice(
                structure.n_constraints + block.start,
                structure.n_constraints + block.stop,
            )
            affine_records[name] = {
                "dual_inf": _maximum(affine_dual[block]),
                "dual_error_from_cpu_inf": _maximum(affine_dual[block] - cpu_dual[dual_block]),
                "natural_inf": _maximum(affine_natural[block]),
            }
        variable_records: dict[str, Any] = {}
        for name, block in variable_blocks.items():
            variable_records[name] = {
                "primal_error_from_cpu_inf": _maximum(primal[block] - cpu.primal[block]),
                "objective_gradient_inf": _maximum(objective_gradient[block]),
                "scalar_adjoint_inf": _maximum(scalar_adjoint[block]),
                "affine_adjoint_inf": _maximum(affine_adjoint[block]),
                "stationarity_inf": _maximum(stationarity[block]),
            }
        top = np.argsort(np.abs(stationarity))[-10:][::-1]
        return {
            "objective": float(0.5 * primal @ (quadratic @ primal) + values.linear @ primal),
            "scalar_blocks": scalar_records,
            "affine_blocks": affine_records,
            "variable_blocks": variable_records,
            "top_stationarity": [
                {
                    "index": int(index),
                    "residual": float(stationarity[index]),
                    "primal": float(primal[index]),
                    "cpu_primal": float(cpu.primal[index]),
                }
                for index in top
            ],
        }

    return {
        "cpu": state_record(cpu.primal, cpu_dual),
        "persistent": state_record(persistent.primal, persistent.dual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--persistent-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze(arguments.dump, arguments.persistent_output)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
