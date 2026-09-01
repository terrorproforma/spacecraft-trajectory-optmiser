#!/usr/bin/env python3
"""Independently diagnose the exact G3 3-DoF CQP dumped by the CUDA gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spacepdhcg.backends import QOCOGPU, PDHCGOneShot, PersistentClarabel
from spacepdhcg.cqp import (
    CanonicalCQP,
    ConeBlock,
    ConeKind,
    CQPSolution,
    CQPStructure,
    CQPValues,
    CSCStructure,
)
from spacepdhcg.models.powered_descent_3dof import PoweredDescent3DOFModel

FloatArray = NDArray[np.float64]

_CONE_KINDS = {
    0: ConeKind.SECOND_ORDER,
    1: ConeKind.ROTATED_SECOND_ORDER,
    2: ConeKind.EXPONENTIAL,
    3: ConeKind.POWER,
    4: ConeKind.POSITIVE_SEMIDEFINITE,
}

_UPSTREAM_VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "ruiz20": {"RuizIters": 20},
    "curtis5": {"CurtisReidIters": 5},
    "no-pc": {"UsePCAlpha": False},
    "pc-half": {"PCAlpha": 0.5},
    "bound-rescale-off": {"BoundObjRescaling": False},
    "restart-fast": {
        "RestartArtificialThresh": 0.20,
        "RestartSufficientReduction": 0.35,
        "RestartNecessaryReduction": 0.90,
    },
    "restart-kp-half": {"RestartKp": 0.5},
    "restart-kp-zero": {"RestartKp": 0.0},
    "reflection-half": {"ReflectionCoeff": 0.5},
}


@dataclass(frozen=True, slots=True)
class IndependentQuality:
    objective: float
    objective_error_from_cpu: float
    scalar_primal_inf: float
    box_primal_inf: float
    cone_primal_inf: float
    stationarity_inf: float
    scalar_natural_inf: float
    cone_natural_inf: float
    natural_inf: float
    relative_natural_inf: float
    complementarity_inf: float
    dual_cone_distance_inf: float
    initial_inf: float
    terminal_inf: float
    linearised_dynamics_inf: float
    nonlinear_dynamics_inf: float


def _numbers(tokens: list[str], *, integer: bool = False) -> NDArray:
    dtype = np.int64 if integer else np.float64
    return np.asarray(tokens, dtype=dtype)


def load_dump(path: Path) -> CanonicalCQP:
    """Load the line-oriented, full-precision CQP emitted by the CUDA test."""

    vectors: dict[str, list[str]] = {}
    dimensions: tuple[int, int, int] | None = None
    affine_cones: list[ConeBlock] = []
    variable_cones: list[ConeBlock] = []
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if line.startswith("PD3DATA "):
                _, name, *tokens = line.split()
                if name == "dimensions":
                    dimensions = tuple(int(token) for token in tokens)  # type: ignore[assignment]
                else:
                    vectors[name] = tokens
            elif line.startswith("PD3CONE "):
                _, axis, kind, start, vector_dimension, alpha = line.split()
                cone = ConeBlock(
                    _CONE_KINDS[int(kind)],
                    int(start),
                    int(vector_dimension),
                    float(alpha),
                )
                (affine_cones if axis == "affine" else variable_cones).append(cone)
    if dimensions is None:
        raise ValueError("dump has no dimensions record")
    variables, scalar_rows, affine_rows = dimensions
    structure = CQPStructure(
        quadratic=CSCStructure(
            (variables, variables),
            _numbers(vectors["q_offsets"], integer=True),
            _numbers(vectors["q_indices"], integer=True),
        ),
        constraint=CSCStructure(
            (scalar_rows, variables),
            _numbers(vectors["a_offsets"], integer=True),
            _numbers(vectors["a_indices"], integer=True),
        ),
        affine_cone=CSCStructure(
            (affine_rows, variables),
            _numbers(vectors["f_offsets"], integer=True),
            _numbers(vectors["f_indices"], integer=True),
        ),
        affine_cones=tuple(affine_cones),
        variable_cones=tuple(variable_cones),
    )
    values = CQPValues(
        quadratic=_numbers(vectors["q"]),
        constraint=_numbers(vectors["a"]),
        affine_cone=_numbers(vectors["f"]),
        linear=_numbers(vectors["c"]),
        lower=_numbers(vectors["scalar_lower"]),
        upper=_numbers(vectors["scalar_upper"]),
        affine_offset=_numbers(vectors["affine_offset"]),
        variable_lower=_numbers(vectors["variable_lower"]),
        variable_upper=_numbers(vectors["variable_upper"]),
    )
    return CanonicalCQP(structure, values)


def load_persistent_solution(path: Path) -> CQPSolution:
    vectors: dict[str, FloatArray] = {}
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line.startswith("PD3DATA persistent_"):
                continue
            _, name, *tokens = line.split()
            vectors[name] = _numbers(tokens)
    if set(vectors) != {"persistent_primal", "persistent_dual"}:
        raise ValueError("persistent output does not contain both iterate vectors")
    return CQPSolution(
        status="IterationLimit",
        primal=vectors["persistent_primal"],
        dual=vectors["persistent_dual"],
        objective=np.nan,
        primal_residual=np.nan,
        dual_residual=np.nan,
        iterations=1_000_000,
        solve_seconds=np.nan,
    )


def _maximum_absolute(values: FloatArray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _project_soc(values: FloatArray) -> FloatArray:
    vector = np.asarray(values, dtype=np.float64)
    spatial = vector[:-1]
    radius = float(vector[-1])
    norm = float(np.linalg.norm(spatial))
    if norm <= radius:
        return vector.copy()
    if norm <= -radius:
        return np.zeros_like(vector)
    result = np.empty_like(vector)
    result[:-1] = 0.5 * (1.0 + radius / norm) * spatial
    result[-1] = 0.5 * (norm + radius)
    return result


def _project_cones(values: FloatArray, cones: tuple[ConeBlock, ...]) -> FloatArray:
    result = np.asarray(values, dtype=np.float64).copy()
    for cone in cones:
        if cone.kind is not ConeKind.SECOND_ORDER:
            raise NotImplementedError(f"diagnostic projection does not support {cone.kind}")
        result[cone.start : cone.stop] = _project_soc(result[cone.start : cone.stop])
    return result


def _finite_scale(lower: FloatArray, upper: FloatArray) -> float:
    finite = np.concatenate((lower[np.isfinite(lower)], upper[np.isfinite(upper)]))
    return max(1.0, _maximum_absolute(finite))


def _rk4_step(
    model: PoweredDescent3DOFModel,
    state: FloatArray,
    control: FloatArray,
    step: float,
) -> FloatArray:
    k1 = model.dynamics(state, control)
    k2 = model.dynamics(state + 0.5 * step * k1, control)
    k3 = model.dynamics(state + 0.5 * step * k2, control)
    k4 = model.dynamics(state + step * k3, control)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def evaluate(
    canonical: CanonicalCQP,
    solution: CQPSolution,
    *,
    normal_dual: FloatArray,
    cpu_objective: float,
    intervals: int,
) -> IndependentQuality:
    """Recompute KKT, trajectory, and objective quantities without backend buffers."""

    structure = canonical.structure
    values = canonical.values
    primal = np.asarray(solution.primal, dtype=np.float64)
    dual = np.asarray(normal_dual, dtype=np.float64)
    quadratic = structure.quadratic.matrix(values.quadratic)
    scalar = structure.constraint.matrix(values.constraint)
    affine = structure.affine_cone.matrix(values.affine_cone)

    scalar_value = np.asarray(scalar @ primal, dtype=np.float64)
    scalar_projection = np.clip(scalar_value, values.lower, values.upper)
    scalar_violation = scalar_value - scalar_projection
    box_violation = primal - np.clip(primal, values.variable_lower, values.variable_upper)
    cone_value = np.asarray(affine @ primal + values.affine_offset, dtype=np.float64)
    cone_projection = _project_cones(cone_value, structure.affine_cones)
    cone_violation = cone_value - cone_projection

    scalar_dual = dual[: structure.n_constraints]
    cone_dual = dual[structure.n_constraints :]
    gradient = np.asarray(quadratic @ primal + values.linear, dtype=np.float64)
    scalar_adjoint = np.asarray(scalar.T @ scalar_dual, dtype=np.float64)
    cone_adjoint = np.asarray(affine.T @ cone_dual, dtype=np.float64)
    stationarity = gradient + scalar_adjoint + cone_adjoint
    stationarity_natural = primal - np.clip(
        primal - stationarity,
        values.variable_lower,
        values.variable_upper,
    )
    scalar_natural = scalar_value - np.clip(
        scalar_value + scalar_dual,
        values.lower,
        values.upper,
    )
    cone_natural = cone_value - _project_cones(
        cone_value + cone_dual,
        structure.affine_cones,
    )

    complementarity = 0.0
    for row, multiplier in enumerate(scalar_dual):
        lower = values.lower[row]
        upper = values.upper[row]
        if np.isfinite(lower) and np.isfinite(upper) and lower == upper:
            continue
        if multiplier >= 0.0 and np.isfinite(upper):
            complementarity = max(
                complementarity,
                abs(float(multiplier * (upper - scalar_value[row]))),
            )
        elif multiplier < 0.0 and np.isfinite(lower):
            complementarity = max(
                complementarity,
                abs(float((-multiplier) * (scalar_value[row] - lower))),
            )
    for cone in structure.affine_cones:
        complementarity = max(
            complementarity,
            abs(float(cone_dual[cone.start : cone.stop] @ cone_value[cone.start : cone.stop])),
        )
    dual_cone_distance = cone_dual + _project_cones(
        -cone_dual,
        structure.affine_cones,
    )

    stationarity_inf = _maximum_absolute(stationarity_natural)
    scalar_natural_inf = _maximum_absolute(scalar_natural)
    cone_natural_inf = _maximum_absolute(cone_natural)
    natural_inf = max(stationarity_inf, scalar_natural_inf, cone_natural_inf)
    stationarity_scale = max(
        1.0,
        _maximum_absolute(gradient),
        _maximum_absolute(scalar_adjoint),
        _maximum_absolute(cone_adjoint),
    )
    row_scale = max(
        _finite_scale(values.lower, values.upper),
        _maximum_absolute(scalar_value),
        _maximum_absolute(scalar_dual),
        _maximum_absolute(cone_value),
        _maximum_absolute(cone_dual),
    )
    relative_natural = max(
        stationarity_inf / stationarity_scale,
        scalar_natural_inf / row_scale,
        cone_natural_inf / row_scale,
    )

    objective = float(0.5 * primal @ (quadratic @ primal) + values.linear @ primal)
    if intervals < 2:
        raise ValueError("powered-descent diagnostics require at least two intervals")
    state_elements = (intervals + 1) * 7
    control_elements = intervals * 4
    initial_rows = slice(0, 7)
    dynamics_rows = slice(7, 7 + intervals * 7)
    terminal_rows = slice(dynamics_rows.stop, dynamics_rows.stop + 6)
    states = primal[:state_elements].reshape(intervals + 1, 7)
    controls = primal[state_elements : state_elements + control_elements].reshape(intervals, 4)
    nonlinear_defect = np.vstack(
        [
            states[index + 1]
            - _rk4_step(
                PoweredDescent3DOFModel(),
                states[index],
                controls[index],
                0.25,
            )
            for index in range(2)
        ]
    )
    return IndependentQuality(
        objective=objective,
        objective_error_from_cpu=abs(objective - cpu_objective) / max(1.0, abs(cpu_objective)),
        scalar_primal_inf=_maximum_absolute(scalar_violation),
        box_primal_inf=_maximum_absolute(box_violation),
        cone_primal_inf=_maximum_absolute(cone_violation),
        stationarity_inf=stationarity_inf,
        scalar_natural_inf=scalar_natural_inf,
        cone_natural_inf=cone_natural_inf,
        natural_inf=natural_inf,
        relative_natural_inf=relative_natural,
        complementarity_inf=complementarity,
        dual_cone_distance_inf=_maximum_absolute(dual_cone_distance),
        initial_inf=_maximum_absolute(scalar_value[initial_rows] - values.lower[initial_rows]),
        terminal_inf=_maximum_absolute(scalar_value[terminal_rows] - values.lower[terminal_rows]),
        linearised_dynamics_inf=_maximum_absolute(
            scalar_value[dynamics_rows] - values.lower[dynamics_rows]
        ),
        nonlinear_dynamics_inf=_maximum_absolute(nonlinear_defect),
    )


def _hash_array(digest: Any, name: str, values: NDArray) -> None:
    contiguous = np.ascontiguousarray(values)
    digest.update(name.encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def cqp_hash(canonical: CanonicalCQP) -> str:
    digest = hashlib.sha256()
    structure = canonical.structure
    values = canonical.values
    for name, array in (
        ("q_offsets", structure.quadratic.indptr),
        ("q_indices", structure.quadratic.indices),
        ("a_offsets", structure.constraint.indptr),
        ("a_indices", structure.constraint.indices),
        ("f_offsets", structure.affine_cone.indptr),
        ("f_indices", structure.affine_cone.indices),
        ("q", values.quadratic),
        ("a", values.constraint),
        ("f", values.affine_cone),
        ("c", values.linear),
        ("lower", values.lower),
        ("upper", values.upper),
        ("affine_offset", values.affine_offset),
        ("variable_lower", values.variable_lower),
        ("variable_upper", values.variable_upper),
    ):
        _hash_array(digest, name, array)
    for cone in (*structure.affine_cones, *structure.variable_cones):
        digest.update(
            f"{cone.kind}:{cone.start}:{cone.vector_dimension}:{cone.power_alpha:.17g}".encode()
        )
    return digest.hexdigest()


def _solution_record(
    canonical: CanonicalCQP,
    solution: CQPSolution,
    *,
    normal_dual: FloatArray,
    cpu_objective: float,
    intervals: int,
) -> dict[str, Any]:
    def finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    return {
        "status": solution.status,
        "iterations": solution.iterations,
        "reported_primal_residual": finite_or_none(solution.primal_residual),
        "reported_dual_residual": finite_or_none(solution.dual_residual),
        "reported_objective": finite_or_none(solution.objective),
        "solve_seconds": finite_or_none(solution.solve_seconds),
        "independent": asdict(
            evaluate(
                canonical,
                solution,
                normal_dual=normal_dual,
                cpu_objective=cpu_objective,
                intervals=intervals,
            )
        ),
    }


def run(
    dump: Path,
    *,
    persistent_output: Path | None,
    upstream_variant: str | None,
    upstream_start: str,
    iteration_limit: int,
    cpu_tolerance: float,
    intervals: int,
    qoco_library: Path | None,
    inspect_variable: int | None,
) -> dict[str, Any]:
    canonical = load_dump(dump)
    cpu = PersistentClarabel(
        canonical,
        tolerance=cpu_tolerance,
        iteration_limit=2_000,
    ).solve()
    if not cpu.solved and cpu.status != "AlmostSolved":
        raise RuntimeError(f"Clarabel failed with status {cpu.status}")
    cpu_normal_dual = cpu.dual.copy()
    cpu_normal_dual[canonical.structure.n_constraints :] *= -1.0
    cpu_objective = float(cpu.objective)
    result: dict[str, Any] = {
        "cqp_sha256": cqp_hash(canonical),
        "dimensions": {
            "variables": canonical.structure.n_variables,
            "scalar_rows": canonical.structure.n_constraints,
            "affine_rows": canonical.structure.n_affine_constraints,
        },
        "cpu_clarabel": _solution_record(
            canonical,
            cpu,
            normal_dual=cpu_normal_dual,
            cpu_objective=cpu_objective,
            intervals=intervals,
        ),
        "upstream_variant": upstream_variant,
    }
    if inspect_variable is not None:
        column = (
            canonical.structure.constraint.matrix(canonical.values.constraint)
            .getcol(inspect_variable)
            .tocoo()
        )
        scalar_value = np.asarray(
            canonical.structure.constraint.matrix(canonical.values.constraint) @ cpu.primal,
            dtype=np.float64,
        )
        result["inspected_variable"] = {
            "index": inspect_variable,
            "primal": float(cpu.primal[inspect_variable]),
            "linear": float(canonical.values.linear[inspect_variable]),
            "quadratic_gradient": float(
                (
                    canonical.structure.quadratic.matrix(canonical.values.quadratic).getrow(
                        inspect_variable
                    )
                    @ cpu.primal
                ).item()
            ),
            "scalar_rows": [
                {
                    "row": int(row),
                    "coefficient": float(value),
                    "activity": float(scalar_value[row]),
                    "lower": (
                        float(canonical.values.lower[row])
                        if np.isfinite(canonical.values.lower[row])
                        else None
                    ),
                    "upper": (
                        float(canonical.values.upper[row])
                        if np.isfinite(canonical.values.upper[row])
                        else None
                    ),
                    "dual": float(cpu_normal_dual[row]),
                    "stationarity": float(value * cpu_normal_dual[row]),
                }
                for row, value in zip(column.row, column.data, strict=True)
            ],
        }
    if persistent_output is not None:
        persistent = load_persistent_solution(persistent_output)
        result["persistent_cuda"] = _solution_record(
            canonical,
            persistent,
            normal_dual=persistent.dual,
            cpu_objective=cpu_objective,
            intervals=intervals,
        )
    if upstream_variant is not None:
        parameters = {
            "LogLevel": 0,
            **_UPSTREAM_VARIANTS[upstream_variant],
        }
        backend = PDHCGOneShot(canonical, params=parameters)
        if upstream_start == "cpu-primal":
            backend.warm_start(primal=cpu.primal)
        elif upstream_start == "cpu-primal-dual":
            backend.warm_start(primal=cpu.primal, dual=-cpu_normal_dual)
        solution = backend.solve(
            tolerance=1.0e-6,
            iteration_limit=iteration_limit,
        )
        result["upstream_pdhcg"] = _solution_record(
            canonical,
            solution,
            normal_dual=-solution.dual,
            cpu_objective=cpu_objective,
            intervals=intervals,
        )
        result["upstream_parameters"] = parameters
        result["upstream_start"] = upstream_start
    if qoco_library is not None:
        with QOCOGPU(
            canonical,
            library_path=qoco_library,
            tolerance=cpu_tolerance,
            iteration_limit=2_000,
        ) as backend:
            qoco = backend.solve()
            report = backend.last_report
        qoco_normal_dual = qoco.dual.copy()
        qoco_normal_dual[canonical.structure.n_constraints :] *= -1.0
        result["qoco_gpu"] = _solution_record(
            canonical,
            qoco,
            normal_dual=qoco_normal_dual,
            cpu_objective=cpu_objective,
            intervals=intervals,
        )
        result["qoco_report"] = asdict(report) if report is not None else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--persistent-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--upstream-variant",
        choices=tuple(_UPSTREAM_VARIANTS),
    )
    parser.add_argument(
        "--upstream-start",
        choices=("cold", "cpu-primal", "cpu-primal-dual"),
        default="cold",
    )
    parser.add_argument("--iteration-limit", type=int, default=1_000_000)
    parser.add_argument("--cpu-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--intervals", type=int, default=2)
    parser.add_argument("--qoco-library", type=Path)
    parser.add_argument("--inspect-variable", type=int)
    arguments = parser.parse_args()
    result = run(
        arguments.dump,
        persistent_output=arguments.persistent_output,
        upstream_variant=arguments.upstream_variant,
        upstream_start=arguments.upstream_start,
        iteration_limit=arguments.iteration_limit,
        cpu_tolerance=arguments.cpu_tolerance,
        intervals=arguments.intervals,
        qoco_library=arguments.qoco_library,
        inspect_variable=arguments.inspect_variable,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        arguments.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
