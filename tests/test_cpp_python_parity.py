from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from spacepdhcg.cqp import ConeBlock, ConeKind, CQPStructure, CSCStructure
from spacepdhcg.distributed import BlockArrowLayout, ScenarioTree
from spacepdhcg.models import PoweredDescent3DOFModel
from spacepdhcg.transcription import (
    PoweredDescent3DOFSubproblem,
    PoweredDescentSCvxConfig,
)

ROOT = Path(__file__).resolve().parents[1]
_FNV_OFFSET = 1_469_598_103_934_665_603
_FNV_PRIME = 1_099_511_628_211
_MASK_64 = (1 << 64) - 1
_CONE_CODE = {
    ConeKind.SECOND_ORDER: 0,
    ConeKind.ROTATED_SECOND_ORDER: 1,
    ConeKind.EXPONENTIAL: 2,
    ConeKind.POWER: 3,
    ConeKind.POSITIVE_SEMIDEFINITE: 4,
}


def _fnv_bytes(current: int, payload: bytes) -> int:
    for value in payload:
        current ^= value
        current = (current * _FNV_PRIME) & _MASK_64
    return current


def _u32(current: int, value: int) -> int:
    return _fnv_bytes(current, struct.pack("<I", value & 0xFFFFFFFF))


def _u64(current: int, value: int) -> int:
    return _fnv_bytes(current, struct.pack("<Q", value))


def _f64(current: int, value: float) -> int:
    return _fnv_bytes(current, struct.pack("<d", value))


def _pattern_fingerprint(current: int, pattern: CSCStructure) -> int:
    rows, columns = pattern.shape
    current = _u32(current, rows)
    current = _u32(current, columns)
    current = _u64(current, pattern.indptr.size)
    for value in pattern.indptr:
        current = _u32(current, int(value))
    current = _u64(current, pattern.indices.size)
    for value in pattern.indices:
        current = _u32(current, int(value))
    return current


def _cone_fingerprint(current: int, cones: tuple[ConeBlock, ...]) -> int:
    current = _u64(current, len(cones))
    for cone in cones:
        current = _fnv_bytes(current, bytes([_CONE_CODE[cone.kind]]))
        current = _u32(current, cone.start)
        current = _u32(current, cone.vector_dimension)
        current = _f64(current, cone.power_alpha)
    return current


def _structure_fingerprint(structure: CQPStructure) -> int:
    current = _FNV_OFFSET
    current = _pattern_fingerprint(current, structure.quadratic)
    current = _pattern_fingerprint(current, structure.constraint)
    current = _fnv_bytes(current, bytes([structure.affine_cone is not None]))
    if structure.affine_cone is not None:
        current = _pattern_fingerprint(current, structure.affine_cone)
    current = _cone_fingerprint(current, structure.affine_cones)
    current = _cone_fingerprint(current, structure.variable_cones)
    return current


def _tiny_structure() -> CQPStructure:
    quadratic = sp.eye(2, format="csc")
    scalar = sp.csc_matrix(np.asarray([[1.0, -1.0]]))
    affine = sp.csc_matrix(
        (
            np.ones(2),
            (np.asarray([0, 1]), np.asarray([0, 1])),
        ),
        shape=(3, 2),
    )
    return CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(scalar),
        affine_cone=CSCStructure.from_matrix(affine),
        affine_cones=(ConeBlock(ConeKind.SECOND_ORDER, 0, 1),),
    )


def _compile_probe(tmp_path: Path) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++20 compiler is required for the cross-language parity gate")
    executable = tmp_path / "spacepdhcg-parity-probe"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "tools" / "parity_probe.cpp"),
            "-o",
            str(executable),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return executable


def test_cpp_core_matches_python_reference(tmp_path: Path) -> None:
    executable = _compile_probe(tmp_path)
    completed = subprocess.run(
        [str(executable)],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["tiny_fingerprint"] == _structure_fingerprint(_tiny_structure())

    model = PoweredDescent3DOFModel()
    state = np.asarray([20.0, -10.0, 120.0, 0.4, -0.2, -7.0, 2_000.0])
    thrust = np.asarray([1_200.0, -500.0, 8_000.0])
    control = np.concatenate((thrust, [np.linalg.norm(thrust)]))
    state_jacobian, control_jacobian = model.jacobians(state, control)
    np.testing.assert_allclose(payload["dynamics"], model.dynamics(state, control), atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(payload["state_jacobian"]).reshape(7, 7),
        state_jacobian,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(payload["control_jacobian"]).reshape(7, 4),
        control_jacobian,
        atol=1.0e-13,
    )

    tree = ScenarioTree.common_open_loop(4, 5, common_prefix=3)
    layout = BlockArrowLayout(
        tree,
        state_dimension=7,
        control_dimension=4,
        local_auxiliary_dimension=12,
    )
    assert payload["scenario"] == {
        "nodes": len(tree.nodes),
        "shared_nodes": len(tree.shared_nodes),
        "variables": layout.total_variables,
        "consensus_dimension": layout.consensus_dimension,
        "nonanticipativity_rows": layout.nonanticipativity_rows,
    }

    np.testing.assert_allclose(
        payload["lambert_departure"],
        [-5.9925, 1.9254, 3.2456],
        atol=5.0e-3,
    )
    np.testing.assert_allclose(
        payload["lambert_arrival"],
        [-3.3125, -4.1966, -0.3853],
        atol=5.0e-3,
    )

    subproblem = PoweredDescent3DOFSubproblem(
        model,
        PoweredDescentSCvxConfig(
            intervals=4,
            step_seconds=1.0,
            trust_radius=1.0,
        ),
    )
    assert payload["transcription"] == {
        "variables": subproblem.layout.n_variables,
        "scalar_rows": subproblem.layout.n_scalar_constraints,
        "affine_rows": subproblem.layout.n_affine_cone_rows,
        "quadratic_nonzeros": subproblem.structure.quadratic.nnz,
        "scalar_nonzeros": subproblem.structure.constraint.nnz,
        "affine_nonzeros": subproblem.structure.affine_cone.nnz,
        "cone_blocks": len(subproblem.structure.affine_cones),
        "fingerprint": _structure_fingerprint(subproblem.structure),
    }
