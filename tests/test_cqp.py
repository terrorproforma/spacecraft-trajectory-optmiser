import numpy as np
import pytest
import scipy.sparse as sp

from spacepdhcg.cqp import (
    ConeBlock,
    ConeKind,
    CQPStructure,
    CQPValues,
    CSCStructure,
)


def _plain_values(structure: CQPStructure) -> CQPValues:
    return CQPValues(
        quadratic=np.ones(structure.quadratic.nnz),
        constraint=np.ones(structure.constraint.nnz),
        linear=np.zeros(structure.n_variables),
        lower=np.zeros(structure.n_constraints),
        upper=np.ones(structure.n_constraints),
        affine_cone=np.empty(0),
        affine_offset=np.empty(0),
        variable_lower=np.full(structure.n_variables, -np.inf),
        variable_upper=np.full(structure.n_variables, np.inf),
    )


def test_csc_structure_round_trip_and_immutability() -> None:
    matrix = sp.csc_matrix(np.array([[2.0, 0.0], [3.0, 4.0]]))
    structure = CSCStructure.from_matrix(matrix)
    rebuilt = structure.matrix(structure.values_from(matrix))

    np.testing.assert_allclose(rebuilt.toarray(), matrix.toarray())
    assert structure.nnz == 3
    with pytest.raises(ValueError):
        structure.indices[0] = 1


def test_cqp_values_reject_pattern_incompatible_shapes() -> None:
    quadratic = sp.eye(2, format="csc")
    constraint = sp.eye(2, format="csc")
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(constraint),
    )
    values = _plain_values(structure)
    values.quadratic = np.ones(1)

    with pytest.raises(ValueError, match="quadratic"):
        values.validated(structure)


def test_cqp_values_allow_infinite_bounds_but_not_nan() -> None:
    matrix = sp.eye(1, format="csc")
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(matrix),
        constraint=CSCStructure.from_matrix(matrix),
    )
    valid = _plain_values(structure)
    valid.lower[0] = -np.inf
    valid.upper[0] = np.inf
    valid.validated(structure)

    invalid = valid.copy()
    invalid.lower[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        invalid.validated(structure)


def test_soc_slot_convention_and_affine_cover_validation() -> None:
    quadratic = sp.eye(2, format="csc")
    constraint = sp.csc_matrix((0, 2))
    affine = sp.csc_matrix(
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
    )
    cone = ConeBlock(ConeKind.SECOND_ORDER, start=0, vector_dimension=2)
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(quadratic),
        constraint=CSCStructure.from_matrix(constraint),
        affine_cone=CSCStructure.from_matrix(affine),
        affine_cones=(cone,),
    )

    assert cone.slot_count == 4
    assert cone.stop == 4
    assert structure.n_affine_constraints == 4
    assert structure.n_duals == 4

    with pytest.raises(ValueError, match="cover"):
        CQPStructure(
            quadratic=CSCStructure.from_matrix(quadratic),
            constraint=CSCStructure.from_matrix(constraint),
            affine_cone=CSCStructure.from_matrix(affine),
            affine_cones=(ConeBlock(ConeKind.SECOND_ORDER, 1, 1),),
        )
