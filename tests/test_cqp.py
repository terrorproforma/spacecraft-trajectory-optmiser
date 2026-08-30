import numpy as np
import pytest
import scipy.sparse as sp

from spacepdhcg.cqp import CQPStructure, CQPValues, CSCStructure


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
    values = CQPValues(
        quadratic=np.ones(1),
        constraint=np.ones(2),
        linear=np.zeros(2),
        lower=np.zeros(2),
        upper=np.ones(2),
    )

    with pytest.raises(ValueError, match="quadratic"):
        values.validated(structure)


def test_cqp_values_allow_infinite_bounds_but_not_nan() -> None:
    matrix = sp.eye(1, format="csc")
    structure = CQPStructure(
        quadratic=CSCStructure.from_matrix(matrix),
        constraint=CSCStructure.from_matrix(matrix),
    )
    valid = CQPValues(
        quadratic=np.ones(1),
        constraint=np.ones(1),
        linear=np.zeros(1),
        lower=np.array([-np.inf]),
        upper=np.array([np.inf]),
    )
    valid.validated(structure)

    invalid = valid.copy()
    invalid.lower[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        invalid.validated(structure)
