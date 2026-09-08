"""Analytic original-coordinate checks independent of native cone conversion."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

path = Path(__file__).resolve().parents[1] / "scripts/gpu/audit_persistent_snapshot.py"
spec = importlib.util.spec_from_file_location("persistent_snapshot_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    # The equality fixes x2=2 when x1=1. Both an upper scalar constraint
    # -x1<=-1 and a radius-first SOC (1,x1,0) are active at the unique optimum.
    ld = np.longdouble
    problem = dict(
        P=sp.csc_matrix([[2, 0.5], [0.5, 1]], dtype=ld),
        A=sp.csc_matrix([[1, 1]], dtype=ld),
        G=sp.csc_matrix([[-1, 0], [0, 0], [-1, 0], [0, 0]], dtype=ld),
        c=np.array([-1.5, -2], dtype=ld),
        b=np.array([3], dtype=ld),
        h=np.array([-1, 1, 0, 0], dtype=ld),
        n=2,
        p=1,
        m=4,
        l=1,
        soc=np.array([3]),
        shift=1,
        origin=np.array([0.25, -0.75], dtype=ld),
    )
    record = dict(x=[1, 2], y=[-0.5], z=[3, 2, -2, 0], s=[0, 1, 1, 0], termination=1)
    return problem, record


def test_mixed_cones_off_diagonal_hessian_known_optimum():
    problem, record = fixture()
    result = module.audit(problem, record, backend="persistent", coordinates="original")
    assert result["qualified"]
    assert result["objective"] == result["dual_objective"] == -1.5
    assert result["gap"] == result["dual"] == result["primal"] == 0


def test_shift_applied_exactly_once():
    problem, record = fixture()
    original = module.audit(problem, record, backend="persistent", coordinates="original")
    record["x"] = [0.75, 2.75]
    translated = module.audit(problem, record, backend="persistent", coordinates="translated")
    assert translated["qualified"]
    assert translated["objective"] == original["objective"]
    assert not module.audit(problem, record, backend="persistent", coordinates="original")[
        "qualified"
    ]


@pytest.mark.parametrize("mutation", ["dual_sign", "radius_order", "primal", "slack"])
def test_incorrect_export_fails_common_gate(mutation):
    problem, record = fixture()
    if mutation == "dual_sign":
        record["z"] = [-v for v in record["z"]]
    elif mutation == "radius_order":
        record["z"] = [3, -2, 0, 2]
    elif mutation == "primal":
        record["x"][1] += 0.1
    else:
        record["s"][0] = 0.1
    assert not module.audit(problem, record, backend="persistent", coordinates="original")[
        "qualified"
    ]


def test_solver_status_and_external_accuracy_are_distinct():
    problem, record = fixture()
    record["termination"] = 2
    result = module.audit(problem, record, backend="persistent", coordinates="original")
    assert result["passes_common_kkt_gate"]
    assert not result["qualified"]
    record["status"] = 2
    assert module.audit(problem, record, backend="qoco", coordinates="original")["qualified"]


@pytest.mark.parametrize("bad", [[np.nan, 2], [np.inf, 2], [1], [[1, 2]]])
def test_nonfinite_and_wrong_dimensions_reject(bad):
    problem, record = fixture()
    record["x"] = bad
    with pytest.raises(ValueError):
        module.audit(problem, record, backend="persistent", coordinates="original")


def test_snapshot_parser_keeps_upper_p_once(tmp_path):
    snapshot = tmp_path / "known.txt"
    snapshot.write_text(
        "\n".join(
            [
                "SPACEPDHCG_QOCO_QP_V1",
                "2 1 4 3 2 2 1 1 0",
                "100 0 0 0",
                "1e-12 1e-8 1e-8 1e-8 1e-12 1e-9 1e-9 1e-6 1e-6",
                "3 0 1 3",
                "3 0 0 1",
                "3 0 1 2",
                "2 0 0",
                "3 0 2 2",
                "2 0 2",
                "1 3",
                "14 2 .5 1 1 1 -1 -1 -1.5 -2 3 -1 1 0 0",
                "0",
                "0",
                "0",
                "",
            ]
        )
    )
    parsed = module.load_snapshot(snapshot)
    _, record = fixture()
    assert np.array_equal(parsed["P"].toarray(), [[2, 0.5], [0.5, 1]])
    assert module.audit(parsed, record, backend="persistent", coordinates="original")["qualified"]


def test_block_complementarity_cannot_cancel():
    ld = np.longdouble
    problem = dict(
        P=sp.csc_matrix((1, 1), dtype=ld),
        A=sp.csc_matrix((0, 1), dtype=ld),
        G=sp.csc_matrix((2, 1), dtype=ld),
        c=np.array([0], dtype=ld),
        b=np.array([], dtype=ld),
        h=np.array([-1e-8, 1], dtype=ld),
        n=1,
        p=0,
        m=2,
        l=2,
        soc=[],
        shift=0,
    )
    record = dict(x=[0], y=[], z=[1e12, 1e4], s=[-1e-8, 1], termination=1)
    result = module.audit(problem, record, backend="persistent", coordinates="original")
    assert result["gap"] < 1e-9
    assert result["cone_violation"] <= 1e-8
    assert result["complementarity_max_absolute"] >= 1e4
    assert not result["passes_common_kkt_gate"]


def test_parser_audits_the_fp64_input_not_the_decimal_real(tmp_path):
    snapshot = tmp_path / "rounded.txt"
    snapshot.write_text(
        "\n".join(
            [
                "SPACEPDHCG_QOCO_QP_V1",
                "1 0 0 1 0 0 0 0 0",
                "100 0 0 0",
                "1e-12 1e-8 1e-8 1e-8 1e-12 1e-9 1e-9 1e-6 1e-6",
                "2 0 1",
                "1 0",
                "2 0 0",
                "0",
                "2 0 0",
                "0",
                "0",
                "2 0.1 0.2",
                "0",
                "0",
                "0",
                "",
            ]
        )
    )
    parsed = module.load_snapshot(snapshot)
    assert parsed["P"][0, 0] == np.longdouble(np.float64("0.1"))


@pytest.mark.parametrize("mismatch", [None, "hash", "coordinates", "missing"])
def test_log_is_bound_to_exact_input_and_coordinates(tmp_path, mismatch):
    problem, record = fixture()
    digest = "a" * 64
    metadata = {"input_sha256": digest, "coordinate_system": "original"}
    if mismatch == "hash":
        metadata["input_sha256"] = "b" * 64
    elif mismatch == "coordinates":
        metadata["coordinate_system"] = "translated"
    log = tmp_path / "run.log"
    lines = [] if mismatch == "missing" else ["PERSISTENT_REPLAY_META " + json.dumps(metadata)]
    lines.append("PERSISTENT_REPLAY " + json.dumps(record))
    log.write_text("\n".join(lines))
    kwargs = dict(backend="persistent", coordinates="original", record_prefix="PERSISTENT_REPLAY")
    if mismatch:
        with pytest.raises(ValueError):
            module.audit_log(problem, log, digest, **kwargs)
    else:
        assert module.audit_log(problem, log, digest, **kwargs)[0]["qualified"]
