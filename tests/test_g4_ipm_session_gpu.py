"""Regression: a pure-gpu-ipm persistent session must run real QOCO solves on every attempt.

Background (integration/single-gpu-v1, campaign g4-claim-core-a08f5e2): every second attempt of
each pure-gpu-ipm group was recorded as an instant ``numerical`` failure with zero QOCO workspace
creations. The reset boundary asked the PDHCG workspace for a FULL_RETAINED warm start although
pure IPM never populates it, the driver returned INVALID_STATE, and the executor mapped that API
error to a solver disposition. This test drives the real executor through the capability probe
(P1-C, N=20, conditioning 0, pure-gpu-ipm) and asserts >= 1 QOCO workspace creation and a solver
disposition on all nine attempts. It needs the CUDA build and the QOCO library, so it skips
unless ``SPACEPDHCG_G4_EXECUTOR`` points at ``device_scvx_integration_test`` and
``SPACEPDHCG_QOCO_LIBRARY`` is set.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = os.environ.get("SPACEPDHCG_G4_EXECUTOR", "")

pytestmark = pytest.mark.skipif(
    not EXECUTOR or not Path(EXECUTOR).is_file() or not os.environ.get("SPACEPDHCG_QOCO_LIBRARY"),
    reason="needs SPACEPDHCG_G4_EXECUTOR (CUDA executor) and SPACEPDHCG_QOCO_LIBRARY",
)


def _generator():
    specification = importlib.util.spec_from_file_location(
        "generate_g4_executor_capability",
        ROOT / "scripts/gpu/generate_g4_executor_capability.py",
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_pure_gpu_ipm_session_constructs_qoco_and_returns_solver_dispositions() -> None:
    generator = _generator()
    policy = json.loads((ROOT / "benchmarks/g4_policy.json").read_text())
    policy_sha256 = (ROOT / "benchmarks/g4_policy.sha256").read_text().split()[0]
    matrix_sha256 = hashlib.sha256(
        generator.canonical_bytes(policy["matrix"]).rstrip(b"\n")
    ).hexdigest()
    probe = generator.run_session_probe(Path(EXECUTOR), policy_sha256, matrix_sha256)
    ipm = probe["pure_gpu_ipm_probe"]
    assert ipm["policy"] == "pure-gpu-ipm"
    assert len(ipm["qoco_workspace_creations"]) == 9
    # The workspace persists across the group, so every attempt reports the single creation.
    assert all(value >= 1 for value in ipm["qoco_workspace_creations"])
    # Warm boundaries (attempts 1..8 under warm_mode=primal) must not turn into API failures.
    assert set(ipm["dispositions"]) <= {"qualified", "unqualified"}
    assert "executor_defect" not in ipm["dispositions"]
    assert "numerical" not in ipm["dispositions"]
    # Amendment single-gpu-v1.2 rule A: the session ran QOCO with its native default
    # equilibration (ruiz_iters = 0 at the pinned commit), echoed that selection on every
    # attempt with the raw QOCO status code, and recorded scaling_mode not_applicable_ipm_native.
    assert ipm["policy_amendment"] == "single-gpu-v1.2"
    assert ipm["ipm_equilibration"] == {
        "mode": "qoco_native_default",
        "ruiz_iterations": 0,
        "requested_ruiz_iterations": 0,
        "scaling_mode": "not_applicable_ipm_native",
    }
    assert len(ipm["qoco_status_codes"]) == 9
    # QOCO_SOLVED (1) or QOCO_SOLVED_INACCURATE (2) behind every solver disposition; never
    # QOCO_NUMERICAL_ERROR (3) or QOCO_MAX_ITER (4) on this known-good coordinate.
    assert all(isinstance(code, int) and code in (1, 2) for code in ipm["qoco_status_codes"])
