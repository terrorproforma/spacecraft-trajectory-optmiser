"""Solver backend adapters."""

from spacepdhcg.backends.base import PersistentCQPBackend
from spacepdhcg.backends.clarabel_backend import PersistentClarabel
from spacepdhcg.backends.osqp_backend import PersistentOSQP
from spacepdhcg.backends.pdhcg_oneshot import PDHCGOneShot, PDHCGUnavailableError
from spacepdhcg.backends.qoco_gpu import (
    QOCOGPU,
    CtypesQOCOAPI,
    PDHCGQOCOHybrid,
    QOCOAdapterError,
    QOCORawSolution,
    QOCORunReport,
    QOCOSettings,
    QOCOSetupError,
    QOCOSolveError,
    QOCOUnavailableError,
    QOCOUnsupportedError,
    UnsupportedQOCOClass,
    canonical_primal_residual,
    convert_to_qoco,
    independent_residuals,
)

__all__ = [
    "QOCOGPU",
    "CtypesQOCOAPI",
    "PDHCGOneShot",
    "PDHCGQOCOHybrid",
    "PDHCGUnavailableError",
    "PersistentCQPBackend",
    "PersistentClarabel",
    "PersistentOSQP",
    "QOCOAdapterError",
    "QOCORawSolution",
    "QOCORunReport",
    "QOCOSettings",
    "QOCOSetupError",
    "QOCOSolveError",
    "QOCOUnavailableError",
    "QOCOUnsupportedError",
    "UnsupportedQOCOClass",
    "canonical_primal_residual",
    "convert_to_qoco",
    "independent_residuals",
]
