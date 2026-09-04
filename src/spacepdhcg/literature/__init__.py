"""Literature reproduction targets for the comparative solver campaign.

The package freezes literature-derived inputs (Phase 0 of
``docs/COMPARATIVE_SOLVER_CAMPAIGN.md``) and reproduces reference results on CPU
(Phase 1).  Every external number is registered in the provenance store and every
runnable target is registered in ``benchmarks/literature/targets.json``.
"""

from spacepdhcg.literature.provenance import (
    EVIDENCE_LABELS,
    ProvenanceError,
    ProvenanceRecord,
    ProvenanceStore,
    load_provenance_store,
    validate_provenance_document,
)
from spacepdhcg.literature.registry import (
    LiteratureTarget,
    TargetRegistry,
    load_target_registry,
)

__all__ = [
    "EVIDENCE_LABELS",
    "LiteratureTarget",
    "ProvenanceError",
    "ProvenanceRecord",
    "ProvenanceStore",
    "TargetRegistry",
    "load_provenance_store",
    "load_target_registry",
    "validate_provenance_document",
]
