"""Versioned campaign scopes and cross-scope claim guards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

HISTORICAL_FULL_SCOPE_ID: Final = "full-multi-gpu-v1"
ACTIVE_SINGLE_GPU_SCOPE_ID: Final = "single-gpu-v1"
SCOPE_IDS: Final = frozenset({HISTORICAL_FULL_SCOPE_ID, ACTIVE_SINGLE_GPU_SCOPE_ID})

ACTIVE_SINGLE_GPU_REQUIREMENTS: Final = (
    "G0 host/native truth and environment freeze",
    "G1 pinned upstream one-shot CUDA correctness on one physical GPU",
    "G2 persistent one-GPU PDHCG workspace correctness and lifecycle",
    "G3 device-resident deterministic SCvx correctness on one physical GPU",
    "G4 one-GPU adaptive, inexact, pure-IPM, and hybrid experiments",
    "Paper 1 one-GPU evidence, H1/H2/H3/H5/H6 decisions, and in-scope products",
    "OrbitWeaver one-GPU coarse-to-refined-to-scenario simulations",
    "OrbitWeaver one-GPU pricing/restricted-master integration and independent certification",
    "OrbitWeaver one-GPU trajectory and mission visualisation",
)
DEFERRED_MULTI_GPU_REQUIREMENTS: Final = (
    "G5 physical 2/4/8-GPU correctness against monolithic CPU/single-GPU truth",
    "G5 physical one-stream and overlap sanitizer/profiler validation",
    "G5 physical cancellation, rank-failure, communicator, checkpoint, and restart validation",
    "G5 scenario-aware versus generic physical partition comparison",
    "G5 same-machine strong scaling at 1/2/4/8 GPUs",
    "G5 weak scaling at fixed per-GPU scenario/nonzero work",
    "G5 physical scaling, communication, memory, energy, and H4 decision evidence",
    "OrbitWeaver distributed route-by-scenario correctness on 2/4/8 physical GPUs",
    "OrbitWeaver physical throughput, scaling, energy, memory-crossover, and tractability claims",
)

SCOPE_DEFINITIONS: Final[dict[str, dict[str, Any]]] = {
    HISTORICAL_FULL_SCOPE_ID: {
        "scope_id": HISTORICAL_FULL_SCOPE_ID,
        "active_hypotheses": ["H1", "H2", "H3", "H4", "H5", "H6"],
        "deferred_hypotheses": [],
        "included_products": [
            *(f"F{index:02d}" for index in range(1, 13)),
            *(f"T{index:02d}" for index in range(1, 9)),
        ],
        "deferred_products": [],
        "allowed_gpu_counts": [0, 1, 2, 4, 8],
        "requires_physical_g5": True,
    },
    ACTIVE_SINGLE_GPU_SCOPE_ID: {
        "scope_id": ACTIVE_SINGLE_GPU_SCOPE_ID,
        "active_hypotheses": ["H1", "H2", "H3", "H5", "H6"],
        "deferred_hypotheses": ["H4"],
        "included_products": [
            "F01",
            "F02",
            "F03",
            "F04",
            "F05",
            "F06",
            "F08",
            "F09",
            "F10",
            "F11",
            "T01",
            "T02",
            "T03",
            "T04",
            "T05",
            "T07",
            "T08",
        ],
        "deferred_products": ["F07", "F12", "T06"],
        "allowed_gpu_counts": [0, 1],
        "requires_physical_g5": False,
        # Preregistered amendments to this scope. Each is a versioned JSON contract with its own
        # lock file; the original single-gpu-v1 rules stay readable and are never rewritten.
        "amendments": [
            {
                "amendment_id": "single-gpu-v1.1",
                "applies_to": "g4-h5-h6-claim-core-v1",
                "path": "benchmarks/g4_claim_core_amendment_v1_1.json",
                "lock": "benchmarks/g4_claim_core_amendment_v1_1.sha256",
                "document": "docs/G4_CLAIM_CORE_AMENDMENT_V1_1.md",
            }
        ],
    },
}


class CampaignScopeError(ValueError):
    """Raised when evidence or claims cross a versioned campaign boundary."""


def scope_definition(scope_id: str) -> Mapping[str, Any]:
    try:
        return SCOPE_DEFINITIONS[scope_id]
    except KeyError as error:
        raise CampaignScopeError(f"unknown campaign scope {scope_id!r}") from error


def effective_scope_id(config: Mapping[str, Any]) -> str:
    """Map historical 1.0 configurations to their original full campaign."""

    if config.get("schema_version") == "1.0.0":
        if "campaign_scope_id" in config:
            raise CampaignScopeError("historical campaign config may not override its scope")
        return HISTORICAL_FULL_SCOPE_ID
    scope_id = config.get("campaign_scope_id")
    if not isinstance(scope_id, str):
        raise CampaignScopeError("versioned campaign config requires campaign_scope_id")
    scope_definition(scope_id)
    return scope_id


def validate_claims_for_scope(
    scope_id: str,
    claims: Mapping[str, Sequence[str]],
) -> None:
    """Require claims for active hypotheses and forbid claims for deferred ones."""

    scope = scope_definition(scope_id)
    expected = {f"H{index}" for index in range(1, 7)}
    if set(claims) != expected:
        raise CampaignScopeError("claims must link exactly H1-H6")
    active = set(scope["active_hypotheses"])
    deferred = set(scope["deferred_hypotheses"])
    for hypothesis, claim_ids in claims.items():
        if not isinstance(claim_ids, Sequence) or isinstance(claim_ids, (str, bytes)):
            raise CampaignScopeError(f"{hypothesis} claims must be an array")
        valid = all(isinstance(claim_id, str) and claim_id for claim_id in claim_ids)
        if not valid:
            raise CampaignScopeError(f"{hypothesis} claim IDs must be non-empty strings")
        if hypothesis in active and not claim_ids:
            raise CampaignScopeError(f"{hypothesis} requires an in-scope manuscript claim")
        if hypothesis in deferred and claim_ids:
            raise CampaignScopeError(
                f"{hypothesis} is deferred in {scope_id}; cross-scope claims are forbidden"
            )


def validate_run_scope(scope_id: str, *, family: str, gpus: int) -> None:
    scope = scope_definition(scope_id)
    if gpus not in scope["allowed_gpu_counts"]:
        raise CampaignScopeError(
            f"{scope_id} cannot consume a {gpus}-GPU record; cross-scope evidence is forbidden"
        )
    if scope_id == ACTIVE_SINGLE_GPU_SCOPE_ID and family.startswith("P1-F"):
        raise CampaignScopeError(
            f"{scope_id} cannot consume G5/P1-F evidence; it belongs to the deferred campaign"
        )
