"""Authoritative dependency-free JSON contracts for OrbitWeaver G7 records."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Frozen Paper 2 matrix digests. The first entry is the matrix the sealed G7 evidence was
# produced against (families P2-A..P2-E). The second entry adds the comparative-campaign P2-F
# historical GTOC replay family (docs/COMPARATIVE_SOLVER_CAMPAIGN.md) plus its metrics and
# archive pointers; it does not alter any P2-A..P2-E coordinate. Records may carry either digest so
# sealed historical evidence stays valid while new records reference the extended matrix.
PAPER2_MATRIX_SHA256_HISTORY: tuple[str, ...] = (
    "78c4e33e4aabcd85d63ba3f1e03aa2214b3ab207e680bcaaf347516802b2f6a2",
    "108f16e07e3cbef647b3b7080746c1fae3670a6a9fa61282776690aac73d17fc",
)
PAPER2_MATRIX_SHA256 = PAPER2_MATRIX_SHA256_HISTORY[-1]
CAMPAIGN_SCOPE_IDS = ["single-gpu-v1", "full-multi-gpu-v1"]
EVIDENCE_LEVELS = [
    "implemented_compiled",
    "cpu_correctness_tested",
    "one_gpu_correctness_tested",
    "physical_multi_gpu_tested",
]
RESULT_STATUSES = [
    "converged",
    "iteration_limit",
    "infeasible",
    "cancelled",
    "failed",
    "censored",
    "unsupported",
    "oom",
    "timeout",
]
FAILURE_STATUSES = [
    "infeasible",
    "cancelled",
    "failed",
    "censored",
    "unsupported",
    "oom",
    "timeout",
]

_CHECKS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "dynamics_defect",
        "path_violation",
        "terminal_error",
        "uncertainty_violation",
        "integration_error",
    ],
    "properties": {
        name: {"type": "number", "minimum": 0}
        for name in [
            "dynamics_defect",
            "path_violation",
            "terminal_error",
            "uncertainty_violation",
            "integration_error",
        ]
    },
}

_CERTIFICATION_SCHEMA = {
    "type": ["object", "null"],
    "additionalProperties": False,
    "required": ["accepted", "checks", "backend_identifier", "diagnostic"],
    "properties": {
        "accepted": {"type": "boolean"},
        "checks": {"anyOf": [_CHECKS_SCHEMA, {"type": "null"}]},
        "backend_identifier": {"type": "string", "minLength": 1},
        "diagnostic": {"type": "string", "minLength": 1},
    },
}

_TELEMETRY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "submitted",
        "completed",
        "feasible",
        "failed",
        "cancelled",
        "batches",
        "maximum_observed_batch",
        "estimated_peak_buffer_bytes",
        "group_batches",
        "ownership_batches",
    ],
    "properties": {
        **{
            name: {"type": "integer", "minimum": 0}
            for name in [
                "submitted",
                "completed",
                "feasible",
                "failed",
                "cancelled",
                "batches",
                "maximum_observed_batch",
                "estimated_peak_buffer_bytes",
            ]
        },
        "group_batches": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "ownership_batches": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
    },
}

_FAILURE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "diagnostic", "deterministic_id"],
    "properties": {
        "status": {"enum": FAILURE_STATUSES},
        "diagnostic": {"type": "string", "minLength": 1},
        "deterministic_id": {"type": ["integer", "null"], "minimum": 0},
    },
}

G7_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://spacepdhcg.dev/schema/orbitweaver-g7-config-v1.json",
    "title": "OrbitWeaver G7 configuration",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "seed",
        "repeat_count",
        "maximum_batch_size",
        "maximum_buffered_arcs",
        "maximum_workspace_bytes",
        "top_k",
        "risk_measure",
        "certification_tolerance",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "seed": {"type": "integer", "minimum": 0},
        "repeat_count": {"type": "integer", "minimum": 1},
        "maximum_batch_size": {"type": "integer", "minimum": 1},
        "maximum_buffered_arcs": {"type": "integer", "minimum": 1},
        "maximum_workspace_bytes": {"type": "integer", "minimum": 1},
        "top_k": {"type": "integer", "minimum": 1},
        "risk_measure": {"enum": ["expected", "worst_case", "cvar_0.9", "cvar_0.99"]},
        "certification_tolerance": {"type": "number", "exclusiveMinimum": 0},
    },
}

_G7_MANIFEST_PROPERTIES = {
    "run_id": {"type": "string", "minLength": 1},
    "repository_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
    "seed": {"type": "integer", "minimum": 0},
    "repeat_count": {"type": "integer", "minimum": 1},
    "backend": {"type": "string", "minLength": 1},
    "ownership": {"enum": ["single_gpu", "logical_rank_mock", "g5_distributed"]},
    "device_ids": {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"type": "integer", "minimum": 0},
    },
    "config_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "paper2_matrix_sha256": {"enum": list(PAPER2_MATRIX_SHA256_HISTORY)},
    "toolchain": {
        "type": "object",
        "additionalProperties": False,
        "required": ["python", "compiler", "cmake", "cuda"],
        "properties": {
            name: {"type": ["string", "null"], "minLength": 1}
            for name in ["python", "compiler", "cmake", "cuda"]
        },
    },
    "hardware": {
        "type": "object",
        "additionalProperties": False,
        "required": ["os", "cpu", "gpus"],
        "properties": {
            "os": {"type": "string", "minLength": 1},
            "cpu": {"type": ["string", "null"], "minLength": 1},
            "gpus": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
        },
    },
    "evidence_level": {"enum": EVIDENCE_LEVELS},
}
_G7_MANIFEST_REQUIRED = [
    "schema_version",
    "run_id",
    "repository_commit",
    "seed",
    "repeat_count",
    "backend",
    "ownership",
    "device_ids",
    "config_sha256",
    "paper2_matrix_sha256",
    "toolchain",
    "hardware",
    "evidence_level",
]
G7_MANIFEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://spacepdhcg.dev/schema/orbitweaver-g7-manifest.json",
    "title": "OrbitWeaver G7 run manifest (historical v1 or scoped v2)",
    "anyOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": _G7_MANIFEST_REQUIRED,
            "properties": {
                "schema_version": {"const": 1},
                **_G7_MANIFEST_PROPERTIES,
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": [*_G7_MANIFEST_REQUIRED, "campaign_scope_id"],
            "properties": {
                "schema_version": {"const": 2},
                "campaign_scope_id": {"enum": CAMPAIGN_SCOPE_IDS},
                **_G7_MANIFEST_PROPERTIES,
            },
        },
    ],
}

G7_CHECKPOINT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://spacepdhcg.dev/schema/orbitweaver-g7-checkpoint-v1.json",
    "title": "OrbitWeaver G7 checkpoint",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "run_id",
        "manifest_sha256",
        "paper2_matrix_sha256",
        "seed",
        "repeat_index",
        "completed_batches",
        "incumbent",
        "lower_bound",
        "completed_arc_ids",
        "warm_tokens",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "run_id": {"type": "string", "minLength": 1},
        "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "paper2_matrix_sha256": {"enum": list(PAPER2_MATRIX_SHA256_HISTORY)},
        "seed": {"type": "integer", "minimum": 0},
        "repeat_index": {"type": "integer", "minimum": 0},
        "completed_batches": {"type": "integer", "minimum": 0},
        "incumbent": {"type": ["number", "null"], "minimum": 0},
        "lower_bound": {"type": ["number", "null"], "minimum": 0},
        "completed_arc_ids": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 0},
        },
        "warm_tokens": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
    },
}

G7_RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://spacepdhcg.dev/schema/orbitweaver-g7-result-v1.json",
    "title": "OrbitWeaver G7 result",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "run_id",
        "manifest_sha256",
        "paper2_matrix_sha256",
        "seed",
        "repeat_index",
        "status",
        "incumbent",
        "lower_bound",
        "optimality_gap",
        "certified",
        "certification",
        "telemetry",
        "failures",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "run_id": {"type": "string", "minLength": 1},
        "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "paper2_matrix_sha256": {"enum": list(PAPER2_MATRIX_SHA256_HISTORY)},
        "seed": {"type": "integer", "minimum": 0},
        "repeat_index": {"type": "integer", "minimum": 0},
        "status": {"enum": RESULT_STATUSES},
        "incumbent": {"type": ["number", "null"], "minimum": 0},
        "lower_bound": {"type": ["number", "null"], "minimum": 0},
        "optimality_gap": {"type": ["number", "null"], "minimum": 0},
        "certified": {"type": "boolean"},
        "certification": _CERTIFICATION_SCHEMA,
        "telemetry": _TELEMETRY_SCHEMA,
        "failures": {"type": "array", "items": _FAILURE_SCHEMA},
    },
    "allOf": [
        {
            "if": {
                "required": ["status"],
                "properties": {"status": {"const": "converged"}},
            },
            "then": {
                "properties": {
                    name: {"type": "number", "minimum": 0}
                    for name in ["incumbent", "lower_bound", "optimality_gap"]
                }
            },
        },
        {
            "if": {
                "required": ["status"],
                "properties": {
                    "status": {
                        "enum": [
                            "infeasible",
                            "cancelled",
                            "failed",
                            "censored",
                            "unsupported",
                            "oom",
                            "timeout",
                        ]
                    }
                },
            },
            "then": {
                "properties": {
                    "certified": {"const": False},
                    "certification": {"type": "null"},
                    "failures": {"type": "array", "minItems": 1},
                }
            },
        },
        {
            "if": {
                "required": ["certified"],
                "properties": {"certified": {"const": True}},
            },
            "then": {
                "properties": {
                    "status": {"enum": ["converged", "iteration_limit"]},
                    "incumbent": {"type": "number", "minimum": 0},
                    "lower_bound": {"type": "number", "minimum": 0},
                    "certification": {
                        "type": "object",
                        "properties": {
                            "accepted": {"const": True},
                            "checks": _CHECKS_SCHEMA,
                        },
                    },
                }
            },
        },
    ],
}

SCHEMAS = {
    "config": G7_CONFIG_SCHEMA,
    "manifest": G7_MANIFEST_SCHEMA,
    "checkpoint": G7_CHECKPOINT_SCHEMA,
    "result": G7_RESULT_SCHEMA,
}


class ContractError(ValueError):
    """Raised when a G7 payload violates its frozen JSON contract."""


def validate_instance(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the JSON-Schema subset used by the frozen G7 contracts."""

    if "anyOf" in schema:
        errors = []
        for candidate in schema["anyOf"]:
            try:
                validate_instance(instance, candidate, path)
                return
            except ContractError as error:
                errors.append(str(error))
        raise ContractError(f"{path}: no anyOf alternative matched: {errors}")
    for clause in schema.get("allOf", []):
        if "if" not in clause:
            validate_instance(instance, clause, path)
            continue
        try:
            validate_instance(instance, clause["if"], path)
        except ContractError:
            if "else" in clause:
                validate_instance(instance, clause["else"], path)
        else:
            if "then" in clause:
                validate_instance(instance, clause["then"], path)
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ContractError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        raise ContractError(f"{path}: value is outside the enum")
    if "type" in schema:
        types = schema["type"]
        if isinstance(types, str):
            types = [types]
        if not any(_matches_type(instance, value) for value in types):
            raise ContractError(f"{path}: expected type {types}")
    if isinstance(instance, float) and not math.isfinite(instance):
        raise ContractError(f"{path}: number must be finite")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise ContractError(f"{path}: value is below minimum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise ContractError(f"{path}: value is below exclusive minimum")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ContractError(f"{path}: string is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], instance) is None:
            raise ContractError(f"{path}: string does not match pattern")
    if isinstance(instance, Mapping):
        required = schema.get("required", [])
        missing = sorted(set(required) - instance.keys())
        if missing:
            raise ContractError(f"{path}: missing required fields {missing}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child = properties.get(key)
            if child is None:
                if additional is False:
                    raise ContractError(f"{path}: unknown field {key!r}")
                if isinstance(additional, Mapping):
                    validate_instance(value, additional, f"{path}.{key}")
            else:
                validate_instance(value, child, f"{path}.{key}")
    if _is_array(instance):
        if len(instance) < schema.get("minItems", 0):
            raise ContractError(f"{path}: array is too short")
        if schema.get("uniqueItems") and len({_freeze(value) for value in instance}) != len(
            instance
        ):
            raise ContractError(f"{path}: array items must be unique")
        if "items" in schema:
            for index, value in enumerate(instance):
                validate_instance(value, schema["items"], f"{path}[{index}]")


def validate_named(instance: Any, name: str) -> None:
    try:
        schema = SCHEMAS[name]
    except KeyError as error:
        raise ValueError(f"unknown G7 contract {name!r}") from error
    validate_instance(instance, schema)


def _matches_type(instance: Any, expected: str) -> bool:
    if expected == "null":
        return instance is None
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "object":
        return isinstance(instance, Mapping)
    if expected == "array":
        return _is_array(instance)
    raise ContractError(f"unsupported schema type {expected!r}")


def _is_array(instance: Any) -> bool:
    return isinstance(instance, Sequence) and not isinstance(instance, (str, bytes, bytearray))


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if _is_array(value):
        return tuple(_freeze(item) for item in value)
    return value


def _json_equal(left: Any, right: Any) -> bool:
    numeric = (int, float)
    if (
        isinstance(left, numeric)
        and not isinstance(left, bool)
        and isinstance(right, numeric)
        and not isinstance(right, bool)
    ):
        return left == right
    return type(left) is type(right) and left == right
