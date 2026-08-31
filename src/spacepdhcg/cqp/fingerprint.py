"""Stable CQP topology fingerprints shared with the C++ production core.

The byte-level format is intentionally simple and versioned by the implementation. It hashes
only immutable sparse and cone topology; numerical coefficient values are excluded. The C++
implementation in ``core/fixed_cqp.hpp`` uses the same little-endian field sequence.
"""

from __future__ import annotations

import struct

from .problem import ConeKind, CQPStructure, CSCStructure

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


def _pattern(current: int, pattern: CSCStructure) -> int:
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


def structure_fingerprint(structure: CQPStructure) -> int:
    """Return a deterministic unsigned 64-bit fingerprint of immutable CQP topology."""

    current = _FNV_OFFSET
    current = _pattern(current, structure.quadratic)
    current = _pattern(current, structure.constraint)
    current = _fnv_bytes(current, bytes([structure.affine_cone is not None]))
    if structure.affine_cone is not None:
        current = _pattern(current, structure.affine_cone)
    for cones in (structure.affine_cones, structure.variable_cones):
        current = _u64(current, len(cones))
        for cone in cones:
            current = _fnv_bytes(current, bytes([_CONE_CODE[cone.kind]]))
            current = _u32(current, cone.start)
            current = _u32(current, cone.vector_dimension)
            current = _f64(current, cone.power_alpha)
    return current


def structure_fingerprint_hex(structure: CQPStructure) -> str:
    """Return the topology fingerprint as a fixed-width hexadecimal string."""

    return f"{structure_fingerprint(structure):016x}"
