"""Safe one-shot extraction of standard DLPack managed-tensor capsules."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol


class DLPackKind(IntEnum):
    LEGACY = 0
    VERSIONED = 1


class DLPackProducer(Protocol):
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] = ...,
    ) -> object:
        """Produce a DLPack capsule."""


@dataclass(frozen=True)
class ConsumedDLPack:
    """Pointer whose ownership has transferred away from its Python capsule."""

    managed_tensor: int
    kind: DLPackKind


_PY_CAPSULE_IS_VALID = ctypes.pythonapi.PyCapsule_IsValid
_PY_CAPSULE_IS_VALID.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PY_CAPSULE_IS_VALID.restype = ctypes.c_int
_PY_CAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
_PY_CAPSULE_GET_POINTER.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PY_CAPSULE_GET_POINTER.restype = ctypes.c_void_p
_PY_CAPSULE_SET_NAME = ctypes.pythonapi.PyCapsule_SetName
_PY_CAPSULE_SET_NAME.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PY_CAPSULE_SET_NAME.restype = ctypes.c_int

_LEGACY_NAME = b"dltensor"
_LEGACY_USED_NAME = b"used_dltensor"
_VERSIONED_NAME = b"dltensor_versioned"
_VERSIONED_USED_NAME = b"used_dltensor_versioned"


def consume_capsule(capsule: object) -> ConsumedDLPack:
    """Consume a DLPack capsule once and transfer its managed-tensor pointer."""

    if _PY_CAPSULE_IS_VALID(capsule, _VERSIONED_NAME):
        name = _VERSIONED_NAME
        used_name = _VERSIONED_USED_NAME
        kind = DLPackKind.VERSIONED
    elif _PY_CAPSULE_IS_VALID(capsule, _LEGACY_NAME):
        name = _LEGACY_NAME
        used_name = _LEGACY_USED_NAME
        kind = DLPackKind.LEGACY
    else:
        raise ValueError("expected an unconsumed DLPack managed-tensor capsule")
    pointer = _PY_CAPSULE_GET_POINTER(capsule, name)
    if not pointer:
        raise ValueError("DLPack capsule contains a null managed-tensor pointer")
    if _PY_CAPSULE_SET_NAME(capsule, used_name) != 0:
        raise RuntimeError("failed to mark DLPack capsule as consumed")
    return ConsumedDLPack(int(pointer), kind)


def consume_producer(
    producer: DLPackProducer,
    *,
    stream: int,
    max_version: tuple[int, int] = (1, 0),
) -> ConsumedDLPack:
    """Request producer-to-consumer stream ordering and consume its capsule."""

    return consume_capsule(producer.__dlpack__(stream=stream, max_version=max_version))
