from __future__ import annotations

import ctypes

import pytest

from spacepdhcg.backends.dlpack_capsule import DLPackKind, consume_capsule

_CAPSULE_NEW = ctypes.pythonapi.PyCapsule_New
_CAPSULE_NEW.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_CAPSULE_NEW.restype = ctypes.py_object


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        (b"dltensor", DLPackKind.LEGACY),
        (b"dltensor_versioned", DLPackKind.VERSIONED),
    ],
)
def test_consume_capsule_is_one_shot(name: bytes, kind: DLPackKind) -> None:
    storage = ctypes.c_int(42)
    capsule = _CAPSULE_NEW(ctypes.addressof(storage), name, None)

    consumed = consume_capsule(capsule)

    assert consumed.managed_tensor == ctypes.addressof(storage)
    assert consumed.kind is kind
    with pytest.raises(ValueError, match="unconsumed DLPack"):
        consume_capsule(capsule)
