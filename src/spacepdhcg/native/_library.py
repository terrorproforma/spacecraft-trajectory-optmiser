"""Locate and load the SpacePDHCG native C ABI shipped inside the wheel."""

from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Final

_EXPECTED_C_API_VERSION: Final = 1


class NativeLibraryError(RuntimeError):
    """Raised when the packaged native library is missing or ABI-incompatible."""


def _candidate_names() -> tuple[str, ...]:
    if sys.platform == "win32":
        return ("spacepdhcg.dll", "libspacepdhcg.dll")
    if sys.platform == "darwin":
        return ("libspacepdhcg.dylib", "spacepdhcg.dylib")
    return ("libspacepdhcg.so", "spacepdhcg.so")


def _matches_platform_library(name: str) -> bool:
    if sys.platform == "win32":
        return name.lower().endswith(".dll") and "spacepdhcg" in name.lower()
    if sys.platform == "darwin":
        return name.startswith("libspacepdhcg") and name.endswith(".dylib")
    return name.startswith("libspacepdhcg.so")


def packaged_library_path() -> Path:
    """Return the packaged native library, preferring the unversioned loader name."""

    override = os.environ.get("SPACEPDHCG_NATIVE_LIBRARY")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise NativeLibraryError(f"SPACEPDHCG_NATIVE_LIBRARY does not name a file: {path}")
        return path

    package = resources.files("spacepdhcg.native")
    entries = [entry for entry in package.iterdir() if entry.is_file()]
    by_name = {entry.name: entry for entry in entries}
    for name in _candidate_names():
        if name in by_name:
            return Path(str(by_name[name]))

    candidates = sorted(
        (entry for entry in entries if _matches_platform_library(entry.name)),
        key=lambda entry: (len(entry.name), entry.name),
    )
    if candidates:
        return Path(str(candidates[0]))
    names = ", ".join(sorted(entry.name for entry in entries)) or "<empty package directory>"
    raise NativeLibraryError(
        "the SpacePDHCG wheel contains no native library for this platform; "
        f"package entries: {names}"
    )


def _configure(library: ctypes.CDLL) -> ctypes.CDLL:
    library.spacepdhcg_c_api_version.argtypes = []
    library.spacepdhcg_c_api_version.restype = ctypes.c_uint32
    library.spacepdhcg_native_version.argtypes = []
    library.spacepdhcg_native_version.restype = ctypes.c_char_p
    library.spacepdhcg_last_error.argtypes = []
    library.spacepdhcg_last_error.restype = ctypes.c_char_p
    version = int(library.spacepdhcg_c_api_version())
    if version != _EXPECTED_C_API_VERSION:
        raise NativeLibraryError(
            f"native C API version {version} is incompatible with expected "
            f"version {_EXPECTED_C_API_VERSION}"
        )
    return library


@lru_cache(maxsize=1)
def load_native_library() -> ctypes.CDLL:
    """Load, configure, and ABI-check the packaged native library once per process."""

    path = packaged_library_path()
    try:
        return _configure(ctypes.CDLL(str(path)))
    except OSError as error:
        raise NativeLibraryError(f"failed to load native library {path}: {error}") from error


def native_available() -> bool:
    try:
        load_native_library()
    except NativeLibraryError:
        return False
    return True


def native_version() -> str:
    raw = load_native_library().spacepdhcg_native_version()
    if raw is None:
        raise NativeLibraryError("native version function returned a null pointer")
    return raw.decode("utf-8")


def c_api_version() -> int:
    return int(load_native_library().spacepdhcg_c_api_version())
