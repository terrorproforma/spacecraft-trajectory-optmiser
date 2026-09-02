"""Shared fixtures.

``planner_native_library`` guarantees a ``libspacepdhcg`` build that exports the planner
transcription ABI: it uses ``SPACEPDHCG_NATIVE_LIBRARY`` or the packaged wheel library when
they already export it, and otherwise compiles ``cpp/src/c_api.cpp`` once per session
(exactly like ``tests/test_cpp_c_api.py``) into the pytest temporary directory.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _exports_planner(path: Path) -> bool:
    try:
        library = ctypes.CDLL(str(path))
    except OSError:
        return False
    return hasattr(library, "spacepdhcg_planner_create")


def _compile_planner_library(directory: Path) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++20 compiler is required to build the planner native library")
    library_path = directory / "libspacepdhcg_planner.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            "-I",
            str(ROOT / "cpp" / "include"),
            str(ROOT / "cpp" / "src" / "c_api.cpp"),
            "-o",
            str(library_path),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return library_path


@pytest.fixture(scope="session")
def planner_native_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override = os.environ.get("SPACEPDHCG_NATIVE_LIBRARY")
    if override:
        path = Path(override)
        if not _exports_planner(path):
            pytest.fail(f"SPACEPDHCG_NATIVE_LIBRARY={path} does not export the planner ABI")
        return path
    from spacepdhcg.native import NativeLibraryError, packaged_library_path

    try:
        packaged = packaged_library_path()
    except NativeLibraryError:
        packaged = None
    if packaged is not None and _exports_planner(packaged):
        return packaged
    library = _compile_planner_library(tmp_path_factory.mktemp("planner-native"))
    os.environ["SPACEPDHCG_NATIVE_LIBRARY"] = str(library)
    from spacepdhcg.native import _library as native_module
    from spacepdhcg.planner import native_library as planner_module

    native_module.load_native_library.cache_clear()
    planner_module._LIBRARY = None
    return library
