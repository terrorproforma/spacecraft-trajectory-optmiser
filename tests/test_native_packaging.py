from __future__ import annotations

import spacepdhcg
from spacepdhcg.native import (
    c_api_version,
    native_available,
    native_version,
    packaged_library_path,
)


def test_native_library_is_packaged_and_abi_compatible() -> None:
    path = packaged_library_path()
    assert path.is_file()
    assert native_available()
    assert c_api_version() == 1
    assert native_version() == spacepdhcg.__version__
