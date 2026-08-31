"""SpacePDHCG research software."""

from spacepdhcg.cqp import CanonicalCQP, CQPStructure, CQPValues, CSCStructure
from spacepdhcg.native import (
    NativeLibraryError,
    c_api_version,
    native_available,
    native_version,
    packaged_library_path,
)

__all__ = [
    "CQPStructure",
    "CQPValues",
    "CSCStructure",
    "CanonicalCQP",
    "NativeLibraryError",
    "c_api_version",
    "native_available",
    "native_version",
    "packaged_library_path",
]
__version__ = "0.1.0.dev0"
