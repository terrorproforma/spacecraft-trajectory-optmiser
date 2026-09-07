"""Select the retained GPU trajectory loop at the command boundary.

The native extensions currently read process-level switches. Keep that transport
inside the CLI's single-worker command scope, restore the caller's environment,
and expose a production option instead of requiring diagnostic flag knowledge.
Library users that manage execution switches directly retain their existing API.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from threading import RLock

_GRAPH_SWITCHES = (
    "GTOC12_OUTER_GRAPH",
    "GTOC12_DEVICE_SCHEDULING",
    "GTOC12_DEFERRED_REPORTS",
    "GTOC12_DEVICE_REFRESH",
    "GTOC12_DEVICE_ASSEMBLY_VALIDATION",
    "GTOC12_DEVICE_QUALIFICATION",
    "QOCO_DEVICE_VALIDATION",
    "QOCO_NATIVE_NUMERIC_REPLAY",
    "QOCO_NATIVE_REPLAY",
    "QOCO_IPM_GRAPH",
)
_EXECUTION_LOCK = RLock()


def selected_execution(args) -> str:
    requested = getattr(args, "gpu_execution", "auto")
    if requested not in {"auto", "graph", "dispatch"}:
        raise ValueError("--gpu-execution must be auto, graph or dispatch")
    outer = getattr(args, "outer_loop_backend", "python")
    if requested == "graph" and outer != "cuda":
        raise ValueError("--gpu-execution graph requires --outer-loop-backend cuda")
    graph = requested == "graph" or (requested == "auto" and outer == "cuda")
    return "graph" if graph else "dispatch"


@contextmanager
def using_gpu_execution(args):
    """Apply one command's execution policy without leaking it to the next call."""
    selected = selected_execution(args)
    # Serialize command scopes because native switches are process-wide. The
    # reentrant lock also permits nested commands to restore their caller's mode.
    with _EXECUTION_LOCK:
        # Ordinary CPU commands have no GPU policy to install. Explicit dispatch
        # is useful for comparisons even if graph flags were inherited.
        if selected == "dispatch" and getattr(args, "gpu_execution", "auto") == "auto":
            yield selected
            return
        if selected == "graph" and getattr(args, "workers", 1) != 1:
            raise ValueError("GPU graph execution requires --workers 1")
        names = ["SPACEPDHCG_TEST_" + name for name in _GRAPH_SWITCHES]
        previous = {name: os.environ.get(name) for name in names}
        try:
            os.environ.update({name: "1" if selected == "graph" else "0" for name in names})
            yield selected
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
