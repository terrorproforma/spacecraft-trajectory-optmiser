"""GPU preflight for the literature GPU legs.

The RTX 5090 in the measurement host is owned by the Gate G4 measured campaign
(``device_scvx_integration_test --g4-session`` / ``--g4-server``).  Any literature GPU leg
(P1-C pure-QOCO SCvx, P1-D-MC pure-QOCO batch, CUDA correctness tests) must refuse to touch the
device while that session runs, so its timing runs are never contaminated.

The preflight is deliberately conservative:

* ``nvidia-smi --query-compute-apps`` must be available and succeed;
* every compute process is resolved through ``/proc/<pid>/cmdline`` (``nvidia-smi`` prints
  ``[Not Found]`` for processes in another mount namespace, e.g. WSL);
* any process whose command line names ``device_scvx_integration_test`` or a ``--g4-``
  argument marks the device as owned by G4 -> refused;
* other compute processes are reported and, by default, also refuse (a shared device is not a
  clean measurement either) unless ``allow_shared=True``.

No environment variable can override a G4 refusal.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

G4_MARKERS = ("device_scvx_integration_test", "--g4-session", "--g4-server", "--g4-sample")


@dataclass(slots=True)
class ComputeProcess:
    pid: int
    reported_name: str
    command_line: str
    g4_owner: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GpuPreflight:
    ok: bool
    reason: str
    nvidia_smi: str | None
    processes: list[ComputeProcess] = field(default_factory=list)
    qoco_library: str | None = None

    @property
    def g4_owned(self) -> bool:
        return any(process.g4_owner for process in self.processes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "nvidia_smi": self.nvidia_smi,
            "g4_owned": self.g4_owned,
            "processes": [process.as_dict() for process in self.processes],
            "qoco_library": self.qoco_library,
        }


class GpuPreflightRefused(RuntimeError):
    """Raised by :func:`require_gpu` when the device must not be used."""

    def __init__(self, preflight: GpuPreflight) -> None:
        super().__init__(preflight.reason)
        self.preflight = preflight


def _read_command_line(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)


def parse_compute_apps(
    text: str, *, command_line_of: Callable[[int], str] | None = None
) -> list[ComputeProcess]:
    """Parse ``nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader``."""

    if command_line_of is None:
        command_line_of = _read_command_line
    processes: list[ComputeProcess] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, name = line.partition(",")
        try:
            pid = int(pid_text.strip())
        except ValueError:
            continue
        name = name.strip()
        command_line = command_line_of(pid)
        haystack = f"{name} {command_line}"
        processes.append(
            ComputeProcess(
                pid=pid,
                reported_name=name,
                command_line=command_line,
                g4_owner=any(marker in haystack for marker in G4_MARKERS),
            )
        )
    return processes


def query_compute_apps(
    *, runner: Callable[[Sequence[str]], str] | None = None
) -> tuple[str | None, list[ComputeProcess], str | None]:
    """Return ``(nvidia_smi_path, processes, error)``."""

    executable = shutil.which("nvidia-smi")
    if runner is None and executable is None:
        return None, [], "nvidia-smi is not on PATH"
    command = [
        executable or "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader",
    ]
    try:
        if runner is not None:
            output = runner(command)
        else:
            output = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=20
            ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        return executable, [], f"nvidia-smi failed: {error!r}"
    return executable, parse_compute_apps(output), None


def preflight(
    *,
    allow_shared: bool = False,
    qoco_library: str | os.PathLike[str] | None = None,
    runner: Callable[[Sequence[str]], str] | None = None,
) -> GpuPreflight:
    """Decide whether a literature GPU leg may run right now."""

    library = qoco_library or os.environ.get("SPACEPDHCG_QOCO_LIBRARY")
    library_text = str(library) if library else None
    smi, processes, error = query_compute_apps(runner=runner)
    if error is not None:
        return GpuPreflight(False, f"refused: {error}", smi, processes, library_text)
    owners = [process for process in processes if process.g4_owner]
    if owners:
        described = ", ".join(f"pid {p.pid} ({p.command_line or p.reported_name})" for p in owners)
        return GpuPreflight(
            False,
            f"refused: the G4 measured campaign owns the device ({described})",
            smi,
            processes,
            library_text,
        )
    if processes and not allow_shared:
        described = ", ".join(f"pid {p.pid}" for p in processes)
        return GpuPreflight(
            False,
            f"refused: other compute processes hold the device ({described}); "
            "pass allow_shared to override",
            smi,
            processes,
            library_text,
        )
    if library_text is not None and not Path(library_text).is_file():
        return GpuPreflight(
            False,
            f"refused: QOCO library not found at {library_text}",
            smi,
            processes,
            library_text,
        )
    return GpuPreflight(True, "device free", smi, processes, library_text)


def require_gpu(**kwargs: Any) -> GpuPreflight:
    result = preflight(**kwargs)
    if not result.ok:
        raise GpuPreflightRefused(result)
    return result
