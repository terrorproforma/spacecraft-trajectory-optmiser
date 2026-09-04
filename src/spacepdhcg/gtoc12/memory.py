"""Process-memory accounting for the pricing workers.

The v6 campaign's four collector workers peaked at 3.04 GB PSS although the live data of a
single cluster pricing is a few hundred MB.  Two mechanisms, both fixed here:

* glibc's *dynamic mmap threshold*: every time a large mmap'd block is freed the threshold is
  raised (up to 32 MB), after which the multi-MB Held-Karp tables and Lambert batches of the
  beam come from the ``brk`` heap and, once freed, stay resident behind later allocations.
  ``bound_heap_growth`` pins the threshold; ``release_heap`` (``malloc_trim``) hands the freed
  pages back at phase boundaries.
* transient working sets that are simply too large (unbounded geometry cache, int64
  back-pointers): capped in ``collectdp``.

``PhaseMemory`` attributes the high-water mark to phases so a campaign's peak can be read off
``bundle.json`` after the fact, and ``MemoryBudget`` is the regression guard the tests assert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MemoryBudget",
    "PhaseMemory",
    "bound_heap_growth",
    "current_rss_mb",
    "peak_rss_mb",
    "release_heap",
]


def peak_rss_mb() -> float:
    """High-water mark of this process' resident size (MB; ``ru_maxrss``)."""

    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover - non-POSIX
        return float("nan")


def current_rss_mb() -> float:
    """Resident set size of this process now (Linux ``/proc``; NaN elsewhere)."""

    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:  # pragma: no cover - non-Linux
        pass
    return float("nan")


_LIBC: Any = None


def _libc() -> Any:
    global _LIBC
    if _LIBC is None:
        import ctypes
        import ctypes.util

        name = ctypes.util.find_library("c")
        try:
            _LIBC = ctypes.CDLL(name) if name else False
        except OSError:  # pragma: no cover - no glibc
            _LIBC = False
    return _LIBC


def release_heap() -> bool:
    """Return freed heap pages to the OS (``malloc_trim(0)``); False where unavailable.

    The pricing worker allocates and frees tens of MB per beam candidate (Held-Karp tables,
    Lambert batches); glibc's dynamic mmap threshold moves such blocks onto the brk heap where
    freed pages stay resident, so a worker's RSS ratchets up although nothing is live.
    """

    libc = _libc()
    if not libc or not hasattr(libc, "malloc_trim"):
        return False
    try:
        libc.malloc_trim(0)
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def bound_heap_growth(mmap_threshold_bytes: int = 256 * 1024) -> bool:
    """Pin glibc's mmap threshold so large numpy blocks are mmap'd and unmapped on free.

    By default glibc raises the threshold (up to 32 MB) each time a large mmap'd block is
    freed, after which the multi-MB DP tables and Lambert batches of the beam come from the
    brk heap and, once freed, remain in the process (the 3 GB PSS of four v6 workers).  A fixed
    threshold also disables that adaptation.  Idempotent; False where ``mallopt`` is missing.
    """

    libc = _libc()
    if not libc or not hasattr(libc, "mallopt"):
        return False
    m_trim_threshold, m_mmap_threshold = -1, -3
    try:
        ok = libc.mallopt(m_mmap_threshold, int(mmap_threshold_bytes)) == 1
        ok = libc.mallopt(m_trim_threshold, int(mmap_threshold_bytes)) == 1 and ok
    except Exception:  # pragma: no cover - defensive
        return False
    return bool(ok)


class PhaseMemory:
    """Attributes the process high-water mark to pricing phases.

    ``ru_maxrss`` only ever grows, so the phase during which it grew is the phase that hosted
    the new peak; ``mark`` records, per phase, the resident size at its end and the growth of
    the high-water mark since the previous mark.  Cheap (two ``/proc`` reads per mark), no
    sampling thread, and it lands in ``bundle.json`` so a campaign's peaks can be attributed
    after the fact.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._peak = peak_rss_mb()
        self._started = time.perf_counter()
        self.mark("start")

    def mark(self, phase: str) -> None:
        peak = peak_rss_mb()
        rss = current_rss_mb()
        released = release_heap()
        self.records.append(
            {
                "phase": phase,
                "rss_mb": round(rss, 1),
                "rss_after_trim_mb": round(current_rss_mb(), 1) if released else None,
                "peak_mb": round(peak, 1),
                "peak_growth_mb": round(peak - self._peak, 1),
                "elapsed_seconds": round(time.perf_counter() - self._started, 1),
            }
        )
        self._peak = peak

    def hottest(self) -> str:
        """Phase that grew the high-water mark the most (empty when nothing grew)."""

        best = max(self.records, key=lambda r: r["peak_growth_mb"], default=None)
        return best["phase"] if best is not None and best["peak_growth_mb"] > 0 else ""


@dataclass(slots=True)
class MemoryBudget:
    """Declared per-worker budget of a campaign and the check the tests/campaigns assert.

    ``workers x slot_peak_mb + parent_mb`` must stay under ``tree_pss_mb`` (the operator's
    process-tree limit); ``slot_peak_mb`` is the high-water mark one cluster pricing may reach.
    The declared numbers are what the memphase probe measured on the 26-member family 54 after
    the fixes (single slot: 329 MB peak, all of it the Earth-leg SCvx phase; the beam adds
    nothing once the DP's fraction cache is bounded - it was 695 MB before), with headroom.
    """

    tree_pss_mb: float = 2048.0
    workers: int = 3
    parent_mb: float = 250.0
    slot_peak_mb: float = 450.0
    notes: list[str] = field(default_factory=list)

    @property
    def projected_tree_mb(self) -> float:
        return self.workers * self.slot_peak_mb + self.parent_mb

    def fits(self) -> bool:
        return self.projected_tree_mb <= self.tree_pss_mb

    def check_slot(self, peak_mb: float, *, baseline_mb: float = 0.0) -> bool:
        """True when a measured slot peak (net of ``baseline_mb`` the process already held)
        is inside the declared slot budget."""

        return (peak_mb - baseline_mb) <= self.slot_peak_mb
