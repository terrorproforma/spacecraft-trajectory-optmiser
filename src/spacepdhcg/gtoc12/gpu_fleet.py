"""Device-controlled packing of certified GTOC12 fleet columns.

Python assembles immutable conflict/dependency topology and materializes the final
selection. Greedy seeds, feasibility, bounds and bounded branch-and-bound run on
CUDA, without an LP or CPU search fallback.
"""

import ctypes as ct
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import constants as C

COLUMN = np.dtype(
    [("value", "f8"), ("mass", "f8"), ("ships", "i4"), ("reserved", "i4"), ("identifier", "i8")],
    align=True,
)
REPORT = np.dtype(
    [
        ("objective", "f8"),
        ("upper_bound", "f8"),
        ("greedy_objective", "f8"),
        ("nodes", "u8"),
        ("exhaustive", "i4"),
        ("tasks", "i4"),
    ],
    align=True,
)
EXCHANGE_REPORT = np.dtype([("proposals", "u8"), ("moves", "i4"), ("rounds", "i4")], align=True)


def _csr(rows):
    offsets = [0]
    items = []
    for row in rows:
        items.extend(row)
        offsets.append(len(items))
    if len(items) > np.iinfo(np.int32).max:
        raise ValueError("CUDA fleet topology exceeds int32 capacity")
    return np.asarray(offsets, dtype=np.int32), np.asarray(items, dtype=np.int32)


def pack_columns(columns, weights, incumbent):
    from .cooperative import EPOCH_TOLERANCE_DAYS

    identifiers = [c.identifier for c in columns]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("CUDA fleet columns must have unique identifiers")
    data = np.zeros(len(columns), dtype=COLUMN)
    owners = {}
    deployers = {}
    for i, c in enumerate(columns):
        data[i] = (c.value(weights), c.collected_kg, c.ships, 0, c.identifier)
        for a, epoch in c.deploys.items():
            owners.setdefault(("d", a), []).append(i)
            deployers.setdefault(a, []).append((i, epoch))
        for a in c.collects:
            owners.setdefault(("c", a), []).append(i)
    conflicts = [set() for _ in columns]
    for group in owners.values():
        for i in group:
            conflicts[i].update(j for j in group if j != i)
    requirements = []
    providers = []
    for c in columns:
        groups = []
        for a, epoch in c.foreign.items():
            groups.append(len(providers))
            providers.append(
                [i for i, e in deployers.get(a, []) if abs(epoch - e) <= EPOCH_TOLERANCE_DAYS]
            )
        requirements.append(groups)
    co, ci = _csr(sorted(row) for row in conflicts)
    ro, _ = _csr(requirements)
    po, pi = _csr(providers)
    warm = np.zeros(len(columns), dtype=np.uint8)
    if incumbent:
        old_ids = {c.identifier for c in incumbent}
        if old_ids.issubset(identifiers):
            warm[:] = [c.identifier in old_ids for c in columns]
    return data, co, ci, ro, po, pi, warm


def solve_fleet_cuda(
    columns,
    *,
    weights=None,
    max_ships=C.MAX_SHIPS,
    node_cap=200_000,
    incumbent=None,
    prefix_bits=8,
    exchange_rounds=16,
):
    from .cooperative import usable_columns

    if not isinstance(max_ships, int) or not 0 <= max_ships <= C.MAX_SHIPS:
        raise ValueError("CUDA fleet max_ships must be in [0,100]")
    if not isinstance(node_cap, int) or not 0 <= node_cap <= 2**64 - 1:
        raise ValueError("CUDA fleet node_cap must fit uint64")
    if not isinstance(prefix_bits, int) or not 0 <= prefix_bits <= 10:
        raise ValueError("CUDA fleet prefix_bits must be in [0,10]")
    if not isinstance(exchange_rounds, int) or not 0 <= exchange_rounds <= 100:
        raise ValueError("CUDA fleet exchange_rounds must be in [0,100]")
    usable, rejected = usable_columns(columns, weights)
    if len(usable) > 4096:
        raise ValueError("CUDA fleet currently supports at most 4096 usable columns")
    arrays = pack_columns(usable, weights, incumbent)
    path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
    if not path:
        raise RuntimeError("CUDA fleet requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
    library = ct.CDLL(str(Path(path).resolve(strict=True)))
    symbol = (
        "spacepdhcg_gtoc12_fleet_search_v2_host"
        if exchange_rounds
        else "spacepdhcg_gtoc12_fleet_search_host"
    )
    native = getattr(library, symbol, None)
    if native is None:
        raise RuntimeError("CUDA fleet requires a rebuilt native library")
    native.argtypes = [ct.c_int32] * 3 + [ct.c_uint64] + [ct.c_void_p] * 9
    if exchange_rounds:
        native.argtypes += [ct.c_int32, ct.c_void_p]
    native.restype = ct.c_int
    selected = np.empty(len(usable), dtype=np.uint8)
    report = np.zeros(1, dtype=REPORT)
    exchanges = np.zeros(1, dtype=EXCHANGE_REPORT)
    started = time.perf_counter()
    status = native(
        len(usable),
        max_ships,
        prefix_bits,
        node_cap,
        *(a.ctypes.data for a in arrays),
        selected.ctypes.data,
        report.ctypes.data,
        *((exchange_rounds, exchanges.ctypes.data) if exchange_rounds else ()),
    )
    seconds = time.perf_counter() - started
    if status:
        raise RuntimeError(f"CUDA fleet search failed (native status {status})")
    return _materialize(usable, rejected, weights, max_ships, selected, report, exchanges, seconds)


def _materialize(usable, rejected, weights, max_ships, selected, report, exchanges, seconds):
    from .cooperative import FleetMasterResult, fleet_feasible

    chosen = tuple(c for i, c in enumerate(usable) if selected[i])
    error = fleet_feasible(chosen)
    row = report[0]
    objective = sum(c.value(weights) for c in chosen)
    if (
        error
        or sum(c.ships for c in chosen) > max_ships
        or abs(objective - row["objective"]) > 1e-8
    ):
        raise RuntimeError(f"CUDA fleet output failed independent packing checks: {error}")
    for i, c in enumerate(usable):
        if not selected[i]:
            rejected.append(
                dict(
                    identifier=c.identifier,
                    label=c.label,
                    reason="not selected by CUDA fleet search",
                    value_kg=c.value(weights),
                )
            )
    return FleetMasterResult(
        tuple(sorted(chosen, key=lambda c: c.identifier)),
        float(row["objective"]),
        float(row["upper_bound"]),
        int(row["nodes"]),
        bool(row["exhaustive"]),
        sorted(rejected, key=lambda r: r["identifier"]),
        float(row["greedy_objective"]),
        root_bound=float(row["upper_bound"]),
        backend="cuda",
        device_tasks=int(row["tasks"]),
        native_seconds=seconds,
        exchange_proposals=int(exchanges[0]["proposals"]),
        exchange_moves=int(exchanges[0]["moves"]),
        exchange_rounds=int(exchanges[0]["rounds"]),
    )


def _snapshot_column(column):
    """Copy packing data while retaining opaque route artifacts by reference."""
    return replace(
        column,
        deploys=dict(column.deploys),
        collects=dict(column.collects),
        foreign=dict(column.foreign),
        collected_mass=dict(column.collected_mass),
        members=tuple(_snapshot_column(c) for c in column.members),
    )


class CudaFleetWorkspace:
    """Retain a fixed column/weight snapshot on CUDA across repeated searches.

    Use as a context manager. Each solve can change the incumbent, ship limit,
    node budget and exchange limit. Rebuild the workspace when the pool or
    weights change. Creation includes packing/upload/ranking; native_seconds on
    each result measures the subsequent solve only. No CPU search fallback.
    """

    def __init__(self, columns, *, weights=None, prefix_bits=8):
        from .cooperative import usable_columns

        self._handle = ct.c_void_p()
        self._lock = threading.RLock()
        if not isinstance(prefix_bits, int) or not 0 <= prefix_bits <= 10:
            raise ValueError("CUDA fleet prefix_bits must be in [0,10]")
        started = time.perf_counter()
        self._weights = None if weights is None else dict(weights)
        self._columns, self._rejected = usable_columns(
            [_snapshot_column(c) for c in columns], self._weights
        )
        if len(self._columns) > 4096:
            raise ValueError("CUDA fleet currently supports at most 4096 usable columns")
        arrays = pack_columns(self._columns, self._weights, None)
        self._identifiers = {c.identifier for c in self._columns}
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA fleet requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self._library = ct.CDLL(str(Path(path).resolve(strict=True)))
        try:
            create = self._library.spacepdhcg_gtoc12_fleet_workspace_create_host
            self._solve = self._library.spacepdhcg_gtoc12_fleet_workspace_solve_host
            self._destroy = self._library.spacepdhcg_gtoc12_fleet_workspace_destroy_host
        except AttributeError as error:
            raise RuntimeError("CUDA fleet workspace requires a rebuilt native library") from error
        create.argtypes = [ct.c_int32] * 2 + [ct.c_void_p] * 7
        create.restype = ct.c_int
        self._solve.argtypes = (
            [ct.c_void_p, ct.c_int32, ct.c_uint64] + [ct.c_void_p] * 3 + [ct.c_int32, ct.c_void_p]
        )
        self._solve.restype = ct.c_int
        self._destroy.argtypes = [ct.c_void_p]
        self._destroy.restype = None
        status = create(
            len(self._columns),
            prefix_bits,
            *(a.ctypes.data for a in arrays[:-1]),
            ct.byref(self._handle),
        )
        if status:
            raise RuntimeError(f"CUDA fleet workspace creation failed (native status {status})")
        self.setup_seconds = time.perf_counter() - started

    def solve(self, *, incumbent=None, max_ships=C.MAX_SHIPS, node_cap=200_000, exchange_rounds=16):
        if not isinstance(max_ships, int) or not 0 <= max_ships <= C.MAX_SHIPS:
            raise ValueError("CUDA fleet max_ships must be in [0,100]")
        if not isinstance(node_cap, int) or not 0 <= node_cap <= 2**64 - 1:
            raise ValueError("CUDA fleet node_cap must fit uint64")
        if not isinstance(exchange_rounds, int) or not 0 <= exchange_rounds <= 100:
            raise ValueError("CUDA fleet exchange_rounds must be in [0,100]")
        with self._lock:
            if not self._handle.value:
                raise RuntimeError("CUDA fleet workspace is closed")
            ids = {c.identifier for c in incumbent} if incumbent else set()
            if not ids.issubset(self._identifiers):
                ids = set()  # Match the one-shot incumbent policy.
            warm = np.asarray([c.identifier in ids for c in self._columns], dtype=np.uint8)
            selected = np.empty(len(self._columns), dtype=np.uint8)
            report = np.zeros(1, dtype=REPORT)
            exchanges = np.zeros(1, dtype=EXCHANGE_REPORT)
            started = time.perf_counter()
            status = self._solve(
                self._handle,
                max_ships,
                node_cap,
                warm.ctypes.data,
                selected.ctypes.data,
                report.ctypes.data,
                exchange_rounds,
                exchanges.ctypes.data,
            )
            seconds = time.perf_counter() - started
            if status:
                raise RuntimeError(f"CUDA fleet workspace solve failed (native status {status})")
            result = _materialize(
                self._columns,
                [dict(r) for r in self._rejected],
                self._weights,
                max_ships,
                selected,
                report,
                exchanges,
                seconds,
            )
            # A caller may mutate dictionaries in the returned column objects.
            # Never expose those belonging to the retained input snapshot.
            result.selected = tuple(_snapshot_column(c) for c in result.selected)
            return result

    def close(self):
        with self._lock:
            if self._handle.value:
                self._destroy(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        if not self._handle.value:
            raise RuntimeError("CUDA fleet workspace is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
