"""Device-controlled packing of certified GTOC12 fleet columns.

Python serializes route records and independently checks/materializes the final
selection. Scoring, eligibility, ordering, conflict/provider topology, seeds,
feasibility, bounds and search run on CUDA, without a CPU search fallback.
"""

import ctypes as ct
import math
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
EVENT = np.dtype([("asteroid", "i8"), ("epoch", "f8")], align=True)
MASS = np.dtype([("mass", "f8"), ("weight", "f8")], align=True)


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


def _pack_routes(columns, weights):
    """Serialize records without calculating scores or column relationships."""
    if len({c.identifier for c in columns}) != len(columns):
        raise ValueError("CUDA fleet columns must have unique identifiers")
    data = np.zeros(len(columns), dtype=COLUMN)
    offsets = [[0] for _ in range(4)]
    deploy, collect, foreign, mass = [], [], [], []
    for i, column in enumerate(columns):
        data[i] = (0, 0, column.ships, int(column.certified), column.identifier)
        deploy.extend(column.deploys.items())
        collect.extend(column.collects)
        foreign.extend(column.foreign.items())
        mass.extend(
            (m, 1.0 if weights is None else weights.get(a, 1.0))
            for a, m in column.collected_mass.items()
        )
        for starts, records in zip(offsets, (deploy, collect, foreign, mass), strict=True):
            if len(records) >= np.iinfo(np.int32).max:
                raise ValueError("CUDA fleet route records exceed int32 capacity")
            starts.append(len(records))
    arrays = [data]
    for starts, records, dtype in zip(
        offsets, (deploy, collect, foreign, mass), (EVENT, np.int64, EVENT, MASS), strict=True
    ):
        arrays.extend((np.asarray(starts, dtype=np.int32), np.asarray(records, dtype=dtype)))
    return arrays


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
    if not isinstance(max_ships, int) or not 0 <= max_ships <= C.MAX_SHIPS:
        raise ValueError("CUDA fleet max_ships must be in [0,100]")
    if not isinstance(node_cap, int) or not 0 <= node_cap <= 2**64 - 1:
        raise ValueError("CUDA fleet node_cap must fit uint64")
    if not isinstance(prefix_bits, int) or not 0 <= prefix_bits <= 10:
        raise ValueError("CUDA fleet prefix_bits must be in [0,10]")
    if not isinstance(exchange_rounds, int) or not 0 <= exchange_rounds <= 100:
        raise ValueError("CUDA fleet exchange_rounds must be in [0,100]")
    workspace = CudaFleetWorkspace(columns, weights=weights, prefix_bits=prefix_bits)
    try:
        result = workspace.solve(
            incumbent=incumbent,
            max_ships=max_ships,
            node_cap=node_cap,
            exchange_rounds=exchange_rounds,
        )
    finally:
        started = time.perf_counter()
        workspace.close()
        close_seconds = time.perf_counter() - started
    result.native_seconds += workspace._native_setup_seconds + close_seconds
    return result


def _materialize(usable, rejected, weights, max_ships, selected, report, exchanges, seconds):
    from .cooperative import FleetMasterResult, fleet_feasible

    chosen = tuple(c for i, c in enumerate(usable) if selected[i])
    error = fleet_feasible(chosen)
    row = report[0]

    def value(c):
        return math.fsum(
            float(m) * (1.0 if weights is None else float(weights.get(a, 1.0)))
            for a, m in c.collected_mass.items()
        )

    # Independent accurate sums also work on Python 3.11, whose built-in sum
    # lacks compensation. This is validation/reporting, not search arithmetic.
    objective = math.fsum(value(c) for c in chosen)
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
                    value_kg=value(c),
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
        self._handle = ct.c_void_p()
        self._lock = threading.RLock()
        if not isinstance(prefix_bits, int) or not 0 <= prefix_bits <= 10:
            raise ValueError("CUDA fleet prefix_bits must be in [0,10]")
        started = time.perf_counter()
        self._weights = None if weights is None else dict(weights)
        snapshots = [_snapshot_column(c) for c in columns]
        if len(snapshots) > 4096:
            raise ValueError("CUDA fleet currently supports at most 4096 input columns")
        arrays = _pack_routes(snapshots, self._weights)
        path = os.environ.get("SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        if not path:
            raise RuntimeError("CUDA fleet requires SPACEPDHCG_GTOC12_CUDA_LIBRARY")
        self._library = ct.CDLL(str(Path(path).resolve(strict=True)))
        try:
            create = self._library.spacepdhcg_gtoc12_fleet_workspace_create_routes_host
            self._solve = self._library.spacepdhcg_gtoc12_fleet_workspace_solve_host
            self._destroy = self._library.spacepdhcg_gtoc12_fleet_workspace_destroy_host
        except AttributeError as error:
            raise RuntimeError("CUDA fleet workspace requires a rebuilt native library") from error
        create.argtypes = [ct.c_int32] * 2 + [ct.c_void_p] * 12
        create.restype = ct.c_int
        self._solve.argtypes = (
            [ct.c_void_p, ct.c_int32, ct.c_uint64] + [ct.c_void_p] * 3 + [ct.c_int32, ct.c_void_p]
        )
        self._solve.restype = ct.c_int
        self._destroy.argtypes = [ct.c_void_p]
        self._destroy.restype = None
        permutation = np.empty(len(snapshots), dtype=np.int32)
        count = ct.c_int32()
        native_started = time.perf_counter()
        status = create(
            len(snapshots),
            prefix_bits,
            *(a.ctypes.data for a in arrays),
            ct.byref(self._handle),
            permutation.ctypes.data,
            ct.byref(count),
        )
        self._native_setup_seconds = time.perf_counter() - native_started
        if status:
            raise RuntimeError(f"CUDA fleet workspace creation failed (native status {status})")
        indices = permutation[: count.value].tolist()
        self._columns = [snapshots[i] for i in indices]
        self._identifiers = {c.identifier for c in self._columns}
        kept = set(indices)
        self._rejected = [
            dict(
                identifier=c.identifier,
                label=c.label,
                reason="not certified"
                if not c.certified
                else "no certified column supplies all foreign miners",
            )
            for i, c in enumerate(snapshots)
            if i not in kept
        ]
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
