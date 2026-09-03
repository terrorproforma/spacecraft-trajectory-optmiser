"""Crash-safe, append-only scheduling for the frozen G4 Cartesian campaign."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Final

from .g4 import (
    POLICY_NAMES,
    QUALITY_TIERS,
    SCALING_MODES,
    WARM_MODES,
    G4ContractError,
    coverage_count,
)
from .g4_execution_contract import (
    ExecutionGroup,
    make_execution_group,
    physical_instance_id,
    solver_rotation,
)

SCHEMA_VERSION: Final = 1
TERMINAL_STATES: Final = ("completed", "quarantined")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def coordinate_id(coordinate: Mapping[str, Any]) -> str:
    """Return the content address of one complete frozen coordinate."""

    return hashlib.sha256(_canonical_json(coordinate)).hexdigest()


def _family_coordinates(policy: Mapping[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    result: list[tuple[str, int, dict[str, Any]]] = []
    for family, values in policy["matrix"]["families"].items():
        if family == "P1-C-pd3":
            classes = (
                {"dispersion_class": dispersion} for dispersion in values["dispersion_classes"]
            )
        elif family == "P1-D-pd6":
            classes = (
                {"attitude_class": attitude, "rate_class": rate}
                for attitude, rate in product(
                    values["attitude_dispersion_radians"],
                    values["angular_rate_dispersion"],
                )
            )
        elif family == "P1-E-low-thrust":
            classes = (
                {"trust_class": trust, "transfer_class": transfer}
                for trust, transfer in product(
                    values["trust_radii"],
                    values["transfer_classes"],
                )
            )
        else:
            raise G4ContractError(f"unsupported G4 family {family!r}")
        frozen_classes = tuple(classes)
        for class_values in frozen_classes:
            for intervals in values["intervals"]:
                result.append((family, int(intervals), class_values))
    return result


def coordinate_at(policy: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    """Unrank one row without materialising the 24,883,200-row ledger."""

    total = coverage_count(policy)
    if ordinal < 0 or ordinal >= total:
        raise IndexError(f"G4 coordinate ordinal {ordinal} outside [0, {total})")
    matrix = policy["matrix"]
    seeds = tuple(policy["tuning_evaluation_split"]["evaluation_seeds"])[
        : matrix["randomised_instances_per_coordinate"]
    ]
    repeats = (
        *(("warmup", index) for index in range(matrix["warmup_repeats"])),
        *(("measured", index) for index in range(matrix["measured_repeats"])),
    )
    dimensions: tuple[Sequence[Any], ...] = (
        POLICY_NAMES,
        QUALITY_TIERS,
        tuple(matrix["conditioning_log10_spans"]),
        SCALING_MODES,
        WARM_MODES,
        seeds,
        repeats,
    )
    inner_count = 1
    for dimension in dimensions:
        inner_count *= len(dimension)
    family_coordinates = _family_coordinates(policy)
    family, intervals, classes = family_coordinates[ordinal // inner_count]
    remainder = ordinal % inner_count
    selected: list[Any] = []
    for dimension in reversed(dimensions):
        remainder, index = divmod(remainder, len(dimension))
        selected.append(dimension[index])
    policy_name, quality, conditioning, scaling, warm, seed, repeat = reversed(selected)
    repeat_kind, repeat_index = repeat
    base = {
        "schema_version": SCHEMA_VERSION,
        "ordinal": ordinal,
        "family": family,
        "intervals": intervals,
        "policy": policy_name,
        "quality_tier": quality,
        "quality_tolerance": float(policy["quality_tiers"][quality]),
        "conditioning": conditioning,
        "scaling_mode": scaling,
        "warm_mode": warm,
        **classes,
        "seed": seed,
        "repeat_kind": repeat_kind,
        "repeat": repeat_index,
    }
    base["instance"] = physical_instance_id(base)
    rotation = solver_rotation(int(policy["randomisation"]["solver_order_seed"]), base)
    base["solver_order"] = (POLICY_NAMES.index(policy_name) + rotation) % len(POLICY_NAMES)
    return base


def execution_group_count(policy: Mapping[str, Any]) -> int:
    """Return persistent tasks without changing the frozen logical row count."""

    attempts = policy["matrix"]["warmup_repeats"] + policy["matrix"]["measured_repeats"]
    if attempts != 9:
        raise G4ContractError("G4 execution groups require exactly 2 warmups and 7 measurements")
    return coverage_count(policy) // attempts


def execution_group_at(policy: Mapping[str, Any], ordinal: int) -> ExecutionGroup:
    """Unrank one same-process, same-workspace nine-attempt task."""

    total = execution_group_count(policy)
    if ordinal < 0 or ordinal >= total:
        raise IndexError(f"G4 execution-group ordinal {ordinal} outside [0, {total})")
    return make_execution_group(coordinate_at(policy, ordinal * 9))


def scheduled_group_ordinal_at(policy: Mapping[str, Any], schedule_index: int) -> int:
    """Rotate policies while retaining all nine attempts in one task."""

    total = execution_group_count(policy)
    if schedule_index < 0 or schedule_index >= total:
        raise IndexError(f"G4 group schedule index {schedule_index} outside [0, {total})")
    policies = len(POLICY_NAMES)
    nonpolicy_count = total // (len(_family_coordinates(policy)) * policies)
    family_index, within_family = divmod(schedule_index, nonpolicy_count * policies)
    nonpolicy_index, solver_slot = divmod(within_family, policies)
    provisional = family_index * nonpolicy_count * policies + nonpolicy_index
    rotation = execution_group_at(policy, provisional).coordinate["solver_order"]
    policy_index = (solver_slot - rotation) % policies
    return (
        family_index * nonpolicy_count * policies + policy_index * nonpolicy_count + nonpolicy_index
    )


def scheduled_ordinal_at(policy: Mapping[str, Any], schedule_index: int) -> int:
    """Map execution order to ledger order using the frozen solver rotation."""

    total = coverage_count(policy)
    if schedule_index < 0 or schedule_index >= total:
        raise IndexError(f"G4 schedule index {schedule_index} outside [0, {total})")
    policies = len(POLICY_NAMES)
    nonpolicy_count = total // (len(_family_coordinates(policy)) * policies)
    family_index, within_family = divmod(schedule_index, nonpolicy_count * policies)
    nonpolicy_index, solver_slot = divmod(within_family, policies)
    provisional = family_index * nonpolicy_count * policies + nonpolicy_index
    rotation = coordinate_at(policy, provisional)["solver_order"]
    policy_index = (solver_slot - rotation) % policies
    return (
        family_index * nonpolicy_count * policies + policy_index * nonpolicy_count + nonpolicy_index
    )


def atomic_create(path: Path, payload: bytes) -> None:
    """Create and fsync a file, refusing every overwrite."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class Claim:
    ordinal: int
    coordinate_id: str
    attempt_id: str
    coordinate: dict[str, Any]


class CampaignStore:
    """Sparse SQLite checkpoint plus an fsynced append-only audit journal."""

    def __init__(
        self,
        root: Path,
        policy: Mapping[str, Any],
        policy_sha256: str,
        source_commit: str,
        *,
        grouped: bool = False,
        groups: Sequence[ExecutionGroup] | None = None,
        schedule_sha256: str | None = None,
        extra_metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.root = root
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self.source_commit = source_commit
        self.extra_metadata = dict(extra_metadata or {})
        if "next_ordinal" in self.extra_metadata:
            raise G4ContractError("next_ordinal is scheduler-owned metadata")
        if groups is not None and not grouped:
            raise G4ContractError("explicit G4 groups require grouped scheduling")
        self.grouped = grouped
        self.groups = tuple(groups) if groups is not None else None
        self.schedule_sha256 = schedule_sha256 or policy_sha256
        self.total = (
            len(self.groups)
            if self.groups is not None
            else execution_group_count(policy)
            if grouped
            else coverage_count(policy)
        )
        root.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(root / "checkpoint.sqlite3", timeout=30.0)
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("PRAGMA synchronous=FULL")
        try:
            self._create_schema()
        except Exception:
            self.database.close()
            raise

    def _create_schema(self) -> None:
        self.database.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS coordinates (
                ordinal INTEGER PRIMARY KEY,
                coordinate_id TEXT NOT NULL UNIQUE,
                state TEXT NOT NULL,
                latest_attempt_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL,
                coordinate_id TEXT NOT NULL,
                state TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                run_directory TEXT NOT NULL UNIQUE,
                disposition TEXT,
                reason TEXT,
                FOREIGN KEY(ordinal) REFERENCES coordinates(ordinal)
            );
            """
        )
        expected = {
            "schema_version": str(SCHEMA_VERSION),
            "schedule_kind": (
                "claim_core_execution_groups"
                if self.groups is not None
                else "execution_groups"
                if self.grouped
                else "logical_rows"
            ),
            "policy_sha256": self.policy_sha256,
            "schedule_sha256": self.schedule_sha256,
            "source_commit": self.source_commit,
            "total_rows": str(self.total),
            **self.extra_metadata,
            "next_ordinal": "0",
        }
        with self.database:
            for key, value in expected.items():
                row = self.database.execute(
                    "SELECT value FROM metadata WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self.database.execute(
                        "INSERT INTO metadata(key, value) VALUES (?, ?)",
                        (key, value),
                    )
                elif key != "next_ordinal" and row["value"] != value:
                    raise G4ContractError(f"campaign metadata mismatch for {key}")

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _journal(self, event: Mapping[str, Any]) -> None:
        payload = _canonical_json(event) + b"\n"
        descriptor = os.open(
            self.root / "journal.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def claim(self) -> Claim | None:
        """Claim the next row or resume the oldest interrupted row."""

        now = self._now()
        with self.database:
            running = self.database.execute(
                """
                SELECT ordinal, coordinate_id
                FROM coordinates
                WHERE state = 'running'
                ORDER BY ordinal
                LIMIT 1
                """
            ).fetchone()
            if running is None:
                next_row = self.database.execute(
                    "SELECT value FROM metadata WHERE key = 'next_ordinal'"
                ).fetchone()
                schedule_index = int(next_row["value"])
                while True:
                    if schedule_index >= self.total:
                        return None
                    ordinal = (
                        schedule_index
                        if self.groups is not None
                        else scheduled_group_ordinal_at(self.policy, schedule_index)
                        if self.grouped
                        else scheduled_ordinal_at(self.policy, schedule_index)
                    )
                    schedule_index += 1
                    self.database.execute(
                        "UPDATE metadata SET value = ? WHERE key = 'next_ordinal'",
                        (str(schedule_index),),
                    )
                    if (
                        self.database.execute(
                            "SELECT 1 FROM coordinates WHERE ordinal = ?",
                            (ordinal,),
                        ).fetchone()
                        is None
                    ):
                        break
                coordinate, identifier = self._coordinate_and_id(ordinal)
            else:
                ordinal = int(running["ordinal"])
                coordinate, expected_identifier = self._coordinate_and_id(ordinal)
                identifier = str(running["coordinate_id"])
                if expected_identifier != identifier:
                    raise G4ContractError("checkpoint coordinate content address drift")
                self.database.execute(
                    """
                    UPDATE attempts
                    SET state = 'interrupted', finished_at = ?,
                        disposition = 'error',
                        reason = 'worker exited before committing a terminal record'
                    WHERE attempt_id = (
                        SELECT latest_attempt_id
                        FROM coordinates
                        WHERE ordinal = ?
                    ) AND state = 'running'
                    """,
                    (now, ordinal),
                )
            attempt_id = uuid.uuid4().hex
            run_directory = self.root / "runs" / identifier / attempt_id
            run_directory.mkdir(parents=True, exist_ok=False)
            atomic_create(
                run_directory / "coordinate.json",
                _canonical_json(coordinate) + b"\n",
            )
            if running is None:
                self.database.execute(
                    """
                    INSERT INTO coordinates(
                        ordinal, coordinate_id, state, latest_attempt_id, updated_at
                    ) VALUES (?, ?, 'running', ?, ?)
                    """,
                    (ordinal, identifier, attempt_id, now),
                )
            else:
                self.database.execute(
                    """
                    UPDATE coordinates
                    SET latest_attempt_id = ?, updated_at = ?
                    WHERE ordinal = ?
                    """,
                    (attempt_id, now, ordinal),
                )
            self.database.execute(
                """
                INSERT INTO attempts(
                    attempt_id, ordinal, coordinate_id, state, started_at, run_directory
                ) VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (attempt_id, ordinal, identifier, now, str(run_directory)),
            )
        claim = Claim(ordinal, identifier, attempt_id, coordinate)
        self._journal(
            {
                "event": "claimed",
                "at": now,
                "ordinal": ordinal,
                "coordinate_id": identifier,
                "attempt_id": attempt_id,
            }
        )
        return claim

    def _coordinate_and_id(self, ordinal: int) -> tuple[dict[str, Any], str]:
        if not self.grouped:
            coordinate = coordinate_at(self.policy, ordinal)
            return coordinate, coordinate_id(coordinate)
        group = (
            self.groups[ordinal]
            if self.groups is not None
            else execution_group_at(self.policy, ordinal)
        )
        coordinate = {
            "schema_version": "1.0.0",
            "record_kind": "execution_group",
            "group_id": group.group_id,
            "physical_instance_id": group.physical_instance_id,
            "coordinate": group.coordinate,
            "process_contract": {
                "processes": 1,
                "persistent_session": True,
                "persistent_workspace": True,
                "policy_reset_between_attempts": True,
            },
            "attempts": list(group.attempts),
        }
        return coordinate, group.group_id

    def import_terminal(
        self,
        *,
        ordinal: int,
        identifier: str,
        state: str,
        disposition: str,
        reason: str,
        coordinate_payload: bytes,
        result_payload: bytes,
        source_campaign: str,
        source_attempt_id: str,
    ) -> bool:
        """Import one immutable terminal attempt without changing row identity."""

        if state not in TERMINAL_STATES:
            raise G4ContractError("only terminal campaign rows may be imported")
        coordinate, expected_identifier = self._coordinate_and_id(ordinal)
        if expected_identifier != identifier:
            raise G4ContractError("imported coordinate differs from frozen ledger")
        if _canonical_json(coordinate) + b"\n" != coordinate_payload:
            raise G4ContractError("imported coordinate payload is not canonical")
        now = self._now()
        attempt_id = f"import-{uuid.uuid4().hex}"
        run_directory = self.root / "runs" / identifier / attempt_id
        with self.database:
            existing = self.database.execute(
                "SELECT state FROM coordinates WHERE ordinal = ?", (ordinal,)
            ).fetchone()
            if existing is not None:
                return False
            run_directory.mkdir(parents=True, exist_ok=False)
            atomic_create(run_directory / "coordinate.json", coordinate_payload)
            atomic_create(run_directory / "result.json", result_payload)
            atomic_create(
                run_directory / "migration.json",
                _canonical_json(
                    {
                        "source_campaign": source_campaign,
                        "source_attempt_id": source_attempt_id,
                        "imported_at": now,
                    }
                )
                + b"\n",
            )
            self.database.execute(
                """
                INSERT INTO coordinates(
                    ordinal, coordinate_id, state, latest_attempt_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ordinal, identifier, state, attempt_id, now),
            )
            self.database.execute(
                """
                INSERT INTO attempts(
                    attempt_id, ordinal, coordinate_id, state, started_at,
                    finished_at, run_directory, disposition, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    ordinal,
                    identifier,
                    state,
                    now,
                    now,
                    str(run_directory),
                    disposition,
                    reason,
                ),
            )
        self._journal(
            {
                "event": "imported",
                "at": now,
                "ordinal": ordinal,
                "coordinate_id": identifier,
                "attempt_id": attempt_id,
                "source_campaign": source_campaign,
                "source_attempt_id": source_attempt_id,
                "state": state,
                "disposition": disposition,
            }
        )
        return True

    def finish(
        self,
        claim: Claim,
        *,
        disposition: str,
        reason: str,
        record: Mapping[str, Any],
        valid: bool,
    ) -> None:
        state = "completed" if valid else "quarantined"
        run_directory = self.root / "runs" / claim.coordinate_id / claim.attempt_id
        atomic_create(run_directory / "result.json", _canonical_json(record) + b"\n")
        now = self._now()
        with self.database:
            row = self.database.execute(
                "SELECT state, latest_attempt_id FROM coordinates WHERE ordinal = ?",
                (claim.ordinal,),
            ).fetchone()
            if (
                row is None
                or row["state"] != "running"
                or row["latest_attempt_id"] != claim.attempt_id
            ):
                raise G4ContractError("attempt no longer owns its coordinate lease")
            self.database.execute(
                """
                UPDATE attempts
                SET state = ?, finished_at = ?, disposition = ?, reason = ?
                WHERE attempt_id = ?
                """,
                (state, now, disposition, reason, claim.attempt_id),
            )
            self.database.execute(
                """
                UPDATE coordinates
                SET state = ?, updated_at = ?
                WHERE ordinal = ?
                """,
                (state, now, claim.ordinal),
            )
        self._journal(
            {
                "event": state,
                "at": now,
                "ordinal": claim.ordinal,
                "coordinate_id": claim.coordinate_id,
                "attempt_id": claim.attempt_id,
                "disposition": disposition,
                "reason": reason,
            }
        )

    def retry_quarantined(self, ordinal: int) -> None:
        with self.database:
            updated = self.database.execute(
                """
                UPDATE coordinates
                SET state = 'running', updated_at = ?
                WHERE ordinal = ? AND state = 'quarantined'
                """,
                (self._now(), ordinal),
            )
            if updated.rowcount != 1:
                raise G4ContractError("only quarantined coordinates may be retried")

    def status(self) -> dict[str, int]:
        counts = {
            row["state"]: int(row["count"])
            for row in self.database.execute(
                "SELECT state, COUNT(*) AS count FROM coordinates GROUP BY state"
            )
        }
        completed = counts.get("completed", 0)
        return {
            "total": self.total,
            "completed": completed,
            "running": counts.get("running", 0),
            "quarantined": counts.get("quarantined", 0),
            "remaining": self.total - completed,
        }

    def close(self) -> None:
        self.database.close()

    def __enter__(self) -> CampaignStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
