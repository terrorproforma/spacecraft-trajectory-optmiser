from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacepdhcg.experiments.g4 import (
    POLICY_NAMES,
    G4ContractError,
    coverage_count,
    load_policy,
)
from spacepdhcg.experiments.g4_scheduler import (
    INVALID_EXECUTOR_DEFECT,
    INVALIDATED_STATE,
    CampaignStore,
    atomic_create,
    coordinate_at,
    coordinate_id,
    scheduled_ordinal_at,
)

ROOT = Path(__file__).resolve().parents[1]


def policy():
    digest = (ROOT / "benchmarks/g4_policy.sha256").read_text().split()[0]
    return load_policy(ROOT / "benchmarks/g4_policy.json", expected_sha256=digest)


def test_coordinate_unranking_covers_frozen_ledger_and_rotation() -> None:
    loaded = policy()
    total = coverage_count(loaded.values)
    assert total == 24_883_200
    first = coordinate_at(loaded.values, 0)
    last = coordinate_at(loaded.values, total - 1)
    assert first["family"] == "P1-C-pd3"
    assert first["dispersion_class"] == 0.0
    assert first["repeat_kind"] == "warmup"
    assert last["family"] == "P1-E-low-thrust"
    assert last["transfer_class"] == "combined"
    assert last["repeat_kind"] == "measured"
    assert last["repeat"] == 6

    scheduled = [
        coordinate_at(loaded.values, scheduled_ordinal_at(loaded.values, index))
        for index in range(len(POLICY_NAMES))
    ]
    assert {row["policy"] for row in scheduled} == set(POLICY_NAMES)
    assert [row["solver_order"] for row in scheduled] == list(range(len(POLICY_NAMES)))
    assert len({coordinate_id(row) for row in scheduled}) == len(POLICY_NAMES)


def test_store_recovers_interrupted_attempt_without_overwrite(tmp_path: Path) -> None:
    loaded = policy()
    store = CampaignStore(tmp_path, loaded.values, loaded.sha256, "a" * 40)
    first = store.claim()
    assert first is not None
    first_directory = tmp_path / "runs" / first.coordinate_id / first.attempt_id
    store.close()

    with CampaignStore(tmp_path, loaded.values, loaded.sha256, "a" * 40) as recovered:
        second = recovered.claim()
        assert second is not None
        assert second.ordinal == first.ordinal
        assert second.coordinate_id == first.coordinate_id
        assert second.attempt_id != first.attempt_id
        assert first_directory.is_dir()
        recovered.finish(
            second,
            disposition="timeout",
            reason="actual launched-process timeout",
            record={"coordinate_id": second.coordinate_id, "disposition": "timeout"},
            valid=True,
        )
        assert recovered.status() == {
            "total": 24_883_200,
            "completed": 1,
            "running": 0,
            "quarantined": 0,
            "invalidated": 0,
            "remaining": 24_883_199,
        }
        attempts = recovered.database.execute(
            "SELECT state FROM attempts ORDER BY started_at, attempt_id"
        ).fetchall()
        assert sorted(row["state"] for row in attempts) == ["completed", "interrupted"]

    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["claimed", "claimed", "completed"]


def test_invalid_record_is_quarantined_and_atomic_create_refuses_reuse(
    tmp_path: Path,
) -> None:
    loaded = policy()
    with CampaignStore(tmp_path, loaded.values, loaded.sha256, "b" * 40) as store:
        claim = store.claim()
        assert claim is not None
        store.finish(
            claim,
            disposition="unqualified",
            reason="schema validation failed",
            record={"invalid": True},
            valid=False,
        )
        status = store.status()
        assert status["completed"] == 0
        assert status["quarantined"] == 1
        assert status["remaining"] == status["total"]

    target = tmp_path / "immutable.txt"
    atomic_create(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_create(target, b"second")
    assert target.read_bytes() == b"first"
    with pytest.raises(G4ContractError, match="source_commit"):
        CampaignStore(tmp_path, loaded.values, loaded.sha256, "c" * 40)


def test_imported_terminal_row_is_exactly_once_and_skipped(tmp_path: Path) -> None:
    loaded = policy()
    source_root = tmp_path / "source"
    with CampaignStore(source_root, loaded.values, loaded.sha256, "a" * 40) as source:
        claim = source.claim()
        assert claim is not None
        source.finish(
            claim,
            disposition="qualified",
            reason="source result",
            record={"coordinate_id": claim.coordinate_id, "disposition": "qualified"},
            valid=True,
        )
    source_run = source_root / "runs" / claim.coordinate_id / claim.attempt_id

    with CampaignStore(tmp_path / "target", loaded.values, loaded.sha256, "b" * 40) as target:
        arguments = {
            "ordinal": claim.ordinal,
            "identifier": claim.coordinate_id,
            "state": "completed",
            "disposition": "qualified",
            "reason": "source result",
            "coordinate_payload": (source_run / "coordinate.json").read_bytes(),
            "result_payload": (source_run / "result.json").read_bytes(),
            "source_campaign": str(source_root),
            "source_attempt_id": claim.attempt_id,
        }
        assert target.import_terminal(**arguments)
        assert not target.import_terminal(**arguments)
        next_claim = target.claim()
        assert next_claim is not None
        assert next_claim.ordinal != claim.ordinal
        assert target.status()["completed"] == 1


def test_invalidation_retains_evidence_and_leaves_completed_set(tmp_path: Path) -> None:
    """Executor-defect hygiene: records stay on disk, the ledger row leaves ``completed``."""

    loaded = policy()
    with CampaignStore(tmp_path, loaded.values, loaded.sha256, "a" * 40) as store:
        claim = store.claim()
        assert claim is not None
        record = {"coordinate_id": claim.coordinate_id, "disposition": "numerical"}
        store.finish(claim, disposition="completed_group", reason="ok", record=record, valid=True)
        run_directory = tmp_path / "runs" / claim.coordinate_id / claim.attempt_id
        before = (run_directory / "result.json").read_bytes()

        # Only completed rows can be invalidated, and a reason is mandatory.
        with pytest.raises(G4ContractError, match="requires a reason"):
            store.invalidate(claim.ordinal, reason="", provenance={})
        with pytest.raises(G4ContractError, match="only completed"):
            store.invalidate(claim.ordinal + 1, reason="defect", provenance={})

        written = store.invalidate(
            claim.ordinal,
            reason="pure-gpu-ipm warm boundary defect",
            provenance={"fix_commit": "f" * 40, "superseded_by": "/tmp/next"},
        )
        assert written["disposition"] == INVALID_EXECUTOR_DEFECT
        assert written["prior_disposition"] == "completed_group"
        # Nothing was deleted or rewritten; the invalidation sits beside the result.
        assert (run_directory / "result.json").read_bytes() == before
        invalidation = json.loads((run_directory / "invalidation.json").read_text())
        assert invalidation["fix_commit"] == "f" * 40
        assert invalidation["attempt_id"] == claim.attempt_id
        status = store.status()
        assert status["completed"] == 0
        assert status["invalidated"] == 1
        assert status["remaining"] == status["total"]
        row = store.database.execute(
            "SELECT state, disposition FROM attempts WHERE attempt_id = ?", (claim.attempt_id,)
        ).fetchone()
        assert (row["state"], row["disposition"]) == (INVALIDATED_STATE, INVALID_EXECUTOR_DEFECT)
        # A second invalidation of the same evidence is refused (create-only sidecar, state).
        with pytest.raises(G4ContractError, match="only completed"):
            store.invalidate(claim.ordinal, reason="again", provenance={})
        # The invalidated row is never claimed again by this checkpoint: a fresh campaign owns it.
        following = store.claim()
        assert following is not None and following.ordinal != claim.ordinal

    events = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "claimed",
        "completed",
        INVALIDATED_STATE,
        "claimed",
    ]
    assert events[2]["reason"] == "pure-gpu-ipm warm boundary defect"
