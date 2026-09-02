from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, ValidationError

from spacepdhcg.experiments import (
    iter_claim_core_groups,
    load_claim_core,
    make_execution_group,
    physical_instance_id,
    validate_paper1_result_schema,
)
from spacepdhcg.experiments.g4 import load_policy
from spacepdhcg.experiments.g4_scheduler import CampaignStore

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_SPEC = importlib.util.spec_from_file_location(
    "g4_capability",
    ROOT / "scripts/gpu/generate_g4_executor_capability.py",
)
assert CAPABILITY_SPEC is not None and CAPABILITY_SPEC.loader is not None
CAPABILITY = importlib.util.module_from_spec(CAPABILITY_SPEC)
CAPABILITY_SPEC.loader.exec_module(CAPABILITY)
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "g4_runner",
    ROOT / "scripts/gpu/run_g4_campaign.py",
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_probe_manifest_is_exact_authoritative_session() -> None:
    manifest = CAPABILITY.probe_manifest()
    assert manifest["process_contract"] == {
        "processes": 1,
        "persistent_session": True,
        "persistent_workspace": True,
        "policy_reset_between_attempts": True,
    }
    assert [
        (attempt["repeat_kind"], attempt["repeat"], attempt["statistics_eligible"])
        for attempt in manifest["attempts"]
    ] == [
        ("warmup", 0, False),
        ("warmup", 1, False),
        *[("measured", repeat, True) for repeat in range(7)],
    ]


def _fake_probe(path: Path, *, varying_workspace: bool = False) -> None:
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import os
ready = {{"case": "g4_session_ready", "pid": os.getpid()}}
print(json.dumps(ready), flush=True)
for ordinal in range(9):
    measured = ordinal >= 2
    session = {{
        "pid": os.getpid(),
        "workspace_address": "0x" + ("2" if {varying_workspace!r} and ordinal == 8 else "1"),
        "topology_fingerprint": "a" * 16,
        "cuda_context_generation": 1,
        "workspace_generation": 1,
        "topology_allocations_after_create": 0,
        "topology_index_copies_after_create": 0,
    }}
    print(json.dumps({{
        "case": "g4_attempt",
        "repeat_kind": "measured" if measured else "warmup",
        "repeat": ordinal - 2 if measured else ordinal,
        "statistics_eligible": measured,
        "session": session,
        **({{"paper1_result": {{}}}} if measured else {{}}),
    }}), flush=True)
print(json.dumps({{"case": "g4_session_complete"}}), flush=True)
"""
    )
    path.chmod(path.stat().st_mode | 0o111)


def test_capability_probe_proves_same_process_context_workspace(tmp_path: Path) -> None:
    executable = tmp_path / "executor"
    _fake_probe(executable)
    result = CAPABILITY.run_session_probe(executable, "a" * 64, "b" * 64)
    assert result == {
        "kind": "real_cuda_session",
        "attempt_count": 9,
        "warmup_count": 2,
        "measured_count": 7,
        "same_process": True,
        "same_context": True,
        "same_workspace": True,
        "zero_post_create_topology_allocations": True,
        "zero_post_create_topology_index_copies": True,
    }


def test_capability_probe_rejects_workspace_recreation(tmp_path: Path) -> None:
    executable = tmp_path / "executor"
    _fake_probe(executable, varying_workspace=True)
    with pytest.raises(SystemExit, match="workspace topology invariants"):
        CAPABILITY.run_session_probe(executable, "a" * 64, "b" * 64)


def test_crashed_group_with_partial_stdout_restarts_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "crashing-executor"
    counter = tmp_path / "counter"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
counter = Path(os.environ["G4_TEST_COUNTER"])
generation = int(counter.read_text()) + 1 if counter.exists() else 0
counter.write_text(str(generation))
print(json.dumps({"case": "g4_attempt", "repeat_kind": "warmup",
                  "repeat": 0, "generation": generation}), flush=True)
raise SystemExit(9)
"""
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    monkeypatch.setenv("G4_TEST_COUNTER", str(counter))
    manifest = CAPABILITY.probe_manifest()
    claim = SimpleNamespace(
        coordinate_id=manifest["group_id"],
        attempt_id="attempt",
        coordinate=manifest,
        ordinal=0,
    )
    (tmp_path / "runs" / manifest["group_id"] / "attempt").mkdir(parents=True)

    class Store:
        root = tmp_path
        record: dict[str, object] | None = None

        def finish(self, _claim: object, **kwargs: object) -> None:
            self.record = kwargs["record"]  # type: ignore[assignment]

    class ConstantPower:
        def watts(self) -> float:
            return 100.0

    store = Store()
    RUNNER.execute_group(
        store,
        claim,
        executable,
        None,
        ConstantPower(),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1,
        None,
    )
    assert store.record is not None
    assert store.record["restart_count"] == 1
    history = store.record["restart_history"]
    assert isinstance(history, list)
    assert [entry["generation"] for entry in history] == [0, 1]
    assert [entry["partial_attempt_count"] for entry in history] == [1, 1]
    assert (
        tmp_path / "runs" / manifest["group_id"] / "attempt" / "stdout.restart-0.jsonl"
    ).is_file()


PMON_SAMPLE = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0        484     C      -      -      -      -      -      -    python.exe
    0        485     C     37      4      -      -      -      -    python.exe
    0      11368   C+G      0      2      -      -      -      -    explorer.exe
    0      40688   C+G      0      5      -      -      -      -    Cursor.exe
"""


def test_pmon_parser_separates_active_idle_and_graphics_contexts() -> None:
    rows = RUNNER.parse_pmon(PMON_SAMPLE)
    assert [row["pid"] for row in rows] == [484, 485, 11368, 40688]
    assert rows[0]["sm_percent"] is None and rows[1]["sm_percent"] == 37
    active, idle = RUNNER.host_compute_activity(rows)
    assert [row["pid"] for row in active] == [485]
    assert [row["pid"] for row in idle] == [484]


def test_contamination_monitor_excludes_own_descendants_and_nvidia_smi(
    tmp_path: Path,
) -> None:
    monitor = RUNNER.GpuContaminationMonitor(host_nvidia_smi=None, interval_seconds=0.01)
    child = subprocess.Popen(["sleep", "30"])
    try:
        assert monitor._is_own(child.pid)
        assert monitor._is_own(os.getpid())
        assert not monitor._is_own(1)
    finally:
        child.kill()
        child.wait()


class _ForeignMonitor(RUNNER.GpuContaminationMonitor):
    """Deterministic monitor: one foreign VM holder during the group, none at boundaries."""

    def __init__(self) -> None:
        super().__init__(host_nvidia_smi=None, interval_seconds=0.01)
        self.watching = False

    def sample_nvidia_smi(self) -> dict[str, object]:
        return {"at": "t", "compute_apps": []}

    def dxg_holders(self) -> list[dict[str, object]]:
        if self.watching:
            return [{"pid": 99999, "comm": "ctest", "cmdline": "ctest --test-dir foreign"}]
        return []

    def start(self) -> None:
        self.watching = True
        super().start()

    def stop(self) -> dict[str, object]:
        summary = super().stop()
        self.watching = False
        return summary


def test_contaminated_group_is_quarantined_with_evidence_and_retryable(tmp_path: Path) -> None:
    executable = tmp_path / "executor"
    _fake_probe(executable)
    policy = load_policy(ROOT / "benchmarks/g4_policy.json")
    core = load_claim_core(ROOT / "benchmarks/g4_h5_h6_claim_core.json")
    groups = tuple(iter_claim_core_groups(core.values))
    with CampaignStore(
        tmp_path / "campaign",
        policy.values,
        policy.sha256,
        "a" * 40,
        grouped=True,
        groups=groups,
        schedule_sha256=core.sha256,
    ) as store:
        claim = store.claim()
        assert claim is not None
        monitor = _ForeignMonitor()

        class ConstantPower:
            def watts(self) -> float:
                return 100.0

        disposition = RUNNER.execute_group(
            store,
            claim,
            executable,
            None,
            ConstantPower(),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            1,
            None,
            monitor,
        )
        assert disposition == "contaminated"
        run_directory = tmp_path / "campaign" / "runs" / claim.coordinate_id / claim.attempt_id
        evidence = json.loads((run_directory / "gpu-contamination.json").read_text())
        assert evidence["foreign_detected"] is True
        assert evidence["during"]["wsl_foreign_processes"][0]["pid"] == 99999
        assert evidence["before"]["foreign"] is False and evidence["after"]["foreign"] is False
        result = json.loads((run_directory / "result.json").read_text())
        assert result["gpu_contamination"]["foreign_detected"] is True
        assert (run_directory / "stdout.jsonl").is_file()
        assert store.status()["quarantined"] == 1 and store.status()["completed"] == 0

        store.retry_quarantined(claim.ordinal)
        retry = store.claim()
        assert retry is not None
        assert retry.ordinal == claim.ordinal
        assert retry.coordinate_id == claim.coordinate_id
        assert retry.attempt_id != claim.attempt_id
        states = {
            row["attempt_id"]: (row["state"], row["disposition"])
            for row in store.database.execute("SELECT attempt_id, state, disposition FROM attempts")
        }
        assert states[claim.attempt_id] == ("quarantined", "contaminated")
        assert states[retry.attempt_id] == ("running", None)
    events = [
        json.loads(line)
        for line in (tmp_path / "campaign" / "journal.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["claimed", "quarantined", "claimed"]
    assert events[1]["disposition"] == "contaminated"


def test_clean_group_keeps_its_disposition_and_records_idle_host_context(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "executor"
    _fake_probe(executable)
    manifest = CAPABILITY.probe_manifest()
    claim = SimpleNamespace(
        coordinate_id=manifest["group_id"],
        attempt_id="attempt",
        coordinate=manifest,
        ordinal=0,
    )
    (tmp_path / "runs" / manifest["group_id"] / "attempt").mkdir(parents=True)

    class Store:
        root = tmp_path
        disposition: str | None = None
        valid: bool | None = None

        def finish(self, _claim: object, **kwargs: object) -> None:
            self.disposition = kwargs["disposition"]  # type: ignore[assignment]
            self.valid = kwargs["valid"]  # type: ignore[assignment]

    class IdleHostMonitor(RUNNER.GpuContaminationMonitor):
        def __init__(self) -> None:
            super().__init__(host_nvidia_smi=None, interval_seconds=0.01)

        def sample_nvidia_smi(self) -> dict[str, object]:
            return {"at": "t"}

        def dxg_holders(self) -> list[dict[str, object]]:
            return []

        def host_compute_contexts(
            self,
        ) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
            idle_only = PMON_SAMPLE.replace("37      4", " -      -")
            active, idle = RUNNER.host_compute_activity(RUNNER.parse_pmon(idle_only))
            return active, idle, ""

    class ConstantPower:
        def watts(self) -> float:
            return 100.0

    store = Store()
    disposition = RUNNER.execute_group(
        store,
        claim,
        executable,
        None,
        ConstantPower(),
        "a" * 64,
        "b" * 64,
        "c" * 64,
        1,
        None,
        IdleHostMonitor(),
    )
    # The fake executor emits schema-incomplete records, so the group is invalid evidence,
    # but an idle host context must never escalate that to contamination.
    assert disposition == "invalid_evidence" and store.valid is False
    evidence = json.loads(
        (
            tmp_path / "runs" / manifest["group_id"] / "attempt" / "gpu-contamination.json"
        ).read_text()
    )
    assert evidence["foreign_detected"] is False
    assert {row["pid"] for row in evidence["during"]["host_idle_compute_contexts"]} == {484, 485}


def test_raw_attempt_schema_is_closed_and_locks_topology_reuse() -> None:
    schema = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text())
    assert schema["additionalProperties"] is False
    assert schema["properties"]["session"]["properties"]["topology_allocations_after_create"] == {
        "const": 0
    }
    validator = Draft202012Validator(schema)
    minimal = {
        "schema_version": "1.0.0",
        "record_kind": "raw_attempt",
        "group_id": "g4-group-v1-" + "a" * 64,
        "attempt_id": "attempt",
        "family": "P1-C-pd3",
        "intervals": 20,
        "policy": "fixed-tight",
        "instance": "g4-instance-v2-" + "b" * 64,
        "seed": 59,
        "repeat_kind": "warmup",
        "repeat": 0,
        "launched": True,
        "statistics_eligible": False,
        "disposition": "unqualified",
        "failure_class": "max_iterations",
        "reason": "launched attempt missed quality",
        "timing": {"elapsed_seconds": 1.0},
    }
    validator.validate(minimal)
    minimal["unexpected"] = True
    with pytest.raises(ValidationError):
        validator.validate(minimal)


def test_seed_replay_and_cross_seed_physical_isolation() -> None:
    coordinate = CAPABILITY.probe_manifest()["coordinate"]
    first = make_execution_group(coordinate)
    replay = make_execution_group(dict(coordinate))
    changed = make_execution_group({**coordinate, "seed": 71})
    assert first == replay
    assert first.physical_instance_id == physical_instance_id(coordinate)
    assert first.physical_instance_id != changed.physical_instance_id
    assert first.group_id != changed.group_id


def test_claim_core_checkpoint_schedules_exactly_360_groups(tmp_path: Path) -> None:
    policy = load_policy(ROOT / "benchmarks/g4_policy.json")
    core = load_claim_core(ROOT / "benchmarks/g4_h5_h6_claim_core.json")
    groups = tuple(iter_claim_core_groups(core.values))
    with CampaignStore(
        tmp_path,
        policy.values,
        policy.sha256,
        "a" * 40,
        grouped=True,
        groups=groups,
        schedule_sha256=core.sha256,
    ) as store:
        assert store.status()["total"] == 360
        claim = store.claim()
        assert claim is not None
        assert claim.coordinate_id == groups[0].group_id
        assert len(claim.coordinate["attempts"]) == 9


def test_native_reset_api_declares_all_retention_boundaries() -> None:
    header = (ROOT / "cpp/cuda/include/spacepdhcg/cuda/device_scvx_driver_c_api.h").read_text()
    implementation = (ROOT / "cpp/cuda/src/device_scvx.cu").read_text()
    assert "spacepdhcg_cuda_scvx_driver_reset_attempt" in header
    assert "SPACEPDHCG_CUDA_WARM_START_NONE" in implementation
    assert "SPACEPDHCG_CUDA_WARM_START_PRIMAL" in implementation
    warm_header = (ROOT / "cpp/cuda/include/spacepdhcg/cuda/persistent_pdhcg_c_api.h").read_text()
    assert "SPACEPDHCG_CUDA_WARM_START_PRIMAL_DUAL" in warm_header
    assert "SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED" in warm_header
    assert "SPACEPDHCG_CUDA_WARM_START_FULL_RETAINED" in implementation
    assert "spacepdhcg_native_qoco_reset_warm_state" in implementation


@pytest.mark.skipif(
    "SPACEPDHCG_G4_SESSION_EXECUTABLE" not in os.environ,
    reason="native CUDA session executable not supplied",
)
def test_real_native_session_is_schema_valid_and_persistent(tmp_path: Path) -> None:
    executable = Path(os.environ["SPACEPDHCG_G4_SESSION_EXECUTABLE"])
    manifest = CAPABILITY.probe_manifest()
    manifest_path = tmp_path / "execution-group.json"
    manifest_path.write_bytes(CAPABILITY.canonical_bytes(manifest))
    environment = dict(os.environ)
    environment.update(
        {
            "SPACEPDHCG_G4_GROUP_ID": manifest["group_id"],
            "SPACEPDHCG_G4_CAPABILITY_PROBE": "1",
            "SPACEPDHCG_G4_OUTER_ITERATIONS": "1",
            "SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS": "10",
            "SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS": "120",
        }
    )
    completed = subprocess.run(
        [
            executable,
            "--g4-session",
            manifest_path,
            "9ab3b444e3dd21fdd2a75c3cebfe8fd8374f9e5ff672cd757afaeb6036530024",
            "b" * 64,
            "c" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=130,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    attempts = [record for record in records if record.get("case") == "g4_attempt"]
    assert len(attempts) == 9
    raw_schema = json.loads((ROOT / "experiments/schema/g4_raw_attempt.schema.json").read_text())
    validator = Draft202012Validator(raw_schema)
    for attempt in attempts:
        validator.validate(attempt)
        if attempt["repeat_kind"] == "measured":
            validate_paper1_result_schema(
                attempt["paper1_result"],
                ROOT / "experiments/schema/paper1_result.schema.json",
            )
    assert len({attempt["session"]["pid"] for attempt in attempts}) == 1
    assert len({attempt["session"]["workspace_address"] for attempt in attempts}) == 1
    assert all(
        attempt["session"]["topology_allocations_after_create"] == 0
        and attempt["session"]["topology_index_copies_after_create"] == 0
        for attempt in attempts
    )
    old = subprocess.run(
        [
            executable,
            "--g4-sample",
            "P1-C-pd3",
            "20",
            "pure-gpu-ipm",
            "primal",
            "1e-6",
            "1",
            "0.05",
            "not_applicable",
            "tight",
            "refresh_if_needed",
            "9ab3b444e3dd21fdd2a75c3cebfe8fd8374f9e5ff672cd757afaeb6036530024",
            "0",
            "59",
            "measured",
            "0",
            "0",
            "d" * 64,
            "b" * 64,
            "c" * 64,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    assert old.returncode == 0, old.stderr
    old_records = [json.loads(line) for line in old.stdout.splitlines() if line.startswith("{")]
    old_axis = next(record for record in old_records if record.get("case") == "g4_axis_application")
    measured_zero = next(
        attempt
        for attempt in attempts
        if attempt["repeat_kind"] == "measured" and attempt["repeat"] == 0
    )
    assert old_axis["coefficient_hash"] == measured_zero["source_hashes"]["coefficient_hash"]
    assert old_axis["problem_hash"] == measured_zero["source_hashes"]["problem_hash"]
    assert old_axis["instance_hash"] == measured_zero["source_hashes"]["instance_hash"]
