from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from spacepdhcg.experiments.g5_campaign import (
    FAILURE_MODES,
    CampaignCoordinate,
    PreflightError,
    assert_physical_execution_permitted,
    build_partial_evidence,
    expand_cpu_list,
    generate_coordinates,
    logical_topology,
    make_command_manifest,
    make_monolithic_reference_manifest,
    parse_nvidia_gpu_csv,
    parse_topology_matrix,
    rank_bindings,
    summarize_gpu_samples,
    validate_command_manifest,
    validate_preflight_record,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4cc2c23840fe73a5b56e02bb58313911b9cf57b1"
UPSTREAM_COMMIT = "167c8b72b4b96d2f94d405b8763e485514192b81"
UPSTREAM_TREE = "62b05e6c1bedd385f6c267af3645ae4aae0421b4"


def _device(index: int) -> dict[str, object]:
    first_cpu = index * 8
    return {
        "index": index,
        "uuid": f"GPU-{index:04d}",
        "model": "NVIDIA H100 80GB HBM3",
        "driver_version": "595.97",
        "memory_total_mib": 81920,
        "memory_free_mib": 81000,
        "pci_bus_id": f"00000000:{index + 1:02X}:00.0",
        "pcie_generation": 5,
        "pcie_width": 16,
        "persistence_mode": "Enabled",
        "ecc_mode": "Enabled",
        "mig_mode": "Disabled",
        "compute_mode": "Exclusive_Process",
        "power_limit_w": "700.00",
        "power_draw_w": "65.00",
        "sm_clock_mhz": 210,
        "memory_clock_mhz": 1593,
        "cpu_affinity": f"{first_cpu}-{first_cpu + 7}",
        "numa_node": index,
    }


def _physical_record(gpu_count: int) -> dict[str, object]:
    devices = [_device(index) for index in range(gpu_count)]
    return {
        "schema_version": "1.0.0",
        "record_type": "g5-physical-preflight",
        "status": "passed",
        "fingerprint": "a" * 64,
        "hostname": "gpu-node",
        "repository": {"commit": COMMIT, "branch": "feat/test", "dirty": False},
        "upstream": {"commit": UPSTREAM_COMMIT, "tree": UPSTREAM_TREE},
        "toolchain": {
            "driver": "595.97",
            "cuda": "Cuda compilation tools, release 12.8",
            "nccl": "2.26.2-1+cuda12.8",
            "mpi": "mpirun (Open MPI) 4.1.2",
        },
        "build": {
            "cmake_cache_sha256": "b" * 64,
            "harness_sha256": "c" * 64,
            "cmake_flags": {"CMAKE_BUILD_TYPE": "Release"},
        },
        "gpus": devices,
        "topology": {
            "rows": [
                {
                    "gpu": f"GPU{index}",
                    "links": ["X" if index == peer else "NV18" for peer in range(gpu_count)],
                    "cpu_affinity": device["cpu_affinity"],
                    "numa_node": index,
                }
                for index, device in enumerate(devices)
            ]
        },
        "network": {"interfaces": [{"name": "ib0"}]},
        "active_compute_processes": [],
    }


def _coordinate(gpu_count: int) -> CampaignCoordinate:
    return CampaignCoordinate(
        scaling="strong",
        gpu_count=gpu_count,
        scenarios=max(8, gpu_count),
        nodes=100,
        partition="scenario_aware",
        risk="cvar_0.9",
        seed=17,
    )


def _manifest(gpu_count: int, *, failure_mode: str | None = None) -> dict[str, object]:
    return make_command_manifest(
        _coordinate(gpu_count),
        topology_record=logical_topology(gpu_count),
        repository_commit=COMMIT,
        executable="/opt/spacepdhcg/bin/g5_physical_validation_harness",
        output_root="/evidence/g5",
        warmups=0 if failure_mode else 2,
        repeats=1 if failure_mode else 7,
        timeout_seconds=60,
        failure_mode=failure_mode,
    )


def test_nvidia_csv_and_topology_parsing() -> None:
    csv_payload = (
        "0, GPU-a, NVIDIA H100 80GB HBM3, 595.97, 81920, 81000, "
        "00000000:01:00.0, 5, 16, Enabled, Enabled, Disabled, "
        "Exclusive_Process, 700.00, 65.00, 210, 1593\n"
        "1, GPU-b, NVIDIA H100 80GB HBM3, 595.97, 81920, 80900, "
        "00000000:02:00.0, 5, 16, Enabled, Enabled, Disabled, "
        "Exclusive_Process, 700.00, 64.00, 210, 1593\n"
    )
    devices = parse_nvidia_gpu_csv(csv_payload)
    topology = parse_topology_matrix(
        """
                GPU0    GPU1    CPU Affinity    NUMA Affinity    GPU NUMA ID
        GPU0     X      NV18    0-7             0                N/A
        GPU1     NV18   X       8-15            1                N/A

        Legend:
          X = Self
          NV# = Bonded set of # NVLinks
        """,
        2,
    )
    assert [item["uuid"] for item in devices] == ["GPU-a", "GPU-b"]
    assert topology["rows"][1]["links"] == ["NV18", "X"]
    assert topology["rows"][1]["cpu_affinity"] == "8-15"
    assert expand_cpu_list("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]


@pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
def test_preflight_and_rank_bindings_for_supported_counts(gpu_count: int) -> None:
    record = _physical_record(gpu_count)
    assert not validate_preflight_record(
        record,
        expected_gpu_count=gpu_count,
        primary=True,
    )
    bindings = rank_bindings(record, gpu_count)
    assert [item["rank"] for item in bindings] == list(range(gpu_count))
    assert len({item["gpu_uuid"] for item in bindings}) == gpu_count
    assert len({item["cpu_set"] for item in bindings}) == gpu_count


def test_primary_preflight_fails_closed() -> None:
    record = _physical_record(4)
    record["gpus"][1]["model"] = "different"
    record["gpus"][2]["memory_free_mib"] = 1
    record["gpus"][3]["mig_mode"] = "Enabled"
    record["repository"]["dirty"] = True
    record["active_compute_processes"] = ["1234, solver, GPU-0000, 1024"]
    failures = validate_preflight_record(
        record,
        expected_gpu_count=8,
        primary=True,
    )
    assert any("at least 8" in failure for failure in failures)
    assert any("homogeneous GPU model" in failure for failure in failures)
    assert any("free-memory fraction" in failure for failure in failures)
    assert any("MIG must be disabled" in failure for failure in failures)
    assert any("repository must be clean" in failure for failure in failures)
    assert any("active GPU compute processes" in failure for failure in failures)


def test_shared_numa_affinity_is_split_into_disjoint_rank_bindings() -> None:
    record = _physical_record(4)
    for device in record["gpus"]:
        device["cpu_affinity"] = "0-15"
        device["numa_node"] = 0
    bindings = rank_bindings(record, 4)
    assert [item["cpu_set"] for item in bindings] == [
        "0-3",
        "4-7",
        "8-11",
        "12-15",
    ]


def test_coordinate_matrix_is_deterministic_and_complete() -> None:
    config = {
        "gpu_counts": [1, 2, 4, 8],
        "nodes": [100],
        "partitions": ["scenario_aware", "nonzero_balanced"],
        "risks": ["expected"],
        "seeds": [17],
        "strong": {"scenario_counts": [16]},
        "weak": {"scenarios_per_gpu": [8]},
    }
    first = generate_coordinates(config)
    second = generate_coordinates(config)
    assert first == second
    assert len(first) == 16
    assert {item.gpu_count for item in first} == {1, 2, 4, 8}
    assert {item.partition for item in first} == {
        "scenario_aware",
        "nonzero_balanced",
    }
    assert {item.scenarios for item in first if item.scaling == "weak"} == {8, 16, 32, 64}


def test_one_monolithic_reference_can_match_all_distributed_gpu_counts() -> None:
    manifests = [
        make_monolithic_reference_manifest(
            _coordinate(gpu_count),
            topology_record=logical_topology(1),
            repository_commit=COMMIT,
            executable="/opt/spacepdhcg/bin/g5_physical_validation_harness",
            output_root="/evidence/g5",
            warmups=2,
            repeats=7,
            timeout_seconds=60,
        )
        for gpu_count in (1, 2, 4, 8)
    ]
    assert len({item["run_id"] for item in manifests}) == 1
    assert manifests[0]["run_id"] != _manifest(1)["run_id"]
    assert manifests[0]["comparison"]["monolithic_reference"]["reference_for_gpu_counts"] == [
        1,
        2,
        4,
        8,
    ]


@pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
def test_logical_commands_are_non_executable_and_match_snapshots(gpu_count: int) -> None:
    manifest = _manifest(gpu_count)
    validate_command_manifest(manifest)
    assert manifest["logical_only"]
    assert not manifest["physical_execution_permitted"]
    assert manifest["argv"][:5] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        "60s",
        "mpirun",
    ]
    assert manifest["argv"][manifest["argv"].index("--np") + 1] == str(gpu_count)
    snapshot = json.loads(
        (ROOT / "tests" / "snapshots" / "g5_commands.json").read_text(encoding="utf-8")
    )
    assert {
        "run_id": manifest["run_id"],
        "rankfile": manifest["rankfile"]["content"],
        "argv": manifest["argv"],
    } == snapshot[str(gpu_count)]
    with pytest.raises(PreflightError, match="non-executable"):
        assert_physical_execution_permitted(manifest, _physical_record(gpu_count))


def test_physical_manifest_pins_executable_and_preflight(tmp_path: Path) -> None:
    executable = tmp_path / "g5-harness"
    executable.write_bytes(b"pinned binary")
    preflight = _physical_record(2)
    manifest = make_command_manifest(
        _coordinate(2),
        topology_record=preflight,
        repository_commit=COMMIT,
        executable=str(executable),
        output_root="/evidence/g5",
        warmups=0,
        repeats=1,
        timeout_seconds=60,
    )
    assert manifest["physical_execution_permitted"]
    assert manifest["executable"]["sha256"]
    assert_physical_execution_permitted(manifest, preflight)
    executable.write_bytes(b"changed binary")
    with pytest.raises(PreflightError, match="hash changed"):
        assert_physical_execution_permitted(manifest, preflight)


@pytest.mark.parametrize("failure_mode", FAILURE_MODES)
def test_failure_injection_requires_explicit_test_mode(failure_mode: str) -> None:
    manifest = _manifest(2, failure_mode=failure_mode)
    validate_command_manifest(manifest)
    assert "--test-mode" in manifest["argv"]
    assert manifest["environment"]["SPACEPDHCG_G5_FAILURE_TEST"] == "1"
    broken = copy.deepcopy(manifest)
    broken["argv"].remove("--test-mode")
    with pytest.raises(ValueError, match="--test-mode"):
        validate_command_manifest(broken)


def test_partial_evidence_preserves_failure_and_missing_rank_data() -> None:
    manifest = _manifest(4)
    evidence = build_partial_evidence(
        manifest,
        preflight=_physical_record(4),
        return_code=137,
        timed_out=False,
        stdout="rank 0 complete",
        stderr="rank 3 CUDA out of memory; NCCL remote process exited",
        rank_telemetry=[{"rank": 0, "complete": True}],
    )
    assert evidence["status"] == "partial"
    assert evidence["missing_ranks"] == [1, 2, 3]
    assert {"oom", "nccl_error"} <= set(evidence["failure"]["kinds"])
    assert evidence["partial_logs"] == {
        "stdout_captured": True,
        "stderr_captured": True,
    }
    assert not evidence["multi_gpu_scaling_verified"]


def test_gpu_power_samples_are_integrated_without_spanning_large_gaps(tmp_path: Path) -> None:
    samples = tmp_path / "gpu.csv"
    samples.write_text(
        "timestamp, index, uuid, memory.used [MiB], power.draw [W]\n"
        "2026/09/01 00:00:00.000, 0, GPU-a, 100 MiB, 50 W\n"
        "2026/09/01 00:00:00.200, 0, GPU-a, 120 MiB, 70 W\n"
        "2026/09/01 00:00:10.000, 0, GPU-a, 110 MiB, 60 W\n",
        encoding="utf-8",
    )
    summary = summarize_gpu_samples(samples)
    assert summary == [
        {
            "index": 0,
            "uuid": "GPU-a",
            "sample_count": 3,
            "energy_joules": pytest.approx(12.0),
            "peak_device_bytes": 120 * 1024**2,
        }
    ]


def test_campaign_records_validate_against_frozen_schema() -> None:
    schema = json.loads(
        (ROOT / "experiments" / "schema" / "g5_campaign.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    preflight = _physical_record(2)
    preflight.update(
        {
            "captured_at_utc": "2026-09-01T00:00:00Z",
            "primary_campaign": True,
            "expected_gpu_count": 2,
            "failures": [],
            "raw_commands": {},
        }
    )
    validator.validate(preflight)
    manifest = _manifest(2)
    validator.validate(manifest)
    evidence = build_partial_evidence(
        manifest,
        preflight=_physical_record(2),
        return_code=1,
        timed_out=False,
        stdout="",
        stderr="rank failure",
    )
    validator.validate(evidence)
