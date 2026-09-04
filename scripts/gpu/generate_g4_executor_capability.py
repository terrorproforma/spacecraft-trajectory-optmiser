#!/usr/bin/env python3
"""Generate a content-addressed G4 executor capability record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
# Amendment single-gpu-v1.2 constants, mirrored from spacepdhcg.experiments.g4_execution_contract
# (this script is dependency-free so it can run from any pinned worktree).
AMENDMENT_ID_V1_2 = "single-gpu-v1.2"
IPM_EQUILIBRATION_MODE = "qoco_native_default"
IPM_NATIVE_RUIZ_ITERATIONS = 0
IPM_NATIVE_SCALING_MODE = "not_applicable_ipm_native"
DEADLINE_CLASSIFICATION_RULE = "measured_wall_exceeds_attempt_deadline"
EXPECTED_AXES = {
    "family",
    "intervals",
    "policy",
    "quality_tier",
    "conditioning",
    "scaling_mode",
    "warm_start_mode",
    "family_classes",
    "evaluation_seed",
    "repeat",
    "solver_order",
}
EXPECTED_EXECUTION_CONTRACT = {
    "version": "g4-persistent-group-v1",
    "one_process_per_group": True,
    "persistent_session": True,
    "persistent_workspace": True,
    "separate_attempt_records": True,
    "policy_reset_between_attempts": True,
}
PINNED_CONTRACTS = {
    "applicability": "benchmarks/g4_applicability.json",
    "claim_core": "benchmarks/g4_h5_h6_claim_core.json",
    "claim_core_amendment": "benchmarks/g4_claim_core_amendment_v1_2.json",
    "execution_group_schema": "experiments/schema/g4_execution_group.schema.json",
    "raw_attempt_schema": "experiments/schema/g4_raw_attempt.schema.json",
    "paper1_result_schema": "experiments/schema/paper1_result.schema.json",
}
PROBE_GROUP_DEADLINE_SECONDS = 900
PINNED_LOCKS = {
    "applicability": "benchmarks/g4_applicability.sha256",
    "claim_core": "benchmarks/g4_h5_h6_claim_core.sha256",
    "claim_core_amendment": "benchmarks/g4_claim_core_amendment_v1_2.sha256",
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_libraries(executable: Path) -> dict[str, str]:
    """SHA-256 of every SpacePDHCG shared library the executable resolves through ``ldd``.

    The executor links ``libspacepdhcg_cuda.so`` dynamically, so the executable hash alone
    cannot pin the solver kernels; a statically linked or non-ELF executable yields an empty map.
    """

    try:
        completed = subprocess.run(
            ["ldd", str(executable)], check=False, capture_output=True, text=True
        )
    except OSError:
        return {}
    libraries: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "spacepdhcg" not in line or "=>" not in line:
            continue
        target = line.split("=>", 1)[1].split("(", 1)[0].strip()
        if target and Path(target).is_file():
            libraries[Path(target).name] = sha256_path(Path(target))
    return libraries


def probe_manifest() -> dict[str, Any]:
    group_id = "g4-group-v1-" + "a" * 64
    instance = "g4-instance-v2-" + "b" * 64
    coordinate = {
        "family": "P1-C-pd3",
        "intervals": 20,
        "dispersion_class": 0.05,
        "seed": 59,
        "policy": "pure-gpu-ipm",
        "quality_tier": "tight",
        "quality_tolerance": 1e-6,
        "conditioning": 0.0,
        "scaling_mode": "refresh_if_needed",
        "warm_mode": "primal",
        "solver_order": 0,
    }
    attempts = []
    for kind, repeat in (
        ("warmup", 0),
        ("warmup", 1),
        *(("measured", repeat) for repeat in range(7)),
    ):
        attempts.append(
            {
                **coordinate,
                "group_id": group_id,
                "instance": instance,
                "repeat_kind": kind,
                "repeat": repeat,
                "statistics_eligible": kind == "measured",
            }
        )
    return {
        "schema_version": "1.0.0",
        "record_kind": "execution_group",
        "group_id": group_id,
        "physical_instance_id": instance,
        "coordinate": coordinate,
        "process_contract": {
            "processes": 1,
            "persistent_session": True,
            "persistent_workspace": True,
            "policy_reset_between_attempts": True,
        },
        "attempts": attempts,
    }


def run_session_probe(
    executable: Path,
    policy_sha256: str,
    matrix_sha256: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="spacepdhcg-g4-probe-") as directory:
        manifest = Path(directory) / "execution-group.json"
        manifest.write_bytes(canonical_bytes(probe_manifest()))
        environment = dict(os.environ)
        environment.update(
            {
                "SPACEPDHCG_G4_GROUP_ID": "g4-group-v1-" + "a" * 64,
                "SPACEPDHCG_G4_CAPABILITY_PROBE": "1",
                "SPACEPDHCG_G4_OUTER_ITERATIONS": "1",
                # One SCvx iteration per attempt. The QOCO solve inside a pure-gpu-ipm attempt
                # cannot be cancelled mid-factorisation and a foreign GPU job can slow it by
                # >10x (observed 12-20 s per attempt on a busy GPU for the N=20 probe), so the
                # probe deadlines are generous: they exist to catch hangs, not to time anything.
                "SPACEPDHCG_G4_ATTEMPT_DEADLINE_SECONDS": "60",
                "SPACEPDHCG_G4_GROUP_DEADLINE_SECONDS": str(PROBE_GROUP_DEADLINE_SECONDS),
                # Amendment single-gpu-v1.2 rule A: the probe must run QOCO with the recorded
                # native equilibration and echo it, exactly as campaign IPM attempts will.
                "SPACEPDHCG_G4_POLICY_AMENDMENT": AMENDMENT_ID_V1_2,
                "SPACEPDHCG_G4_CENSORING_STRATUM": "claim_core",
                "SPACEPDHCG_G4_INNER_ITERATION_CAP": "200000",
                "SPACEPDHCG_G4_DETERMINISTIC_REPLAY": "1",
            }
        )
        completed = subprocess.run(
            [
                executable,
                "--g4-session",
                manifest,
                policy_sha256,
                matrix_sha256,
                "d" * 64,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PROBE_GROUP_DEADLINE_SECONDS + 120,
            env=environment,
        )
    if completed.returncode != 0:
        raise SystemExit(
            "executor failed real persistent-session probe: "
            f"exit={completed.returncode}; stderr={completed.stderr[-1000:]}"
        )
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    attempts = [record for record in records if record.get("case") == "g4_attempt"]
    ready = [record for record in records if record.get("case") == "g4_session_ready"]
    complete = [record for record in records if record.get("case") == "g4_session_complete"]
    if len(ready) != 1 or len(complete) != 1 or len(attempts) != 9:
        raise SystemExit(
            "executor session probe did not emit one ready, nine attempts, one complete"
        )
    expected = [("warmup", 0), ("warmup", 1), *(("measured", index) for index in range(7))]
    if [(record["repeat_kind"], record["repeat"]) for record in attempts] != expected:
        raise SystemExit("executor session probe emitted the wrong attempt order")
    sessions = [record.get("session", {}) for record in attempts]
    if (
        len({record.get("pid") for record in sessions}) != 1
        or len({record.get("workspace_address") for record in sessions}) != 1
        or len({record.get("topology_fingerprint") for record in sessions}) != 1
        or any(record.get("cuda_context_generation") != 1 for record in sessions)
        or any(record.get("workspace_generation") != 1 for record in sessions)
        or any(record.get("topology_allocations_after_create") != 0 for record in sessions)
        or any(record.get("topology_index_copies_after_create") != 0 for record in sessions)
    ):
        raise SystemExit(
            "executor session probe failed process/context/workspace topology invariants"
        )
    measured = [record for record in attempts if record["repeat_kind"] == "measured"]
    if any(
        record.get("statistics_eligible") is not True
        or not isinstance(record.get("paper1_result"), dict)
        for record in measured
    ):
        raise SystemExit("executor session probe omitted strict measured records")
    if any(record.get("statistics_eligible") is not False for record in attempts[:2]):
        raise SystemExit("executor session probe included warmups in statistics")
    # The probe coordinate is pure-gpu-ipm: every launched attempt must have run on a real
    # QOCO workspace (>= 1 creation in the persistent session) and end in a solver outcome.
    # An executor whose IPM adapter cannot initialise, whose warm boundary fails, or whose
    # library is missing is refused here instead of producing fake failures for a campaign.
    dispositions = [record.get("disposition") for record in attempts]
    if any(record.get("launched") is not True for record in attempts):
        raise SystemExit(f"pure-gpu-ipm probe did not launch every attempt: {dispositions}")
    if any(disposition not in {"qualified", "unqualified"} for disposition in dispositions):
        raise SystemExit(
            "pure-gpu-ipm probe produced non-solver dispositions (IPM adapter or warm boundary "
            f"defect, or QOCO unavailable): {dispositions}"
        )
    creations = [record.get("qoco_workspace_creations") for record in sessions]
    if any(not isinstance(value, int) or value < 1 for value in creations):
        raise SystemExit(f"pure-gpu-ipm probe never constructed a QOCO workspace: {creations}")
    expected_echo = {
        "mode": IPM_EQUILIBRATION_MODE,
        "ruiz_iterations": IPM_NATIVE_RUIZ_ITERATIONS,
        "requested_ruiz_iterations": IPM_NATIVE_RUIZ_ITERATIONS,
        "scaling_mode": IPM_NATIVE_SCALING_MODE,
    }
    echoes = [record.get("amendment", {}).get("ipm_equilibration") for record in attempts]
    for echo in echoes:
        stripped = (
            {key: value for key, value in echo.items() if key != "qoco_status_code"}
            if isinstance(echo, dict)
            else echo
        )
        if stripped != expected_echo or not isinstance(echo.get("qoco_status_code"), int):
            raise SystemExit(
                f"pure-gpu-ipm probe did not run with the amended native equilibration: {echo!r}"
            )
    if any(record.get("policy_amendment") != AMENDMENT_ID_V1_2 for record in attempts):
        raise SystemExit("pure-gpu-ipm probe records do not carry policy_amendment single-gpu-v1.2")
    identities = [
        (record.get("paper1_result") or {}).get("identity", {}).get("scaling_mode")
        for record in attempts
        if record["repeat_kind"] == "measured"
    ]
    if any(value != IPM_NATIVE_SCALING_MODE for value in identities):
        raise SystemExit(f"pure-gpu-ipm probe recorded scaling_mode {identities!r}")
    return {
        "kind": "real_cuda_session",
        "attempt_count": 9,
        "warmup_count": 2,
        "measured_count": 7,
        "same_process": True,
        "same_context": True,
        "same_workspace": True,
        "zero_post_create_topology_allocations": True,
        "zero_post_create_topology_index_copies": True,
        "pure_gpu_ipm_probe": {
            "policy": "pure-gpu-ipm",
            "dispositions": dispositions,
            "qoco_workspace_creations": creations,
            "qoco_numeric_updates": [record.get("qoco_numeric_updates") for record in sessions],
            "policy_amendment": AMENDMENT_ID_V1_2,
            "ipm_equilibration": expected_echo,
            "qoco_status_codes": [echo.get("qoco_status_code") for echo in echoes],
        },
    }


def ipm_library_identity() -> dict[str, str]:
    """Pin the QOCO library the executor will dlopen; refuse when it is not usable."""

    path = os.environ.get("SPACEPDHCG_QOCO_LIBRARY", "")
    if not path:
        raise SystemExit(
            "SPACEPDHCG_QOCO_LIBRARY is unset: the pure-gpu-ipm and hybrid policies cannot run; "
            "export the QOCO shared library path before generating a capability"
        )
    library = Path(path)
    if not library.is_file():
        raise SystemExit(f"SPACEPDHCG_QOCO_LIBRARY={path} is not a file")
    return {"path": str(library.resolve()), "sha256": sha256_path(library)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    executable = arguments.executable.resolve()
    policy_path = repository / "benchmarks/g4_policy.json"
    lock = (repository / "benchmarks/g4_policy.sha256").read_text().split()
    if len(lock) != 2 or lock[0] != sha256_path(policy_path):
        raise SystemExit("G4 policy lock mismatch")
    policy = json.loads(policy_path.read_text())
    split = policy["tuning_evaluation_split"]
    if set(split["tuning_seeds"]) & set(split["evaluation_seeds"]):
        raise SystemExit("G4 tuning and evaluation seed sets overlap")
    if (
        len(set(split["evaluation_seeds"]))
        != policy["matrix"]["randomised_instances_per_coordinate"]
    ):
        raise SystemExit("G4 evaluation seed cardinality mismatch")
    matrix_sha256 = hashlib.sha256(canonical_bytes(policy["matrix"]).rstrip(b"\n")).hexdigest()
    source_commit = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", repository, "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("capability generation requires a clean source commit")
    emitted = subprocess.run(
        [executable, "--g4-capabilities"],
        check=True,
        capture_output=True,
        text=True,
    )
    capability = json.loads(emitted.stdout)
    if set(capability.get("axes", {})) != EXPECTED_AXES:
        raise SystemExit("executor did not audit every frozen axis")
    if any(
        value.get("status") not in {"applied", "execution_only"}
        for value in capability["axes"].values()
    ):
        raise SystemExit("executor reports an unapplied roadmap-required axis")
    if capability.get("execution_contract") != EXPECTED_EXECUTION_CONTRACT:
        raise SystemExit("executor lacks the authoritative persistent nine-attempt group contract")
    contract_hashes = {
        name: sha256_path(repository / relative) for name, relative in PINNED_CONTRACTS.items()
    }
    for contract_name, relative in PINNED_LOCKS.items():
        expected = (repository / relative).read_text().split()[0]
        if contract_hashes[contract_name] != expected:
            raise SystemExit(f"{contract_name} lock mismatch")
    if capability.get("policy_amendments_supported") != ["single-gpu-v1.1", "single-gpu-v1.2"]:
        raise SystemExit("executor does not declare support for amendments single-gpu-v1.1/v1.2")
    declared = capability.get("ipm_equilibration", {})
    if (
        declared.get("mode") != IPM_EQUILIBRATION_MODE
        or declared.get("ruiz_iterations") != IPM_NATIVE_RUIZ_ITERATIONS
        or declared.get("recorded_scaling_mode") != IPM_NATIVE_SCALING_MODE
    ):
        raise SystemExit(
            f"executor declares IPM equilibration {declared!r}, amendment v1.2 requires "
            f"{IPM_EQUILIBRATION_MODE}/{IPM_NATIVE_RUIZ_ITERATIONS}/{IPM_NATIVE_SCALING_MODE}"
        )
    if capability.get("deadline_classification", {}).get("rule") != DEADLINE_CLASSIFICATION_RULE:
        raise SystemExit("executor does not declare amendment v1.2 rule B deadline classification")
    # The executor bakes SPACEPDHCG_SOURCE_COMMIT at CMake *configure* time and echoes it as
    # identity.repository_commit in every measured record; the decision step rejects records
    # whose commit differs from the campaign's. A tree that was only rebuilt (not
    # re-configured) since an older commit would pass every other check and fail at the end
    # of the campaign, so refuse it here.
    if capability.get("compiled_source_commit") != source_commit:
        raise SystemExit(
            "executor was configured at commit "
            f"{capability.get('compiled_source_commit')!r}, not HEAD {source_commit}; "
            "re-run the CMake configure step and rebuild before generating a capability"
        )
    ipm_library = ipm_library_identity()
    session_probe = run_session_probe(executable, lock[0], matrix_sha256)
    capability.update(
        {
            "source_commit": source_commit,
            "executable_sha256": sha256_path(executable),
            "runtime_library_sha256": runtime_libraries(executable),
            "ipm_library": ipm_library,
            "policy_sha256": lock[0],
            "matrix_sha256": matrix_sha256,
            "contract_hashes": contract_hashes,
            "session_probe": session_probe,
        }
    )
    capability["capability_sha256"] = hashlib.sha256(
        canonical_bytes(capability).rstrip(b"\n")
    ).hexdigest()
    encoded = canonical_bytes(capability)
    output = arguments.output.resolve()
    if arguments.check:
        if not output.is_file() or output.read_bytes() != encoded:
            raise SystemExit("G4 executor capability record drift")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    print(capability["capability_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
