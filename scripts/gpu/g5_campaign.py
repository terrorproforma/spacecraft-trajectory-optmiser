#!/usr/bin/env python3
"""Capture, plan, validate, and execute fail-closed Gate G5 campaigns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]
MODULE_PATH = REPOSITORY / "src" / "spacepdhcg" / "experiments" / "g5_campaign.py"
SPEC = importlib.util.spec_from_file_location("spacepdhcg_g5_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load G5 campaign module: {MODULE_PATH}")
G5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = G5
SPEC.loader.exec_module(G5)

FAILURE_MODES = G5.FAILURE_MODES
PreflightError = G5.PreflightError
assert_physical_execution_permitted = G5.assert_physical_execution_permitted
build_partial_evidence = G5.build_partial_evidence
capture_preflight = G5.capture_preflight
generate_coordinates = G5.generate_coordinates
logical_topology = G5.logical_topology
make_command_manifest = G5.make_command_manifest
make_monolithic_reference_manifest = G5.make_monolithic_reference_manifest
summarize_gpu_samples = G5.summarize_gpu_samples
validate_command_manifest = G5.validate_command_manifest
verify_installed_distributed_stack = G5.verify_installed_distributed_stack
write_json = G5.write_json


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _write_raw_capture(output: Path, record: dict[str, Any]) -> None:
    raw_directory = output.parent / "preflight-raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    for name, capture in record["raw_commands"].items():
        (raw_directory / f"{name}.stdout.txt").write_text(
            capture["stdout"],
            encoding="utf-8",
        )
        (raw_directory / f"{name}.stderr.txt").write_text(
            capture["stderr"],
            encoding="utf-8",
        )


def command_capture(args: argparse.Namespace) -> int:
    record = capture_preflight(
        args.repository,
        expected_gpu_count=args.expected_gpus,
        primary=not args.non_primary,
        minimum_free_fraction=args.minimum_free_fraction,
        build_directory=args.build_directory,
    )
    write_json(args.output, record)
    _write_raw_capture(args.output, record)
    print(f"{record['status']}: {args.output}")
    for failure in record["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if record["status"] == "passed" else 2


def _logical_record(gpu_count: int) -> dict[str, Any]:
    return logical_topology(gpu_count)


def command_plan(args: argparse.Namespace) -> int:
    config = _read_json(args.config)
    preflight = None if args.logical_dry_run else _read_json(args.preflight)
    coordinates = generate_coordinates(config)
    if args.gpu_count:
        selected_counts = set(args.gpu_count)
        coordinates = [
            coordinate
            for coordinate in coordinates
            if coordinate.gpu_count in selected_counts
        ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    monolithic_keys: set[tuple[Any, ...]] = set()
    for coordinate in coordinates:
        topology = _logical_record(coordinate.gpu_count) if args.logical_dry_run else preflight
        if topology is None:
            raise AssertionError("physical planning requires a preflight record")
        manifest = make_command_manifest(
            coordinate,
            topology_record=topology,
            repository_commit=args.repository_commit,
            executable=args.executable,
            output_root=args.run_root,
            warmups=config["warmups"],
            repeats=config["repeats"],
            timeout_seconds=config["timeout_seconds"],
            failure_mode=args.failure_mode,
        )
        run_directory = output / manifest["run_id"]
        write_json(run_directory / "command.json", manifest)
        (run_directory / "rankfile").write_text(
            manifest["rankfile"]["content"],
            encoding="utf-8",
        )
        manifests.append(manifest)
        if args.failure_mode is not None or not config.get("include_monolithic", True):
            continue
        reference_key = (
            coordinate.scaling,
            coordinate.scenarios,
            coordinate.nodes,
            coordinate.risk,
            coordinate.seed,
        )
        if reference_key in monolithic_keys:
            continue
        monolithic_keys.add(reference_key)
        reference_topology = _logical_record(1) if args.logical_dry_run else preflight
        if reference_topology is None:
            raise AssertionError("physical planning requires a preflight record")
        reference = make_monolithic_reference_manifest(
            coordinate,
            topology_record=reference_topology,
            repository_commit=args.repository_commit,
            executable=args.executable,
            output_root=args.run_root,
            warmups=config["warmups"],
            repeats=config["repeats"],
            timeout_seconds=config["timeout_seconds"],
            failure_mode=None,
        )
        reference_directory = output / reference["run_id"]
        write_json(reference_directory / "command.json", reference)
        (reference_directory / "rankfile").write_text(
            reference["rankfile"]["content"],
            encoding="utf-8",
        )
        manifests.append(reference)
    plan = {
        "schema_version": "1.0.0",
        "record_type": "g5-launch-plan",
        "logical_only": args.logical_dry_run,
        "physical_execution_permitted": not args.logical_dry_run,
        "qualification_permitted": False,
        "config": str(args.config),
        "repository_commit": args.repository_commit,
        "preflight_fingerprint": None if preflight is None else preflight.get("fingerprint"),
        "manifest_count": len(manifests),
        "manifests": [
            {
                "run_id": item["run_id"],
                "record_type": item["record_type"],
                "command": f"{item['run_id']}/command.json",
                "manifest_sha256": item["manifest_sha256"],
            }
            for item in manifests
        ],
    }
    write_json(output / "launch-plan.json", plan)
    print(f"generated {len(manifests)} command manifests in {output}")
    return 0


def command_verify(args: argparse.Namespace) -> int:
    plan = _read_json(args.plan)
    root = args.plan.parent
    errors: list[str] = []
    first_manifest: dict[str, Any] | None = None
    for item in plan["manifests"]:
        manifest = _read_json(root / item["command"])
        validate_command_manifest(manifest)
        if item["manifest_sha256"] != manifest["manifest_sha256"]:
            errors.append(f"{manifest['run_id']}: launch-plan digest mismatch")
        first_manifest = first_manifest or manifest
    if first_manifest is None:
        errors.append("launch plan has no command manifests")
    else:
        errors.extend(verify_installed_distributed_stack(first_manifest))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"validated {len(plan['manifests'])} commands against installed "
        "OpenMPI/CUDA/NCCL"
    )
    return 0


def _live_repository_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _live_repository_dirty(repository: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip())


def command_execute(args: argparse.Namespace) -> int:
    if args.confirm != "EXECUTE-PHYSICAL-G5":
        raise PreflightError("physical execution requires --confirm EXECUTE-PHYSICAL-G5")
    manifest = _read_json(args.manifest)
    preflight = _read_json(args.preflight)
    assert_physical_execution_permitted(manifest, preflight)
    if _live_repository_commit(args.repository) != manifest["repository_commit"]:
        raise PreflightError("live repository commit differs from command manifest")
    if _live_repository_dirty(args.repository):
        raise PreflightError("live repository is dirty after physical preflight")
    run_directory = Path(manifest["evidence"]["directory"]).resolve()
    run_directory.mkdir(parents=True, exist_ok=False)
    (run_directory / "rankfile").write_text(
        manifest["rankfile"]["content"],
        encoding="utf-8",
    )
    write_json(run_directory / "command.json", manifest)
    write_json(run_directory / "preflight.json", preflight)
    sample_stream = (run_directory / "gpu-samples.csv").open("w", encoding="utf-8")
    sampler = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,memory.used,power.draw,"
            "clocks.current.sm,clocks.current.memory,temperature.gpu",
            "--format=csv",
            "--loop-ms=200",
        ],
        stdout=sample_stream,
        stderr=subprocess.STDOUT,
        text=True,
    )
    stdout = ""
    stderr = ""
    return_code: int | None = None
    timed_out = False
    launch_started = time.monotonic()
    try:
        completed = subprocess.run(
            manifest["argv"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **manifest["environment"]},
        )
        return_code = completed.returncode
        timed_out = completed.returncode == 124
        stdout = completed.stdout
        stderr = completed.stderr
    except KeyboardInterrupt:
        timed_out = True
        stderr = "launcher interrupted by operator\n"
    finally:
        sampler.terminate()
        try:
            sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:
            sampler.kill()
            sampler.wait()
        sample_stream.close()
    rank_telemetry = []
    for path in sorted(run_directory.glob("rank-*.json")):
        try:
            rank_telemetry.append(_read_json(path))
        except (json.JSONDecodeError, OSError, TypeError) as error:
            stderr += f"partial rank telemetry {path.name}: {error}\n"
    (run_directory / "launcher.stdout.log").write_text(stdout, encoding="utf-8")
    (run_directory / "launcher.stderr.log").write_text(stderr, encoding="utf-8")
    evidence = build_partial_evidence(
        manifest,
        preflight=preflight,
        return_code=return_code,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        rank_telemetry=rank_telemetry,
        gpu_samples=summarize_gpu_samples(run_directory / "gpu-samples.csv"),
        launcher_seconds=time.monotonic() - launch_started,
    )
    evidence_name = "evidence.json" if evidence["status"] == "complete" else "evidence.partial.json"
    write_json(run_directory / evidence_name, evidence)
    print(f"{evidence['status']}: {run_directory}")
    return return_code or (124 if timed_out else 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-preflight")
    capture.add_argument("--repository", type=Path, default=REPOSITORY)
    capture.add_argument("--expected-gpus", type=int, required=True)
    capture.add_argument("--minimum-free-fraction", type=float, default=0.90)
    capture.add_argument("--build-directory", type=Path, required=True)
    capture.add_argument("--non-primary", action="store_true")
    capture.add_argument("--output", type=Path, required=True)
    capture.set_defaults(handler=command_capture)

    plan = subparsers.add_parser("generate-plan")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--preflight", type=Path)
    plan.add_argument("--logical-dry-run", action="store_true")
    plan.add_argument("--repository-commit", required=True)
    plan.add_argument("--executable", required=True)
    plan.add_argument("--run-root", required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--failure-mode", choices=FAILURE_MODES)
    plan.add_argument(
        "--gpu-count",
        action="append",
        type=int,
        choices=(1, 2, 4, 8),
        help="restrict output to one or more rank counts",
    )
    plan.set_defaults(handler=command_plan)

    verify = subparsers.add_parser("verify-plan")
    verify.add_argument("plan", type=Path)
    verify.set_defaults(handler=command_verify)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--preflight", type=Path, required=True)
    execute.add_argument("--repository", type=Path, default=REPOSITORY)
    execute.add_argument("--confirm", required=True)
    execute.set_defaults(handler=command_execute)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "generate-plan" and not args.logical_dry_run and args.preflight is None:
        raise SystemExit("--preflight is required unless --logical-dry-run is set")
    try:
        return int(args.handler(args))
    except (PreflightError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
