#!/usr/bin/env python3
"""Seal a G5 run directory into a write-once archive and digest manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def seal(args: argparse.Namespace) -> int:
    run_directory = args.run_directory.resolve()
    archive = args.archive.resolve()
    seal_path = archive.with_suffix(archive.suffix + ".seal.json")
    if archive.exists() or seal_path.exists():
        raise FileExistsError("archive or seal already exists; G5 sealing is write-once")
    evidence_path = run_directory / "evidence.json"
    partial_path = run_directory / "evidence.partial.json"
    if evidence_path.exists():
        evidence = _read(evidence_path)
    elif args.allow_partial_failure_evidence and partial_path.exists():
        evidence_path = partial_path
        evidence = _read(partial_path)
    else:
        raise FileNotFoundError("complete evidence is required unless partial failure is explicit")
    if evidence.get("qualification_claim") is not False:
        raise ValueError("G5 tooling evidence must not contain a qualification claim")
    command = [
        sys.executable,
        str(REPOSITORY / "scripts" / "gpu" / "archive_run.py"),
        str(run_directory),
        "--repository",
        str(args.repository),
        "--require-clean-repository",
        "--archive",
        str(archive),
    ]
    completed = subprocess.run(command, check=False, text=True)
    if completed.returncode != 0:
        return completed.returncode
    index_path = run_directory / "evidence-index.json"
    seal_record = {
        "schema_version": "1.0.0",
        "record_type": "g5-immutable-archive-seal",
        "sealed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
        },
        "evidence_index": {
            "path": str(index_path),
            "sha256": sha256(index_path),
        },
        "evidence": {
            "path": str(evidence_path),
            "sha256": sha256(evidence_path),
            "status": evidence.get("status"),
            "run_id": evidence.get("run_id"),
        },
        "repository_commit": evidence.get("repository_commit")
        or _read(run_directory / "preflight.json")["repository"]["commit"],
        "write_once": True,
    }
    descriptor = (
        json.dumps(seal_record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor_fd = os.open(seal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor_fd, "wb") as stream:
        stream.write(descriptor)
    archive.chmod(0o444)
    index_path.chmod(0o444)
    print(f"{seal_record['archive']['sha256']}  {archive}")
    print(f"{sha256(seal_path)}  {seal_path}")
    return 0


def verify(args: argparse.Namespace) -> int:
    seal_path = args.seal.resolve()
    seal_record = _read(seal_path)
    failures: list[str] = []
    for key in ("archive", "evidence_index", "evidence"):
        item = seal_record[key]
        path = Path(item["path"])
        if not path.is_file():
            failures.append(f"{key} is missing: {path}")
        elif sha256(path) != item["sha256"]:
            failures.append(f"{key} digest mismatch: {path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    print(f"verified immutable G5 seal: {seal_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("run_directory", type=Path)
    seal_parser.add_argument("--archive", type=Path, required=True)
    seal_parser.add_argument("--repository", type=Path, default=REPOSITORY)
    seal_parser.add_argument("--allow-partial-failure-evidence", action="store_true")
    seal_parser.set_defaults(handler=seal)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("seal", type=Path)
    verify_parser.set_defaults(handler=verify)
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        return int(args.handler(args))
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
