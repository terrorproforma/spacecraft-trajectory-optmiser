#!/usr/bin/env python3
"""Hash a completed experiment directory and write an immutable evidence index."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: Path, *arguments: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    completed = subprocess.run(
        [git, "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def files(directory: Path, excluded: set[Path]) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.resolve() not in excluded
    )


def relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "sha256": sha256(path),
        "mode": oct(stat.st_mode & 0o777),
    }


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def create_reproducible_tar(source: Path, destination: Path) -> None:
    """Create a byte-reproducible gzip-compressed POSIX tar archive.

    Tar member ordering is lexical, uid/gid/user/group and mtimes are normalised, and the gzip
    header uses an empty filename plus mtime zero. File modes are preserved because executable
    experiment scripts and captured permissions are meaningful evidence.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded = {destination.resolve()}
    with tempfile.NamedTemporaryFile(
        suffix=".tar",
        dir=destination.parent,
        delete=False,
    ) as temporary_stream:
        temporary_tar = Path(temporary_stream.name)
    try:
        with tarfile.open(temporary_tar, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in files(source, excluded | {temporary_tar.resolve()}):
                relative = path.relative_to(source)
                info = archive.gettarinfo(str(path), arcname=relative.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                # Suppress platform-dependent high-resolution timestamp PAX records.
                info.pax_headers = {}
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
        with temporary_tar.open("rb") as uncompressed, destination.open("wb") as output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=output,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                shutil.copyfileobj(uncompressed, compressed, length=_CHUNK)
    finally:
        temporary_tar.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--index-name", default="evidence-index.json")
    parser.add_argument(
        "--archive",
        type=Path,
        help="optional byte-reproducible .tar.gz destination",
    )
    parser.add_argument(
        "--require-clean-repository",
        action="store_true",
        help="fail when the repository has uncommitted changes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_directory = args.run_directory.resolve()
    repository = args.repository.resolve()
    if not run_directory.is_dir():
        raise SystemExit(f"run directory does not exist: {run_directory}")
    index_path = run_directory / args.index_name
    archive_path = args.archive.resolve() if args.archive is not None else None
    excluded = {index_path.resolve()}
    if archive_path is not None:
        excluded.add(archive_path)

    repository_status = git_output(repository, "status", "--porcelain=v1")
    if args.require_clean_repository and repository_status:
        print("ERROR: repository has uncommitted changes", file=sys.stderr)
        return 2

    artifact_records = [
        relative_artifact(path, run_directory) for path in files(run_directory, excluded)
    ]
    payload = {
        "schema_version": "1.0.0",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "repository": {
            "root": str(repository),
            "commit": git_output(repository, "rev-parse", "HEAD"),
            "branch": git_output(repository, "branch", "--show-current"),
            "dirty": bool(repository_status),
            "status_porcelain": repository_status,
        },
        "run_directory": str(run_directory),
        "artifacts": artifact_records,
    }
    write_atomic(
        index_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    index_digest = sha256(index_path)
    print(f"{index_digest}  {index_path}")

    if archive_path is not None:
        create_reproducible_tar(run_directory, archive_path)
        print(f"{sha256(archive_path)}  {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
