"""Fetch the pinned GTOC12 official data into the data directory.

Every file listed in ``benchmarks/gtoc12/pins.json`` is downloaded from the first reachable URL,
its byte size and SHA-256 are checked against the pin, and the verifier archive is extracted.
Nothing is kept when a digest disagrees; the offending file is removed and reported.  The
destination defaults to :func:`spacepdhcg.gtoc12.data.data_directory`, so the same code serves
``spacepdhcg gtoc12 fetch`` from an installed wheel and ``scripts/gtoc12/fetch_gtoc12_data.py``
from a checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .data import data_directory, load_pins, sha256_file

USER_AGENT = "spacepdhcg-gtoc12-fetch/1.0"


def download(url: str, destination: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
            temporary = Path(handle.name)
    temporary.replace(destination)


def fetch_entry(entry: dict[str, Any], directory: Path, timeout: float) -> tuple[bool, str]:
    destination = directory / entry["name"]
    if destination.is_file() and sha256_file(destination) == entry["sha256"]:
        return True, "present"
    errors: list[str] = []
    for url in entry["urls"]:
        try:
            download(url, destination, timeout)
        except Exception as error:
            errors.append(f"{url}: {error}")
            continue
        size = destination.stat().st_size
        digest = sha256_file(destination)
        if size == int(entry["bytes"]) and digest == entry["sha256"]:
            return True, f"fetched from {url}"
        destination.unlink(missing_ok=True)
        errors.append(f"{url}: size {size} sha256 {digest} disagree with pin")
    return False, "; ".join(errors)


def extract_verifier(directory: Path, entry: dict[str, Any]) -> list[str]:
    archive = directory / entry["name"]
    target = directory / "verifier"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)
    problems: list[str] = []
    for member, expected in entry["members"].items():
        path = target / member
        if not path.is_file():
            problems.append(f"missing member {member}")
            continue
        digest = sha256_file(path)
        if digest != expected:
            problems.append(f"{member}: {digest} != {expected}")
        if path.name.startswith("GTOC12_Verify"):
            path.chmod(path.stat().st_mode | 0o111)
    return problems


def fetch_pinned_data(
    directory: Path | None = None,
    *,
    only: Iterable[str] | None = None,
    skip_optional: bool = False,
    timeout: float = 600.0,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Download and verify the pinned files; return ``(report, failure_count)``.

    The report is also written to ``<directory>/fetch_report.json``.
    """

    emit = log or (lambda line: print(line, file=sys.stderr))
    pins = load_pins()
    directory = directory or data_directory()
    directory.mkdir(parents=True, exist_ok=True)
    selected = None if only is None else set(only)
    report: dict[str, dict[str, Any]] = {}
    failures = 0
    for entry in pins["files"]:
        if selected is not None and entry["name"] not in selected:
            continue
        if skip_optional and entry.get("optional"):
            continue
        ok, message = fetch_entry(entry, directory, timeout)
        record: dict[str, Any] = {"ok": ok, "message": message, "sha256": entry["sha256"]}
        if ok and entry["role"] == "official_verifier":
            problems = extract_verifier(directory, entry)
            record["verifier_members_ok"] = not problems
            if problems:
                record["problems"] = problems
                ok = False
        report[entry["name"]] = record
        failures += 0 if ok else 1
        emit(f"{'OK ' if ok else 'ERR'} {entry['name']}: {message}")
    (directory / "fetch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report, failures


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``fetch`` options shared by the script and ``spacepdhcg gtoc12 fetch``."""

    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="pinned file names to fetch")
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)


def run(args: argparse.Namespace) -> int:
    _, failures = fetch_pinned_data(
        args.data_dir,
        only=args.only,
        skip_optional=args.skip_optional,
        timeout=args.timeout,
    )
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))
