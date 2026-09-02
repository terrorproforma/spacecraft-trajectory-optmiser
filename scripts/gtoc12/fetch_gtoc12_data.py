#!/usr/bin/env python3
"""Fetch the pinned GTOC12 official data into the ignored data directory.

Every file listed in ``benchmarks/gtoc12/pins.json`` is downloaded from the first reachable URL,
its byte size and SHA-256 are checked against the pin, and the verifier archive is extracted.
Nothing is written when a digest disagrees; the offending file is removed and reported.

Usage::

    python scripts/gtoc12/fetch_gtoc12_data.py [--data-dir DIR] [--only NAME ...] [--skip-optional]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PINS = ROOT / "benchmarks" / "gtoc12" / "pins.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, timeout: float) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "spacepdhcg-gtoc12-fetch/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
            temporary = Path(handle.name)
    temporary.replace(destination)


def fetch_entry(entry: dict, directory: Path, timeout: float) -> tuple[bool, str]:
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


def extract_verifier(directory: Path, entry: dict) -> list[str]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-optional", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    pins = json.loads(PINS.read_text(encoding="utf-8"))
    directory = args.data_dir or (ROOT / pins["data_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    failures = 0
    for entry in pins["files"]:
        if args.only and entry["name"] not in args.only:
            continue
        if args.skip_optional and entry.get("optional"):
            continue
        ok, message = fetch_entry(entry, directory, args.timeout)
        record = {"ok": ok, "message": message, "sha256": entry["sha256"]}
        if ok and entry["role"] == "official_verifier":
            problems = extract_verifier(directory, entry)
            record["verifier_members_ok"] = not problems
            if problems:
                record["problems"] = problems
                ok = False
        report[entry["name"]] = record
        failures += 0 if ok else 1
        print(f"{'OK ' if ok else 'ERR'} {entry['name']}: {message}", file=sys.stderr)
    (directory / "fetch_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
