"""The GPU-deferred validation manifest must stay well-formed, cross-linked and truthful.

It pins the artefacts that the v2 candidate integration promised to leave byte-identical; if one of
those files changes, this test fails before anyone re-uses the manifest's hash evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "gpu_deferred_validation_v2.json"
DOCUMENT = ROOT / "docs" / "GPU_DEFERRED_VALIDATION_V2.md"

REQUIRED_ITEM_KEYS = {"id", "track", "title", "blocked_by", "commands", "expected", "evidence"}
EXPECTED_TRACKS = {"planner", "literature", "gtoc12", "integration"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_manifest_shape(manifest: dict) -> None:
    assert manifest["schema_version"] == 1
    assert manifest["candidate"]["branch"] == "integration/single-gpu-v2-candidate"
    assert set(manifest) >= {
        "candidate",
        "gpu_ownership",
        "environment",
        "items",
        "frozen_hashes",
        "promotion",
    }
    ids = [item["id"] for item in manifest["items"]]
    assert len(ids) == len(set(ids)) >= 8
    for item in manifest["items"]:
        assert REQUIRED_ITEM_KEYS <= set(item), item["id"]
        assert item["track"] in EXPECTED_TRACKS, item["id"]
        assert item["commands"] and item["expected"], item["id"]
    assert {item["track"] for item in manifest["items"]} == EXPECTED_TRACKS


def test_every_item_is_documented(manifest: dict) -> None:
    text = DOCUMENT.read_text(encoding="utf-8")
    for item in manifest["items"]:
        assert f"## {item['id']}" in text, item["id"]
    assert manifest["promotion"]["fast_forward_command"].split("&&")[-1].strip() in text


def test_gpu_lock_is_never_created_by_the_manifest(manifest: dict) -> None:
    lock = manifest["gpu_ownership"]["lock_file"]
    for item in manifest["items"]:
        for command in item["commands"]:
            assert lock not in command, item["id"]


def test_referenced_repository_paths_exist(manifest: dict) -> None:
    pattern = re.compile(
        r"(?<![\w/.-])((?:tests|examples|scripts|cpp)/[\w./-]+\.(?:py|json|sh|cu))"
    )
    referenced: set[str] = set()
    for item in manifest["items"]:
        for command in item["commands"]:
            referenced.update(pattern.findall(command))
    assert referenced
    missing = sorted(path for path in referenced if not (ROOT / path).exists())
    assert not missing, missing


def test_frozen_files_are_byte_identical_to_the_recorded_blobs(manifest: dict) -> None:
    blobs = manifest["frozen_hashes"]["identical_blobs_base_vs_candidate"]
    assert len(blobs) >= 10
    for relative, blob in blobs.items():
        assert _git_blob_id(ROOT / relative) == blob, relative


def test_sha256_locks_match_repository_lock_files(manifest: dict) -> None:
    locks = manifest["frozen_hashes"]["sha256_locks"]
    for relative, digest in locks.items():
        lock_file = (ROOT / relative).with_suffix(".sha256")
        recorded = lock_file.read_text(encoding="utf-8").split()[0]
        assert recorded == digest, relative
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == digest, relative


def test_promotion_is_a_fast_forward_onto_the_recorded_base(manifest: dict) -> None:
    promotion = manifest["promotion"]
    assert "--ff-only" in promotion["fast_forward_command"]
    assert manifest["candidate"]["base"] in promotion["fast_forward_command"]
    assert "--force" not in promotion["fast_forward_command"]
