"""Pinned GTOC12 data access: catalogue, bonus coefficients, verifier binaries.

Multi-megabyte official files are never committed.  ``benchmarks/gtoc12/pins.json`` records
URLs, byte sizes and SHA-256 digests; ``scripts/gtoc12/fetch_gtoc12_data.py`` downloads them into
the ignored data directory and this module refuses to load anything whose digest disagrees.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constants import ASTEROID_COUNT, ASTEROID_ELEMENT_EPOCH_MJD, AU_KM

FloatArray = NDArray[np.float64]

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PINS_PATH = REPOSITORY_ROOT / "benchmarks" / "gtoc12" / "pins.json"
RULES_PATH = REPOSITORY_ROOT / "benchmarks" / "gtoc12" / "gtoc12_rules.json"
DATA_ENVIRONMENT_VARIABLE = "SPACEPDHCG_GTOC12_DATA"


class Gtoc12DataError(RuntimeError):
    """Raised when pinned data is missing, unreadable or fails its checksum."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_pins() -> dict[str, Any]:
    payload = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0" or "files" not in payload:
        raise Gtoc12DataError("benchmarks/gtoc12/pins.json has an unexpected schema")
    return payload


def pinned_file(name: str) -> dict[str, Any]:
    for entry in load_pins()["files"]:
        if entry["name"] == name:
            return entry
    raise Gtoc12DataError(f"{name} is not a pinned GTOC12 file")


def data_directory() -> Path:
    override = os.environ.get(DATA_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser().resolve()
    return REPOSITORY_ROOT / load_pins()["data_directory"]


def verified_path(name: str, *, directory: Path | None = None) -> Path:
    """Return the local path of a pinned file after checking size and SHA-256."""

    entry = pinned_file(name)
    path = (directory or data_directory()) / name
    if not path.is_file():
        raise Gtoc12DataError(
            f"{name} is not present in {path.parent}; run "
            "`python scripts/gtoc12/fetch_gtoc12_data.py` first"
        )
    size = path.stat().st_size
    if size != int(entry["bytes"]):
        raise Gtoc12DataError(f"{name} has {size} bytes, pinned {entry['bytes']}")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise Gtoc12DataError(f"{name} SHA-256 {digest} disagrees with pin {entry['sha256']}")
    return path


@dataclass(frozen=True, slots=True)
class AsteroidCatalogue:
    """The 60,000 official asteroids with elements converted to km and radians.

    Row ``k`` holds asteroid ID ``ids[k]``; IDs are 1..60000 in file order so
    ``index = id - 1``.  Angles are radians and ``semi_major_axis_km`` is in km.
    """

    ids: NDArray[np.int64]
    epoch_mjd: FloatArray
    semi_major_axis_km: FloatArray
    eccentricity: FloatArray
    inclination_rad: FloatArray
    ascending_node_rad: FloatArray
    argument_of_perihelion_rad: FloatArray
    mean_anomaly_rad: FloatArray
    source_sha256: str

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def index_of(self, asteroid_id: int | NDArray[np.int64]) -> NDArray[np.int64]:
        index = np.asarray(asteroid_id, dtype=np.int64) - 1
        if np.any(index < 0) or np.any(index >= len(self)):
            raise KeyError("asteroid ID outside 1..60000")
        if not np.array_equal(self.ids[index], np.asarray(asteroid_id, dtype=np.int64)):
            raise KeyError("catalogue row order is not identity")
        return index

    def semi_major_axis_au(self) -> FloatArray:
        return self.semi_major_axis_km / AU_KM


def parse_catalogue_text(text: str, *, source_sha256: str = "") -> AsteroidCatalogue:
    rows = np.loadtxt(text.splitlines(), skiprows=1, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise Gtoc12DataError("asteroid catalogue must have eight columns")
    ids = rows[:, 0].astype(np.int64)
    if ids.shape[0] != ASTEROID_COUNT or not np.array_equal(ids, np.arange(1, ASTEROID_COUNT + 1)):
        raise Gtoc12DataError("asteroid catalogue IDs must be exactly 1..60000 in order")
    if not np.all(rows[:, 1] == ASTEROID_ELEMENT_EPOCH_MJD):
        raise Gtoc12DataError("asteroid catalogue epochs must all equal 64328 MJD")
    return AsteroidCatalogue(
        ids=ids,
        epoch_mjd=rows[:, 1].copy(),
        semi_major_axis_km=rows[:, 2] * AU_KM,
        eccentricity=rows[:, 3].copy(),
        inclination_rad=np.deg2rad(rows[:, 4]),
        ascending_node_rad=np.deg2rad(rows[:, 5]),
        argument_of_perihelion_rad=np.deg2rad(rows[:, 6]),
        mean_anomaly_rad=np.deg2rad(rows[:, 7]),
        source_sha256=source_sha256,
    )


@lru_cache(maxsize=1)
def load_catalogue() -> AsteroidCatalogue:
    path = verified_path("GTOC12_Asteroids_Data.txt")
    return parse_catalogue_text(
        path.read_text(encoding="utf-8"), source_sha256=pinned_file(path.name)["sha256"]
    )


@dataclass(frozen=True, slots=True)
class BonusTable:
    """Frozen end-of-competition ``bonus_coefficients.txt`` (row ``i`` is asteroid ``i+1``)."""

    coefficient: FloatArray
    already_mined_kg: FloatArray
    source_sha256: str

    def for_asteroid(self, asteroid_id: int) -> float:
        if not 1 <= asteroid_id <= ASTEROID_COUNT:
            raise KeyError("asteroid ID outside 1..60000")
        return float(self.coefficient[asteroid_id - 1])


def parse_bonus_text(text: str, *, source_sha256: str = "") -> BonusTable:
    rows = np.loadtxt(text.splitlines(), dtype=np.float64)
    if rows.shape != (ASTEROID_COUNT, 2):
        raise Gtoc12DataError("bonus_coefficients.txt must contain 60000 rows of two columns")
    if np.any(rows[:, 0] <= 0.0) or np.any(rows[:, 0] > 1.0) or np.any(rows[:, 1] < 0.0):
        raise Gtoc12DataError("bonus coefficients must lie in (0, 1] with non-negative masses")
    return BonusTable(rows[:, 0].copy(), rows[:, 1].copy(), source_sha256)


@lru_cache(maxsize=1)
def load_bonus_table() -> BonusTable:
    path = verified_path("bonus_coefficients.txt")
    return parse_bonus_text(
        path.read_text(encoding="utf-8"), source_sha256=pinned_file(path.name)["sha256"]
    )


def official_verifier_binary(*, extract: bool = True) -> Path:
    """Extract (once) and return the pinned Linux ``GTOC12_Verify`` binary."""

    archive = verified_path("GTOC12_Verification_Program.zip")
    member = "GTOC12_Verification/Linux/GTOC12_Verify"
    expected = pinned_file(archive.name)["members"][member]
    target = archive.parent / "verifier" / member
    if not target.is_file() and extract:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(archive.parent / "verifier")
    if not target.is_file():
        raise Gtoc12DataError("official verifier binary is not extracted")
    digest = sha256_file(target)
    if digest != expected:
        raise Gtoc12DataError(f"official verifier binary digest {digest} disagrees with pin")
    target.chmod(target.stat().st_mode | 0o111)
    return target


def official_example_solution() -> Path:
    archive = verified_path("GTOC12_Verification_Program.zip")
    member = "GTOC12_Verification/Linux/Result.txt"
    target = archive.parent / "verifier" / member
    if not target.is_file():
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(archive.parent / "verifier")
    if sha256_file(target) != pinned_file(archive.name)["members"][member]:
        raise Gtoc12DataError("official example Result.txt digest disagrees with pin")
    return target


def data_available(*names: str) -> bool:
    """True when every named pinned file is present and passes its checksum."""

    try:
        for name in names or ("GTOC12_Asteroids_Data.txt",):
            verified_path(name)
    except Gtoc12DataError:
        return False
    return True
