"""Emit GTOC12 ship histories in the ``web/trajectory-viewer`` dataset schema.

The viewer consumes ``trajectories.json`` records with a dense ``replay`` series and a sparse
``transcription`` series (``points_txyz`` + ``selected_indices`` into the original sample set), a
``viewer`` block describing the physical frame, and validation metadata.  The dense samples here
come from the independent verifier's propagation of the emitted solution (the same model that
scores it), decimated deterministically to at most 512 points with endpoints and every event
preserved; no interpolation is performed.  Asteroid and Earth orbits are added as context traces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import constants as C
from .data import AsteroidCatalogue
from .ephemeris import asteroid_state, earth_state
from .solution import Solution
from .verifier import PropagatedHistory

FloatArray = NDArray[np.float64]
VIEWER_SCHEMA_VERSION = "1.0.0"
MAX_REPLAY_POINTS = 512


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_indices(count: int, keep: set[int], maximum: int = MAX_REPLAY_POINTS) -> list[int]:
    """Deterministic decimation: endpoints, mandatory indices, then evenly spaced samples."""

    if count <= maximum:
        return list(range(count))
    mandatory = {0, count - 1} | {index for index in keep if 0 <= index < count}
    remaining = maximum - len(mandatory)
    even = np.linspace(0, count - 1, max(remaining, 2)).round().astype(int).tolist()
    chosen = sorted(mandatory | set(even))
    while len(chosen) > maximum:
        # drop the non-mandatory point closest to its neighbour
        gaps = [
            (chosen[i + 1] - chosen[i], i)
            for i in range(len(chosen) - 1)
            if chosen[i + 1] not in mandatory
        ]
        _, index = min(gaps)
        chosen.pop(index + 1)
    return chosen


def _series(times: FloatArray, points: FloatArray, keep: set[int]) -> dict[str, Any]:
    original = [[float(t), *map(float, p)] for t, p in zip(times, points, strict=True)]
    indices = select_indices(len(original), keep)
    return {
        "original_point_count": len(original),
        "original_sha256": _sha256(_canonical(original)),
        "point_count": len(indices),
        "points_txyz": [original[index] for index in indices],
        "selected_indices": indices,
    }


def ship_record(
    ship_id: int,
    history: PropagatedHistory,
    solution: Solution,
    catalogue: AsteroidCatalogue,
    *,
    run_id: str,
    commit: str,
    verification: dict[str, Any],
    solution_sha256: str,
) -> dict[str, Any]:
    arrays = history.arrays()
    times = arrays["epochs_mjd"]
    positions = arrays["positions_km"]
    ship = solution.ships[ship_id - 1]
    event_epochs = sorted({event.epoch for event in ship.events})
    keep = {int(np.argmin(np.abs(times - epoch))) for epoch in event_epochs}
    replay = _series(times, positions, keep)
    # transcription: the event states themselves (exact body states at rendezvous/flyby/launch)
    event_points = []
    event_times = []
    for event in ship.events:
        event_times.append(event.epoch)
        event_points.append(event.before.position)
    transcription = _series(np.asarray(event_times), np.asarray(event_points), set())
    thrust = arrays["thrust_n"]
    masses = arrays["masses_kg"]
    visited = [event.event_id for event in ship.events if event.is_asteroid]
    context = []
    for asteroid in sorted(set(visited)):
        grid = np.linspace(C.MISSION_START_MJD, C.MISSION_END_MJD, 361)
        r, _ = asteroid_state(catalogue, np.full(grid.shape[0], asteroid), grid)
        context.append(
            {
                "body": f"asteroid {asteroid}",
                "points_txyz": [[float(t), *map(float, p)] for t, p in zip(grid, r, strict=True)],
            }
        )
    grid = np.linspace(C.MISSION_START_MJD, C.MISSION_START_MJD + 366.0, 122)
    r, _ = earth_state(grid)
    context.append(
        {
            "body": "Earth",
            "points_txyz": [[float(t), *map(float, p)] for t, p in zip(grid, r, strict=True)],
        }
    )
    return {
        "trajectory_id": f"gtoc12_{run_id}_ship{ship_id}",
        "family": "GTOC12",
        "physical_family": "GTOC12 sustainable asteroid mining ship",
        "frame": "J2000 heliocentric ecliptic Cartesian [x, y, z]",
        "position_units": "km",
        "time_units": "MJD (days)",
        "state_order": ["x", "y", "z", "vx", "vy", "vz", "m"],
        "control_order": ["Tx", "Ty", "Tz"],
        "initial_state": [
            *map(float, positions[0]),
            *map(float, arrays["velocities_km_s"][0]),
            float(masses[0]),
        ],
        "terminal_target": [
            *map(float, positions[-1]),
            *map(float, arrays["velocities_km_s"][-1]),
            float(masses[-1]),
        ],
        "events": [
            {
                "event_id": event.event_id,
                "kind": event.kind,
                "epoch_mjd": event.epoch,
                "mass_before_kg": event.before.mass,
                "mass_after_kg": event.after.mass,
            }
            for event in ship.events
        ],
        "asteroids_visited": visited,
        "controls_summary": {
            axis: {
                "minimum": float(np.min(thrust[:, k])),
                "maximum": float(np.max(thrust[:, k])),
                "mean": float(np.mean(thrust[:, k])),
            }
            for k, axis in enumerate(("Tx", "Ty", "Tz"))
        },
        "mass_summary": {
            "initial_kg": float(masses[0]),
            "final_kg": float(masses[-1]),
            "minimum_kg": float(np.min(masses)),
        },
        "attitude_summary": None,
        "iteration_history": None,
        "path_constraint_bounds": {
            "thrust_magnitude_max_n": C.THRUST_MAX_N,
            "minimum_sun_distance_au": C.MIN_SUN_DISTANCE_AU,
        },
        "qualification": {
            "qualified": bool(verification.get("ok", False)),
            "label": "official GTOC12 verifier pass; independent verifier pass"
            if verification.get("ok")
            else "unverified",
        },
        "raw_evidence_sha256": solution_sha256,
        "replay": replay,
        "transcription": transcription,
        "context_orbits": context,
        "validation": {
            "finite": bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(times))),
            **{key: value for key, value in verification.items() if key != "violations"},
        },
        "source": {
            "campaign": "gtoc12-reduced-v1",
            "commit": commit,
            "run_id": run_id,
            "generator": "spacepdhcg.gtoc12.pipeline",
        },
        "viewer": {
            "scene_kind": "heliocentric",
            "body_label": "Sun (not to scale); asteroid and Earth orbits shown as context",
            "body_radius": 696000.0,
            "radius_label": "Sun photospheric radius 696,000 km",
            "axes": ["X ecliptic (vernal equinox)", "Y ecliptic", "Z ecliptic north"],
            "frame_choice": "GTOC12 states are defined in the J2000 heliocentric ecliptic frame; "
            "hops are shown in that inertial frame.",
        },
    }


def write_viewer_dataset(
    directory: Path,
    solution: Solution,
    histories: dict[int, PropagatedHistory],
    catalogue: AsteroidCatalogue,
    *,
    run_id: str,
    commit: str,
    verification: dict[str, Any],
    solution_path: Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    solution_sha256 = hashlib.sha256(solution_path.read_bytes()).hexdigest()
    trajectories = [
        ship_record(
            ship_id,
            histories[ship_id],
            solution,
            catalogue,
            run_id=run_id,
            commit=commit,
            verification=verification,
            solution_sha256=solution_sha256,
        )
        for ship_id in sorted(histories)
    ]
    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "viewer_schema_version": VIEWER_SCHEMA_VERSION,
        "title": "GTOC12 reduced-instance OrbitWeaver solution",
        "generated_by_commit": commit,
        "imported_source_sha256": solution_sha256,
        "prohibitions": {
            "visual_interpolation_included": False,
            "aggregate_metric_path_fabrication": True,
            "gpu_execution_during_extraction": True,
        },
        "archive": {
            "data_dictionary": {
                "point_encoding": {
                    "points_txyz": "[MJD, x, y, z] exact propagated samples (km)",
                    "decimation": "deterministic endpoint/event preservation plus evenly spaced "
                    "exact samples; no interpolation",
                    "original_sha256": "SHA-256 of canonical JSON of all propagated samples",
                }
            },
            "validation_report": verification,
        },
        "trajectories": trajectories,
    }
    text = json.dumps(payload, sort_keys=True, indent=1, allow_nan=False)
    data_path = directory / "trajectories.json"
    data_path.write_text(text + "\n", encoding="utf-8")
    raw = data_path.read_bytes()
    manifest = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "files": {
            "trajectories.json": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        },
        "source": {
            "path_basename": solution_path.name,
            "sha256": solution_sha256,
            "bytes": solution_path.stat().st_size,
        },
        "transform": "Verifier-model propagation of the emitted GTOC12 solution; stable-key "
        "JSON; deterministic decimation.",
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
