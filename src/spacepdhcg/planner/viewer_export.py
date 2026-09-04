"""Export a plan into the dataset format consumed by ``web/trajectory-viewer``.

The export writes ``data/trajectories.json`` and ``data/manifest.json`` (viewer schema
1.0.0, ``dataset_kind: planner-export``) and, when the viewer sources are available,
copies the static viewer files next to them so ``npm run check`` and ``npm run serve``
work directly inside the exported directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from spacepdhcg import resources
from spacepdhcg.planner.problem import FAMILY_INFO
from spacepdhcg.planner.result import PlanResult, json_safe

VIEWER_SCHEMA_VERSION = "1.0.0"
DATASET_KIND = "planner-export"
VIEWER_SOURCE_ENVIRONMENT = "SPACEPDHCG_VIEWER_SOURCE"
# Static files every bundle carries.  The ES-module graph rooted at ``app.js`` is discovered
# from the sources (``viewer_modules``) rather than listed here: the first real-GPU sweep
# found this tuple frozen at ``app.js``/``math.js`` after the viewer had grown ``gtoc12.js``,
# ``webgl.js``, ``kepler.js``, ``camera.js`` and ``dom.js``, so every export failed
# ``scripts/check.mjs`` and would not have loaded in a browser.
_VIEWER_FILES = ("index.html", "app.js", "styles.css", "package.json", "README.md")
# Script roots; their own import graphs are discovered the same way (``check.mjs`` imports
# ``palette.mjs`` for the ship-palette regeneration check), so nothing under ``scripts/`` is
# listed twice or forgotten.
_VIEWER_SCRIPTS = ("check.mjs", "serve.mjs")
_VIEWER_ENTRY = "app.js"
_RELATIVE_IMPORT = re.compile(r"""\bfrom\s+["'](\./[A-Za-z0-9_./-]+\.m?js)["']""")


def viewer_modules(source: Path, entry: str = _VIEWER_ENTRY) -> tuple[str, ...]:
    """Relative ES-module files reachable from ``entry`` inside the viewer ``source`` tree.

    Follows ``from "./x.js"`` imports transitively (the viewer has no bundler, so the browser
    resolves exactly these paths).  The entry itself is included first; a missing module raises
    so a broken import graph fails the export instead of producing a bundle that cannot load.
    """

    ordered: list[str] = []
    pending = [entry]
    while pending:
        name = pending.pop(0)
        if name in ordered:
            continue
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"viewer module {name!r} imported but missing under {source}")
        ordered.append(name)
        for match in _RELATIVE_IMPORT.finditer(path.read_text(encoding="utf-8")):
            relative = os.path.normpath(os.path.join(os.path.dirname(name), match.group(1)))
            relative = relative.replace(os.sep, "/")
            if relative not in ordered and relative not in pending:
                pending.append(relative)
    return tuple(ordered)


def viewer_scripts(source: Path) -> tuple[str, ...]:
    """``scripts/`` files a bundle needs: each root in ``_VIEWER_SCRIPTS`` plus its imports.

    Roots that the source tree lacks are skipped (older viewer copies); a root that is present
    but imports a missing module raises like ``viewer_modules``.
    """

    ordered: list[str] = []
    for script in _VIEWER_SCRIPTS:
        root = f"scripts/{script}"
        if not (source / root).is_file():
            continue
        for name in viewer_modules(source, root):
            if name not in ordered:
                ordered.append(name)
    return tuple(ordered)


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_stable(item) for item in value]
    return value


def _serialize(value: Any) -> str:
    return json.dumps(_stable(json_safe(value)), indent=2, allow_nan=False) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_viewer_source() -> Path | None:
    """``$SPACEPDHCG_VIEWER_SOURCE``, else ``web/trajectory-viewer`` of a source checkout.

    The static viewer is not packaged in the wheel, so an installed package without the
    override returns ``None`` and the export simply omits the viewer files.
    """

    override = os.environ.get(VIEWER_SOURCE_ENVIRONMENT)
    candidates = []
    if override:
        candidates.append(Path(override))
    root = resources.repository_root()
    if root is not None:
        candidates.append(root / "web" / "trajectory-viewer")
    for candidate in candidates:
        if (candidate / "app.js").is_file() and (candidate / "scripts" / "check.mjs").is_file():
            return candidate
    return None


def _viewer_metadata(result: PlanResult) -> dict[str, Any]:
    problem = result.document.get("problem", {})
    vehicle = problem.get("vehicle", {})
    family = result.family
    if family == "hcw":
        return {
            "scene_kind": "hcw",
            "axes": ["X radial outward", "Y along-track", "Z cross-track"],
            "body_label": "Local orbital frame; no globe",
            "body_radius": None,
            "radius_label": "Not applicable in relative HCW coordinates",
            "gravity_label": f"Mean motion {vehicle.get('mean_motion')} rad/s",
            "frame_choice": (
                "HCW is relative motion about a reference orbit. The scene shows an LVLH plane "
                "and Earthward -X direction, never a globe."
            ),
        }
    if family in {"powered_descent_3dof", "powered_descent_6dof"}:
        gravity = vehicle.get("gravity", [0.0, 0.0, 0.0])
        return {
            "scene_kind": "local-surface",
            "axes": ["X local", "Y local", "Z altitude"],
            "body_label": "Generic local planetary surface",
            "body_radius": 0,
            "radius_label": "Local tangent surface at model altitude Z = 0 m",
            "gravity_label": f"Uniform gravity {json.dumps(gravity)} m/s²",
            "frame_choice": (
                "The planner problem names no body or radius. A generic local tangent surface "
                "at physical Z = 0 is shown; no globe is inferred."
            ),
        }
    minimum_radius = float(vehicle.get("minimum_radius", 0.0))
    mu = vehicle.get("gravitational_parameter")
    return {
        "scene_kind": "central-body",
        "axes": ["X inertial", "Y inertial", "Z inertial"],
        "body_label": "Unnamed central body",
        "body_radius": minimum_radius,
        "radius_label": (
            f"Rendered r_min = {minimum_radius:,.0f} km constraint sphere; not a claimed surface"
        ),
        "gravity_label": f"μ = {mu} km³/s²; body unnamed",
        "frame_choice": (
            "The sphere is the planner minimum-radius constraint boundary. It is not labelled "
            "as a physical globe."
        ),
    }


def _series(times: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    points = [
        [float(time), *[float(value) for value in row[:3]]]
        for time, row in zip(times, states, strict=True)
    ]
    encoded = json.dumps(points, allow_nan=False).encode("utf-8")
    return {
        "point_count": len(points),
        "original_point_count": len(points),
        "selected_indices": list(range(len(points))),
        "original_sha256": _sha256(encoded),
        "points_txyz": points,
    }


def _summary(values: np.ndarray, names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    if values.size == 0:
        return {}
    return {
        name: {
            "minimum": float(np.min(values[:, index])),
            "mean": float(np.mean(values[:, index])),
            "maximum": float(np.max(values[:, index])),
        }
        for index, name in enumerate(names)
    }


def build_viewer_dataset(result: PlanResult, *, run_id: str | None = None) -> dict[str, Any]:
    """Construct the viewer dataset document for one plan result."""

    if "trajectory" not in result.document:
        raise ValueError("the plan result contains no trajectory to export")
    info = FAMILY_INFO[result.family]
    problem = result.document.get("problem", {})
    independent = result.independent_replay
    certificate = result.document.get("certificate", {})
    result_bytes = result.to_json().encode("utf-8")
    evidence_sha = _sha256(result_bytes)
    states = result.states
    controls = result.controls
    replay_states = result.replay_states if result.replay_states.size else states
    replay_times = result.replay_times if result.replay_times.size else result.times
    physical_path = independent.get("continuous_time_components_physical", {}) or {}
    dense_path_inf = max([float(value) for value in physical_path.values()] or [0.0])
    for value in np.concatenate(
        (states.reshape(-1), controls.reshape(-1), replay_states.reshape(-1))
    ):
        if not math.isfinite(float(value)):
            raise ValueError(
                "plan result contains non-finite trajectory values; refusing to export"
            )
    label = (
        "certified: converged and independently replayed within tolerance"
        if result.certified
        else f"not certified: {result.status} ({', '.join(result.failed_gates) or result.message})"
    )
    trajectory = {
        "trajectory_id": run_id or f"plan_{result.family}_{evidence_sha[:12]}",
        "family": result.family,
        "physical_family": info.physical_family,
        "frame": info.frame,
        "position_units": info.units["position"],
        "time_units": info.units["time"],
        "state_order": list(info.state_names),
        "control_order": list(info.control_names),
        "initial_state": json_safe(states[0]) if states.size else [],
        "terminal_target": json_safe(problem.get("terminal", {}).get("state", [])),
        "qualification": {"qualified": bool(result.certified), "label": label},
        "raw_evidence_sha256": evidence_sha,
        "source": {
            "run_id": run_id or f"plan-{evidence_sha[:16]}",
            "commit": str(result.document.get("source_commit") or "unknown"),
            "solver": result.backend_execution,
            "policy": str(result.document.get("backend", {}).get("device_policy", "")),
            "status": result.status,
            "requested_backend": result.requested_backend,
            "problem_name": problem.get("name", ""),
        },
        "replay": _series(replay_times, replay_states),
        "transcription": _series(result.times, states),
        "validation": {
            "finite": True,
            "dense_replay_terminal_inf": independent.get("terminal_position_error"),
            "dense_replay_physical_path_inf": dense_path_inf,
            "path_inf": independent.get("path_violation"),
            "terminal_inf": independent.get("terminal_residual"),
            "dynamics_inf": independent.get("dynamics_defect"),
            "continuous_time_inf": independent.get("continuous_time_violation"),
            "replay_parity": independent.get("replay_parity"),
            "certified": bool(result.certified),
            "failed_gates": result.failed_gates,
            "certificate_tolerance": certificate.get("tolerance"),
        },
        "viewer": _viewer_metadata(result),
        "controls_summary": _summary(controls, info.control_names),
        "mass_summary": (
            {
                "initial": float(states[0, -1]),
                "final": float(states[-1, -1]),
                "minimum": float(np.min(states[:, -1])),
            }
            if info.state_kinds[-1] == "mass" and states.size
            else None
        ),
        "attitude_summary": None,
        "path_constraint_bounds": json_safe(problem.get("vehicle", {})),
        "iteration_history": [
            {
                key: record.get(key)
                for key in (
                    "outer_iteration",
                    "phase",
                    "requested_tolerance",
                    "accepted",
                    "restoration_accepted",
                    "trust_action",
                    "trust_radius_before",
                    "trust_radius_after",
                    "reduction_ratio",
                )
            }
            for record in result.iterations
        ],
    }
    return {
        "schema_version": "1.0.0",
        "viewer_schema_version": VIEWER_SCHEMA_VERSION,
        "dataset_kind": DATASET_KIND,
        "title": "SpacePDHCG planner export",
        "generated_by": "spacepdhcg plan --export-viewer",
        "generated_by_commit": str(result.document.get("source_commit") or "unknown"),
        "imported_source_sha256": evidence_sha,
        "prohibitions": {
            "visual_interpolation_included": False,
            "aggregate_metric_path_fabrication": True,
        },
        "trajectories": [trajectory],
    }


def export_viewer_bundle(
    result: PlanResult, target: str | Path, *, viewer_source: Path | None = None
) -> Path:
    """Write the viewer data (and static viewer files when available) into ``target``."""

    target_path = Path(target)
    data_directory = target_path / "data"
    data_directory.mkdir(parents=True, exist_ok=True)
    dataset = build_viewer_dataset(result)
    data_bytes = _serialize(dataset).encode("utf-8")
    (data_directory / "trajectories.json").write_bytes(data_bytes)
    manifest = {
        "schema_version": "1.0.0",
        "dataset_kind": DATASET_KIND,
        "files": {"trajectories.json": {"bytes": len(data_bytes), "sha256": _sha256(data_bytes)}},
        "source": {
            "bytes": len(result.to_json().encode("utf-8")),
            "path_basename": "plan-result.json",
            "sha256": dataset["imported_source_sha256"],
        },
        "transform": (
            "Planner result converted losslessly to viewer txyz series; viewer metadata added "
            "by family."
        ),
    }
    (data_directory / "manifest.json").write_text(_serialize(manifest), encoding="utf-8")
    source = viewer_source or default_viewer_source()
    if source is not None:
        modules = viewer_modules(source) if (source / _VIEWER_ENTRY).is_file() else ()
        (target_path / "scripts").mkdir(exist_ok=True)
        for name in (*_VIEWER_FILES, *modules, *viewer_scripts(source)):
            if (source / name).is_file():
                (target_path / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target_path / name)
    return target_path
