"""Viewer export: dataset structure and the viewer's own ``npm run check`` round trip."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from spacepdhcg.planner import PlanOptions, load_problem, plan
from spacepdhcg.planner.viewer_export import (
    build_viewer_dataset,
    export_viewer_bundle,
    viewer_modules,
)

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "web" / "trajectory-viewer"

pytestmark = pytest.mark.usefixtures("planner_native_library")


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    output = tmp_path_factory.mktemp("viewer-export")
    document = load_problem(ROOT / "examples" / "planner" / "powered_descent_3dof.json")
    document["horizon"]["intervals"] = 16
    result = plan(document, PlanOptions(backend="cpu_reference", output_directory=output / "plan"))
    bundle = export_viewer_bundle(result, output / "viewer", viewer_source=VIEWER)
    dataset = json.loads((bundle / "data" / "trajectories.json").read_text(encoding="utf-8"))
    return bundle, dataset


def test_dataset_matches_the_viewer_contract(exported: tuple[Path, dict]) -> None:
    bundle, dataset = exported
    assert dataset["dataset_kind"] == "planner-export"
    assert dataset["viewer_schema_version"] == "1.0.0"
    assert dataset["prohibitions"]["visual_interpolation_included"] is False
    [trajectory] = dataset["trajectories"]
    assert trajectory["family"] == "powered_descent_3dof"
    assert trajectory["viewer"]["scene_kind"] == "local-surface"
    assert trajectory["qualification"]["qualified"] is True
    replay = trajectory["replay"]
    assert replay["point_count"] == 16 * 10 + 1 == len(replay["points_txyz"])
    assert replay["selected_indices"] == list(range(replay["point_count"]))
    assert all(len(point) == 4 for point in replay["points_txyz"])
    assert replay["points_txyz"][0][1:] == trajectory["initial_state"][:3]
    assert trajectory["transcription"]["point_count"] == 17
    manifest = json.loads((bundle / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["files"]["trajectories.json"]["bytes"]
        == (bundle / "data" / "trajectories.json").stat().st_size
    )
    for name in (
        "index.html",
        "app.js",
        "math.js",
        "styles.css",
        "package.json",
        "scripts/check.mjs",
    ):
        assert (bundle / name).is_file(), name


def test_export_carries_the_whole_viewer_module_graph(exported: tuple[Path, dict]) -> None:
    # The bundle is a self-contained viewer: every ES module reachable from app.js and every
    # non-data file scripts/check.mjs reads must be present (the first real-GPU sweep found
    # gtoc12.js/webgl.js/kepler.js/camera.js/dom.js missing, so `node scripts/check.mjs` and a
    # browser load both failed on each export).
    bundle, _ = exported
    modules = viewer_modules(VIEWER)
    assert set(modules) >= {"app.js", "math.js", "gtoc12.js", "webgl.js", "kepler.js", "camera.js"}
    for name in modules:
        assert (bundle / name).is_file(), name
    check_source = (VIEWER / "scripts" / "check.mjs").read_text(encoding="utf-8")
    read_targets = set(re.findall(r'read\("([^"]+)"\)', check_source))
    assert read_targets, "check.mjs read() calls not found"
    for name in sorted(read_targets):
        if name.startswith("data/"):
            continue
        assert (bundle / name).is_file(), f"check.mjs reads {name} but the export lacks it"


def test_viewer_modules_rejects_a_missing_import(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text('import { x } from "./gone.js";\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"gone\.js"):
        viewer_modules(tmp_path)


def test_export_refuses_results_without_trajectories() -> None:
    from spacepdhcg.planner.result import PlanResult

    with pytest.raises(ValueError, match="no trajectory"):
        build_viewer_dataset(
            PlanResult(
                document={"status": {"code": "invalid_problem"}, "problem": {"family": "hcw"}}
            )
        )


def test_viewer_check_accepts_the_export(exported: tuple[Path, dict]) -> None:
    bundle, _ = exported
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable; the viewer's npm run check cannot execute here")
    completed = subprocess.run(
        [node, "scripts/check.mjs"],
        cwd=bundle,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "planner-export" in completed.stdout


def test_viewer_check_still_accepts_the_archive() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is unavailable; the viewer's npm run check cannot execute here")
    completed = subprocess.run(
        [node, "scripts/check.mjs"],
        cwd=VIEWER,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "archive" in completed.stdout
