"""Check exact locally served display bytes; no solver or propagation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "results/lambda/2026-09-06/visualiser"
OUTPUT = ROOT / "results/local/2026-09-09/frontier-viewer-v629"
DATASET = "gtoc12-frontier-v629"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError("Assertions must be enabled")
    OUTPUT.mkdir(exist_ok=False)
    paths = ["index.html", "app.js", "gtoc12.js", "replay.js", "kepler.js"]
    paths += [f"data/{DATASET}/{p}" for p in (
        "manifest.json", "fleet.json", "compute.json", "checker-binding.json", "saved-node-provenance.json")]
    checks = []
    for name in paths:
        local = (VIEWER / name).read_bytes()
        with urlopen(f"http://127.0.0.1:4173/{name}", timeout=10) as response:
            actual = response.read()
            assert response.status == 200 and actual == local
            checks.append({"path": name, "status": response.status, "bytes": len(actual), "sha256": digest(actual)})
    app = (VIEWER / "app.js").read_text(encoding="utf-8")
    html = (VIEWER / "index.html").read_text(encoding="utf-8")
    assert f'const initialDataset = availableFleets.has(datasetParam) ? datasetParam : "{DATASET}"' in app
    assert f'<option value="{DATASET}" selected disabled>Latest verified fleet' in html
    for name in ("gtoc12-current-union-v629", "gtoc12-collect-composition-v807", "gtoc12-current-composition-v628"):
        assert name in app
    manifest = json.loads((VIEWER / f"data/{DATASET}/manifest.json").read_bytes())
    assert manifest["source"]["solution_sha256"] == "765cb7ef97d38926317dbbf4ae8700626dffec06c48af3d6f68dae47c154f3da"
    assert manifest["summary"]["ships"] == 23 and manifest["summary"]["replay_points"] == 11652
    report = {
        "passed": True, "utc": datetime.now(timezone.utc).isoformat(), "http_assets": checks,
        "default_dataset": DATASET, "solution_sha256": manifest["source"]["solution_sha256"],
        "scope": "Exact HTTP bytes and registration; no new propagation or GPU work",
        "prior_verification": {
            "node_tests": {"passed": 16, "failed": 0, "skipped": 1,
                           "skip_reason": "Historical fleet_master_v1 source export unavailable",
                           "files": ["tests/replay.test.mjs", "tests/geometry.test.mjs", "tests/gtoc12-import.test.mjs"]},
            "schema_check": "node scripts/check.mjs: exit 0; all installed datasets validate",
            "independent_read_only_review": "Gap rendering and 142 source/data hashes approved; unchanged 21 histories and original bindings verified",
            "browser": {"tab_id": "31", "url": f"http://127.0.0.1:4173/?dataset={DATASET}&ship=8&epoch=69807&preset=oblique&z=1",
                        "observed_accessibility_state": "Correct selected dataset, 23 ships/199 asteroids, source label, ship8 missing intervals2, Active WebGL2 on RTX5090",
                        "pixel_screenshot_review": False},
        },
    }
    source_paths = ["app.js", "gtoc12.js", "index.html", "replay.js", "scripts/import-gtoc12.mjs", "tests/replay.test.mjs"]
    report["reviewed_source"] = {name: digest((VIEWER / name).read_bytes()) for name in source_paths}
    (OUTPUT / "verify_http.py").write_bytes(Path(__file__).read_bytes())
    (OUTPUT / "http-and-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / ".gitattributes").write_text("* -text -whitespace\n", encoding="utf-8")
    print(json.dumps({"passed": True, "http_assets": len(checks), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
