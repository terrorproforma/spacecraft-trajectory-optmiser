"""Freeze a prespecified CPU-only comparison; no search or native execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
BASE = ROOT / "build/performance/depth-diverse-generation-v616"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    assert not (KIT / "selection.json").exists(), "Never overwrite frozen preparation"
    ready = read(BASE / "ready-manifest.json")
    for name, digest in ready["files"].items():
        assert sha(BASE / name) == digest, name
    inventory_path = BASE / "inputs/incumbent-inventory.json"
    inventory = read(inventory_path)
    selected = []
    (KIT / "inputs").mkdir()
    for ship in (23, 1, 4, 7, 10):
        row = next(row for row in inventory["routes"] if row["ship"] == ship)
        assert row["independent_inventory"] and row["cargo_matches_retained_Result"]
        source = Path(row["path"])
        assert sha(source) == row["sha256"]
        dest = KIT / "inputs" / f"ship-{ship:02}.json"
        shutil.copyfile(source, dest)
        selected.append({**row, "purpose": "target" if ship == 23 else "prespecified control"})
    shutil.copyfile(inventory_path, KIT / "inputs/incumbent-inventory.json")
    fit = ROOT / "results/gtoc12/hop_inflation_fit.json"
    assert sha(fit) == "b23afd0272c35197b14c2f0c85e5ff8b3a063e29867425835966de88e70d096e"
    assert read(fit) == read(BASE / "source/results/gtoc12/hop_inflation_fit.json")
    shutil.copyfile(fit, KIT / "inputs/hop_inflation_fit.json")
    write(KIT / "selection.json", {
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "routes": selected,
        "selection_before_replay": True,
        "models": ["v616_no_fit", "existing_fit_no_refit"],
        "prefixes": ["flat_deployment_proxy", "measured_deployment_prefix"],
        "implementations": ["baseline", "candidate"],
        "scalar_bridge_calls": 40,
        "GPU_calls": 0, "Lambert_calls": 0, "DP_solves": 0, "refinements": 0,
        "fit_sha256": sha(fit),
        "fit_historical_training_and_holdout": {
            key: value for key, value in read(fit).items()
            if key in ("commit", "train_sources", "train_hops", "holdout_sources", "holdout_hops")
        },
        "fit_scope": "Existing coefficients unchanged; these five controls are not claimed statistically held out from the historical fit.",
        "scope": "CPU bridge replay on archived epochs and saved Lambert DVs. No grid, pruning, native solver, trajectory recertification, fleet emission or promotion.",
    })
    old_files = read(BASE / "source-sha256.json")
    assert len(old_files) == 190
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    search_rel = "src/spacepdhcg/gtoc12/search.py"
    head_search = subprocess.check_output(["git", "show", f"HEAD:{search_rel}"], cwd=ROOT)
    assert head_search.replace(b"\r\n", b"\n") == (BASE / "source" / search_rel).read_bytes().replace(b"\r\n", b"\n")
    indices = {}
    for arm in ("baseline", "candidate"):
        for name, digest in old_files.items():
            source = BASE / "source" / name
            assert sha(source) == digest
            dest = KIT / "source" / arm / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        if arm == "candidate":
            shutil.copyfile(ROOT / search_rel, KIT / "source" / arm / search_rel)
        indices[arm] = {name: sha(KIT / "source" / arm / name) for name in old_files}
    changed = [name for name in old_files if indices["baseline"][name] != indices["candidate"][name]]
    assert changed == [search_rel]
    write(KIT / "source-sha256.json", indices)
    shutil.copyfile(ROOT / "tests/test_gtoc12_completion_costs.py", KIT / "test_gtoc12_completion_costs.py")
    shutil.copyfile(BASE / "output/report.json", KIT / "inputs/v616-report.json")
    shutil.copyfile(BASE / "profile.json", KIT / "inputs/v616-profile.json")
    root_report = ROOT / "build/performance/return-mass-bias-v619/report.json"
    assert sha(root_report) == "54aad2bb302738883ba9aedfcd1def85b4277eba40d48daf14d07056b7e08048"
    shutil.copyfile(root_report, KIT / "inputs/root-return-components.json")
    write(KIT / "provenance.json", {
        "source_base": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7",
        "workspace_HEAD_at_freeze": head,
        "baseline_search_matches_workspace_HEAD_after_newline_normalization": True,
        "changed_source_files": changed,
        "baseline_source_files": 190, "candidate_source_files": 190,
        "v616_ready_sha256": sha(BASE / "ready-manifest.json"),
        "source_indices_sha256": sha(KIT / "source-sha256.json"),
        "selection_sha256": sha(KIT / "selection.json"),
        "inputs": {p.name: sha(p) for p in sorted((KIT / "inputs").iterdir())},
        "focused_test_sha256": sha(KIT / "test_gtoc12_completion_costs.py"),
        "native_libraries": "Forbidden. v616 profile retained only as settings/data lineage, not runtime evidence.",
    })
    print(json.dumps({"prepared": True, "selection_sha256": sha(KIT / "selection.json"), "source_files_per_arm": 190}))


if __name__ == "__main__":
    main()
