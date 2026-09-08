"""Freeze only the reviewed v606 source, profiles and prefix inputs for v607b."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "truth-set-v606"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ready = json.loads((OLD / "ready-manifest.json").read_text())
assert (
    digest(OLD / "ready-manifest.json")
    == "7e87c5360efaee14ddaec94602ac4132b610073d782118d6e4cb122312c4d2cc"
)
for name, value in ready["files"].items():
    assert digest(OLD / name) == value
manifest = json.loads((OLD / "source-sha256.json").read_text())
for name, value in manifest.items():
    source = OLD / "source" / name
    destination = ROOT / "source" / name
    assert digest(source) == value
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists()
    shutil.copy2(source, destination)
for name in ("source-sha256.json", "profiles.json", "preparation.json", "fixed_refine.py"):
    shutil.copy2(OLD / name, ROOT / name)
inputs = ROOT / "inputs"
inputs.mkdir(exist_ok=False)
origins = {}


def copy(source, relative):
    target = inputs / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    origins[relative] = {"origin": str(source), "sha256": digest(target)}


copy(OLD / "inputs/Result.txt", "Result.txt")
copy(OLD / "output/report.json", "v606-report.json")
copy(OLD / "ready-manifest.json", "v606-ready-manifest.json")
copy(ROOT.parent / "return-window-rescue-v607/plan.json", "v607-plan.json")
copy(ROOT.parent / "return-coast-review-v607/report.json", "coast-review.json")
for short, case in (("control", "control_ship_07"), ("probe", "ship_07_replace_15206_with_3150")):
    directory = OLD / "output/cases" / case
    for name in ("prescription.json", "refinement.json"):
        copy(directory / name, f"{short}/{name}")
    for index in range(17):
        copy(directory / f"leg-{index:02d}.json", f"{short}/leg-{index:02d}.json")
        if short == "control" or index < 16:
            copy(
                directory / f"leg-{index:02d}-solution.npz", f"{short}/leg-{index:02d}-solution.npz"
            )
copy(OLD / "output/cases/control_ship_07/route/Result.txt", "control/Result.txt")
copy(OLD / "output/cases/control_ship_07/verification.json", "control/verification.json")
(ROOT / "inputs-sha256.json").write_text(json.dumps(origins, indent=2) + "\n")
print({"source_files": len(manifest), "input_files": len(origins), "GPU_calls": 0})
