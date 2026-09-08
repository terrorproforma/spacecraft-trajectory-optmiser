"""Copy immutable v595 inputs only; no imports from the solver or GPU access."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGET = HERE / "inputs"
TARGET.mkdir(exist_ok=False)
original = REPO / "results/lambda/2026-09-08/fleet-objective-v229/incumbent-sources"
promoted = REPO / "build/performance/orphan-recovery-v595/output-compatible/candidates/ship_15_attempt_01/route/route_summary.json"
sources = {f"ship-{ship:02d}.json": promoted if ship == 15 else original / f"ship-{ship:02d}.json"
           for ship in range(1, 24)}
sources["Result.txt"] = REPO / "results/local/2026-09-09/orphan-recovery-v595/Result.txt"
hashes = {}
for name, source in sources.items():
    target = TARGET / name
    shutil.copy2(source, target)
    hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
assert hashes["Result.txt"] == "33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8"
(TARGET / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
(TARGET / "provenance.json").write_text(json.dumps({k: str(v.relative_to(REPO)) for k, v in sources.items()}, indent=2) + "\n")
print(json.dumps({"files": len(hashes), "selected_summary_sha256": hashes["ship-15.json"],
                  "incumbent_sha256": hashes["Result.txt"]}, indent=2))
