"""Complete the newly created v599 fixture with its published scoring audit."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
inputs = root / "inputs"
target = inputs / "fresh-verification.json"
if target.exists():
    raise FileExistsError(target)
shutil.copy2(
    root.parents[2] / "results/local/2026-09-09/orphan-recovery-v595/audit/fresh-verification.json",
    target,
)
manifest = json.loads((inputs / "sha256.json").read_text())
manifest[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
(inputs / "sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
record = json.loads((root / "preparation.json").read_text())
record["input_manifest_sha256"] = hashlib.sha256((inputs / "sha256.json").read_bytes()).hexdigest()
(root / "preparation.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
