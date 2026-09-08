"""Read-only inspection of the actual frozen local QOCO540 source."""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
path = Path("/home/angus/build-qoco-scaled-pool-v540/source/algebra/cuda/cudss_backend.cu")
lines = path.read_text().splitlines()
markers = ("DETERMINISTIC", "deterministic_option", "CUDSS_CONFIG_USE_SUPERPANELS", "IR_N_STEPS")
matches = [i for i, line in enumerate(lines) if any(marker in line for marker in markers)]
included = sorted({j for i in matches for j in range(max(0, i - 2), min(len(lines), i + 4))})
report = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "excerpt": [{"line": i + 1, "text": lines[i]} for i in included],
          "interpretation": "Configuration and retained vendor paths are hypotheses to isolate; no v598 factor/solve dump or workspace-reset comparison locates the cause."}
(here / "vendor-source-inspection.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
