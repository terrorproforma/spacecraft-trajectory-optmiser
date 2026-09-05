import json
from pathlib import Path

root = Path.home() / "spacepdhcg/v2/results/gpu/h100-deferred-3373988/literature-rerun-5aabbfc"
print((root / "status.txt").read_text())
for name in ("blackmore-2010-pd3-case1", "chari-2024-pd6-monte-carlo"):
    d = json.loads((root / f"{name}.json").read_text())
    print("==", name, "status:", d.get("status"), "support:", d.get("support"))
    m = d.get("measured", {})
    print("  measured keys:", sorted(m.keys())[:40])
    for k, v in m.items():
        if "gpu" in k.lower():
            print("  ", k, "=", json.dumps(v)[:600])
    det = d.get("details", {})
    print("  details keys:", list(det.keys()))
    for k, v in det.items():
        if "gpu" in k.lower() or "batch" in k.lower() or "qoco" in k.lower():
            print("  details.", k, "=", json.dumps(v)[:1500])
    for note in d.get("notes", []):
        if "gpu" in note.lower() or "GPU" in note:
            print("  note:", note[:300])
