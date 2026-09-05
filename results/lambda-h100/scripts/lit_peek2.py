import json, sys
from pathlib import Path

root = Path.home() / "spacepdhcg/v2/results/literature"
for name in ("acikmese-ploen-2007-pd3", "blackmore-2010-pd3-case1", "chari-2024-pd6-monte-carlo"):
    p = root / f"{name}.json"
    if not p.exists():
        print(name, "missing"); continue
    d = json.loads(p.read_text())
    m = d.get("measured", {})
    print("==", name, "status:", d.get("status"))
    for k, v in m.items():
        if "gpu" in k or k in ("scvx_qoco_fuel_used_kg", "fuel_used_kg", "published_fuel_used_kg"):
            print("  ", k, "=", v if not isinstance(v, (dict, list)) else json.dumps(v)[:400])
    gp = d.get("details", {}).get("gpu_persistent_batch")
    if gp:
        print("   gpu_persistent_batch:", json.dumps(gp)[:1200])
