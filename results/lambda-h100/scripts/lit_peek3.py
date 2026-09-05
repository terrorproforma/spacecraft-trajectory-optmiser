import json
from pathlib import Path

root = Path.home() / "spacepdhcg/v2/results/literature"
b = json.loads((root / "blackmore-2010-pd3-case1.json").read_text())
print("== blackmore keys with gpu/defer:")
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o[:3]):
            walk(v, f"{path}[{i}]")
    else:
        s = str(o)
        if "gpu" in path.lower() or "defer" in s.lower() or "gpu" in s.lower():
            print(" ", path, "=", s[:300])
walk(b)
c = json.loads((root / "chari-2024-pd6-monte-carlo.json").read_text())
gp = c["details"]["gpu_persistent_batch"]
print("== chari pure_qoco_native_pd6_fft:")
print(json.dumps(gp.get("pure_qoco_native_pd6_fft"), indent=1)[:3000])
print("== chari measured summary:")
print(json.dumps({k: v for k, v in c.get("measured", {}).items() if not isinstance(v, (list, dict))}, indent=1)[:1500])
