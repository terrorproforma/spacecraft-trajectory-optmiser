import json, statistics
from pathlib import Path

p = Path.home() / "spacepdhcg/v2/results/gpu/h100-deferred-3373988/literature-rerun-5aabbfc/chari-2024-pd6-monte-carlo.json"
d = json.loads(p.read_text())
g = d["details"]["gpu_pure_qoco_native_pd6_fft"]
for size, rows in sorted(g.items(), key=lambda kv: int(kv[0])):
    conv = [r for r in rows if r["status"] == "converged"]
    print(f"batch {size}: {len(conv)}/{len(rows)} converged; fuel median {statistics.median(r['fuel_used'] for r in rows):.4f}; "
          f"max replay defect {max(r['replay_defect_inf'] for r in rows):.2e}; max path viol {max(r['max_path_violation'] for r in rows):.2e}; "
          f"wall median {statistics.median(r['wall_seconds'] for r in rows):.1f}s; total wall {sum(r['wall_seconds'] for r in rows):.0f}s")
m = d["measured"]["gpu_persistent_batch"]["pure_qoco_native_pd6_fft"]
print({k: v for k, v in m.items() if k not in ("preflight",)})
c = d["measured"]["cpu_independent_batch"]
print("cpu batch:", json.dumps(c)[:600])
