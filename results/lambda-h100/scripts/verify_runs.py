"""Official GTOC12_Verify over every Result.txt below results/gtoc12/runs/<run> for the given runs."""
import json, pathlib, sys
from spacepdhcg.gtoc12.official import run_official_verifier
rows = []
for run in sys.argv[1:]:
    root = pathlib.Path("results/gtoc12/runs", run)
    if not root.exists():
        print("missing run", run); continue
    for path in sorted(root.rglob("Result.txt")):
        s = run_official_verifier(path).summary()
        rows.append({"run": run, "path": str(path), **s})
ok = sum(r["ok"] for r in rows)
print(f"official verifier: {ok}/{len(rows)} Result.txt files pass")
by_run = {}
for r in rows:
    b = by_run.setdefault(r["run"], {"total": 0, "ok": 0, "fleet_files": []})
    b["total"] += 1; b["ok"] += int(r["ok"])
    if "/fleet/" in r["path"] or "/fleets/" in r["path"]: b["fleet_files"].append((r["path"], r["ok"], r.get("total_mass_kg")))
for run, b in by_run.items():
    print(run, b["ok"], "/", b["total"], "pass;", sum(1 for f in b["fleet_files"] if f[1]), "/", len(b["fleet_files"]), "fleet files pass")
bad = [r for r in rows if not r["ok"]]
for r in bad[:15]: print("FAIL", r["path"], {k: r[k] for k in r if k not in ("path", "run")})
fleet = [r for r in rows if r["path"].endswith("fleet_master_h100_v2/fleet/Result.txt")]
print("master fleet:", fleet)
pathlib.Path("results/gtoc12/runs/fleet_master_h100_v2/official_verification.json").write_text(json.dumps({"passed": ok, "total": len(rows), "rows": rows}, indent=1))
# per-ship diagnostic files of cooperative members fail Error803 by construction; the fleet files must all pass
fleet_ok = all(f[1] for b in by_run.values() for f in b["fleet_files"])
raise SystemExit(0 if fleet_ok and fleet and fleet[0]["ok"] else 1)