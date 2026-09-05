import json
from pathlib import Path

root = Path.home() / "spacepdhcg/gtoc12/results/gtoc12/runs"
for run in sorted(root.iterdir()):
    rep = run / "run_report.json"
    if not rep.exists():
        continue
    try:
        d = json.loads(rep.read_text())
    except Exception as e:
        print(run.name, "unreadable", e); continue
    m = d.get("master") or {}
    score = d.get("score_kg") or m.get("collected_kg") or (d.get("final_fleet") or {}).get("collected_kg")
    ships = m.get("ships") or d.get("ships") or (d.get("final_fleet") or {}).get("ships")
    print(f"{run.name:45s} score_kg={score} ships={ships} columns={d.get('columns')} status={d.get('status')}")
