#!/bin/bash
# Status of the GTOC12 H100 v2 campaign: bash ~/s/v2_status.sh
cd "$HOME/spacepdhcg/gtoc12"
echo "== $(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg) mem_used=$(free -g | awk '/Mem:/{print $3}')G gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
cat "$HOME/logs/gtoc12-v2-RESULT"
echo "processes: $(pgrep -fc 'spacepdhcg gtoc12 cluster-fleet') cluster-fleet, $(pgrep -fc 'spacepdhcg gtoc12 joint-itinerary') joint, $(pgrep -fc 'spacepdhcg gtoc12 fleet-master') master"
echo "-- orchestrator:"; tail -4 "$HOME/logs/gtoc12_v2_campaign.sh.log" | cut -c1-220
echo "-- cluster_fleet_h100_v2: families done $(grep -c '"family"' "$HOME/logs/gtoc12-cluster_fleet_h100_v2.log" 2>/dev/null)"
grep '"family"' "$HOME/logs/gtoc12-cluster_fleet_h100_v2.log" 2>/dev/null | tail -3 | cut -c1-260
python3 - <<'PY' 2>/dev/null
import json, pathlib
p = pathlib.Path("results/gtoc12/runs/cluster_fleet_h100_v2/run_report.json")
if p.exists():
    d = json.load(open(p))
    fl = d.get("fleets", [])
    print(f"   wall {d.get('wall_seconds_total',0)/60:.1f} min, bundles {len(d.get('bundles',[]))}, pss_peak {d.get('memory_total_pss_peak_mb')} MB, verified fleets {len(fl)}")
    if fl: f = fl[-1]; print(f"   incumbent {f['score_kg']:.1f} kg {f['ships']} ships {f['asteroids']} asteroids avg {f['average_collected_kg']:.1f} at {f['elapsed_seconds']/60:.0f} min")
    marks = {m['elapsed_seconds']//1800*30: m['score_kg'] for m in fl}
    parts = d.get("instance", {}).get("partitions")
    if parts: print("   partitions:", [(x['name'], x['families']) for x in parts])
    ships = [s for b in d.get("bundles", []) for s in b.get("ships", [])]
    m = [s["collected_kg"] for s in ships]
    print(f"   ships {len(m)} >=550 {sum(x>=550 for x in m)} >=600 {sum(x>=600 for x in m)} >=650 {sum(x>=650 for x in m)} >=700 {sum(x>=700 for x in m)} max {max(m) if m else 0:.1f}")
    st = [b.get("stopped") for b in d.get("bundles", [])]
    print("   stopped:", {s: st.count(s) for s in set(st)})
for run in ("joint_itinerary_h100_v1", "joint_itinerary_h100_v2"):
    q = pathlib.Path("results/gtoc12/runs", run, "ships.jsonl")
    if q.exists():
        rows = [json.loads(l) for l in q.read_text().splitlines() if l.strip()]
        imp = [r for r in rows if r.get("gain_kg", 0) > 1e-6]
        before = [r["before_kg"] for r in rows]; after = [r["after_kg"] for r in rows]
        print(f"-- {run}: {len(rows)} ships done ({rows[-1].get('done')}/{rows[-1].get('total')} tasks), improved {len(imp)}, gain {sum(r['gain_kg'] for r in rows):.1f} kg; >=600 before {sum(b>=600 for b in before)} after {sum(a>=600 for a in after)}; >=650 after {sum(a>=650 for a in after)}; best gain {max((r['gain_kg'] for r in rows), default=0):.1f}")
PY
echo "-- joint v1 log:"; tail -2 "$HOME/logs/gtoc12-joint_itinerary_h100_v1.log" 2>/dev/null | cut -c1-200
[ -f "$HOME/logs/gtoc12-fleet_master_h100_v2.log" ] && { echo "-- master log:"; tail -3 "$HOME/logs/gtoc12-fleet_master_h100_v2.log" | cut -c1-200; }
echo "-- tests: $(tail -c 120 $HOME/logs/gtoc12-tests-merged.log | tr '\n' ' ')"