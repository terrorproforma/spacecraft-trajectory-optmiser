set -u
cd /home/angus/worktrees/spacepdhcg-gtoc12
echo "== running python gtoc12 processes in WSL:"; ps -eo pid,ni,pcpu,etime,cmd | grep -E "spacepdhcg gtoc12|gtoc12" | grep -v grep | head
echo "== cluster_fleet_v8 contents:"; ls -la results/gtoc12/runs/cluster_fleet_v8 | head; 
[ -f results/gtoc12/runs/cluster_fleet_v8/run_report.json ] && python3 -c "import json;d=json.load(open('results/gtoc12/runs/cluster_fleet_v8/run_report.json'));print('status',d.get('status'),'wall',d.get('wall_seconds_total'));print('best',json.dumps(d.get('best'))[:400]);print('final_fleet',json.dumps(d.get('final_fleet'))[:400]);print('marks',json.dumps(d.get('budget_marks'))[:600])"
echo "== v8 tracked?"; git ls-files results/gtoc12/runs/cluster_fleet_v8 | wc -l
echo "== git diff docs (uncommitted) head:"; git diff -- docs/GTOC12_TRACK.md | head -80
echo "== hop_inflation_fit exists:"; ls -la results/gtoc12/hop_inflation_fit.json
echo "== leg_stats:"; ls results/gtoc12/leg_stats/