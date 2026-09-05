#!/bin/bash
cd "$HOME/spacepdhcg/gtoc12" || exit 1
start=$(grep -n 'def cmd_fleet_master' src/spacepdhcg/gtoc12/cli.py | cut -d: -f1)
echo "cmd_fleet_master at $start"
sed -n "${start},$((start+120))p" src/spacepdhcg/gtoc12/cli.py | grep -n 'columns\|exists\|cache\|is_file\|json\|price\|Pool\|workers' | head -40
echo "== campaign fleet-master invocation"; sed -n 36,48p "$HOME/s/gtoc12_campaign.sh"
echo "== columns dir sample"; ls results/gtoc12/runs/fleet_master_h100_v1/columns/cluster_fleet_h100_v1__clusters__family_0000/ | head
echo "== columns count"; ls results/gtoc12/runs/fleet_master_h100_v1/columns | wc -l
echo "== recursion limit"; .venv/bin/python -c 'import sys; print(sys.getrecursionlimit(), sys.version)'
echo "== tests touching solve_fleet_master"; grep -ln 'solve_fleet_master' tests/*.py
echo "== ship_count def"; sed -n 250,262p src/spacepdhcg/gtoc12/cooperative.py
echo "== solve_fleet_master signature"; grep -n 'def solve_fleet_master' -A 14 src/spacepdhcg/gtoc12/cooperative.py
echo "== imports"; sed -n 1,40p src/spacepdhcg/gtoc12/cooperative.py | grep -n '^import\|^from'
