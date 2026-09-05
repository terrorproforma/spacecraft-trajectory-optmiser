#!/bin/bash
cd "$HOME/spacepdhcg/gtoc12" || exit 1
echo "== cmd_fleet_master 1010-1060"; sed -n 1010,1060p src/spacepdhcg/gtoc12/cli.py
echo "== FleetColumn"; grep -n 'class FleetColumn' -A 30 src/spacepdhcg/gtoc12/cooperative.py | head -45
echo "== test helpers"; grep -n 'def _column\|def make_column\|FleetColumn(' tests/test_gtoc12_cooperative.py | head -5
first=$(grep -n 'def _column\|def make_column\|def _col' tests/test_gtoc12_cooperative.py | head -1 | cut -d: -f1)
[ -n "$first" ] && sed -n "${first},$((first+30))p" tests/test_gtoc12_cooperative.py
echo "== a solve_fleet_master test"; grep -n 'solve_fleet_master' tests/test_gtoc12_cooperative.py | head -3
