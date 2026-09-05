cd ~/spacepdhcg/gtoc12
echo "tracked h100_v1: $(git ls-files results/gtoc12/runs/cluster_fleet_h100_v1 | wc -l) fm_h100_v1: $(git ls-files results/gtoc12/runs/fleet_master_h100_v1 | wc -l)"
git ls-files results/gtoc12/runs/cluster_fleet_h100_v1 | sed 's#/[^/]*$##' | sort | uniq -c | sort -rn | head -5
git ls-files results/gtoc12/runs/fleet_master_h100_v1
git log --oneline -3 -- results/gtoc12/runs/cluster_fleet_h100_v1 | cat
echo "== ignored rules for results"; grep -n "results\|Result" .gitignore | head -20
git status --porcelain=v1 --ignored -- results/gtoc12/runs/cluster_fleet_h100_v1 | cut -c1-90 | head -5
git check-ignore -v results/gtoc12/runs/cluster_fleet_h100_v1/fleet/Result.txt results/gtoc12/runs/cluster_fleet_h100_v1/fleets/fleet_000_02ships/Result.txt results/gtoc12/runs/cluster_fleet_h100_v1/clusters/family_0001/ship_01/Result.txt results/gtoc12/runs/fleet_master_h100_v1/columns 2>&1 | head