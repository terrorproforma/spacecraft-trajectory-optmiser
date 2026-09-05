set -u
cd /home/angus/worktrees/spacepdhcg-gtoc12
for f in bundles.py search.py collectdp.py cooperative.py memory.py returnsweep.py archive.py; do cp src/spacepdhcg/gtoc12/$f /mnt/c/Users/Angus/h100work/src/$f 2>/dev/null || echo "no $f"; done
ls src/spacepdhcg/gtoc12/
git ls-files results/gtoc12 | grep -v runs/ 
echo "== tracked files per run dir:"; for d in results/gtoc12/runs/*/; do echo "$d $(git ls-files $d | wc -l)"; done