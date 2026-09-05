set -u
mkdir -p /mnt/c/Users/Angus/h100work/src
cd /home/angus/worktrees/spacepdhcg-gtoc12
cp src/spacepdhcg/gtoc12/cli.py /mnt/c/Users/Angus/h100work/src/cli.py
cp docs/GTOC12_TRACK.md /mnt/c/Users/Angus/h100work/src/GTOC12_TRACK.md
git show ba9b764 -- src/spacepdhcg/gtoc12/cooperative.py > /mnt/c/Users/Angus/h100work/src/ba9b764.cooperative.diff
git show refs/h100/gtoc12-asteroid-mining -- src/spacepdhcg/gtoc12/cooperative.py > /mnt/c/Users/Angus/h100work/src/c4e2c31.cooperative.diff
git show 7d2e301 --stat | head -60 > /mnt/c/Users/Angus/h100work/src/7d2e301.stat
git show 7d2e301 -- src/spacepdhcg/gtoc12/cli.py > /mnt/c/Users/Angus/h100work/src/7d2e301.cli.diff
ls results/gtoc12/runs/ > /mnt/c/Users/Angus/h100work/src/wsl_runs.txt
du -sh results/gtoc12/runs/* >> /mnt/c/Users/Angus/h100work/src/wsl_runs.txt
git status --porcelain=v1 -- results/ | head >> /mnt/c/Users/Angus/h100work/src/wsl_runs.txt
echo "== ignored/untracked runs" >> /mnt/c/Users/Angus/h100work/src/wsl_runs.txt
git status --porcelain=v1 --ignored -- results/gtoc12/runs | head -40 >> /mnt/c/Users/Angus/h100work/src/wsl_runs.txt
cd /home/angus/worktrees/spacepdhcg-gtoc12-methods
git show f81e834 --stat | head -40 > /mnt/c/Users/Angus/h100work/src/f81e834.stat
ls results/gtoc12/runs/ > /mnt/c/Users/Angus/h100work/src/methods_runs.txt 2>&1
cd /home/angus/worktrees/spacepdhcg-gtoc12-arcs
git log --oneline -3 > /mnt/c/Users/Angus/h100work/src/arcs.log
echo done