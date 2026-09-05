set -u
cd /home/angus/worktrees/spacepdhcg-gtoc12
for f in clusters.py fleet.py pipeline.py; do cp src/spacepdhcg/gtoc12/$f /mnt/c/Users/Angus/h100work/src/$f; done
echo ok