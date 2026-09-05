set -u
cd /home/angus/worktrees/spacepdhcg-gtoc12
sed -n 1240,1290p tests/test_gtoc12_bundles.py
echo "== fixture"; grep -rn "def catalogue" tests/*.py | head -5
sed -n 1,40p tests/test_gtoc12_bundles.py