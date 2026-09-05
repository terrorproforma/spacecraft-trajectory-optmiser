set -u
cd /home/angus/worktrees/spacepdhcg-gtoc12
ls tests | grep gtoc12
echo "== family_clusters / ClusterBands usage in tests"
grep -n "family_clusters\|ClusterBands\|rank_families\|cluster_radius\|collect_epoch_families" tests/test_gtoc12_*.py | head -30
echo "== catalogue fixture"
grep -n "def .*catalogue\|load_catalogue\|synthetic" tests/conftest.py tests/test_gtoc12_clusters.py 2>/dev/null | head -20
mkdir -p /mnt/c/Users/Angus/h100work/src/tests
cp tests/test_gtoc12_clusters.py /mnt/c/Users/Angus/h100work/src/tests/ 2>/dev/null
cp tests/conftest.py /mnt/c/Users/Angus/h100work/src/tests/ 2>/dev/null
grep -n "BUDGET_MARKS" -r src tests | head