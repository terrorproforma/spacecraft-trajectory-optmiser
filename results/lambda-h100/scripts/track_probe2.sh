cd ~/spacepdhcg/gtoc12
echo "== v7 tracked file kinds"; git ls-files results/gtoc12/runs/cluster_fleet_v7 | sed 's#.*/##' | sort | uniq -c
echo "== joint_itinerary_v2 kinds"; git ls-files results/gtoc12/runs/joint_itinerary_v2 | sed 's#.*/##' | sort | uniq -c
echo "== return_sweep_v2 kinds"; git ls-files results/gtoc12/runs/return_sweep_v2 | sed 's#.*/##' | sort | uniq -c
echo "== fm_v7"; git ls-files results/gtoc12/runs/fleet_master_v7
echo "== docs section 7 anchors"; grep -n "^## 8\|^### 7\|^\*\*Eighth\|joint_itinerary_v2\b" docs/GTOC12_TRACK.md | head
bash ~/s/v2_status.sh 2>&1 | tail -12