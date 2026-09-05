#!/bin/bash
for w in v1 v2 gtoc12; do echo "== $w"; git -C "$HOME/spacepdhcg/$w" remote -v | head -2; git -C "$HOME/spacepdhcg/$w" for-each-ref --format='%(refname) %(objectname:short)' | grep -v tags | head -8; done
echo "== campaign verify step"; grep -n 'Verify\|verify\|official' "$HOME/s/gtoc12_campaign.sh" | head -12
echo "== manifests"; ls -la "$HOME/manifests/" 2>/dev/null
echo "== deferred dir size"; du -sh "$HOME/spacepdhcg/v2/results/gpu/h100-deferred-3373988" "$HOME/spacepdhcg/v2/results/gpu/h100-deferred-3373988/build-v2-gpu-deferred" 2>/dev/null
echo "== reseal dir size (excl big)"; du -sh "$HOME/spacepdhcg/v1/results/gpu/current-head-9e75b47-h100"; find "$HOME/spacepdhcg/v1/results/gpu/current-head-9e75b47-h100" -size +2M -type f | head
echo "== gtoc12 run dirs"; ls "$HOME/spacepdhcg/gtoc12/results/gtoc12/runs/" | tail -5; ls "$HOME/spacepdhcg/gtoc12/results/gtoc12/runs/cluster_fleet_h100_v1/fleet/"
