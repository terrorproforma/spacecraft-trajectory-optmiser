#!/bin/bash
cd "$HOME/spacepdhcg/v2" || exit 1
out=results/gpu/h100-deferred-3373988
echo "== items"; cat "$out/items.tsv" 2>/dev/null
echo "== status"; cat "$out/status.txt" 2>/dev/null
echo "== log tail"; tail -3 "$HOME/logs/v2_deferred.sh.log"
echo "== failures"
awk -F'\t' '$2=="FAIL"{print $1}' "$out/items.tsv" 2>/dev/null | while read -r id; do
  echo "--- $id"; head -25 "$out/$id.log"
done
