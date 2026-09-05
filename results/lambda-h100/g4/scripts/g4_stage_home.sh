#!/bin/bash
# Stage the compact G4 evidence (capability, checkpoint snapshot, deadline-test results, first group
# records, scripts, status) into ~/evidence-home-g4 as one tarball for scp to the Windows workspace.
set -uo pipefail
source /home/ubuntu/s/g4env-h100.sh
dst=/home/ubuntu/evidence-home-g4
rm -rf "$dst"; mkdir -p "$dst/fix-verification-${head7}" "$dst/campaign" "$dst/scripts" "$dst/logs"
cp /home/ubuntu/g4/STATUS.txt /home/ubuntu/g4/qoco-pin-${head7}-h100.txt "$dst/" 2>/dev/null
cp "$capability" "$dst/"
# fix verification: everything except the raw stdout of the repro is small
cp -r /home/ubuntu/g4/fix-verification-${head7}/. "$dst/fix-verification-${head7}/"
# consistent checkpoint snapshot via the sqlite backup API (the worker may be writing)
python3 - "$campaign/checkpoint.sqlite3" "$dst/campaign/checkpoint.sqlite3" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst); dst.close(); src.close()
print("checkpoint snapshot ok")
PY
cp "$campaign/hardware.txt" "$campaign/journal.jsonl" "$dst/campaign/" 2>/dev/null
cp "$campaign"/environment.txt "$campaign"/inputs.sha256 "$dst/campaign/" 2>/dev/null
# first group records: ordinal 0 (first IPM group), 5 (first IPM twin), 66 + 67 (first PDHCG groups),
# 73 (first PDHCG censoring twin, the RTX 5090 defect coordinate) - whichever have finished.
mkdir -p "$dst/campaign/runs"
python3 - "$g4logs/worker.err" "$campaign" "$dst/campaign/runs" ${G4_HOME_ORDINALS:-0 5 66 67 73} <<'PY'
import json, shutil, sys
from pathlib import Path
events, campaign, dst, *ordinals = sys.argv[1:]
want = {int(o) for o in ordinals}
for line in open(events):
    if '"group_finished"' not in line:
        continue
    e = json.loads(line)
    if e["ordinal"] in want:
        src = Path(campaign) / "runs" / e["group_id"]
        out = Path(dst) / f"ordinal-{e['ordinal']:03d}-{e['group_id']}"
        if src.is_dir() and not out.exists():
            shutil.copytree(src, out)
            print("copied ordinal", e["ordinal"], e["disposition"], round(e["elapsed_seconds"], 1), "s")
PY
cp "$g4logs/worker.err" "$dst/logs/worker-events.jsonl" 2>/dev/null
cp /home/ubuntu/g4/progress/*.txt "$dst/" 2>/dev/null
cp /home/ubuntu/s/g4env-h100.sh /home/ubuntu/s/g4-*.sh /home/ubuntu/s/host-pmon-linux.sh /home/ubuntu/s/g4_build_verify.sh /home/ubuntu/s/ordinal73_repro.py /home/ubuntu/s/g4_stage_home.sh "$dst/scripts/"
cp "$g4logs"/*.log /home/ubuntu/logs/g4_build_verify.sh.log "$dst/logs/" 2>/dev/null
"$root/.venv/bin/python" "$root/scripts/gpu/run_g4_campaign.py" status --claim-core --amendment "$root/benchmarks/g4_claim_core_amendment_v1_2.json" --repository "$root" --campaign "$campaign" > "$dst/campaign/status.json" 2>/dev/null
date -u +%FT%TZ > "$dst/STAGED_UTC.txt"
tar -C /home/ubuntu -czf /home/ubuntu/evidence-home-g4.tar.gz evidence-home-g4
du -sh "$dst" /home/ubuntu/evidence-home-g4.tar.gz; find "$dst" -type f | wc -l
