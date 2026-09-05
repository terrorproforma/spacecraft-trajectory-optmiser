import hashlib, json
from pathlib import Path

ROOT = Path("/home/ubuntu/spacepdhcg/v1/results/gpu/current-head-9e75b47-h100")
index = json.loads((ROOT / "evidence-index.json").read_text())
indexed = set()
for rec in index["artifacts"]:
    p = ROOT / rec["path"]
    indexed.add(p.resolve())
    if not p.exists():
        print("MISSING", rec["path"]); continue
    if p.stat().st_size != rec["bytes"]:
        print("SIZE", rec["path"], rec["bytes"], p.stat().st_size)
    elif hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
        print("HASH", rec["path"])
excluded = {(ROOT / "evidence-index.json").resolve(), (ROOT / "evidence-index.json.sha256").resolve()}
actual = {p.resolve() for p in ROOT.rglob("*") if p.is_file()} - excluded
for p in sorted(actual - indexed):
    print("UNINDEXED", p.relative_to(ROOT))
for p in sorted(indexed - actual):
    print("INDEXED-BUT-ABSENT", p.relative_to(ROOT))
