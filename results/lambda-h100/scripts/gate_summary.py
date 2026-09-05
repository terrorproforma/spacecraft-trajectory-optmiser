import json, sys
from pathlib import Path

root = Path(sys.argv[1])
skip = {"artifacts", "steps", "checks", "source_commit", "source_tree", "local_only",
        "immutable_uri", "schema_version", "branch", "campaign_scope_id"}
for g in ["g0", "g1", "g2", "g3"]:
    s = json.loads((root / g / "summary.json").read_text())
    keep = {k: v for k, v in s.items() if k not in skip}
    print(f"== {g}")
    print(json.dumps(keep, indent=1, sort_keys=True)[:2500])
