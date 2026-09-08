"""Package existing CPU-only route-family diagnosis and exact frozen source evidence."""

import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIGIN = REPO / "build/performance/route-family-cap-audit-v611"
KIT = REPO / "build/performance/coupled-fuel-search-v611"
DEST = REPO / "results/local/2026-09-09/route-family-cap-audit-v611"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(origin, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    assert sha(origin) == sha(target)


def main():
    if DEST.exists():
        raise FileExistsError("Preserve existing publication")
    DEST.mkdir(parents=True)
    report, conclusion = read(ORIGIN / "report.json"), read(ORIGIN / "conclusion.json")
    assert report["certified_closed_records"] == 1374
    assert report["unique_certified_closed_plans"] == 1282
    assert conclusion["actual_fleet_collected_miners"] == 195
    for name in ("inventory.py", "followup.py", "report.json", "conclusion.json"):
        copy(ORIGIN / name, DEST / name)
    copy(Path(__file__), DEST / "package.py")
    pinned = read(KIT / "source-sha256.json")
    files = {}
    for name in ("search.py", "bundles.py", "cooperative.py", "collectdp.py", "constants.py"):
        relative = "src/spacepdhcg/gtoc12/" + name
        origin = KIT / "source" / relative
        assert sha(origin) == pinned[relative]
        copy(origin, DEST / "frozen-source" / name)
        files[relative] = {"sha256": pinned[relative], "published_path": "frozen-source/" + name}
    write(DEST / "source-provenance.json", {
        "commit": "f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7", "files": files,
        "source_manifest_sha256": sha(KIT / "source-sha256.json"),
        "source_kind": "Exact published Git archive from immutable v611 kit; no live worktree mixing",
        "ready_manifest_sha256": sha(KIT / "ready-manifest.json"),
        "retained_Result_sha256": conclusion["retained_Result_sha256"],
        "retained_Result_location": "../coupled-fuel-search-v611/inputs.tar.gz member Result.txt",
    })
    settings = []
    for relative in (
        "results/gtoc12/runs/cluster_fleet_v7/run_report.json",
        "results/gtoc12/runs/cluster_fleet_h100_v2/run_report.json",
        "results/gtoc12/runs/cluster_fleet_v10/run_report.json",
        "build/performance/family-gpu-v588/output/run_report.json",
    ):
        origin = REPO / relative
        data = read(origin)
        settings.append({"path": relative, "sha256": sha(origin), "settings": data["settings"]})
        assert data["settings"]["max_deploys"] == 10
    write(DEST / "retained-campaign-settings.json", settings)
    (DEST / ".gitattributes").write_text("# Preserve exact evidence bytes and indexed hashes.\n* -text\n")
    (DEST / "README.md").write_text("""# Route-family cap audit v611

**Nine- and ten-miner routes are not excluded by a global eight-miner limit.** Frozen route generation and collection-DP defaults both permit10 deployments. The inclusive expansion loop attempts depths2 through10. Cluster settings propagate that limit, and four retained campaign configurations also used10. The fleet master has asteroid/dependency conflicts and the official raw-mass ship-count rule; it does not impose an eight-miner limit per route.

The exact retained23-ship Result contains196 deployments and195 collections. Twelve ships deploy9 miners and11 deploy8;11 ships collect9 and12 collect8. Its independently and officially checked raw mass is14,051.854893908598kg, with fixed-bonus weighted score12,810.135953048577kg. These are distinct quantities. The current fleet already demonstrates nine-miner operation.

The read-only archive inventory examined2,469 `route_summary.json` files under `results/gtoc12/runs`;2,287 included deployment/collection event tables. It found **1,374 records marked certified and independently closed**, representing **1,282 unique plan JSONs** after canonical-plan hashing. These are not1,374 independent solutions. The nine-miner subset contains175 records /158 unique plan JSONs. No independently closed certified ten-miner plan was found in this inspected pool. Forty records deploy10 miners, but they collect only5–9 each. Their complete fleet dependencies were not re-certified in this audit.

**Archived certification was not recomputed.** The inventory reads retained flags, events, cargo and exact file hashes. Packed-only artifacts and newer files outside the named archive subtree were not searched. Differences in full plan JSON, including numeric proxy fields, count as different plan hashes; this is not a count of unique asteroid orders or physical trajectories.

## Specific selection restriction

`frozen-source/bundles.py:657–680` contains `refine_candidates`. It traverses candidates in beam order, keeps only the first chain for each `(Earth target, departure, time of flight)`, puts previously certified Earth legs first, then returns at most `refine_top` candidates (default2). A longer9/10-miner alternative sharing that Earth leg can therefore be discarded before low-thrust refinement even when the generator produced it. This is a possible selection bottleneck, not evidence that a particular discarded ten-miner route would certify.

Other finite-search restrictions include beam width24, at most2 variants per deployed set,8 per first asteroid, estimated collection reserve and return feasibility pruning, and disabled joint asteroid insertion. These restrictions need measured attrition before increasing budgets. Simply changing `max_deploys` from8 to10 would not change the existing default or the inspected campaign settings.

The recommended next comparison uses identical generated candidates and a fixed refinement budget: current selection versus selection that preserves miner-count diversity per certified Earth leg. Include separate9/10-miner deploy-and-collect chains, rank using complete collection/return cost and true weighted contribution, and preserve all raw ship-rule and physical/final-fleet gates. This expands mission route families rather than repeating the fixed eight-miner timing search. No new search, refinement or GPU call was prepared or launched by this audit.

## Evidence

- `report.json`: full inventory, every retained route-record path/hash, duplicate-aware counts and archived incumbent summaries.
- `conclusion.json`: independently counted exact Result event pairs, ten-deployer collection counts, source locations and scoped recommendation.
- `inventory.py` and `followup.py`: CPU-only readers. Citation line numbers in the conclusion were corrected after checking the frozen files; numeric inventory was unchanged.
- `frozen-source/` and `source-provenance.json`: five exact source files from commit `f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7`, verified against the immutable v611 source manifest.
- `retained-campaign-settings.json`: exact settings subsets and source report hashes, including the historical H100 campaigns and the local v588 family pilot.
- The exact retained Result is preserved in [the adjacent coupled-search bundle](../coupled-fuel-search-v611/README.md), `inputs.tar.gz`, member `Result.txt`; SHA256 `33701ef2b797f44ef2e8aa50a2dd59cb238df9aab604e7cef25ead6cbdd669e8`.
- `sha256.json`: every publication file except itself, with byte-preserving `.gitattributes`.
""")
    indexed = {str(path.relative_to(DEST)): {"sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(DEST.rglob("*")) if path.is_file()}
    write(DEST / "sha256.json", {"files": indexed, "file_count": len(indexed),
          "total_bytes": sum(entry["bytes"] for entry in indexed.values())})
    for name, entry in indexed.items():
        assert sha(DEST / name) == entry["sha256"]
    print(json.dumps({"destination": str(DEST), "indexed_files": len(indexed),
          "indexed_bytes": sum(entry["bytes"] for entry in indexed.values()),
          "index_sha256": sha(DEST / "sha256.json"),
          "conclusion_sha256": sha(DEST / "conclusion.json")}, indent=2))


if __name__ == "__main__":
    main()
