# Route-family cap audit v611

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
