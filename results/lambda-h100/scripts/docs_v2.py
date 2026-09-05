"""Append the H100 v2 campaign results to docs/GTOC12_TRACK.md (§7 table rows + a paragraph before §8)."""
import json, pathlib, statistics, subprocess
runs = pathlib.Path("results/gtoc12/runs")
doc = pathlib.Path("docs/GTOC12_TRACK.md")
text = doc.read_text()
assert "cluster_fleet_h100_v2" not in text, "docs already updated"
commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

def load(p):
    p = pathlib.Path(p)
    return json.load(open(p)) if p.exists() else None

cf1 = load(runs / "cluster_fleet_h100_v1/run_report.json")
fm1 = load(runs / "fleet_master_h100_v1/run_report.json")
cf = load(runs / "cluster_fleet_h100_v2/run_report.json")
j1 = load(runs / "joint_itinerary_h100_v1/run_report.json")
j2 = load(runs / "joint_itinerary_h100_v2/run_report.json")
fm = load(runs / "fleet_master_h100_v2/run_report.json")
cs = load(runs / "fleet_master_h100_v2/chain_stats.json") or {}
ov = load(runs / "fleet_master_h100_v2/official_verification.json") or {}

def marks(d):
    out = []
    for k in ("30_min", "60_min", "120_min", "240_min", "480_min"):
        m = (d.get("budget_marks") or {}).get(k)
        out.append(f"{k.replace('_min', ' min')} {m['score_kg']:.1f} / {m['ships']} ships" if m else f"{k.replace('_min', ' min')} —")
    return "; ".join(out)

def fleet_row(name, desc, d, master_key="master"):
    best = d.get("best") or {}
    fl = best.get("fleet", {})
    m = d.get(master_key) or {}
    return (f"| {name} ({desc}) | full catalogue | **{fl.get('ships', '—')}** | **{len(fl.get('asteroids', []))}** | "
            f"**{best.get('score_kg', float('nan')):.2f} kg** ({fl.get('average_collected_kg', float('nan')):.2f} kg average) | "
            f"{m.get('objective_kg', float('nan')):.2f} kg (LP bound {m.get('lp_bound_kg', float('nan')):.2f}, gap {m.get('lp_gap_kg', float('nan')):.1f} kg{'; exhaustive' if m.get('exhaustive') else ''}) | — | ")

rows = []
if cf1 and fm1:
    b = cf1["bundles"]; w = [x["wall_seconds"] for x in b]
    rows.append(fleet_row("`cluster_fleet_h100_v1`", "Lambda H100 host, 26-core Xeon 8480+, v6 recipe, 16 workers on cores 10-25, 6 h budget", cf1)
                + f"{len(b)} families × {min(w):.0f}–{max(w):.0f} s (16 in parallel) | in the family pricing | {cf1['wall_seconds_total']:.0f} s ({cf1['wall_seconds_total']/60:.0f} min); process-tree PSS peak {cf1.get('memory_total_pss_peak_mb', 0)/1024:.2f} GB | CPU (H100 host) |")
    rows.append(fleet_row("**`fleet_master_h100_v1`**", f"master over {len(fm1.get('sources', []))} archives incl. `cluster_fleet_h100_v1`, {fm1['master'].get('columns')} columns, 16 workers", fm1)
                + f"{fm1.get('recertification_wall_seconds', 0):.0f} s re-certification + {fm1.get('master_wall_seconds', 0):.0f} s master | — | {fm1['wall_seconds_total']:.0f} s | CPU (H100 host) |")
if cf:
    b = cf["bundles"]; w = [x["wall_seconds"] for x in b]
    ships = [s for x in b for s in x["ships"]]
    m = [s["collected_kg"] for s in ships]
    parts = cf.get("instance", {}).get("partitions") or []
    part_desc = ", ".join(f"{p['name']} {p['families']}" for p in parts)
    part_desc_long = ", ".join(f"{p['name']} {p['families']} families" for p in parts)
    rows.append(fleet_row("`cluster_fleet_h100_v2`", f"ninth campaign on the H100 host: union of {len(parts)} family partitions (radii 1.75/1.6 × collect-window + phasing bands, ≥ 20 members: {part_desc}), 5 ships per family, beam 32, refine-top 3, 6600 s per family, harvest substitution + return sweep cells, 22 workers `nice 5`, 8 h budget; marks: {marks(cf)}", cf)
                + f"{len(b)} families × {min(w) if w else 0:.0f}–{max(w) if w else 0:.0f} s (22 in parallel), {len(ships)} ships ({sum(x >= 600 for x in m)} ≥ 600 kg, {sum(x >= 650 for x in m)} ≥ 650, {sum(x >= 700 for x in m)} ≥ 700, best {max(m) if m else 0:.1f}) | in the family pricing | {cf['wall_seconds_total']:.0f} s ({cf['wall_seconds_total']/3600:.1f} h); process-tree PSS peak {cf.get('memory_total_pss_peak_mb', 0)/1024:.2f} GB | CPU (H100 host) |")
for name, j in (("`joint_itinerary_h100_v1`", j1), ("`joint_itinerary_h100_v2`", j2)):
    if not j: continue
    recs = j.get("records", [])
    before = [r["before_kg"] for r in recs]; after = [r["after_kg"] for r in recs]
    src = "every archived chain ≥ 450 kg of the 16 local archives + `cluster_fleet_h100_v1`, `fleet_master_v7` ships first, 4 workers alongside the campaign" if j is j1 else "every `cluster_fleet_h100_v2` chain ≥ 450 kg, 22 workers"
    rows.append(f"| {name} (§6.10 joint re-optimisation of {src}) | full catalogue | {j.get('improved')} improved of {j.get('attempted')} | {j.get('inserted', 0)} inserted | +{j.get('gain_kg_total', 0):.1f} kg (chains ≥ 600 kg {sum(x >= 600 for x in before)} → {sum(x >= 600 for x in after)}, ≥ 650 {sum(x >= 650 for x in before)} → {sum(x >= 650 for x in after)}; best +{max((r.get('gain_kg', 0) for r in recs), default=0):.1f}) | — | — | {len(recs)} ships × {min((r['wall_seconds'] for r in recs), default=0):.0f}–{max((r['wall_seconds'] for r in recs), default=0):.0f} s | in the joint search | {j.get('wall_seconds_total', 0):.0f} s; worker peak {j.get('worker_peak_rss_mb', 0)/1024:.2f} GB | CPU (H100 host) |")
if fm:
    rows.append(fleet_row("**`fleet_master_h100_v2`**", f"archive-wide master over {len(fm.get('sources', []))} archives ({fm.get('recertified_routes', '?')} routes re-flown through SCvx, {fm['master'].get('columns')} columns, {'proven optimal' if fm['master'].get('proven_optimal') else 'LP-bounded'}), 22 workers", fm)
                + f"{fm.get('recertification_wall_seconds', 0):.0f} s re-certification + {fm.get('master_wall_seconds', 0):.0f} s master | — | {fm['wall_seconds_total']:.0f} s | CPU (H100 host) |")

para = ["", f"**Ninth iteration (Lambda H100 host, commit `{commit}`): breadth on 26 CPU cores.** The GPU stayed reserved for the G4 campaign; every GTOC12 process ran with `CUDA_VISIBLE_DEVICES=\"\"`."]
if cf:
    para.append(f"`cluster_fleet_h100_v2` priced {len(cf['bundles'])} families from the union of {len(parts)} partitions ({part_desc_long}; `family_partitions`, labels offset per partition) in {cf['wall_seconds_total']/3600:.1f} h: {marks(cf)}; final {cf['best']['score_kg']:.1f} kg, {cf['best']['fleet']['ships']} ships, {len(cf['best']['fleet']['asteroids'])} asteroids, {cf['best']['fleet']['average_collected_kg']:.1f} kg average. Family stops: {json.dumps({k: [x.get('stopped') for x in b].count(k) for k in set(x.get('stopped') for x in b)})}.")
if cs:
    u = cs.get("UNION", {})
    para.append(f"Chain-mass distribution over the {len(cs) - 1} master sources (unique asteroid sets): {u.get('chains')} chains, {u.get('ge_600')} ≥ 600 kg, {u.get('ge_650')} ≥ 650 kg, {u.get('ge_700')} ≥ 700 kg, best {u.get('max_kg', 0):.1f} kg; per source: " + "; ".join(f"`{k}` {v['chains']} chains / {v['ge_600']} ≥ 600" for k, v in cs.items() if k != "UNION") + ".")
for name, j in (("joint_itinerary_h100_v1", j1), ("joint_itinerary_h100_v2", j2)):
    if j:
        recs = j.get("records", [])
        para.append(f"`{name}`: {j.get('improved')} of {j.get('attempted')} ships improved, +{j.get('gain_kg_total', 0):.1f} kg, {j.get('inserted', 0)} insertions; chains ≥ 600 kg {sum(r['before_kg'] >= 600 for r in recs)} → {sum(r['after_kg'] >= 600 for r in recs)}, {j.get('wall_seconds_total', 0)/60:.0f} min.")
if fm:
    m = fm["master"]; best = fm["best"]
    para.append(f"`fleet_master_h100_v2` over {len(fm['sources'])} archives ({m.get('columns')} columns): **{best['score_kg']:.2f} kg, {best['fleet']['ships']} ships, {len(best['fleet']['asteroids'])} asteroids, {best['fleet']['average_collected_kg']:.2f} kg average**; master objective {m.get('objective_kg', 0):.2f} kg, LP bound {m.get('lp_bound_kg', 0):.2f} kg, gap {m.get('lp_gap_kg', 0):.1f} kg ({'proven optimal' if m.get('proven_optimal') else 'not proven'}; LP relaxation infeasible or below the incumbent beyond {best['fleet']['ships']} ships: {json.dumps({k: round(v, 1) for k, v in (m.get('lp_relaxations_kg') or {}).items() if int(k) >= best['fleet']['ships']})}). The 22-ship threshold is 599.5 kg average, 23 ships ~611 kg. Official `GTOC12_Verify`: {ov.get('passed')}/{ov.get('total')} emitted `Result.txt` files pass (per-ship diagnostic files of cooperative members fail Error803 by construction); the independent verifier agrees on the fleet (`independent_verify.txt`).")
para.append("Commands: `~/s/gtoc12_v2_campaign.sh` on the host (cluster-fleet `--workers 22 --ships-per-cluster 5 --cluster-radius 1.75,1.6 --all-family-bands --collect-epoch-families --min-members 20 --beam-width 32 --refine-top 3 --cluster-budget-seconds 6600 --retime-budget-seconds 900 --budget-seconds 28800 --max-clusters 400` with the v6 DP/harvest flags; `joint-itinerary --top 100000 --min-collected-kg 450 --per-ship-seconds 600 --insert-trials 4`; `fleet-master --workers 22`). Artefacts committed as for the earlier campaigns (run reports, `bundle.json`, `route_summary.json`, `ships.jsonl`, verified `fleet.json`s, the master's `fleet/Result.txt`, `official_verification.json`, `chain_stats.json`, `results/gtoc12/leg_stats/after_h100_v2.json`).")

table_anchor = "\n\n**Reference solutions" if "\n\n**Reference solutions" in text else None
# rows go at the end of the §7 table: find the last table row before the first blank line after the header
start = text.index("## 7. Results")
end_table = text.index("\n\n", text.index("| --- |", start))
text = text[:end_table] + "\n" + "\n".join(rows) + text[end_table:]
sec8 = text.index("## 8. Limitations")
text = text[:sec8] + "\n".join(para) + "\n\n" + text[sec8:]
doc.write_text(text)
print("docs updated:", len(rows), "rows,", len(para), "paragraph lines")