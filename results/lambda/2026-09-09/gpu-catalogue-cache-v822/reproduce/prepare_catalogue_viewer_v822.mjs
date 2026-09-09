import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(),evidence=join(repo,'results/lambda/2026-09-09/gpu-catalogue-cache-v822'),origin=join(evidence,'h100-best');
const report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
if(!report.complete||!report.success||!report.qualified||!report.independent.ok||!report.official.ok||report.improved)throw Error('Expected verified unchanged-score replay');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.independent.ship_limit}};
await writeFile(join(evidence,'viewer-fleet-input.json'),JSON.stringify(fleet,null,2));
const out=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-catalogue-v822');await mkdir(out,{recursive:true});
const meta={run_id:'gpu_catalogue_cache_v822_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:'5f698731',result_sha256:report.solution_sha256,
 source_revision_note:'Frozen final host cache v819 and unchanged integrated CUDA core v808. Fleet replay v817 retains the frontier score and has fresh full-fleet checks on both hosts. Frontier ships 8/19 retain their local GPU-refinement provenance.',
 weighted_score_fixed_bonus_kg:report.score_kg,raw_kg_per_ship:report.total_mass_kg/23,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Paired two-route benchmark: 517 + 472 surrogate candidates; exact files in all repeats. H100 search throughput improves 7.3–7.5%.'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s CUDA selection/full-fleet checks/export; measured route searches separately take median 18.086 s and 17.379 s on H100`},
 model:{dynamics:'Original GTOC12 physics and tolerances; both complete-fleet checkers pass on both hosts',local_refine:'Integrated-core replay: 147 native QOCO/SCvx attempts, 138 converge, five complete routes certify. No meaningful new fleet score. Frontier routes retain their recorded source identities.'},
 optimisation:{strategy:'Retain 13,023.705 fixed-bonus weighted kg. Four catalogue hashes replace 1,302 scans across the benchmark routes; mutable input changes remain detectable. Python orchestration and request packing remain. Dense verifier replay includes coast intervals.',proven_optimal:false}};
await writeFile(join(out,'compute.json'),JSON.stringify(meta,null,2)+'\n');await writeFile(join(out,'.gitattributes'),'* -text\n');
console.log(JSON.stringify({ships:mass.length,asteroids:asteroids.length,score:report.score_kg,run:meta.fleet_run_id}));
