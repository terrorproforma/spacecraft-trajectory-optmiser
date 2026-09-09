import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(),evidence=join(repo,'results/lambda/2026-09-09/gpu-collect-composition-v807');
const report=JSON.parse(await readFile(join(evidence,'h100/report.json'),'utf8'));
const source=JSON.parse(await readFile(join(evidence,'h100/viewer/trajectories.json'),'utf8'));
if(!report.complete||!report.success||!report.qualified||!report.independent.ok||!report.official.ok)throw Error('Unverified candidate');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.independent.ship_limit}};
await writeFile(join(evidence,'viewer-fleet-input.json'),JSON.stringify(fleet,null,2));
const out=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-collect-composition-v807');await mkdir(out,{recursive:true});
const meta={run_id:'gpu_collect_composition_v807',fleet_run_id:'gpu_collect_composition_v807',commit:'7f07c6b6',result_sha256:report.solution_sha256,
 source_revision_note:'Mixed provenance: H100 v804 fleet, locally GPU-refined endpoint-merit-corrected ship 8. Both full-fleet checkers rerun locally and on Lambda. No new GPU solve in composition.',
 weighted_score_fixed_bonus_kg:report.score_kg,raw_kg_per_ship:report.total_mass_kg/23,
 hardware:{gpu:'Lambda H100 fleet + RTX 5090 ship-8 correction',upstream_search:'H100 v802: 11,007 candidate routes in 480.684 seconds; 454,295,808 logical Lambert branch requests'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s full-fleet checks/export on Lambda; composition performs no new GPU solve`},
 model:{dynamics:'Original GTOC12 physics and tolerances; both complete-fleet checkers pass on both hosts',local_refine:'Upstream v803: 147 native QOCO/SCvx attempts, 138 converge, five complete routes certify; new ship 4 improves the fleet. Ship 8 was separately GPU-refined locally.'},
 optimisation:{strategy:'12,999.825 fixed-bonus weighted kg. Ships 4 and 8 improve H100 v799; other 21 sections are byte-identical. Retained CUDA collection buffers reduce workspace creations 97.7% in paired route tests. Python orchestration and host packing remain.',proven_optimal:false}};
await writeFile(join(out,'compute.json'),JSON.stringify(meta,null,2)+'\n');
await writeFile(join(out,'.gitattributes'),'* -text\n');
console.log(JSON.stringify({ships:mass.length,asteroids:asteroids.length,score:report.score_kg}));
