import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(),evidence=join(repo,'results/lambda/2026-09-09/gpu-regeneration-v799');
const origin=join(evidence,'h100-best'),report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const audit=JSON.parse(await readFile(join(evidence,'h100/audit.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
if(!report.complete||!report.qualified||!report.independent.ok||!report.official.ok||!report.improved)throw Error('Unverified candidate');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.independent.ship_limit}};
await writeFile(join(repo,'build/performance/regeneration-viewer-fleet-v799.json'),JSON.stringify(fleet,null,2));
const out=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-regeneration-v799');await mkdir(out);
const seconds=audit.v794.search_seconds+audit.v795.seconds+report.seconds;
const meta={run_id:'gpu_regeneration_v799_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:'ccf5de37',
 source_revision_note:'Frozen v788 source and CUDA core, retained QOCO v686. Raw-mass beam search, native route refinement, CUDA fleet selection; exact full-fleet checks pass.',
 weighted_score_fixed_bonus_kg:report.score_kg,raw_kg_per_ship:report.total_mass_kg/23,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'23 ships; 10,971 surrogate route candidates; 454,203,618 logical Lambert branch requests'},
 timing:{wall_seconds_total:seconds,wall_human:`${audit.v794.search_seconds.toFixed(1)} s search + ${audit.v795.seconds.toFixed(1)} s refinement/fleet checks + ${report.seconds.toFixed(1)} s deeper selection/recheck/export; earlier pilot excluded`},
 model:{dynamics:'Official GTOC12 dynamics; both full-fleet checkers pass at unchanged tolerances',local_refine:'373 native QOCO/SCvx leg attempts; 365 converge. 18 of 26 routes certify; nine replacements selected. Fourteen incumbent routes retained.'},
 optimisation:{strategy:'Gain: 148.480 weighted kg and 226.694 raw kg. Wider beam with raw-mass objective restores the 23-ship margin. CUDA fleet search exhausts the 44-column certified pool; global mission optimality is not claimed. Python orchestration and CPU independent audits remain.',proven_optimal:false}};
await writeFile(join(out,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({ships:mass.length,asteroids:asteroids.length,score:report.score_kg,source:meta.fleet_run_id,seconds}));
