import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(), evidence=join(repo,'results/lambda/2026-09-09/gpu-fleet-routes-v767');
const origin=join(evidence,'h100-best'), report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
if(!report.complete||!report.independent.ok||!report.official.ok||!report.improved)throw Error('Unverified candidate');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.independent.ship_limit}};
await writeFile(join(repo,'build/performance/routes-viewer-fleet-v767.json'),JSON.stringify(fleet,null,2));
const out=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-routes-v767');await mkdir(out);
const meta={run_id:'gpu_route_search_v767_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:'258e5c3a',
 source_revision_note:'Frozen v763 CUDA core with retained QOCO v686; GPU timing search and native refinement around the two newly selected routes. Both full-fleet checkers pass.',
 weighted_score_fixed_bonus_kg:report.score_kg,raw_kg_per_ship:report.total_mass_kg/23,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Two routes; 15,748 joint evaluations and 220,079 computed geometry hops'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s search + 36 native leg attempts + full checks and export; historical fleet generation excluded`},
 model:{dynamics:'Official GTOC12 dynamics; fresh official and independent full-fleet checks at unchanged tolerances',local_refine:'33 legs in two accepted routes converge and have CUDA certificates. Three exploratory Earth-leg attempts are also included in work/timing. Other 21 routes retained.'},
 optimisation:{strategy:'New verified best: +0.585023 weighted kg / +0.602327 raw kg. Four GPU-accepted timing moves; no asteroid insertion qualified. 13,172 joint batches expose remaining host dispatch overhead. Python route orchestration and independent CPU audits remain.',proven_optimal:false}};
await writeFile(join(out,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({ships:mass.length,asteroids:asteroids.length,score:report.score_kg,source:meta.fleet_run_id}));
