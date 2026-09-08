import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
const repo=process.cwd();
const origin=join(repo,'results/lambda/2026-09-09/gpu-joint-itinerary-v642/h100-best');
const report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
const manifest=JSON.parse(await readFile(join(origin,'viewer/manifest.json'),'utf8'));
if (!report.complete || !report.best.official.ok || !report.best.independent.ok) throw new Error('Unverified H100 result');
const mass=source.trajectories.map(record=>record.events.reduce((sum,event)=>sum+(event.kind==='rendezvous'?Math.max(0,event.mass_after_kg-event.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(record=>record.events.map(event=>event.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report.best,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.best.independent.ship_limit},viewer_manifest:manifest};
await writeFile(join(repo,'build/performance/joint-viewer-fleet-v642.json'),JSON.stringify(fleet,null,2));
const destination=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-joint-v642');
await mkdir(destination);
const meta={run_id:'gpu_joint_v642_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:source.generated_by_commit,
  source_revision_note:'Frozen bd944af9 plus native joint evaluation and GPU winner selection. Exact core/Python hashes, complete repeated campaigns and verifier reports are archived in gpu-joint-itinerary-v642.',
  weighted_score_fixed_bonus_kg:report.best.score_kg,raw_kg_per_ship:report.best.total_mass_kg/mass.length,
  hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Retained historical 23-ship fleet'},
  timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s recovery replay including CPU checks and viewer export; historical fleet search excluded`},
  model:{dynamics:'Official GTOC12 dynamics; both complete-fleet checkers passed at unchanged tolerances',local_refine:'CUDA seed, SCvx outer loop, dynamics, conic assembly and QOCO; GPU Lambert/DP/joint evaluation and winner selection'},
  optimisation:{strategy:'H100 reproduction of the abandoned-miner improvement. 18 orders, 36 DP grids, 23,142 joint candidates, 36 native leg solves. Device result downloads: 88,576 bytes. Same 12,810.136 weighted kg record; Python orchestration and CPU independent verification remain.',proven_optimal:false}};
await writeFile(join(destination,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({destination,ships:mass.length,asteroids:asteroids.length,score:report.best.score_kg}));
