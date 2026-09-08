import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';
const repo=process.cwd();
const origin=join(repo,'results/lambda/2026-09-09/gpu-device-search-v707/h100-best');
const report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
const manifest=JSON.parse(await readFile(join(origin,'viewer/manifest.json'),'utf8'));
if (!report.complete || !report.best.official.ok || !report.best.independent.ok) throw new Error('Unverified H100 result');
const mass=source.trajectories.map(record=>record.events.reduce((sum,event)=>sum+(event.kind==='rendezvous'?Math.max(0,event.mass_after_kg-event.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(record=>record.events.map(event=>event.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report.best,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.best.independent.ship_limit},viewer_manifest:manifest};
await writeFile(join(repo,'build/performance/search-viewer-fleet-v707.json'),JSON.stringify(fleet,null,2));
const destination=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-search-v707');
await mkdir(destination);
const meta={run_id:'gpu_device_search_v707_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:source.generated_by_commit,
 source_revision_note:'Frozen 3175fb44 plus the complete CUDA epoch-search controller. Exact source/binary hashes and full evidence are archived in gpu-device-search-v707.',
 weighted_score_fixed_bonus_kg:report.best.score_kg,raw_kg_per_ship:report.best.total_mass_kg/mass.length,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Retained historical 23-ship fleet'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s recovery replay including CPU checks and export; historical fleet search excluded`},
 model:{dynamics:'Official GTOC12 dynamics; both complete-fleet checkers passed at unchanged tolerances',local_refine:'CUDA seed, SCvx, assembly and GPU QOCO with device-selected retry conditioning; GPU Lambert/DP and whole epoch-search graph, including generation, geometry, winner acceptance, mesh transitions and deadlines'},
 optimisation:{strategy:'18 orders, 23,142 joint candidates and 36 converged native legs. Four GPU-controlled epoch searches replace 132 joint host calls; epoch traffic is 1,248 bytes each way. Same verified fleet score. Broader Python route/fleet orchestration and independent CPU checks remain.',proven_optimal:false}};
await writeFile(join(destination,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({destination,ships:mass.length,asteroids:asteroids.length,score:report.best.score_kg}));

