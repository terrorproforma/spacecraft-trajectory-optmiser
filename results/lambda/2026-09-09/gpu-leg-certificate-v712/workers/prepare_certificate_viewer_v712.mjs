import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(), origin=join(repo,'results/lambda/2026-09-09/gpu-leg-certificate-v712/h100-best');
const report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
const manifest=JSON.parse(await readFile(join(origin,'viewer/manifest.json'),'utf8'));
if(!report.complete||!report.best.official.ok||!report.best.independent.ok)throw Error('Unverified fleet');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report.best,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.best.independent.ship_limit},viewer_manifest:manifest};
await writeFile(join(repo,'build/performance/certificate-viewer-fleet-v712.json'),JSON.stringify(fleet,null,2));
const destination=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-certificate-v712');
await mkdir(destination);
const meta={run_id:'gpu_leg_certificate_v712_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:source.generated_by_commit,
 source_revision_note:'Frozen v709 Python driver with retained CUDA DOP853 certificates; previously qualified v702 native libraries. Exact source and binary hashes are in gpu-leg-certificate-v712.',
 weighted_score_fixed_bonus_kg:report.best.score_kg,raw_kg_per_ship:report.best.total_mass_kg/mass.length,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Retained historical 23-ship fleet'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s replay including independent CPU checks and export; historical search excluded`},
 model:{dynamics:'Official GTOC12 dynamics; both complete-fleet checkers passed at unchanged tolerances',local_refine:'CUDA seed, SCvx, QOCO conditioning retry, whole epoch-search graph and retained CUDA DOP853 per-leg certificates'},
 optimisation:{strategy:'18 orders and 36 converged native solves with CUDA leg certificates. Sequential certificate-stage benchmarks: 19.55–19.58x on H100, 4.88–5.08x locally. Same verified fleet score. Independent CPU fleet audit dominates total runtime; broader Python route/fleet control remains.',proven_optimal:false}};
await writeFile(join(destination,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({destination,ships:mass.length,score:report.best.score_kg}));
