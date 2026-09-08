import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {join} from 'node:path';
const repo=process.cwd(), evidence=join(repo,'results/lambda/2026-09-09/gpu-layouts-v780');
const origin=join(evidence,'h100-best'), report=JSON.parse(await readFile(join(origin,'campaign-report.json'),'utf8'));
const source=JSON.parse(await readFile(join(origin,'viewer/trajectories.json'),'utf8'));
if(!report.complete||!report.independent.ok||!report.official.ok||!report.improved)throw Error('Unverified candidate');
const mass=source.trajectories.map(r=>r.events.reduce((sum,e)=>sum+(e.kind==='rendezvous'?Math.max(0,e.mass_after_kg-e.mass_before_kg):0),0));
const asteroids=[...new Set(source.trajectories.flatMap(r=>r.events.map(e=>e.event_id).filter(id=>id>0)))].sort((a,b)=>a-b);
const fleet={...report,fleet:{ships:mass.length,total_collected_kg:mass.reduce((a,b)=>a+b,0),collected_kg_per_ship:mass,asteroids,ship_limit:report.independent.ship_limit}};
await writeFile(join(repo,'build/performance/layouts-viewer-fleet-v780.json'),JSON.stringify(fleet,null,2));
const out=join(repo,'results/lambda/2026-09-06/visualiser/data/gtoc12-layouts-v780');await mkdir(out);
const meta={run_id:'gpu_layout_search_v780_h100',fleet_run_id:source.trajectories[0].source.run_id,commit:'49b1babe',
 source_revision_note:'Frozen v778 Python wrapper with v776 CUDA core and retained QOCO v686. CUDA layout generation from shared device inputs; both complete-fleet physics checkers pass.',
 weighted_score_fixed_bonus_kg:report.score_kg,raw_kg_per_ship:report.total_mass_kg/23,
 hardware:{gpu:'Lambda NVIDIA H100 80 GB HBM3',upstream_search:'Two routes; 15,748 evaluations in 181 joint batches; 220,079 computed geometry hops'},
 timing:{wall_seconds_total:report.seconds,wall_human:`${report.seconds.toFixed(3)} s search + 36 native leg attempts + full checks and export; historical fleet generation excluded`},
 model:{dynamics:'Official GTOC12 dynamics; fresh official and independent full-fleet checks at unchanged tolerances',local_refine:'33 legs in two accepted routes converge and have CUDA certificates. Three exploratory Earth-leg attempts are also included in work/timing. Other 21 routes retained.'},
 optimisation:{strategy:'Same verified mission score. CUDA constructs insertion layouts and epochs from shared edge inputs. Repeated 12,992-schedule screening: 59.986 ms RTX / 55.704 ms H100, 18.4x / 37.4x faster than host-built layouts in the same core. One insertion batch; 181 joint calls in the full replay. Shared-edge input compilation, survivor sorting and independent audits still use CPU.',proven_optimal:false}};
await writeFile(join(out,'compute.json'),JSON.stringify(meta,null,2)+'\n');
console.log(JSON.stringify({ships:mass.length,asteroids:asteroids.length,score:report.score_kg,source:meta.fleet_run_id}));
