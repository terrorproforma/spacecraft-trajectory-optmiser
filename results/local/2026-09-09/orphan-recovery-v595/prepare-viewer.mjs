import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
const repo = process.cwd();
const origin = join(repo, 'build/performance/orphan-recovery-v595/output-compatible');
const report = JSON.parse(await readFile(join(origin, 'report.json'), 'utf8'));
const source = JSON.parse(await readFile(join(origin, 'best/viewer/trajectories.json'), 'utf8'));
const manifest = JSON.parse(await readFile(join(origin, 'best/viewer/manifest.json'), 'utf8'));
if (!report.complete || !report.best.official.ok || !report.best.independent.ok) throw new Error('Unverified run');
const mass = source.trajectories.map(record => record.events.reduce((sum, event) =>
  sum + (event.kind === 'rendezvous' ? Math.max(0, event.mass_after_kg - event.mass_before_kg) : 0), 0));
const asteroids = [...new Set(source.trajectories.flatMap(record => record.events
  .map(event => event.event_id).filter(id => id > 0)))].sort((a,b) => a-b);
const fleet = {
  ...report.best,
  fleet: { ships: mass.length, total_collected_kg: mass.reduce((a,b) => a+b, 0),
    collected_kg_per_ship: mass, asteroids, ship_limit: report.best.independent.ship_limit },
  viewer_manifest: manifest,
};
await writeFile(join(repo, 'build/performance/orphan-viewer-fleet-v595.json'), JSON.stringify(fleet, null, 2));
const destination = join(repo, 'results/lambda/2026-09-06/visualiser/data/gtoc12-orphan-v595');
await mkdir(destination);
const meta = {
  run_id: 'orphan_recovery_v595', fleet_run_id: source.trajectories[0].source.run_id,
  commit: source.generated_by_commit,
  source_revision_note: 'Frozen bd944af9 Python plus native joint overlays; v590 core ffbae813. Exact source/binary hashes and compatibility flags in the published v595 report. Experimental is the exporter label, not a Git revision.',
  weighted_score_fixed_bonus_kg: report.best.score_kg,
  raw_kg_per_ship: report.best.total_mass_kg / mass.length,
  hardware: { gpu: 'NVIDIA GeForce RTX 5090 (WSL CUDA)', upstream_search: 'Retained historical 23-ship fleet: local RTX 5090 and Lambda H100 campaigns' },
  timing: { wall_seconds_total: report.seconds, wall_human: `${report.seconds.toFixed(3)} s incremental recovery experiment; historical fleet search excluded` },
  model: { dynamics: 'Official GTOC12 low-thrust dynamics; both complete-fleet checkers passed at unchanged tolerances',
    local_refine: 'CUDA seed, SCvx outer loop, dynamics, conic assembly and GPU QOCO; GPU Lambert/DP/joint arithmetic. Native winner reduction disabled for the pinned older core.' },
  optimisation: { strategy: 'Recover own orphan miner 19102 on ship 15. 18 collection orders × 2 DP grids; 23,142 joint candidates; two routes certified; one fleet improvement retained. +4.052 raw kg / +4.942 weighted kg at the same 23 ships. Python orchestration and CPU verification remain.', proven_optimal: false },
};
await writeFile(join(destination, 'compute.json'), JSON.stringify(meta, null, 2) + '\n');
console.log(JSON.stringify({destination, run_id:meta.fleet_run_id, weighted:meta.weighted_score_fixed_bonus_kg, raw_kg_per_ship:meta.raw_kg_per_ship, ships:mass.length, asteroid_footprint:asteroids.length}));
