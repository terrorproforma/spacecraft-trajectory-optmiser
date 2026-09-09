import assert from 'node:assert/strict';
import { readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serialize } from '../../../results/lambda/2026-09-06/visualiser/scripts/import-data.mjs';

const kit = dirname(fileURLToPath(import.meta.url)), root = resolve(kit, '../../..');
const composition = join(root, 'build/performance/seeded-current-composition-v628');
const baseline = join(root, 'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best');
const route = join(root, 'build/performance/seeded-candidate-boundary-merit-v627/campaign');
const routeDisplay = join(root, 'build/performance/seeded-candidate-visualizer-v627');
const output = join(kit, 'export'), runId = 'local_current_fleet_composition_v628';
const resultSha = '63446ebf3ff1298911bccade7b789a8fb68a0f7db4171906c8e6783a7f174bab';
const oldSha = '97d1f351bf6ad4907ddce887aa47d1fa974f587ab270374f4bb46a5898491d48';
const routeSha = '931defc362c9299d28d089aa46130c05f92bf9ee38ea03d870b4dc2a83915946';
const digest = b => createHash('sha256').update(b).digest('hex');
const read = async p => JSON.parse(await readFile(p, 'utf8'));
const write = async (p, x) => writeFile(p, serialize(x), { flag: 'wx' });
function sections(bytes) {
  const result = new Map();
  for (const line of bytes.toString('utf8').match(/[^\n]*\n|[^\n]+$/g)) {
    const id = Number(line.trim().split(/\s+/)[0]);
    if (id) result.set(id, (result.get(id) ?? '') + line);
  }
  return result;
}
await mkdir(output);
const started = performance.now();
const report = await read(join(composition, 'output/report.json'));
const binding = await read(join(composition, 'output/checker-binding.json'));
const result = await readFile(join(composition, 'inputs/Result.txt'));
assert.equal(digest(result), resultSha);
assert(report.qualified && report.complete && report.independent_calls === 1 && report.official_calls === 1);
assert.equal(binding.result_sha256, resultSha);
for (const key of ['independent', 'official']) assert(binding[key].ok && binding[key].result_sha256 === resultSha);
const oldResult = await readFile(join(baseline, 'Result.txt'));
const routeResult = await readFile(join(route, 'output/fleet/Result.txt'));
assert.equal(digest(oldResult), oldSha);
assert.equal(digest(routeResult), routeSha);
const currentSections = sections(result), oldSections = sections(oldResult), routeSections = sections(routeResult);
assert.equal(currentSections.size, 23);
for (let id = 1; id <= 23; id++) assert.equal(currentSections.get(id), (id === 8 ? routeSections : oldSections).get(id));
const manifestOld = await read(join(baseline, 'viewer/manifest.json'));
const bytesOld = await readFile(join(baseline, 'viewer/trajectories.json'));
assert.equal(manifestOld.source.sha256, oldSha);
assert.equal(digest(bytesOld), manifestOld.files['trajectories.json'].sha256);
const routeAudit = await read(join(routeDisplay, 'export-audit.json'));
const bytesRoute = await readFile(join(routeDisplay, 'export/trajectories.json'));
assert.equal(routeAudit.source_result_sha256, routeSha);
assert.equal(digest(bytesRoute), routeAudit.export_files['trajectories.json']);
const source = JSON.parse(bytesOld), replacement = JSON.parse(bytesRoute);
assert.equal(source.trajectories.length, 23);
const originals = structuredClone(source.trajectories);
source.trajectories[7] = structuredClone(replacement.trajectories[7]);
const masses = [], asteroids = new Set();
for (let i = 0; i < 23; i++) {
  const item = source.trajectories[i], prior = i === 7 ? replacement.trajectories[7] : originals[i];
  for (const key of ['replay', 'transcription', 'events']) assert.deepEqual(item[key], prior[key]);
  item.trajectory_id = `gtoc12_${runId}_ship${i + 1}`;
  item.source.run_id = runId;
  item.source.commit = 'v799-H100+v627-local-boundary-merit-composition-v628';
  item.source.display_sample_origin = i === 7 ? 'Exact saved certified local v627 ship8 history' : 'Exact saved H100 v799 history; Result section byte-identical';
  item.raw_evidence_sha256 = resultSha;
  item.validation = { finite: true, ...Object.fromEntries(Object.entries(binding.independent).filter(([key]) => key !== 'violations')) };
  let cargo = 0;
  for (const event of item.events) if (event.event_id > 0) {
    asteroids.add(event.event_id);
    cargo += Math.max(0, event.mass_after_kg - event.mass_before_kg);
  }
  masses.push(cargo);
}
assert.equal(asteroids.size, 200);
assert(Math.abs(masses.reduce((a, b) => a + b, 0) - binding.independent.total_mass_kg) < 1e-8);
source.title = 'Local fleet composition v628 — H100 v799 plus certified local ship 8';
source.generated_by_commit = source.trajectories[0].source.commit;
source.imported_source_sha256 = resultSha;
source.archive.validation_report = binding.independent;
source.display_replay_provenance = { newly_propagated_ships: 0, reused_v799_ship_ids: Array.from({ length: 23 }, (_, i) => i + 1).filter(i => i !== 8), reused_v627_ship_ids: [8], all_histories_reused_exactly: true, no_new_propagation: true };
await write(join(output, 'trajectories.json'), source);
const dataBytes = await readFile(join(output, 'trajectories.json'));
const manifest = { schema_version: '1.0.0', source: { path_basename: 'Result.txt', bytes: result.length, sha256: resultSha }, files: { 'trajectories.json': { bytes: dataBytes.length, sha256: digest(dataBytes) } }, transform: 'Reuse22 existing v799 histories and1 certified v627 ship8 history after exact Result-section equality. No new trajectory samples, interpolation, or propagation. Both complete-fleet checkers independently validate the composition.' };
await write(join(output, 'manifest.json'), manifest);
await write(join(output, 'fleet-summary.json'), { viewer_manifest: manifest, independent: binding.independent, official: binding.official, fleet: { ships: 23, total_collected_kg: binding.independent.total_mass_kg, collected_kg_per_ship: masses, asteroids: [...asteroids].sort((a, b) => a - b), ship_limit: binding.independent.ship_limit } });
await write(join(output, 'checker-binding.json'), binding);
await write(join(output, 'compute.json'), {
  run_id: runId, fleet_run_id: runId, commit: source.generated_by_commit,
  result_sha256: resultSha, weighted_score_fixed_bonus_kg: binding.independent.weighted_score_fixed_bonus_kg,
  raw_kg_per_ship: binding.independent.total_mass_kg / 23,
  hardware: { gpu: 'No GPU work in composition v628', upstream_search: 'H100 v799 fleet plus local RTX5090 v627 ship8 refinement' },
  timing: { wall_seconds_total: report.seconds, wall_human: `${report.seconds.toFixed(3)} s composition verification; no additional GPU solve` },
  model: { dynamics: 'Official GTOC12 dynamics, unchanged physical tolerances; both original complete-fleet checkers passed.', local_refine: 'Reuse the17 certified flights and3 waits solved locally in v627. This composition performs no new refinement.' },
  optimisation: { strategy: 'Insert exact certified v627 ship8 into its unchanged v799 route slot; retain the other22 ship sections byte-for-byte. Separate CPU-only current-fleet verification, not a new H100 optimization run.', proven_optimal: false },
  composition: { source_v799_sha256: oldSha, inserted_v627_sha256: routeSha, replaced_ship: 8, other_ship_sections_exact: 22, raw_gain_kg: report.raw_delta_kg, weighted_gain_kg: report.weighted_delta_kg, qualified: true, original_run_incumbent_promoted: false },
  run_counts: { full_fleet_CPU_checks: report.independent_calls, full_fleet_official_checks: report.official_calls, native_solves_started: 0, search_or_Lambert_calls: 0, GPU_calls: 0, extra_leg_certificates: 0 },
  display_replay_provenance: source.display_replay_provenance,
});
const inputPaths = [join(composition, 'inputs/Result.txt'), join(composition, 'output/report.json'), join(composition, 'output/checker-binding.json'), join(baseline, 'Result.txt'), join(baseline, 'viewer/manifest.json'), join(baseline, 'viewer/trajectories.json'), join(route, 'output/fleet/Result.txt'), join(routeDisplay, 'export-audit.json'), join(routeDisplay, 'export/trajectories.json')];
const audit = { status: 'passed', source_result_sha256: resultSha, bytes_reused_ship_sections: 23, reused_history_series: 23, ships: 23, unique_asteroids: 200, events: source.trajectories.reduce((n, t) => n + t.events.length, 0), replay_points: source.trajectories.reduce((n, t) => n + t.replay.point_count, 0), new_propagations: 0, GPU_calls: 0, seconds: (performance.now() - started) / 1000, source_sha256: {}, export_files: {} };
for (const path of inputPaths) audit.source_sha256[relative(root, path).replaceAll('\\', '/')] = digest(await readFile(path));
for (const name of ['trajectories.json', 'manifest.json', 'fleet-summary.json', 'checker-binding.json', 'compute.json']) audit.export_files[name] = digest(await readFile(join(output, name)));
await write(join(kit, 'export-audit.json'), audit);
console.log(JSON.stringify(audit));
