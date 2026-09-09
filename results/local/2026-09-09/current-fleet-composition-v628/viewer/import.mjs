import assert from 'node:assert/strict';
import { readFile, writeFile, copyFile, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { importGtoc12 } from '../../../results/lambda/2026-09-06/visualiser/scripts/import-gtoc12.mjs';
import { serialize } from '../../../results/lambda/2026-09-06/visualiser/scripts/import-data.mjs';

const kit = dirname(fileURLToPath(import.meta.url)), root = resolve(kit, '../../..');
const viewer = join(root, 'results/lambda/2026-09-06/visualiser');
const dataset = 'gtoc12-current-composition-v628', outputDirectory = join(viewer, 'data', dataset);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const read = async p => JSON.parse(await readFile(p, 'utf8'));
await assert.rejects(stat(outputDirectory), { code: 'ENOENT' });
const started = performance.now();
const manifest = await importGtoc12({
  exportDirectory: join(kit, 'export'), cataloguePath: join(root, 'build/performance/catalogue-v767.txt'),
  solutionPath: join(root, 'build/performance/seeded-current-composition-v628/inputs/Result.txt'),
  fleetPath: join(kit, 'export/fleet-summary.json'), outputDirectory,
});
const importerSeconds = (performance.now() - started) / 1000;
const fleetPath = join(outputDirectory, 'fleet.json'), fleet = await read(fleetPath);
const base = await read(join(viewer, 'data/gtoc12-regeneration-v799/fleet.json'));
const replacement = await read(join(viewer, 'data/gtoc12-seeded-candidate-v627/fleet.json'));
const source = await read(join(kit, 'export/trajectories.json'));
const binding = await read(join(kit, 'export/checker-binding.json'));
const compute = await read(join(kit, 'export/compute.json'));
fleet.title = source.title;
fleet.display_replay_provenance = source.display_replay_provenance;
fleet.composition = compute.composition;
fleet.score.weighted_score_fixed_bonus_kg = compute.weighted_score_fixed_bonus_kg;
assert.equal(fleet.source.solution_sha256, binding.result_sha256);
assert.equal(fleet.score.ships, 23);
assert.equal(fleet.score.unique_asteroids, 200);
assert.equal(fleet.ships.flatMap(s => s.events).length, 446);
assert.equal(fleet.run_id, compute.fleet_run_id);
assert.equal(fleet.generated_by_commit, compute.commit);
assert.equal(fleet.verification.ok, true);
assert.equal(fleet.source.official_verifier_ok, true);
for (let i = 0; i < 23; i++) {
  const previous = (i === 7 ? replacement : base).ships[i];
  for (const key of ['replay', 'transcription', 'events']) assert.deepEqual(fleet.ships[i][key], previous[key]);
}
assert.equal(fleet.ships[7].events.length, 18);
assert.equal(fleet.ships[7].miners_deployed, 8);
assert.equal(fleet.ships[7].collects, 8);
assert(Math.abs(fleet.score.total_collected_kg - 14271.4852840526) < 1e-8);
assert.equal(fleet.score.weighted_score_fixed_bonus_kg, 12992.407741224697);
const bytes = Buffer.from(serialize(fleet));
await writeFile(fleetPath, bytes);
manifest.files['fleet.json'] = { bytes: bytes.length, sha256: digest(bytes) };
for (const file of ['compute.json', 'checker-binding.json']) {
  await copyFile(join(kit, 'export', file), join(outputDirectory, file));
  const data = await readFile(join(outputDirectory, file));
  manifest.files[file] = { bytes: data.length, sha256: digest(data) };
}
await writeFile(join(outputDirectory, '.gitattributes'), '*.json -text -whitespace\n', { flag: 'wx' });
manifest.composition = compute.composition;
await writeFile(join(outputDirectory, 'manifest.json'), serialize(manifest));
const audit = {
  status: 'passed', dataset, importer_seconds: importerSeconds, source_result_sha256: binding.result_sha256,
  summary: manifest.summary, kepler_check: manifest.kepler_check, all_23_saved_histories_exact: true,
  copied_v799_histories: 22, copied_v627_histories: 1, new_propagations: 0, GPU_calls: 0,
  input_sha256: {}, output_sha256: {},
};
for (const path of ['scripts/import-gtoc12.mjs', 'scripts/import-data.mjs', 'kepler.js']) audit.input_sha256[path] = digest(await readFile(join(viewer, path)));
for (const file of ['fleet.json', 'manifest.json', 'compute.json', 'checker-binding.json', '.gitattributes']) audit.output_sha256[file] = digest(await readFile(join(outputDirectory, file)));
await writeFile(join(kit, 'import-audit.json'), serialize(audit), { flag: 'wx' });
console.log(JSON.stringify(audit));
