import assert from 'node:assert/strict';
import { readFile, writeFile, copyFile, mkdir, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { gzipSync, gunzipSync } from 'node:zlib';
import { dirname, resolve, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const kit = dirname(fileURLToPath(import.meta.url));
const root = resolve(kit, '../../..');
const output = join(root, 'results/local/2026-09-09/current-fleet-composition-v628');
const campaign = join(root, 'build/performance/seeded-current-composition-v628');
const review = join(root, 'build/performance/current-composition-review-v628');
const previous = join(root, 'build/performance/seeded-candidate-boundary-merit-v627/campaign');
const base = join(root, 'results/lambda/2026-09-09/gpu-regeneration-v799/h100-best');
const viewer = join(root, 'results/lambda/2026-09-06/visualiser');
const hash = data => createHash('sha256').update(data).digest('hex');
const read = async p => JSON.parse(await readFile(p, 'utf8'));
const write = async (name, value) => {
  const path = join(output, name);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
};
const copied = [];
const copy = async (source, target) => {
  const data = await readFile(source);
  await write(target, data);
  copied.push({ path: target, source: relative(root, source).replaceAll('\\', '/'), bytes: data.length, sha256: hash(data) });
};
async function files(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) result.push(...await files(path));
    else if (entry.isFile()) result.push(path);
    else throw new Error(`Unexpected non-file entry: ${path}`);
  }
  return result.sort();
}
function sections(data) {
  const result = new Map();
  for (const line of data.toString('latin1').match(/[^\n]*\n|[^\n]+$/g) || []) {
    const ship = line.trim().split(/\s+/)[0];
    if (!ship) continue;
    assert(/^\d+$/.test(ship));
    result.set(ship, Buffer.concat([result.get(ship) || Buffer.alloc(0), Buffer.from(line, 'latin1')]));
  }
  return result;
}

await assert.rejects(stat(output), { code: 'ENOENT' });
await mkdir(output);
await write('.gitattributes', '* -text -whitespace\n');
const plan = await read(join(campaign, 'plan.json'));
const sourceBindings = [];
for (const [name, expected] of Object.entries(plan.source_sha256)) {
  const bytes = await readFile(join(root, name));
  assert.equal(hash(bytes), expected, name);
  sourceBindings.push({ path: name, bytes: bytes.length, sha256: expected });
}
await copy(join(campaign, 'inputs/Result.txt'), 'Result.txt');
assert.equal(hash(await readFile(join(output, 'Result.txt'))), plan.composed_result_sha256);
for (const file of ['report.json', 'checker-binding.json']) await copy(join(campaign, 'output', file), file);
for (const file of ['independent-full.json', 'independent.json', 'official-full.json', 'official.json', 'partial-report.json', 'launch-report.json', 'worker.log', 'official/ScoreData.txt']) {
  await copy(join(campaign, 'output', file), `raw/${file}`);
}
for (const file of ['prepare.py', 'run.py', 'plan.json', 'launch-marker.json']) await copy(join(campaign, file), `run/${file}`);
for (const path of await files(review)) await copy(path, `review/${relative(review, path).replaceAll('\\', '/')}`);
for (const [source, target] of [
  [join(base, 'campaign-report.json'), 'lineage/v799-campaign-report.json'],
  [join(previous, 'output/report.json'), 'lineage/v627-report.json'],
  [join(previous, 'output/fleet/checker-binding.json'), 'lineage/v627-checker-binding.json'],
  [join(previous, 'ready.json'), 'lineage/v627-ready.json'],
  [join(previous, 'common.py'), 'lineage/v627-common.py'],
  [join(previous, 'profile.json'), 'lineage/v627-profile.json'],
]) await copy(source, target);
await write('lineage/source-bindings.json', { status: 'all_original_plan_inputs_rehashed', sources: sourceBindings,
  note: 'The original execution scripts retain their workspace-relative paths. Native/source lineage is published separately in endpoint-merit-and-gpu-dual-v627; no native or verifier executables are copied here.' });
const current = sections(await readFile(join(output, 'Result.txt')));
const old = sections(await readFile(join(base, 'Result.txt')));
const replacement = sections(await readFile(join(previous, 'output/fleet/Result.txt')));
const proof = { current_result_sha256: plan.composed_result_sha256, H100_v799_Result_sha256: plan.source_result_sha256,
  local_v627_Result_sha256: plan.certified_route_result_sha256, sections: {} };
assert.equal(current.size, 23);
for (const [ship, data] of current) {
  const selected = (ship === '8' ? replacement : old).get(ship);
  assert(data.equals(selected));
  proof.sections[ship] = { bytes: data.length, composed_sha256: hash(data), selected_source: ship === '8' ? 'local_v627' : 'H100_v799',
    selected_source_sha256: hash(selected), baseline_sha256: hash(old.get(ship)) };
}
await write('lineage/retained-sections.json', proof);
const archives = [];
async function compressed(source, target) {
  const data = await readFile(source), compressedBytes = gzipSync(data, { level: 9 });
  assert(gunzipSync(compressedBytes).equals(data));
  await write(target, compressedBytes);
  archives.push({ path: target, source: relative(root, source).replaceAll('\\', '/'), compressed_bytes: compressedBytes.length,
    compressed_sha256: hash(compressedBytes), uncompressed_bytes: data.length, uncompressed_sha256: hash(data), roundtrip_exact: true });
}
for (const file of ['export.mjs', 'import.mjs', 'verify-http.mjs', 'README.md', 'export-audit.json', 'import-audit.json', 'http-audit.json', 'schema.log', 'syntax.log']) {
  await copy(join(kit, file), `viewer/${file}`);
}
for (const file of ['manifest.json', 'fleet-summary.json', 'checker-binding.json', 'compute.json']) await copy(join(kit, 'export', file), `viewer/export/${file}`);
await compressed(join(kit, 'export/trajectories.json'), 'viewer/export/trajectories.json.gz');
await compressed(join(viewer, 'data/gtoc12-current-composition-v628/fleet.json'), 'viewer/installed/fleet.json.gz');
for (const file of ['manifest.json', 'checker-binding.json', 'compute.json', '.gitattributes']) {
  await copy(join(viewer, 'data/gtoc12-current-composition-v628', file), `viewer/installed/${file}`);
}
for (const file of ['app.js', 'index.html', 'scripts/import-gtoc12.mjs', 'scripts/import-data.mjs', 'kepler.js']) await copy(join(viewer, file), `viewer/source/${file}`);
await write('archive-audit.json', { status: 'all_gzip_roundtrips_byte_exact', files: archives });
await copy(join(kit, 'verify_package.py'), 'reproduce/verify_package.py');
await copy(join(kit, 'package.mjs'), 'reproduce/package.mjs');
await copy(join(kit, 'package-README.md'), 'README.md');
await write('copy-manifest.json', { files: copied, excluded_duplicate_artifacts: [
  { path: 'output/official/Result.txt', reason: 'Exact canonical Result.txt already copied', sha256: plan.composed_result_sha256 },
  { path: 'output/official/GTOC12_Asteroids_Data.txt', reason: 'Official catalogue is an existing pinned external dataset; no duplicate', sha256: (await read(join(campaign, 'output/report.json'))).catalogue_sha256 },
  { path: 'output/official/GTOC12_Verify', reason: 'No executable binaries in this package', sha256: 'd4e4bc81129266420b27c9bde038bce9eda1960e7de9c695772fbfdb1cc82cd6' },
] });
const indexed = {};
for (const path of await files(output)) {
  const bytes = await readFile(path);
  indexed[relative(output, path).replaceAll('\\', '/')] = { bytes: bytes.length, sha256: hash(bytes) };
}
await write('index.json', { file_count: Object.keys(indexed).length, bytes: Object.values(indexed).reduce((a, x) => a + x.bytes, 0), files: indexed });
console.log(JSON.stringify({ output: relative(root, output).replaceAll('\\', '/'), file_count: Object.keys(indexed).length,
  index_sha256: hash(await readFile(join(output, 'index.json'))), bytes: Object.values(indexed).reduce((a, x) => a + x.bytes, 0) }));
