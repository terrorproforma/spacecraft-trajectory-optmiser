import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const kit = dirname(fileURLToPath(import.meta.url));
const viewer = resolve(kit, '../../../results/lambda/2026-09-06/visualiser');
const digest = b => createHash('sha256').update(b).digest('hex');
const stages = [];
for (const [name, args] of [['schema', ['scripts/check.mjs']], ['syntax', ['--check', 'app.js']]]) {
  const result = spawnSync(process.execPath, args, { cwd: viewer, encoding: 'utf8', timeout: 60000 });
  await writeFile(join(kit, `${name}.log`), result.stdout + result.stderr, { flag: 'wx' });
  stages.push({ name, command: ['node', ...args], exit_code: result.status });
  assert.equal(result.status, 0);
}
const paths = ['index.html', 'app.js', ...['manifest.json', 'fleet.json', 'compute.json', 'checker-binding.json'].map(p => `data/gtoc12-current-composition-v628/${p}`)];
const checks = [];
for (const path of paths) {
  const response = await fetch(`http://127.0.0.1:4173/${path}`, { signal: AbortSignal.timeout(10000) });
  assert.equal(response.status, 200);
  const bytes = Buffer.from(await response.arrayBuffer());
  assert.equal(digest(bytes), digest(await readFile(join(viewer, path))));
  checks.push({ path, status: response.status, bytes: bytes.length, sha256: digest(bytes) });
}
const app = await readFile(join(viewer, 'app.js'), 'utf8');
const html = await readFile(join(viewer, 'index.html'), 'utf8');
assert(app.includes('const initialDataset = availableFleets.has(datasetParam) ? datasetParam : "gtoc12-current-composition-v628"'));
assert(html.includes('<option value="gtoc12-current-composition-v628" selected disabled>Local fleet composition v628'));
assert(!html.includes('<option value="gtoc12" selected'));
for (const retained of ['gtoc12-regeneration-v799', 'gtoc12-seeded-candidate-v627']) {
  assert(app.includes(retained) && html.includes(retained));
}
assert(!app.split('\n').find(line => line.includes('"gtoc12-orphan-v595":')).includes('best'));
const report = {
  status: 'passed', checked_utc: new Date().toISOString(), http_checks: checks, stages,
  url: 'http://127.0.0.1:4173/?dataset=gtoc12-current-composition-v628&ship=8&epoch=69807&preset=oblique&z=1',
  default_dataset: 'gtoc12-current-composition-v628',
  H100_v799_and_historical_v627_retained: true, stale_v595_best_label_removed: true,
  independent_history_audit_sha256: 'd22bd60ba92d5caf816fefd9b82acaafb2d0fec4b2b05c6270f206d6fa99f37c',
  server: 'Existing localhost:4173 server returned exact local bytes; no server was started or stopped.',
  browser_visual_check: 'HTTP and data/schema verification only; parent will open the user-facing URL.',
};
await writeFile(join(kit, 'http-audit.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ status: report.status, http_paths_verified: checks.length, default_dataset: report.default_dataset }));
