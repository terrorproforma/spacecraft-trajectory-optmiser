import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (path) => readFile(join(root, path));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const requiredIds = [
  "renderer-status", "renderer-status-text", "error-banner", "inventory-count",
  "trajectory-list", "data-mode", "mode-description", "play-button", "play-icon",
  "play-label", "reset-button", "timeline", "timeline-output", "sample-output",
  "trajectory-canvas", "family-label", "trajectory-title", "qualification-badge",
  "qualification-notice", "frame-overlay", "legend-overlay", "scene-overlay",
  "current-state", "frame-details", "validation-details", "gpu-details",
  "provenance-content", "viewport-wrap",
];

const [dataBytes, manifestBytes, htmlBytes, appBytes, cssBytes] = await Promise.all([
  read("data/trajectories.json"), read("data/manifest.json"), read("index.html"),
  read("app.js"), read("styles.css"),
]);
const data = JSON.parse(dataBytes);
const manifest = JSON.parse(manifestBytes);
const html = htmlBytes.toString();
const app = appBytes.toString();
const css = cssBytes.toString();

assert.equal(data.viewer_schema_version, "1.0.0");
assert.equal(data.imported_source_sha256, "83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2");
assert.equal(data.imported_source_sha256, manifest.source.sha256);
assert.equal(sha256(dataBytes), manifest.files["trajectories.json"].sha256);
assert.equal(dataBytes.length, manifest.files["trajectories.json"].bytes);
assert.equal(data.prohibitions.visual_interpolation_included, false);
assert.ok(data.archive?.data_dictionary && data.archive?.validation_report);
assert.deepEqual(data.trajectories.map((item) => item.family), ["P1-B", "P1-C", "P1-D", "P1-E", "P2"]);

for (const item of data.trajectories) {
  assert.equal(typeof item.qualification.qualified, "boolean", `${item.family} qualification`);
  assert.ok(item.qualification.label && item.raw_evidence_sha256?.length === 64);
  assert.equal(item.validation.finite, true);
  for (const mode of ["replay", "transcription"]) {
    const series = item[mode];
    assert.equal(series.point_count, series.points_txyz.length, `${item.family} ${mode} count`);
    assert.equal(series.point_count, series.selected_indices.length, `${item.family} ${mode} indices`);
    assert.equal(series.selected_indices[0], 0, `${item.family} ${mode} first endpoint`);
    assert.equal(series.selected_indices.at(-1), series.original_point_count - 1, `${item.family} ${mode} last endpoint`);
    assert.ok(series.points_txyz.every((point) =>
      Array.isArray(point) && point.length === 4 && point.every(Number.isFinite)),
    `${item.family} ${mode} finite txyz`);
    assert.ok(series.points_txyz.every((point, index) =>
      index === 0 || point[0] >= series.points_txyz[index - 1][0]),
    `${item.family} ${mode} time order`);
    assert.ok(series.selected_indices.every((value, index) =>
      Number.isInteger(value) && (index === 0 || value > series.selected_indices[index - 1])),
    `${item.family} ${mode} source index order`);
  }
  assert.ok(item.replay.point_count >= 201 && item.replay.point_count <= 512, `${item.family} replay density`);
  assert.deepEqual(item.replay.points_txyz[0].slice(1), item.initial_state.slice(0, 3), `${item.family} replay initial state`);
}

const byFamily = Object.fromEntries(data.trajectories.map((item) => [item.family, item]));
assert.equal(byFamily["P1-B"].viewer.scene_kind, "hcw");
assert.equal(byFamily["P1-B"].viewer.body_radius, null);
for (const family of ["P1-C", "P1-D"]) {
  assert.equal(byFamily[family].viewer.scene_kind, "local-surface");
  assert.equal(byFamily[family].viewer.body_radius, 0);
  assert.match(byFamily[family].viewer.radius_label, /Z = 0/);
}
assert.equal(byFamily["P1-E"].viewer.body_radius, 6500);
assert.match(byFamily["P1-E"].viewer.radius_label, /constraint/i);
assert.equal(byFamily.P2.viewer.body_radius, 6378136.3);
assert.match(byFamily.P2.frame, /Earth-centred inertial/);

for (const id of requiredIds) assert.match(html, new RegExp(`id=["']${id}["']`), `DOM id ${id}`);
assert.match(html, /<script type="module" src="\.\/app\.js"><\/script>/);
assert.match(app, /from "\.\/math\.js"/);
assert.match(app, /getContext\("webgl2"/);
assert.match(app, /webglcontextlost/);
assert.match(app, /webglcontextrestored/);
assert.doesNotMatch(`${html}\n${css}\n${app}`, /https?:\/\/(?!localhost|127\.0\.0\.1)/i, "No external URLs");

console.log(`Validated ${data.trajectories.length} trajectories, ${data.trajectories.reduce((sum, item) => sum + item.replay.point_count, 0)} dense replay points`);
console.log(`Data SHA-256 ${manifest.files["trajectories.json"].sha256}`);
