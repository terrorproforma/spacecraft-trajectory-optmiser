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
  // Dataset selector and GTOC12 fleet panels (viewer works without the optional dataset).
  "dataset-select", "dataset-help", "ship-list", "fleet-count", "mission-timeline",
  "mission-timeline-output", "mission-play-button", "mission-play-icon", "mission-play-label",
  "focus-ship-button", "fleet-reset-button", "fleet-summary", "ship-detail", "ship-detail-title",
  "fleet-legend", "hover-tooltip", "fleet-provenance-content", "event-labels",
  // Perspective 3D controls (camera presets, follow-ship, vertical exaggeration, playback speed).
  "speed-select", "camera-presets", "follow-ship-button", "exaggeration", "exaggeration-output", "legend-ships",
  // Timeline strip tick rows (archive percentages are static HTML; mission years are rendered by app.js).
  "timeline-ticks", "mission-timeline-ticks",
];

const [dataBytes, manifestBytes, htmlBytes, appBytes, cssBytes, gtocBytes, webglBytes, keplerBytes, cameraBytes] = await Promise.all([
  read("data/trajectories.json"), read("data/manifest.json"), read("index.html"),
  read("app.js"), read("styles.css"), read("gtoc12.js"), read("webgl.js"), read("kepler.js"), read("camera.js"),
]);
const data = JSON.parse(dataBytes);
const manifest = JSON.parse(manifestBytes);
const html = htmlBytes.toString();
const app = appBytes.toString();
const css = cssBytes.toString();
const modules = `${app}\n${gtocBytes}\n${webglBytes}\n${keplerBytes}\n${cameraBytes}`;

// Two dataset kinds share the viewer: the verified archive (default) and planner
// exports written by `spacepdhcg plan --export-viewer`.  Archive-specific assertions
// (authoritative SHA, fixed family list, archive companions, 201..512 replay density)
// apply only to the archive; planner exports carry `dataset_kind: "planner-export"`.
const plannerExport = data.dataset_kind === "planner-export";

assert.equal(data.viewer_schema_version, "1.0.0");
if (plannerExport) {
  assert.equal(manifest.dataset_kind, "planner-export");
  assert.match(data.imported_source_sha256, /^[0-9a-f]{64}$/);
  assert.equal(data.generated_by, "spacepdhcg plan --export-viewer");
  assert.ok(Array.isArray(data.trajectories) && data.trajectories.length >= 1, "planner export trajectories");
} else {
  assert.equal(data.imported_source_sha256, "83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2");
  assert.ok(data.archive?.data_dictionary && data.archive?.validation_report);
  assert.deepEqual(data.trajectories.map((item) => item.family), ["P1-B", "P1-C", "P1-D", "P1-E", "P2"]);
}
assert.equal(data.imported_source_sha256, manifest.source.sha256);
assert.equal(sha256(dataBytes), manifest.files["trajectories.json"].sha256);
assert.equal(dataBytes.length, manifest.files["trajectories.json"].bytes);
assert.equal(data.prohibitions.visual_interpolation_included, false);

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
  if (plannerExport) {
    assert.ok(item.replay.point_count >= 2, `${item.family} replay density`);
    assert.ok(item.transcription.point_count >= 3, `${item.family} node count`);
    assert.ok(Array.isArray(item.terminal_target) && item.terminal_target.length >= 3, `${item.family} terminal target`);
    assert.ok(["hcw", "local-surface", "central-body"].includes(item.viewer.scene_kind), `${item.family} scene kind`);
    assert.ok(Array.isArray(item.viewer.axes) && item.viewer.axes.length === 3, `${item.family} axes`);
    assert.ok(typeof item.source.run_id === "string" && typeof item.source.commit === "string", `${item.family} provenance`);
    assert.ok(typeof item.frame === "string" && typeof item.position_units === "string", `${item.family} frame/units`);
    assert.equal(item.replay.points_txyz.at(-1)[0], item.transcription.points_txyz.at(-1)[0], `${item.family} replay reaches final time`);
  } else {
    assert.ok(item.replay.point_count >= 201 && item.replay.point_count <= 512, `${item.family} replay density`);
  }
  assert.deepEqual(item.replay.points_txyz[0].slice(1), item.initial_state.slice(0, 3), `${item.family} replay initial state`);
}

if (!plannerExport) {
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
}

for (const id of requiredIds) assert.match(html, new RegExp(`id=["']${id}["']`), `DOM id ${id}`);
assert.match(html, /<script type="module" src="\.\/app\.js"><\/script>/);
assert.match(app, /from "\.\/math\.js"/);
assert.match(app, /from "\.\/gtoc12\.js"/);
assert.match(modules, /getContext\("webgl2"/);
assert.match(app, /webglcontextlost/);
assert.match(app, /webglcontextrestored/);
assert.match(app, /data\/gtoc12\/fleet\.json/, "GTOC12 dataset is fetched from data/gtoc12/");
assert.match(html, /<option value="gtoc12"/, "dataset selector offers the GTOC12 fleet");
assert.match(html, /segments connect exact archived samples — no interpolation/i, "straight-segment caveat in the fleet legend");
assert.match(html, /Vertical exaggeration — <em>not physical<\/em>/, "exaggeration slider is labelled as not physical");
assert.match(html, /id="exaggeration"[\s\S]*?value="6"/, "the fleet view opens at 6x vertical exaggeration");
assert.match(css, /#trajectory-canvas \{[^}]*height: max\(560px, 72vh\)/, "canvas fills >= 72% of the window height on desktop");
for (const name of ["BACKGROUND_VERTEX", "TUBE_VERTEX", "drawArraysInstanced", "drawElements", "tubeArrays", "fogUniforms"]) {
  assert.ok(gtocBytes.toString().includes(name), `${name} is used by the fleet renderer (instanced spheres, tube arcs, fog, procedural sky)`);
}
for (const preset of ["top", "oblique", "edge", "follow"]) assert.match(html, new RegExp(`data-preset="${preset}"`), `camera preset ${preset}`);
assert.match(app, /from "\.\/camera\.js"/);
for (const name of ["BODY_VERTEX", "STAR_VERTEX", "uZScale", "starField", "concatRibbons"]) assert.match(`${gtocBytes}\n${webglBytes}`, new RegExp(name), `3D scene uses ${name}`);
assert.doesNotMatch(`${html}\n${css}\n${modules}`, /https?:\/\/(?!localhost|127\.0\.0\.1)/i, "No external URLs");
// Ship colours are duplicated between gtoc12.js and styles.css because the CSP forbids inline styles.
const shipColours = [...String(gtocBytes).matchAll(/"(#[0-9a-f]{6})"/g)].map((match) => match[1]).slice(0, 20);
assert.equal(shipColours.length, 20, "twenty ship colours");
assert.equal(new Set(shipColours).size, 20, "ship colours are distinct");
shipColours.forEach((colour, index) => assert.match(css, new RegExp(`\\.ship-colour-${index + 1} \\{ color: ${colour}; \\}`), `ship colour ${index + 1} in styles.css`));
assert.match(css, /\[hidden\] \{ display: none !important; \}/, "hidden panels stay hidden");

console.log(`Validated ${data.trajectories.length} ${plannerExport ? "planner-export" : "archive"} trajectories, ${data.trajectories.reduce((sum, item) => sum + item.replay.point_count, 0)} dense replay points`);
console.log(`Data SHA-256 ${manifest.files["trajectories.json"].sha256}`);

// Optional GTOC12 fleet dataset (data/gtoc12/ is ignored by git; see README "GTOC12 fleet dataset").
let fleetBytes = null, fleetManifestBytes = null;
try {
  [fleetBytes, fleetManifestBytes] = await Promise.all([read("data/gtoc12/fleet.json"), read("data/gtoc12/manifest.json")]);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
if (!fleetBytes) {
  console.log("GTOC12 fleet dataset not installed (optional): run `npm run import-gtoc12 -- --export <dir> --catalogue <GTOC12_Asteroids_Data.txt>`");
} else {
  const fleet = JSON.parse(fleetBytes);
  const fleetManifest = JSON.parse(fleetManifestBytes);
  assert.equal(fleet.viewer_schema_version, "1.0.0");
  assert.equal(fleet.dataset_kind, "gtoc12-fleet");
  assert.equal(fleetManifest.dataset_kind, "gtoc12-fleet");
  assert.equal(sha256(fleetBytes), fleetManifest.files["fleet.json"].sha256, "fleet.json hash matches its manifest");
  assert.equal(fleetBytes.length, fleetManifest.files["fleet.json"].bytes);
  assert.equal(fleet.prohibitions.visual_interpolation_included, false);
  assert.match(fleet.source.solution_sha256, /^[0-9a-f]{64}$/);
  assert.match(fleet.source.export_trajectories_sha256, /^[0-9a-f]{64}$/);
  assert.equal(fleet.source.catalogue.sha256, "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675", "pinned GTOC12 catalogue");
  assert.ok(fleet.kepler_check.asteroid_max_error_km <= fleet.kepler_check.tolerance_km && fleet.kepler_check.earth_max_error_km <= fleet.kepler_check.tolerance_km, "Kepler cross-check within tolerance");
  assert.equal(fleet.constants.mission_start_mjd, 64328);
  assert.equal(fleet.constants.mission_end_mjd, 69807);
  assert.ok(Array.isArray(fleet.ships) && fleet.ships.length >= 1 && fleet.ships.length <= 100, "ship count within GTOC12 bounds");
  assert.equal(fleet.score.ships, fleet.ships.length);
  assert.equal(fleet.score.unique_asteroids, fleet.asteroids.length);
  const asteroidIds = new Set(fleet.asteroids.map((asteroid) => asteroid.id));
  let collected = 0;
  for (const ship of fleet.ships) {
    const label = `ship ${ship.ship_id}`;
    for (const mode of ["replay", "transcription"]) {
      const series = ship[mode];
      assert.equal(series.point_count, series.points_txyz.length, `${label} ${mode} count`);
      assert.equal(series.point_count, series.selected_indices.length, `${label} ${mode} indices`);
      assert.equal(series.selected_indices[0], 0, `${label} ${mode} first endpoint`);
      assert.equal(series.selected_indices.at(-1), series.original_point_count - 1, `${label} ${mode} last endpoint`);
      assert.ok(series.points_txyz.every((point) => point.length === 4 && point.every(Number.isFinite)), `${label} ${mode} finite txyz`);
      assert.ok(series.points_txyz.every((point, index) => index === 0 || point[0] >= series.points_txyz[index - 1][0]), `${label} ${mode} time order`);
      assert.match(series.original_sha256, /^[0-9a-f]{64}$/, `${label} ${mode} original hash`);
    }
    assert.ok(ship.replay.point_count <= 512, `${label} replay density`);
    assert.equal(ship.transcription.point_count, ship.events.length, `${label} one transcription node per event`);
    assert.equal(ship.events[0].role, "launch", `${label} starts with launch`);
    assert.equal(ship.events.at(-1).role, "earth-return", `${label} ends with Earth return`);
    const replayEpochs = new Set(ship.replay.points_txyz.map((point) => point[0]));
    ship.events.forEach((event, index) => {
      assert.ok(replayEpochs.has(event.epoch_mjd), `${label} event ${index} epoch is an archived sample`);
      assert.deepEqual(event.position_km, ship.transcription.points_txyz[index].slice(1), `${label} event ${index} position is the transcription node`);
      assert.ok(event.epoch_mjd >= 64328 && event.epoch_mjd <= 69807, `${label} event ${index} inside mission window`);
      if (event.event_id > 0) assert.ok(asteroidIds.has(event.event_id), `${label} event ${index} asteroid ${event.event_id} has pinned elements`);
      if (event.role === "deploy") assert.ok(Math.abs(event.mass_delta_kg + 40) < 1e-6, `${label} deploy drops one 40 kg miner`);
      if (event.role === "collect") assert.ok(event.mass_delta_kg > 0, `${label} collect gains mass`);
    });
    const shipCollected = ship.events.filter((event) => event.role === "collect").reduce((sum, event) => sum + event.mass_delta_kg, 0);
    assert.ok(Math.abs(shipCollected - ship.collected_kg) < 1e-9, `${label} collected mass from events`);
    collected += shipCollected;
  }
  assert.ok(Math.abs(collected - fleet.score.total_collected_kg) < 1e-6, "fleet collected mass from events");
  // The official verifier prints six significant digits (7575.58, 10700.5), so compare at that precision.
  assert.equal(Number(collected.toPrecision(6)), fleet.score.official_total_mass_kg, "official score equals summed collects to 6 significant digits");
  assert.ok(fleet.ships.length <= 20, "ship palette covers every ship without wrapping");
  for (const asteroid of fleet.asteroids) {
    assert.ok(asteroid.a_au > 0 && asteroid.e >= 0 && asteroid.e < 1 && asteroid.epoch_mjd === 64328, `asteroid ${asteroid.id} elements`);
    assert.ok(asteroid.visited_by.every((shipId) => fleet.ships.some((ship) => ship.ship_id === shipId && ship.asteroids.includes(asteroid.id))), `asteroid ${asteroid.id} visitors`);
  }
  console.log(`Validated GTOC12 fleet ${fleet.run_id}: ${fleet.ships.length} ships, ${fleet.asteroids.length} asteroids, ${fleet.ships.reduce((sum, ship) => sum + ship.replay.point_count, 0)} exact replay samples, ${collected.toFixed(2)} kg collected (official verifier ${fleet.score.official_total_mass_kg} kg)`);
  console.log(`Fleet SHA-256 ${fleetManifest.files["fleet.json"].sha256} · solution ${fleet.source.solution_sha256}`);
}
