import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PALETTE_FLOORS, SHIP_PALETTE_SPEC, generateShipPalette, maxChannelDifference, paletteMetrics } from "./palette.mjs";

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

const [dataBytes, manifestBytes, htmlBytes, appBytes, cssBytes, gtocBytes, webglBytes, keplerBytes, cameraBytes, plotBytes] = await Promise.all([
  read("data/trajectories.json"), read("data/manifest.json"), read("index.html"),
  read("app.js"), read("styles.css"), read("gtoc12.js"), read("webgl.js"), read("kepler.js"), read("camera.js"),
  read("scripts/plot_gtoc12_fleet.py").catch((error) => { if (error.code === "ENOENT") return null; throw error; }),
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
// Ship palette: SHIP_COLOURS in gtoc12.js is the committed output of scripts/palette.mjs and is
// duplicated in styles.css (the CSP forbids inline styles) and in the matplotlib fallback. Its size
// is read from gtoc12.js (never hard-coded here) and must cover both the largest fleet the GTOC12
// solver has produced (fleet_master_v7: 21 ships) and the GTOC12 record fleet (Antipodes: 39 ships).
const colourList = (source, label) => {
  const block = /SHIP_COLOURS = \[([\s\S]*?)\]/.exec(source);
  assert.ok(block, `${label} declares SHIP_COLOURS = [...]`);
  return [...block[1].matchAll(/"(#[0-9a-f]{6})"/g)].map((match) => match[1]);
};
assert.match(String(gtocBytes), /export const SHIP_COLOURS = \[/, "SHIP_COLOURS palette is exported by gtoc12.js");
const shipColours = colourList(String(gtocBytes), "gtoc12.js");
const expectedShipCount = SHIP_PALETTE_SPEC.hues * SHIP_PALETTE_SPEC.bands.length;
assert.equal(shipColours.length, expectedShipCount, `${expectedShipCount} ship colours (${SHIP_PALETTE_SPEC.bands.length} lightness bands x ${SHIP_PALETTE_SPEC.hues} hues)`);
assert.ok(shipColours.length >= 21, `ship palette covers the 21-ship fleet_master_v7 fleet (found ${shipColours.length})`);
assert.ok(shipColours.length >= 40, `ship palette covers the GTOC12 record fleet (39 ships) with one spare (found ${shipColours.length})`);
assert.equal(new Set(shipColours).size, shipColours.length, "ship colours are distinct");
assert.match(String(gtocBytes), /export const MAX_SHIPS = SHIP_COLOURS\.length;/, "MAX_SHIPS is derived from the palette length");
// Regeneration: the committed hex values are the spec's output (per channel within 8-bit rounding).
const regenerated = generateShipPalette(SHIP_PALETTE_SPEC);
const channelDrift = maxChannelDifference(shipColours, regenerated);
assert.ok(channelDrift <= 2, `SHIP_COLOURS regenerate from SHIP_PALETTE_SPEC (max channel difference ${channelDrift}; run node scripts/palette.mjs)`);
// Distinctness (CIE76 dE*ab): consecutive ships, every pair, and against the reserved scene/UI colours.
const palette = paletteMetrics(shipColours, SHIP_PALETTE_SPEC.reserved);
assert.ok(palette.neighbour >= PALETTE_FLOORS.neighbour, `consecutive ships differ by >= ${PALETTE_FLOORS.neighbour} dE (min ${palette.neighbour.toFixed(1)} between ships ${palette.neighbourPair})`);
assert.ok(palette.pairwise >= PALETTE_FLOORS.pairwise, `every pair of ships differs by >= ${PALETTE_FLOORS.pairwise} dE (min ${palette.pairwise.toFixed(1)} between ships ${palette.pairwisePair})`);
assert.ok(palette.reserved >= PALETTE_FLOORS.reserved, `every ship differs from the Sun/Earth/asteroid/UI colours by >= ${PALETTE_FLOORS.reserved} dE (min ${palette.reserved.toFixed(1)}: ship ${palette.reservedPair?.[0]} vs ${palette.reservedPair?.[1]})`);
// Reserved colours in the spec are the ones the scene and stylesheet actually use.
for (const [name, token] of [["verified", "--verified"], ["caution", "--caution"], ["alert", "--alert"], ["focus", "--focus"], ["bone", "--bone"]]) {
  assert.match(css, new RegExp(`${token}: ${SHIP_PALETTE_SPEC.reserved[name]};`), `palette spec reserves the stylesheet's ${token}`);
}
const glColour = (name) => { const match = new RegExp(`const ${name} = \\[([^\\]]+)\\]`).exec(String(gtocBytes)); assert.ok(match, `${name} in gtoc12.js`); return `#${match[1].split(",").slice(0, 3).map((v) => Math.round(Number(v) * 255).toString(16).padStart(2, "0")).join("")}`; };
assert.equal(glColour("EARTH_COLOUR"), SHIP_PALETTE_SPEC.reserved.earth, "palette spec reserves the scene's Earth colour");
assert.equal(glColour("SUN_COLOUR"), SHIP_PALETTE_SPEC.reserved.sun, "palette spec reserves the scene's Sun colour");
assert.equal(glColour("PENDING_ASTEROID"), SHIP_PALETTE_SPEC.reserved.asteroidPending, "palette spec reserves the pending-asteroid colour");
// Mirrors: one `.ship-colour-N` class per palette entry (and none beyond), same order in the matplotlib fallback.
shipColours.forEach((colour, index) => assert.match(css, new RegExp(`\\.ship-colour-${index + 1} \\{ color: ${colour}; \\}`), `ship colour ${index + 1} in styles.css`));
assert.equal([...css.matchAll(/\.ship-colour-(\d+) \{/g)].length, shipColours.length, "styles.css has exactly one class per ship colour");
if (plotBytes) assert.deepEqual(colourList(String(plotBytes), "plot_gtoc12_fleet.py"), shipColours, "matplotlib fallback palette matches gtoc12.js");
assert.match(css, /\.ship-list\.dense \{/, "dense (> 20 ships) rail layout is styled");
assert.match(css, /\.legend-ships\.dense \{/, "dense (> 20 ships) legend layout is styled");
assert.match(css, /\[hidden\] \{ display: none !important; \}/, "hidden panels stay hidden");

/** Ship-count rule shared by the real dataset and the synthetic fleets below: one palette entry per ship, no wrapping. */
function assertPaletteCoversFleet(fleet, label) {
  const count = fleet.ships.length;
  assert.ok(count <= shipColours.length, `${label}: ${count} ships exceed the ${shipColours.length}-colour ship palette (SHIP_COLOURS in gtoc12.js); ship ${shipColours.length + 1} would repeat ship 1's colour`);
  const classes = fleet.ships.map((_, index) => `ship-colour-${index % shipColours.length + 1}`);
  assert.equal(new Set(classes).size, count, `${label}: every ship has its own colour class`);
  return classes;
}
const syntheticFleet = (count) => ({ ships: Array.from({ length: count }, (_, index) => ({ ship_id: index + 1 })) });
assertPaletteCoversFleet(syntheticFleet(21), "synthetic 21-ship fleet (fleet_master_v7 size)");
assertPaletteCoversFleet(syntheticFleet(39), "synthetic 39-ship fleet (GTOC12 record, Antipodes)");
assertPaletteCoversFleet(syntheticFleet(shipColours.length), `synthetic ${shipColours.length}-ship fleet (palette size)`);
const oversized = shipColours.length + 1;
assert.throws(() => assertPaletteCoversFleet(syntheticFleet(oversized), `synthetic ${oversized}-ship fleet`), new RegExp(`${oversized} ships exceed the ${shipColours.length}-colour ship palette`), `a ${oversized}-ship fleet is refused with a clear message`);
console.log(`Ship palette: ${shipColours.length} colours, min dE consecutive ${palette.neighbour.toFixed(1)}, pairwise ${palette.pairwise.toFixed(1)}, to reserved ${palette.reserved.toFixed(1)}; synthetic 21/39/${shipColours.length}-ship fleets pass, ${oversized} refused`);

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
  assertPaletteCoversFleet(fleet, `GTOC12 fleet ${fleet.run_id}`);
  for (const asteroid of fleet.asteroids) {
    assert.ok(asteroid.a_au > 0 && asteroid.e >= 0 && asteroid.e < 1 && asteroid.epoch_mjd === 64328, `asteroid ${asteroid.id} elements`);
    assert.ok(asteroid.visited_by.every((shipId) => fleet.ships.some((ship) => ship.ship_id === shipId && ship.asteroids.includes(asteroid.id))), `asteroid ${asteroid.id} visitors`);
  }
  console.log(`Validated GTOC12 fleet ${fleet.run_id}: ${fleet.ships.length} ships, ${fleet.asteroids.length} asteroids, ${fleet.ships.reduce((sum, ship) => sum + ship.replay.point_count, 0)} exact replay samples, ${collected.toFixed(2)} kg collected (official verifier ${fleet.score.official_total_mass_kg} kg)`);
  console.log(`Fleet SHA-256 ${fleetManifest.files["fleet.json"].sha256} · solution ${fleet.source.solution_sha256}`);
}
