import test from "node:test";
import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { importGtoc12, parseCatalogue, classifyEvent } from "../scripts/import-gtoc12.mjs";
import { serialize } from "../scripts/import-data.mjs";
import { EARTH_ELEMENTS_DEG, MISSION_END_MJD, MISSION_START_MJD, positionAt, prepareElements } from "../kepler.js";

const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const exists = (path) => access(path).then(() => true, () => false);

const ASTEROIDS = {
  7: { a_au: 2.774, e: 0.0858, i_deg: 4.36, node_deg: 196.02, peri_deg: 152.49, m0_deg: 276.7315 },
  9: { a_au: 3.05, e: 0.12, i_deg: 9.1, node_deg: 30.5, peri_deg: 80.2, m0_deg: 12.3 },
};

/** Synthetic 60,000-row catalogue in the official column layout; rows 7 and 9 carry known elements. */
function syntheticCatalogue() {
  const lines = ["  ID    epoch(MJD)       a(AU)            e             i(deg)         LAN(deg)      argperi(deg)       M(deg)"];
  for (let id = 1; id <= 60000; id += 1) {
    const row = ASTEROIDS[id] ?? { a_au: 2.5 + (id % 100) / 100, e: 0.1, i_deg: 5, node_deg: id % 360, peri_deg: (id * 7) % 360, m0_deg: (id * 13) % 360 };
    lines.push(`${String(id).padStart(5)}   64328        ${row.a_au.toExponential(6)}    ${row.e.toExponential(6)}    ${row.i_deg.toExponential(6)}    ${row.node_deg.toExponential(6)}    ${row.peri_deg.toExponential(6)}    ${row.m0_deg.toExponential(6)}`);
  }
  return `${lines.join("\n")}\n`;
}

/** One-ship export in the `spacepdhcg gtoc12 export-viewer` record schema. */
function syntheticExport() {
  const earth = prepareElements(EARTH_ELEMENTS_DEG);
  const bodies = Object.fromEntries(Object.entries(ASTEROIDS).map(([id, elements]) => [id, prepareElements({ ...elements, epoch_mjd: 64328 })]));
  const events = [
    { event_id: 0, kind: "launch", epoch_mjd: 64400, mass_before_kg: 3000, mass_after_kg: 3000 },
    { event_id: 7, kind: "rendezvous", epoch_mjd: 65000, mass_before_kg: 2600, mass_after_kg: 2560 },
    { event_id: 9, kind: "rendezvous", epoch_mjd: 65400, mass_before_kg: 2400, mass_after_kg: 2360 },
    { event_id: 7, kind: "rendezvous", epoch_mjd: 67000, mass_before_kg: 1800, mass_after_kg: 1854.75 },
    { event_id: 9, kind: "rendezvous", epoch_mjd: 67500, mass_before_kg: 1700, mass_after_kg: 1757.5 },
    { event_id: -3, kind: "flyby", epoch_mjd: 69000, mass_before_kg: 1200, mass_after_kg: 1087.75 },
  ];
  const epochs = [...new Set([...events.map((event) => event.epoch_mjd), 64700, 66000, 66500, 68000, 68500])].sort((a, b) => a - b);
  const position = (event) => (event.event_id > 0 ? positionAt(bodies[event.event_id], event.epoch_mjd) : positionAt(earth, event.epoch_mjd));
  const replay = epochs.map((epoch) => {
    const event = events.find((item) => item.epoch_mjd === epoch);
    return [epoch, ...(event ? position(event) : [1e8 * Math.cos(epoch / 400), 1e8 * Math.sin(epoch / 400), 1e5])];
  });
  const transcription = events.map((event) => [event.epoch_mjd, ...position(event)]);
  const grid = Array.from({ length: 37 }, (_, index) => MISSION_START_MJD + (MISSION_END_MJD - MISSION_START_MJD) * index / 36);
  const context = [
    ...Object.entries(bodies).map(([id, body]) => ({ body: `asteroid ${id}`, points_txyz: grid.map((epoch) => [epoch, ...positionAt(body, epoch)]) })),
    { body: "Earth", points_txyz: grid.map((epoch) => [epoch, ...positionAt(earth, epoch)]) },
  ];
  const series = (points) => ({ original_point_count: points.length, original_sha256: hash(JSON.stringify(points)), point_count: points.length, points_txyz: points, selected_indices: points.map((_, index) => index) });
  return {
    viewer_schema_version: "1.0.0", schema_version: "1.0.0", title: "synthetic", generated_by_commit: "deadbeef",
    imported_source_sha256: hash("solution"), prohibitions: { visual_interpolation_included: false },
    archive: { validation_report: { ok: true, total_mass_kg: 112.25, mined_asteroids: 2, ships: 1, ship_limit: 3.1 } },
    trajectories: [{
      trajectory_id: "gtoc12_synthetic_ship1", family: "GTOC12", physical_family: "GTOC12 ship", frame: "J2000 heliocentric ecliptic Cartesian [x, y, z]",
      position_units: "km", time_units: "MJD (days)", events, asteroids_visited: [7, 9, 7, 9],
      mass_summary: { initial_kg: 3000, final_kg: 1087.75, minimum_kg: 1087.75 }, controls_summary: {},
      qualification: { qualified: true, label: "synthetic" }, raw_evidence_sha256: hash("solution"),
      replay: series(replay), transcription: series(transcription), context_orbits: context,
      source: { campaign: "test", commit: "deadbeef", run_id: "synthetic", generator: "test" }, viewer: { scene_kind: "heliocentric" },
    }],
  };
}

async function writeFixture(directory) {
  const exportBytes = Buffer.from(serialize(syntheticExport()));
  const manifest = { schema_version: "1.0.0", files: { "trajectories.json": { bytes: exportBytes.length, sha256: hash(exportBytes) } }, source: { path_basename: "Result.txt", sha256: hash("solution"), bytes: 8 }, transform: "test" };
  const catalogue = Buffer.from(syntheticCatalogue());
  await Promise.all([
    writeFile(join(directory, "trajectories.json"), exportBytes),
    writeFile(join(directory, "manifest.json"), serialize(manifest)),
    writeFile(join(directory, "GTOC12_Asteroids_Data.txt"), catalogue),
    writeFile(join(directory, "Result.txt"), "solution"),
  ]);
  return { cataloguePin: { name: "GTOC12_Asteroids_Data.txt", bytes: catalogue.length, sha256: hash(catalogue) } };
}

test("event roles follow the GTOC12 mass bookkeeping", () => {
  assert.equal(classifyEvent({ kind: "launch", event_id: 0, mass_before_kg: 3000, mass_after_kg: 3000 }, 0, 3), "launch");
  assert.equal(classifyEvent({ kind: "rendezvous", event_id: 5, mass_before_kg: 1000, mass_after_kg: 960 }, 1, 3), "deploy");
  assert.equal(classifyEvent({ kind: "rendezvous", event_id: 5, mass_before_kg: 900, mass_after_kg: 950 }, 1, 3), "collect");
  assert.equal(classifyEvent({ kind: "flyby", event_id: -3, mass_before_kg: 900, mass_after_kg: 850 }, 2, 3), "earth-return");
  assert.throws(() => classifyEvent({ kind: "rendezvous", event_id: 5, mass_before_kg: 1000, mass_after_kg: 990 }, 1, 3), /deploy event/);
});

test("synthetic export imports deterministically with verified hashes and classified events", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gtoc12-import-"));
  try {
    const { cataloguePin } = await writeFixture(directory);
    const options = { exportDirectory: directory, cataloguePath: join(directory, "GTOC12_Asteroids_Data.txt"), solutionPath: join(directory, "Result.txt"), cataloguePin };
    const first = await importGtoc12({ ...options, outputDirectory: join(directory, "out-a") });
    const second = await importGtoc12({ ...options, outputDirectory: join(directory, "out-b") });
    const [aBytes, bBytes] = await Promise.all([readFile(join(directory, "out-a", "fleet.json")), readFile(join(directory, "out-b", "fleet.json"))]);
    assert.equal(hash(aBytes), hash(bBytes));
    assert.equal(hash(aBytes), first.files["fleet.json"].sha256);
    assert.deepEqual(first, second);
    const fleet = JSON.parse(aBytes);
    assert.equal(fleet.dataset_kind, "gtoc12-fleet");
    assert.equal(fleet.ships.length, 1);
    assert.deepEqual(fleet.ships[0].events.map((event) => event.role), ["launch", "deploy", "deploy", "collect", "collect", "earth-return"]);
    assert.deepEqual(fleet.ships[0].asteroids, [7, 9]);
    assert.ok(Math.abs(fleet.ships[0].collected_kg - 112.25) < 1e-9);
    assert.equal(fleet.ships[0].miners_deployed, 2);
    assert.equal(fleet.score.official_total_mass_kg, 112.25);
    assert.deepEqual(fleet.asteroids.map((asteroid) => asteroid.id), [7, 9]);
    assert.deepEqual(fleet.asteroids[0].visited_by, [1]);
    assert.equal(fleet.asteroids[1].a_au, 3.05);
    assert.ok(fleet.kepler_check.asteroid_max_error_km < 1e-6 && fleet.kepler_check.earth_max_error_km < 1e-6);
    assert.equal(fleet.kepler_check.context_points_checked, 3 * 37);
    assert.equal(fleet.source.solution_verified_locally, true);
    assert.equal(fleet.source.catalogue.sha256, cataloguePin.sha256);
    assert.deepEqual(fleet.ships[0].replay, JSON.parse(await readFile(join(directory, "trajectories.json"))).trajectories[0].replay, "replay samples copied verbatim");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("tampered export or wrong catalogue is refused", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gtoc12-import-bad-"));
  try {
    const { cataloguePin } = await writeFixture(directory);
    const options = { exportDirectory: directory, cataloguePath: join(directory, "GTOC12_Asteroids_Data.txt"), cataloguePin, outputDirectory: join(directory, "out") };
    await assert.rejects(importGtoc12({ ...options, cataloguePin: { ...cataloguePin, sha256: "0".repeat(64) } }), /not the pinned GTOC12 catalogue/);
    const bytes = await readFile(join(directory, "trajectories.json"));
    bytes[bytes.length - 3] = bytes[bytes.length - 3] === 0x20 ? 0x09 : 0x20;
    await writeFile(join(directory, "trajectories.json"), bytes);
    await assert.rejects(importGtoc12(options), /SHA-256 .* differs from manifest/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("catalogue parser requires the official 60,000-row layout", () => {
  const rows = parseCatalogue(syntheticCatalogue());
  assert.equal(rows.size, 60000);
  assert.equal(rows.get(7).a_au, 2.774);
  assert.throws(() => parseCatalogue("header\n1 64328 2.5 0.1 5 10 20 30\n"), /expected 60000/);
});

const realExport = process.env.GTOC12_EXPORT_DIR
  || String.raw`\\wsl.localhost\Ubuntu-22.04\home\angus\worktrees\spacepdhcg-gtoc12\results\gtoc12\viewer-exports\fleet_master_v1`;
const realCatalogue = process.env.GTOC12_CATALOGUE
  || String.raw`\\wsl.localhost\Ubuntu-22.04\home\angus\worktrees\spacepdhcg-gtoc12\benchmarks\gtoc12\data\GTOC12_Asteroids_Data.txt`;
const haveReal = await exists(join(realExport, "manifest.json")) && await exists(realCatalogue);

test("fleet_master_v1 export imports to the 15-ship, 109-asteroid, 7575.58 kg fleet", { skip: !haveReal && "GTOC12 export not reachable" }, async () => {
  const directory = await mkdtemp(join(tmpdir(), "gtoc12-real-"));
  try {
    const manifest = await importGtoc12({ exportDirectory: realExport, cataloguePath: realCatalogue, outputDirectory: directory });
    assert.equal(manifest.summary.ships, 15);
    assert.equal(manifest.summary.unique_asteroids, 109);
    assert.equal(manifest.summary.replay_points, 7622);
    assert.equal(manifest.summary.official_total_mass_kg, 7575.58);
    assert.equal(manifest.source.solution_sha256, "61603bb44b2ea7e9f43b45b8899fe437d1f984e9e89db2fd756d76d52bae7c35");
    assert.ok(manifest.kepler_check.asteroid_max_error_km < 1e-4);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
