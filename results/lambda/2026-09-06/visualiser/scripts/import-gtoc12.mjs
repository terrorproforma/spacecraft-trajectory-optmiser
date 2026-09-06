// Build the viewer's GTOC12 fleet dataset (data/gtoc12/fleet.json + manifest.json) from a
// `spacepdhcg gtoc12 export-viewer` directory and the pinned asteroid catalogue.
//
//   node scripts/import-gtoc12.mjs --export <dir with trajectories.json + manifest.json> \
//     --catalogue <GTOC12_Asteroids_Data.txt> [--solution <Result.txt>] [--fleet <fleet.json>] \
//     [--output data/gtoc12]
//
// The importer verifies every hash it can (export manifest, Result.txt, catalogue pin,
// fleet.json cross-references), keeps the exact propagated replay samples, selected indices and
// original-sample hashes verbatim, classifies each archived event (launch / deploy / collect /
// Earth return), attaches the pinned Keplerian elements of every visited asteroid plus Earth, and
// checks the viewer's Kepler propagation against the exporter's context orbits. Nothing is
// interpolated or fabricated; the output is ignored by git and regenerable.

import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  AU_KM, DAY_S, EARTH_ELEMENTS_DEG, ELEMENT_EPOCH_MJD, MISSION_END_MJD, MISSION_START_MJD,
  MU_SUN_KM3_S2, positionAt, prepareElements,
} from "../kepler.js";
import { serialize, sha256 } from "./import-data.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const CATALOGUE_PIN = {
  name: "GTOC12_Asteroids_Data.txt",
  bytes: 6840111,
  sha256: "99a42cc30d4498d99b8acf507790ab74f040ff2e202ef6c8e90bbb39b6c46675",
};
export const KEPLER_TOLERANCE_KM = 1e-2;
export const FLEET_SCHEMA_VERSION = "1.0.0";
const MINER_MASS_KG = 40;

function fail(message) { throw new Error(`import-gtoc12: ${message}`); }
function assert(condition, message) { if (!condition) fail(message); }

export function parseCatalogue(text) {
  const rows = new Map();
  for (const line of text.split(/\r?\n/)) {
    const fields = line.trim().split(/\s+/);
    if (fields.length !== 8 || !/^\d+$/.test(fields[0])) continue;
    const [id, epoch, a, e, i, node, peri, m0] = fields.map(Number);
    assert([id, epoch, a, e, i, node, peri, m0].every(Number.isFinite), `catalogue row ${line} is not numeric`);
    rows.set(id, { id, epoch_mjd: epoch, a_au: a, e, i_deg: i, node_deg: node, peri_deg: peri, m0_deg: m0 });
  }
  assert(rows.size === 60000, `catalogue has ${rows.size} rows, expected 60000`);
  return rows;
}

export function classifyEvent(event, index, count) {
  const delta = event.mass_after_kg - event.mass_before_kg;
  if (event.kind === "launch") return "launch";
  if (event.kind === "rendezvous") {
    if (delta < 0) {
      assert(Math.abs(delta + MINER_MASS_KG) < 1e-6, `deploy event ${index} changes mass by ${delta} kg`);
      return "deploy";
    }
    return delta > 0 ? "collect" : "rendezvous";
  }
  if (event.kind === "flyby") return event.event_id === -3 && index === count - 1 ? "earth-return" : "flyby";
  fail(`unknown event kind ${event.kind}`);
}

function checkSeries(series, label) {
  assert(Number.isInteger(series.point_count) && series.point_count === series.points_txyz.length, `${label} point count`);
  assert(series.selected_indices.length === series.point_count, `${label} selected indices`);
  assert(series.points_txyz.every((point) => point.length === 4 && point.every(Number.isFinite)), `${label} finite txyz`);
  assert(series.points_txyz.every((point, index) => index === 0 || point[0] >= series.points_txyz[index - 1][0]), `${label} time order`);
  assert(series.selected_indices.every((value, index) => Number.isInteger(value) && (index === 0 || value > series.selected_indices[index - 1])), `${label} index order`);
  assert(series.selected_indices[0] === 0 && series.selected_indices.at(-1) === series.original_point_count - 1, `${label} endpoints`);
  assert(/^[0-9a-f]{64}$/.test(series.original_sha256), `${label} original hash`);
}

function keplerCheck(record, catalogue, earth) {
  let asteroidMax = 0, earthMax = 0, checked = 0;
  for (const orbit of record.context_orbits ?? []) {
    const match = /^asteroid (\d+)$/.exec(orbit.body);
    const prepared = match ? prepareElements(catalogue.get(Number(match[1])) ?? fail(`asteroid ${match[1]} missing from catalogue`)) : earth;
    for (const [epoch, x, y, z] of orbit.points_txyz) {
      const [px, py, pz] = positionAt(prepared, epoch);
      const error = Math.hypot(px - x, py - y, pz - z);
      if (match) asteroidMax = Math.max(asteroidMax, error); else earthMax = Math.max(earthMax, error);
      checked += 1;
    }
  }
  return { asteroidMax, earthMax, checked };
}

export async function importGtoc12({
  exportDirectory, cataloguePath, solutionPath = null, fleetPath = null, outputDirectory = join(ROOT, "data", "gtoc12"),
  cataloguePin = CATALOGUE_PIN, // overridable only programmatically (unit tests with synthetic catalogues)
}) {
  assert(exportDirectory && cataloguePath, "--export and --catalogue are required");
  const [exportBytes, manifestBytes, catalogueBytes] = await Promise.all([
    readFile(join(exportDirectory, "trajectories.json")), readFile(join(exportDirectory, "manifest.json")), readFile(cataloguePath),
  ]);
  const manifest = JSON.parse(manifestBytes);
  const exportSha = sha256(exportBytes);
  assert(manifest.files?.["trajectories.json"]?.sha256 === exportSha, `export trajectories.json SHA-256 ${exportSha} differs from manifest ${manifest.files?.["trajectories.json"]?.sha256}`);
  assert(manifest.files["trajectories.json"].bytes === exportBytes.length, "export byte count differs from manifest");
  const catalogueSha = sha256(catalogueBytes);
  assert(catalogueSha === cataloguePin.sha256 && catalogueBytes.length === cataloguePin.bytes, `catalogue SHA-256 ${catalogueSha} (${catalogueBytes.length} bytes) is not the pinned GTOC12 catalogue`);
  const source = JSON.parse(exportBytes);
  assert(source.viewer_schema_version === "1.0.0", "unsupported export schema");
  assert(source.imported_source_sha256 === manifest.source.sha256, "export source hash differs from manifest");
  assert(source.prohibitions?.visual_interpolation_included === false, "export must declare no visual interpolation");
  assert(Array.isArray(source.trajectories) && source.trajectories.length >= 1, "export has no trajectories");

  let solution = null;
  if (solutionPath) {
    const bytes = await readFile(solutionPath);
    const digest = sha256(bytes);
    assert(digest === manifest.source.sha256 && bytes.length === manifest.source.bytes, `${basename(solutionPath)} SHA-256 ${digest} is not the export's source`);
    solution = { basename: basename(solutionPath), sha256: digest, bytes: bytes.length };
  }
  let fleetSummary = null;
  if (fleetPath) {
    const bytes = await readFile(fleetPath);
    const fleet = JSON.parse(bytes);
    assert(fleet.viewer_manifest?.source?.sha256 === manifest.source.sha256, "fleet.json viewer_manifest source differs from the export source");
    assert(fleet.fleet?.ships === source.trajectories.length, "fleet.json ship count differs from export");
    fleetSummary = {
      sha256: sha256(bytes), official_total_mass_kg: fleet.official?.total_mass_kg ?? null, official_ok: fleet.official?.ok ?? null,
      independent_ok: fleet.independent?.ok ?? null, total_collected_kg: fleet.fleet.total_collected_kg,
      collected_kg_per_ship: fleet.fleet.collected_kg_per_ship, asteroids: fleet.fleet.asteroids, ship_limit: fleet.fleet.ship_limit,
    };
  }

  const catalogue = parseCatalogue(catalogueBytes.toString("utf8"));
  const earth = prepareElements(EARTH_ELEMENTS_DEG);
  const visited = new Map();
  let asteroidMax = 0, earthMax = 0, contextChecked = 0;
  const ships = source.trajectories.map((record, shipIndex) => {
    const shipId = shipIndex + 1;
    assert(record.family === "GTOC12" && record.viewer?.scene_kind === "heliocentric", `record ${shipIndex} is not a GTOC12 heliocentric record`);
    assert(record.position_units === "km" && /MJD/.test(record.time_units), `record ${shipIndex} units`);
    checkSeries(record.replay, `ship ${shipId} replay`);
    checkSeries(record.transcription, `ship ${shipId} transcription`);
    assert(record.transcription.point_count === record.events.length, `ship ${shipId} transcription/event count`);
    const replayEpochs = new Set(record.replay.points_txyz.map((point) => point[0]));
    const events = record.events.map((event, index) => {
      const node = record.transcription.points_txyz[index];
      assert(node[0] === event.epoch_mjd, `ship ${shipId} event ${index} epoch differs from transcription node`);
      assert(replayEpochs.has(event.epoch_mjd), `ship ${shipId} event ${index} epoch missing from replay samples`);
      assert(index === 0 || event.epoch_mjd >= record.events[index - 1].epoch_mjd, `ship ${shipId} event order`);
      const role = classifyEvent(event, index, record.events.length);
      if (event.event_id > 0) {
        const entry = visited.get(event.event_id) ?? { ships: new Set(), visits: 0 };
        entry.ships.add(shipId); entry.visits += 1; visited.set(event.event_id, entry);
      }
      return {
        index, epoch_mjd: event.epoch_mjd, event_id: event.event_id, kind: event.kind, role,
        body: event.event_id > 0 ? `asteroid ${event.event_id}` : event.event_id === 0 ? "Earth (launch)" : event.event_id === -3 ? "Earth" : `body ${event.event_id}`,
        mass_before_kg: event.mass_before_kg, mass_after_kg: event.mass_after_kg, mass_delta_kg: event.mass_after_kg - event.mass_before_kg,
        position_km: node.slice(1, 4),
      };
    });
    assert(events[0]?.role === "launch", `ship ${shipId} must start with a launch`);
    const collected = events.filter((event) => event.role === "collect").reduce((sum, event) => sum + event.mass_delta_kg, 0);
    if (fleetSummary) {
      const expected = fleetSummary.collected_kg_per_ship[shipIndex];
      assert(Math.abs(collected - expected) < 1e-6, `ship ${shipId} collected ${collected} kg differs from fleet.json ${expected}`);
    }
    const check = keplerCheck(record, catalogue, earth);
    asteroidMax = Math.max(asteroidMax, check.asteroidMax); earthMax = Math.max(earthMax, check.earthMax); contextChecked += check.checked;
    const asteroids = [...new Set(events.filter((event) => event.event_id > 0).map((event) => event.event_id))];
    return {
      ship_id: shipId, trajectory_id: record.trajectory_id, launch_epoch_mjd: events[0].epoch_mjd,
      return_epoch_mjd: events.at(-1).epoch_mjd, final_sample_epoch_mjd: record.replay.points_txyz.at(-1)[0],
      initial_mass_kg: record.mass_summary.initial_kg, final_mass_kg: record.mass_summary.final_kg,
      collected_kg: collected, miners_deployed: events.filter((event) => event.role === "deploy").length,
      collects: events.filter((event) => event.role === "collect").length, asteroids, events,
      replay: record.replay, transcription: record.transcription, controls_summary: record.controls_summary,
      qualification: record.qualification, raw_evidence_sha256: record.raw_evidence_sha256,
    };
  });
  assert(asteroidMax <= KEPLER_TOLERANCE_KM && earthMax <= KEPLER_TOLERANCE_KM, `Kepler propagation disagrees with the export context orbits (asteroid ${asteroidMax} km, Earth ${earthMax} km)`);
  const asteroidIds = [...visited.keys()].sort((a, b) => a - b);
  if (fleetSummary) {
    assert(JSON.stringify(asteroidIds) === JSON.stringify([...fleetSummary.asteroids].sort((a, b) => a - b)), "visited asteroid set differs from fleet.json");
  }
  const totalCollected = ships.reduce((sum, ship) => sum + ship.collected_kg, 0);
  const validation = source.archive?.validation_report ?? {};
  const dataset = {
    viewer_schema_version: FLEET_SCHEMA_VERSION,
    dataset_kind: "gtoc12-fleet",
    title: `GTOC12 verified fleet ${source.trajectories[0].source.run_id}`,
    export_title: source.title,
    run_id: source.trajectories[0].source.run_id,
    generated_by_commit: source.generated_by_commit,
    frame: source.trajectories[0].frame,
    position_units: "km",
    time_units: "MJD (days)",
    prohibitions: { visual_interpolation_included: false, straight_segments_are_connections_between_archived_samples: true },
    constants: { au_km: AU_KM, mu_sun_km3_s2: MU_SUN_KM3_S2, day_s: DAY_S, mission_start_mjd: MISSION_START_MJD, mission_end_mjd: MISSION_END_MJD, element_epoch_mjd: ELEMENT_EPOCH_MJD },
    score: {
      ships: ships.length, unique_asteroids: asteroidIds.length, total_collected_kg: totalCollected,
      official_total_mass_kg: fleetSummary?.official_total_mass_kg ?? Math.round(totalCollected * 100) / 100,
      independent_total_mass_kg: validation.total_mass_kg ?? null, mined_asteroids: validation.mined_asteroids ?? null,
      ship_limit: validation.ship_limit ?? fleetSummary?.ship_limit ?? null, verifier_ok: Boolean(validation.ok),
    },
    verification: validation,
    earth: { ...EARTH_ELEMENTS_DEG, source: "GTOC12_Problem.pdf Table 2 (Earth elements at 64328 MJD)" },
    asteroids: asteroidIds.map((id) => ({ ...catalogue.get(id), visited_by: [...visited.get(id).ships].sort((a, b) => a - b), visits: visited.get(id).visits })),
    ships,
    source: {
      export_trajectories_sha256: exportSha, export_trajectories_bytes: exportBytes.length, export_manifest_sha256: sha256(manifestBytes),
      solution_basename: manifest.source.path_basename, solution_sha256: manifest.source.sha256, solution_bytes: manifest.source.bytes,
      solution_verified_locally: Boolean(solution), catalogue: { name: basename(cataloguePath), bytes: catalogueBytes.length, sha256: catalogueSha }, fleet_json_sha256: fleetSummary?.sha256 ?? null,
      official_verifier_ok: fleetSummary?.official_ok ?? null, independent_verifier_ok: fleetSummary?.independent_ok ?? validation.ok ?? null,
      generator: source.trajectories[0].source.generator, transform: manifest.transform,
    },
    kepler_check: { context_points_checked: contextChecked, asteroid_max_error_km: asteroidMax, earth_max_error_km: earthMax, tolerance_km: KEPLER_TOLERANCE_KM },
  };
  const dataBytes = Buffer.from(serialize(dataset));
  const outputManifest = {
    schema_version: FLEET_SCHEMA_VERSION,
    dataset_kind: "gtoc12-fleet",
    files: { "fleet.json": { bytes: dataBytes.length, sha256: sha256(dataBytes) } },
    source: dataset.source,
    kepler_check: dataset.kepler_check,
    summary: { ships: ships.length, unique_asteroids: asteroidIds.length, replay_points: ships.reduce((sum, ship) => sum + ship.replay.point_count, 0), events: ships.reduce((sum, ship) => sum + ship.events.length, 0), official_total_mass_kg: dataset.score.official_total_mass_kg },
    transform: "Verified export copied losslessly (exact samples, indices, hashes); events classified; pinned Keplerian elements attached; Kepler propagation cross-checked against exporter context orbits.",
  };
  await mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    writeFile(join(outputDirectory, "fleet.json"), dataBytes),
    writeFile(join(outputDirectory, "manifest.json"), serialize(outputManifest)),
  ]);
  return outputManifest;
}

function parseArguments(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const [flag, inline] = argv[index].split("=");
    const value = inline ?? argv[++index];
    if (flag === "--export") options.exportDirectory = value;
    else if (flag === "--catalogue") options.cataloguePath = value;
    else if (flag === "--solution") options.solutionPath = value;
    else if (flag === "--fleet") options.fleetPath = value;
    else if (flag === "--output") options.outputDirectory = resolve(value);
    else fail(`unknown argument ${flag}`);
  }
  return options;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const options = parseArguments(process.argv.slice(2));
  const manifest = await importGtoc12(options);
  const target = options.outputDirectory ?? join(ROOT, "data", "gtoc12");
  await stat(join(target, "fleet.json"));
  console.log(`Imported GTOC12 fleet: ${manifest.summary.ships} ships, ${manifest.summary.unique_asteroids} asteroids, ${manifest.summary.replay_points} exact replay samples, ${manifest.summary.official_total_mass_kg} kg -> ${join(target, "fleet.json")} (${manifest.files["fleet.json"].sha256})`);
  console.log(`Kepler cross-check: ${manifest.kepler_check.context_points_checked} context points, asteroid max ${manifest.kepler_check.asteroid_max_error_km.toExponential(2)} km, Earth max ${manifest.kepler_check.earth_max_error_km.toExponential(2)} km`);
}
