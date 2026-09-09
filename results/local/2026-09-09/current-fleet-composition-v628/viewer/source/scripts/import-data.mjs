import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const EXPECTED_SOURCE_SHA256 =
  "83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(SCRIPT_DIR, "..");
const DEFAULT_SOURCE = String.raw`\\wsl.localhost\Ubuntu-22.04\home\angus\worktrees\spacepdhcg-trajectory-visualization\visualization-output\final\spacepdhcg_verified_trajectories.compact.json`;

const VIEWER_METADATA = {
  "P1-B": {
    scene_kind: "hcw",
    axes: ["X radial outward", "Y along-track", "Z cross-track"],
    body_label: "Local orbital frame; no globe",
    body_radius: null,
    radius_label: "Not applicable in relative HCW coordinates",
    gravity_label: "Mean motion 1.13e-3 rad/s",
    frame_choice: "HCW is relative motion about a reference orbit. The scene shows an LVLH plane and Earthward -X direction, never a globe.",
  },
  "P1-C": {
    scene_kind: "local-surface",
    axes: ["X local", "Y local", "Z altitude"],
    body_label: "Generic local planetary surface",
    body_radius: 0,
    radius_label: "Local tangent surface at model altitude Z = 0 m",
    gravity_label: "Uniform gravity [0, 0, -3.711] m/s²",
    frame_choice: "The archive names no body or radius. A generic local tangent surface at physical Z = 0 is shown; no globe is inferred.",
  },
  "P1-D": {
    scene_kind: "local-surface",
    axes: ["X local", "Y local", "Z altitude"],
    body_label: "Generic local planetary surface",
    body_radius: 0,
    radius_label: "Local tangent surface at model altitude Z = 0 m",
    gravity_label: "Uniform gravity [0, 0, -3.711] m/s²",
    frame_choice: "The archive names no body or radius. A generic local tangent surface at physical Z = 0 is shown; the target remains above that plane.",
  },
  "P1-E": {
    scene_kind: "central-body",
    axes: ["X inertial", "Y inertial", "Z inertial"],
    body_label: "Unnamed central body",
    body_radius: 6500,
    radius_label: "Rendered r_min = 6,500 km constraint sphere; not a claimed surface",
    gravity_label: "μ = 398,600.4418 km³/s²; body unnamed",
    frame_choice: "The sphere is the archived 6,500 km minimum-radius constraint boundary. It is not labelled as a physical globe.",
  },
  P2: {
    scene_kind: "central-body",
    axes: ["X ECI", "Y ECI", "Z ECI"],
    body_label: "Earth",
    body_radius: 6378136.3,
    radius_label: "Earth equatorial radius = 6,378,136.3 m",
    gravity_label: "μ = 3.986004418e14 m³/s²",
    frame_choice: "The evidence explicitly specifies Earth-centred inertial coordinates. The sphere uses the archived G7 Earth equatorial radius.",
  },
};

export const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

export function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map((key) => [key, stable(value[key])]),
    );
  }
  return value;
}

export function serialize(value) {
  return `${JSON.stringify(stable(value), null, 2)}\n`;
}

export async function importDataset({
  sourcePath = process.env.TRAJECTORY_SOURCE || DEFAULT_SOURCE,
  outputDirectory = join(ROOT, "data"),
} = {}) {
  const sourceBytes = await readFile(sourcePath);
  const sourceSha256 = sha256(sourceBytes);
  if (sourceSha256 !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`Source SHA-256 mismatch: expected ${EXPECTED_SOURCE_SHA256}, received ${sourceSha256}`);
  }
  const source = JSON.parse(sourceBytes);
  if (!Array.isArray(source.trajectories) || source.trajectories.length !== 5) {
    throw new Error("Authoritative compact dataset must contain five trajectories");
  }

  const companionDirectory = dirname(sourcePath);
  const [dictionaryBytes, validationBytes] = await Promise.all([
    readFile(join(companionDirectory, "data_dictionary.json")),
    readFile(join(companionDirectory, "validation_report.json")),
  ]);
  const dataset = {
    ...source,
    archive: {
      data_dictionary: JSON.parse(dictionaryBytes),
      validation_report: JSON.parse(validationBytes),
    },
    imported_source_sha256: sourceSha256,
    trajectories: source.trajectories.map((trajectory) => ({
      ...trajectory,
      viewer: VIEWER_METADATA[trajectory.family],
    })),
    viewer_schema_version: "1.0.0",
  };
  const dataBytes = Buffer.from(serialize(dataset));
  const manifest = {
    files: {
      "trajectories.json": {
        bytes: dataBytes.length,
        sha256: sha256(dataBytes),
      },
    },
    source: {
      bytes: sourceBytes.length,
      data_dictionary_sha256: sha256(dictionaryBytes),
      path_basename: "spacepdhcg_verified_trajectories.compact.json",
      sha256: sourceSha256,
      validation_report_sha256: sha256(validationBytes),
    },
    schema_version: "1.0.0",
    transform: "Stable-key JSON serialization; source fields and numeric values preserved losslessly; viewer metadata added by family.",
  };
  await mkdir(outputDirectory, { recursive: true });
  await Promise.all([
    writeFile(join(outputDirectory, "trajectories.json"), dataBytes),
    writeFile(join(outputDirectory, "manifest.json"), serialize(manifest)),
  ]);
  return manifest;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const sourcePath = process.argv[2] || process.env.TRAJECTORY_SOURCE || DEFAULT_SOURCE;
  const manifest = await importDataset({ sourcePath });
  console.log(`Imported ${manifest.source.sha256} -> data/trajectories.json (${manifest.files["trajectories.json"].sha256})`);
}
