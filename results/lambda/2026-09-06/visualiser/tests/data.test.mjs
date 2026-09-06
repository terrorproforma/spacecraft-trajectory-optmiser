import test from "node:test";
import assert from "node:assert/strict";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { importDataset } from "../scripts/import-data.mjs";

const root = new URL("..", import.meta.url);
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
// The authoritative compact archive lives in the WSL visualisation worktree; override with TRAJECTORY_SOURCE.
const sourcePath = process.env.TRAJECTORY_SOURCE
  || String.raw`\\wsl.localhost\Ubuntu-22.04\home\angus\worktrees\spacepdhcg-trajectory-visualization\visualization-output\final\spacepdhcg_verified_trajectories.compact.json`;
const haveSource = await access(sourcePath).then(() => true, () => false);

test("generated data has dense finite replay and qualifications", async () => {
  const data = JSON.parse(await readFile(new URL("data/trajectories.json", root)));
  assert.equal(data.trajectories.length, 5);
  for (const trajectory of data.trajectories) {
    assert.ok(trajectory.replay.point_count >= 201);
    assert.equal(typeof trajectory.qualification.qualified, "boolean");
    assert.ok(trajectory.replay.points_txyz.flat().every(Number.isFinite));
    assert.equal(trajectory.replay.selected_indices[0], 0);
    assert.equal(trajectory.replay.selected_indices.at(-1), trajectory.replay.original_point_count - 1);
  }
});

test("authoritative import is byte deterministic", { skip: !haveSource && `authoritative source not reachable at ${sourcePath}` }, async () => {
  const first = await mkdtemp(join(tmpdir(), "trajectory-import-a-"));
  const second = await mkdtemp(join(tmpdir(), "trajectory-import-b-"));
  try {
    await importDataset({ sourcePath, outputDirectory: first });
    await importDataset({ sourcePath, outputDirectory: second });
    const [aData, bData, aManifest, bManifest] = await Promise.all([
      readFile(join(first, "trajectories.json")), readFile(join(second, "trajectories.json")),
      readFile(join(first, "manifest.json")), readFile(join(second, "manifest.json")),
    ]);
    assert.equal(hash(aData), hash(bData));
    assert.deepEqual(aData, bData);
    assert.deepEqual(aManifest, bManifest);
  } finally {
    await Promise.all([rm(first, { recursive: true, force: true }), rm(second, { recursive: true, force: true })]);
  }
});
