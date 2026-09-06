import test from "node:test";
import assert from "node:assert/strict";
import {
  BACKGROUND_FRAGMENT, BODY_FRAGMENT, BODY_VERTEX, LINE_VERTEX, RIBBON_FRAGMENT, RIBBON_VERTEX, STAR_VERTEX, SURFACE_VERTEX,
  TUBE_FRAGMENT, TUBE_VERTEX, circleLines, concatRibbons, discTriangles, hex, mulberry32, ribbonArrays, sphere, spokeLines,
  starField, tubeArrays,
} from "../webgl.js";

test("tube meshes carry radial normals from a continuous frame and segment-ordered indices", () => {
  const points = [[0, 0, 0], [1, 0, 0], [2, 0.5, 0], [3, 0.5, 0.5]];
  const tube = tubeArrays(points, [1, 2, 3, 4], 6);
  assert.equal(tube.vertexCount, 24); assert.equal(tube.sides, 6); assert.equal(tube.indicesPerSegment, 36);
  assert.equal(tube.indices.length, 3 * 36, "one quad (two triangles) per side per segment");
  assert.ok(tube.indices instanceof Uint16Array);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (let vertex = 0; vertex < tube.vertexCount; vertex += 1) {
    const sample = Math.floor(vertex / 6);
    const normal = Array.from(tube.normals.subarray(vertex * 3, vertex * 3 + 3));
    assert.ok(Math.abs(Math.hypot(...normal) - 1) < 1e-6, "unit normals (float32)");
    assert.deepEqual(Array.from(tube.positions.subarray(vertex * 3, vertex * 3 + 3)), points[sample], "vertex sits on the axis sample");
    assert.equal(tube.times[vertex], sample + 1);
    const before = points[Math.max(0, sample - 1)], after = points[Math.min(3, sample + 1)];
    const tangent = [after[0] - before[0], after[1] - before[1], after[2] - before[2]];
    assert.ok(Math.abs(dot(normal, tangent)) < 1e-6, "normals are perpendicular to the local tangent");
  }
  // Ring 0 normals are evenly spaced and the frame is transported (side 0 of ring 1 stays close to ring 0).
  const n0 = Array.from(tube.normals.subarray(0, 3)), n3 = Array.from(tube.normals.subarray(9, 12)), n6 = Array.from(tube.normals.subarray(18, 21));
  assert.ok(Math.abs(dot(n0, n3) + 1) < 1e-6, "opposite sides are antiparallel");
  assert.ok(dot(n0, n6) > 0.99, "parallel transport keeps the frame continuous");
  // Segment 0 indices reference rings 0 and 1 only.
  const first = Array.from(tube.indices.subarray(0, 36));
  assert.ok(first.every((index) => index < 12));
  assert.ok(Array.from(tube.indices.subarray(36, 72)).every((index) => index >= 6 && index < 18));
  assert.throws(() => tubeArrays([[0, 0, 0]]), RangeError);
});

test("open ribbons duplicate each point on two sides with clamped tangents and per-vertex times", () => {
  const points = [[0, 0, 0], [1, 0, 0], [2, 1, 0]];
  const ribbon = ribbonArrays(points, [10, 20, 30]);
  assert.equal(ribbon.vertexCount, 6);
  assert.deepEqual([...ribbon.sides], [-1, 1, -1, 1, -1, 1]);
  assert.deepEqual([...ribbon.previous.slice(0, 3)], [0, 0, 0], "first previous clamps to itself");
  assert.deepEqual([...ribbon.next.slice(-3)], [2, 1, 0], "last next clamps to itself");
  assert.deepEqual([...ribbon.times], [10, 10, 20, 20, 30, 30]);
  assert.equal(ribbonArrays(points).times, undefined);
});

test("closed ribbons seal the loop and wrap the tangents", () => {
  const points = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]];
  const ribbon = ribbonArrays(points, null, true);
  assert.equal(ribbon.vertexCount, 10, "n + 1 points, two vertices each");
  assert.deepEqual([...ribbon.positions.slice(-3)], [1, 0, 0], "loop returns to the first point");
  assert.deepEqual([...ribbon.previous.slice(0, 3)], [0, -1, 0], "previous of the first point is the last point");
  assert.deepEqual([...ribbon.next.slice(-3)], [0, 1, 0], "next of the sealing point is the second point");
});

test("concatenated ribbons keep per-ribbon offsets and counts", () => {
  const a = ribbonArrays([[0, 0, 0], [1, 0, 0]]);
  const b = ribbonArrays([[0, 0, 1], [1, 0, 1], [2, 0, 1]], null, true);
  const joined = concatRibbons([a, b]);
  assert.deepEqual(joined.offsets, [0, 4]);
  assert.deepEqual(joined.counts, [4, 8]);
  assert.equal(joined.vertexCount, 12);
  assert.equal(joined.positions.length, 36); assert.equal(joined.sides.length, 12);
  assert.deepEqual([...joined.positions.slice(12, 15)], [0, 0, 1], "second ribbon starts at its offset");
});

test("procedural star field is deterministic, on-sphere and magnitude-bounded", () => {
  const stars = starField(500, 30, 7);
  const again = starField(500, 30, 7);
  assert.deepEqual([...stars.positions], [...again.positions]);
  assert.notDeepEqual([...stars.positions.slice(0, 9)], [...starField(500, 30, 8).positions.slice(0, 9)]);
  for (let index = 0; index < 500; index += 1) {
    const radius = Math.hypot(stars.positions[index * 3], stars.positions[index * 3 + 1], stars.positions[index * 3 + 2]);
    assert.ok(Math.abs(radius - 30) < 1e-3, `star ${index} radius ${radius}`);
    assert.ok(stars.magnitudes[index] > 0 && stars.magnitudes[index] <= 1);
  }
  const above = [...stars.positions].filter((_, index) => index % 3 === 2 && stars.positions[index] > 0).length;
  assert.ok(above > 180 && above < 320, `roughly half the stars above the ecliptic (${above})`);
  const random = mulberry32(1);
  const samples = Array.from({ length: 1000 }, random);
  assert.ok(samples.every((value) => value >= 0 && value < 1));
  assert.ok(new Set(samples).size > 990);
});

test("disc, ring, spoke and sphere primitives have the expected vertex counts", () => {
  assert.equal(discTriangles(1, 12).length, 12 * 3 * 3);
  const disc = discTriangles(2, 4);
  assert.deepEqual([...disc.slice(0, 6)], [0, 0, 0, 2, 0, 0], "wedge starts at the centre and radius");
  assert.equal(circleLines(1, 90).length, 90 * 2 * 3);
  assert.equal(spokeLines(0.1, 1, 30).length, 12 * 2 * 3);
  const unit = sphere(1, [0, 0, 0], 6, 8);
  assert.equal(unit.positions.length, 6 * 8 * 6 * 3);
  assert.deepEqual([...unit.positions], [...unit.normals], "a unit sphere at the origin is its own normal field");
  assert.deepEqual(hex("#ff8000", 0.5).map((value) => Math.round(value * 100) / 100), [1, 0.5, 0, 0.5]);
});

test("shaders expose the vertical-exaggeration, trail and lighting uniforms", () => {
  for (const source of [LINE_VERTEX, SURFACE_VERTEX, RIBBON_VERTEX, BODY_VERTEX]) assert.match(source, /uniform float uZScale/);
  assert.match(RIBBON_VERTEX, /in float aTime/);
  for (const name of ["uEpoch", "uTrail", "uBaseAlpha", "uShade"]) assert.match(RIBBON_FRAGMENT, new RegExp(`uniform float ${name}`));
  for (const name of ["uEye", "uLight", "uAmbient", "uFogColor"]) assert.match(BODY_FRAGMENT, new RegExp(`uniform vec3 ${name}`));
  for (const name of ["aCenter", "aColor"]) assert.match(BODY_VERTEX, new RegExp(`in vec[34] ${name}`), "instanced body attributes");
  for (const name of ["aRadius", "aEmissive"]) assert.match(BODY_VERTEX, new RegExp(`in float ${name}`));
  assert.match(TUBE_VERTEX, /uniform float uRadius/);
  for (const name of ["uFog", "uEpoch", "uTrail"]) assert.match(TUBE_FRAGMENT, new RegExp(`uniform float ${name}`));
  assert.match(RIBBON_FRAGMENT, /uniform float uFog/);
  assert.match(STAR_VERTEX, /in float aMagnitude/);
  assert.match(BACKGROUND_FRAGMENT, /vignette/);
  for (const source of [LINE_VERTEX, RIBBON_VERTEX, BODY_VERTEX, STAR_VERTEX, TUBE_VERTEX, BACKGROUND_FRAGMENT]) assert.match(source, /^#version 300 es/);
});
