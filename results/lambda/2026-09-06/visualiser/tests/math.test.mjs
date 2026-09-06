import test from "node:test";
import assert from "node:assert/strict";
import { clamp, lookAt, multiply, normalizePoint, orbitEye, perspective } from "../math.js";

test("camera helpers clamp and produce finite coordinates", () => {
  assert.equal(clamp(15, 0, 10), 10);
  assert.deepEqual(normalizePoint([5, 7, 9], [1, 3, 5], 2), [2, 2, 2]);
  const eye = orbitEye({ yaw: 0, pitch: 0, distance: 3, target: [1, 2, 3] });
  assert.deepEqual(eye, [4, 2, 3]);
});

test("projection and view matrices compose", () => {
  const projection = perspective(Math.PI / 4, 16 / 9, 0.03, 80);
  const view = lookAt([3, 2, 2], [0, 0, 0], [0, 0, 1]);
  const mvp = multiply(projection, view);
  assert.equal(mvp.length, 16);
  assert.ok([...mvp].every(Number.isFinite));
  assert.ok(mvp.some((value) => value !== 0));
});
