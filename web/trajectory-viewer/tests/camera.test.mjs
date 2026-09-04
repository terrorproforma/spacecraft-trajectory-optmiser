import test from "node:test";
import assert from "node:assert/strict";
import {
  CAMERA_PRESETS, EXAGGERATION, PITCH_LIMIT, applyVelocity, boundsOf, cameraBasis, cameraEquals, clampCamera, createTransition,
  cursorPointOnFocalPlane, decayVelocity, dollyTowards, easeInOut, exaggerate, fitDistance, followCamera, inertiaActive,
  interpolateCamera, normaliseAngle, presetCamera, rescaleTarget, sampleTransition, shortestDelta,
} from "../camera.js";
import { orbitEye } from "../math.js";

const base = { yaw: 0.4, pitch: 0.3, distance: 2.5, target: [0.1, -0.2, 0.05] };

test("angles wrap and the shortest rotation is chosen", () => {
  assert.ok(Math.abs(normaliseAngle(3 * Math.PI) - Math.PI) < 1e-12);
  assert.ok(Math.abs(normaliseAngle(-Math.PI / 2) + Math.PI / 2) < 1e-12);
  assert.ok(Math.abs(shortestDelta(0.1, 2 * Math.PI - 0.1) + 0.2) < 1e-12, "goes backwards through zero");
  assert.ok(Math.abs(shortestDelta(-3, 3) - (6 - 2 * Math.PI)) < 1e-12);
});

test("presets re-aim the camera and keep distance and target", () => {
  for (const name of Object.keys(CAMERA_PRESETS)) {
    const camera = presetCamera(name, base);
    assert.equal(camera.pitch, CAMERA_PRESETS[name].pitch);
    assert.equal(camera.yaw, CAMERA_PRESETS[name].yaw);
    assert.equal(camera.distance, base.distance);
    assert.deepEqual(camera.target, base.target);
    assert.notEqual(camera.target, base.target, "target array is copied");
    assert.ok(Math.abs(camera.pitch) <= PITCH_LIMIT);
  }
  assert.ok(CAMERA_PRESETS.top.pitch > CAMERA_PRESETS.oblique.pitch && CAMERA_PRESETS.oblique.pitch > CAMERA_PRESETS.edge.pitch);
  assert.ok(Math.abs(CAMERA_PRESETS.oblique.pitch - Math.PI / 6) < 1e-12, "oblique preset is 30 degrees");
  assert.throws(() => presetCamera("sideways", base), RangeError);
});

test("interpolation is exact at the ends, eased in the middle and geometric in distance", () => {
  const to = { yaw: base.yaw + 2 * Math.PI - 0.5, pitch: 1.2, distance: 10, target: [1, 1, 1] };
  assert.ok(cameraEquals(interpolateCamera(base, to, 0), base));
  assert.ok(cameraEquals(interpolateCamera(base, to, 1), to));
  const middle = interpolateCamera(base, to, 0.5);
  assert.ok(Math.abs(middle.yaw - (base.yaw - 0.25)) < 1e-12, "yaw takes the short way round");
  assert.ok(Math.abs(middle.distance - Math.sqrt(2.5 * 10)) < 1e-12, "distance interpolates in log space");
  assert.deepEqual(middle.target.map((value) => Math.round(value * 1e6) / 1e6), [0.55, 0.4, 0.525]);
  assert.equal(easeInOut(0), 0); assert.equal(easeInOut(1), 1); assert.equal(easeInOut(0.5), 0.5);
  assert.ok(easeInOut(0.25) < 0.25 && easeInOut(0.75) > 0.75);
});

test("transitions finish exactly on the target and report completion", () => {
  const to = { yaw: -1, pitch: 0.9, distance: 4, target: [0, 0, 0] };
  const transition = createTransition(base, to, 1000, 500);
  assert.equal(sampleTransition(transition, 1000).done, false);
  assert.ok(cameraEquals(sampleTransition(transition, 1000).camera, base));
  const partial = sampleTransition(transition, 1250);
  assert.equal(partial.done, false);
  assert.ok(partial.camera.distance > 2.5 && partial.camera.distance < 4);
  const finished = sampleTransition(transition, 1600);
  assert.equal(finished.done, true);
  assert.ok(cameraEquals(finished.camera, to));
});

test("inertia decays exponentially and stops below the threshold", () => {
  const velocity = { yaw: 0.002, pitch: -0.001 };
  const later = decayVelocity(velocity, 140);
  assert.ok(Math.abs(later.yaw - 0.001) < 1e-12 && Math.abs(later.pitch + 0.0005) < 1e-12, "one half-life halves the speed");
  assert.ok(inertiaActive(velocity));
  assert.equal(inertiaActive(decayVelocity(velocity, 140 * 12)), false);
  const moved = applyVelocity(base, { yaw: 0.001, pitch: 0.2 }, 16);
  assert.ok(Math.abs(moved.yaw - (base.yaw + 0.016)) < 1e-12);
  assert.equal(moved.pitch, PITCH_LIMIT, "pitch is clamped at the pole limit");
  assert.deepEqual(moved.target, base.target);
});

test("bounds and fit distance frame a point cloud inside the frustum", () => {
  const points = [[-1, 0, 0], [1, 0, 0], [0, 0.5, 0], [0, -0.5, 0.25]];
  const { center, radius } = boundsOf(points);
  assert.deepEqual(center, [0, 0, 0.125]);
  assert.ok(Math.abs(radius - Math.hypot(1, 0, 0.125)) < 1e-12);
  assert.deepEqual(boundsOf([]), { center: [0, 0, 0], radius: 0 });
  const square = fitDistance(1, Math.PI / 2, 1, 1);
  assert.ok(Math.abs(square - Math.SQRT2) < 1e-12, "unit sphere at 90 degrees FOV needs sqrt(2)");
  assert.ok(fitDistance(1, Math.PI / 2, 0.5, 1) > square, "narrow portrait aspect needs more distance");
  assert.ok(fitDistance(1, Math.PI / 2, 2, 1) === square, "wide aspect is limited by the vertical FOV");
  assert.ok(fitDistance(2) > fitDistance(1));
});

test("exaggeration scales z only and re-expresses the camera target", () => {
  assert.deepEqual(exaggerate([1, 2, 0.5], 8), [1, 2, 4]);
  assert.deepEqual(exaggerate([1, 2, 0], 20), [1, 2, 0], "the ecliptic plane is invariant");
  const camera = rescaleTarget({ ...base, target: [1, 1, 0.6] }, 3, 12);
  assert.deepEqual(camera.target, [1, 1, 2.4]);
  assert.ok(EXAGGERATION.minimum === 1 && EXAGGERATION.maximum === 20);
  assert.equal(EXAGGERATION.initial, 6, "the fleet view opens at 6x (labelled not physical)");
});

test("camera basis is orthonormal and consistent with the orbit eye", () => {
  const { forward, right, up } = cameraBasis(base);
  const eye = orbitEye(base);
  const toTarget = base.target.map((value, axis) => value - eye[axis]);
  const length = Math.hypot(...toTarget);
  toTarget.forEach((value, axis) => assert.ok(Math.abs(value / length - forward[axis]) < 1e-12, "forward points from eye to target"));
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  for (const [a, b] of [[forward, right], [forward, up], [right, up]]) assert.ok(Math.abs(dot(a, b)) < 1e-12);
  for (const vector of [forward, right, up]) assert.ok(Math.abs(Math.hypot(...vector) - 1) < 1e-12);
  assert.ok(up[2] > 0, "screen-up has a positive world-z component");
});

test("wheel dolly keeps the point under the cursor fixed", () => {
  const camera = { yaw: 0.3, pitch: 0.5, distance: 4, target: [0.2, -0.1, 0.3] };
  const centre = cursorPointOnFocalPlane(camera, 0, 0, Math.PI / 4, 16 / 9);
  assert.deepEqual(centre, camera.target, "the screen centre is the target");
  const corner = cursorPointOnFocalPlane(camera, 1, 1, Math.PI / 2, 2);
  const { right, up } = cameraBasis(camera);
  const expected = camera.target.map((value, axis) => value + right[axis] * 8 + up[axis] * 4);
  corner.forEach((value, axis) => assert.ok(Math.abs(value - expected[axis]) < 1e-12));
  const zoomed = dollyTowards(camera, corner, 0.5);
  assert.equal(zoomed.distance, 2);
  // The cursor point lies on the new focal plane at the same NDC position, so it stays under the cursor.
  const again = cursorPointOnFocalPlane(zoomed, 1, 1, Math.PI / 2, 2);
  again.forEach((value, axis) => assert.ok(Math.abs(value - corner[axis]) < 1e-12));
  const clamped = dollyTowards(camera, corner, 0.01, { minimum: 1, maximum: 12 });
  assert.equal(clamped.distance, 1);
  assert.equal(dollyTowards(camera, corner, 1).target.join(), camera.target.join(), "factor 1 leaves the target alone");
});

test("follow and clamp keep orientation while re-targeting", () => {
  const following = followCamera(base, [2, 3, 4], 0.4);
  assert.deepEqual(following.target, [2, 3, 4]);
  assert.equal(following.distance, 0.4);
  assert.equal(following.yaw, base.yaw); assert.equal(following.pitch, base.pitch);
  const clamped = clampCamera({ ...base, pitch: 9, distance: 100 }, { minimum: 0.1, maximum: 12 });
  assert.equal(clamped.pitch, PITCH_LIMIT); assert.equal(clamped.distance, 12);
});
