// Pure perspective-camera helpers for the 3D scene: orbit-camera presets, eased transitions,
// pointer inertia, fit-to-bounds, follow targets and the vertical (Z) exaggeration used to make
// GTOC12 orbital inclinations legible. No DOM or WebGL; every function returns new objects.

import { clamp } from "./math.js";

export const DEG = Math.PI / 180;
/** Orbit pitch is kept just short of the poles so the [0, 0, 1] up-vector never degenerates. */
export const PITCH_LIMIT = 1.52;
/** Vertical exaggeration range; the fleet view opens at 6x (labelled "not physical") so inclinations read in 3D. */
export const EXAGGERATION = { minimum: 1, maximum: 20, initial: 6, step: 0.5 };
export const TRANSITION_MS = 650;
export const INERTIA = { halfLifeMs: 140, stopBelow: 2e-5 };

/**
 * Camera presets. Angles only: distance and target are preserved by `presetCamera` so a preset
 * re-aims the current view instead of resetting it. `follow` is handled by the caller because it
 * needs the ship position at the current epoch (see `followCamera`).
 */
export const CAMERA_PRESETS = Object.freeze({
  top: { label: "Top-down ecliptic", pitch: PITCH_LIMIT, yaw: -Math.PI / 2 },
  oblique: { label: "30° oblique", pitch: 30 * DEG, yaw: -0.72 },
  edge: { label: "Edge-on (inclinations)", pitch: 1.5 * DEG, yaw: -0.72 },
});

export function cloneCamera(camera) {
  return { yaw: camera.yaw, pitch: camera.pitch, distance: camera.distance, target: [...camera.target] };
}

/** Wrap an angle to (-π, π]. */
export function normaliseAngle(angle) {
  const wrapped = ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  return wrapped === -Math.PI ? Math.PI : wrapped;
}

/** Signed shortest rotation from `from` to `to`. */
export function shortestDelta(from, to) { return normaliseAngle(to - from); }

export function easeInOut(t) {
  const x = clamp(t, 0, 1);
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}

/** Camera aimed by a named preset, keeping the current distance and target. */
export function presetCamera(name, current) {
  const preset = CAMERA_PRESETS[name];
  if (!preset) throw new RangeError(`Unknown camera preset: ${name}`);
  return { ...cloneCamera(current), yaw: preset.yaw, pitch: preset.pitch };
}

/** Blend two cameras: shortest-arc yaw, linear pitch/target, geometric distance. */
export function interpolateCamera(from, to, t) {
  const s = clamp(t, 0, 1);
  return {
    yaw: from.yaw + shortestDelta(from.yaw, to.yaw) * s,
    pitch: from.pitch + (to.pitch - from.pitch) * s,
    distance: Math.exp(Math.log(from.distance) + (Math.log(to.distance) - Math.log(from.distance)) * s),
    target: from.target.map((value, axis) => value + (to.target[axis] - value) * s),
  };
}

export function cameraEquals(a, b, epsilon = 1e-9) {
  return Math.abs(shortestDelta(a.yaw, b.yaw)) <= epsilon && Math.abs(a.pitch - b.pitch) <= epsilon
    && Math.abs(a.distance - b.distance) <= epsilon * Math.max(1, a.distance)
    && a.target.every((value, axis) => Math.abs(value - b.target[axis]) <= epsilon);
}

/** Eased transition record; sample it with `sampleTransition`. */
export function createTransition(from, to, startMs, durationMs = TRANSITION_MS) {
  return { from: cloneCamera(from), to: cloneCamera(to), startMs, durationMs: Math.max(1, durationMs) };
}

export function sampleTransition(transition, nowMs) {
  const t = (nowMs - transition.startMs) / transition.durationMs;
  if (t >= 1) return { camera: cloneCamera(transition.to), done: true };
  return { camera: interpolateCamera(transition.from, transition.to, easeInOut(t)), done: false };
}

/** Exponentially decayed angular velocity (rad/ms) after `dtMs`. */
export function decayVelocity(velocity, dtMs, halfLifeMs = INERTIA.halfLifeMs) {
  const factor = Math.pow(0.5, Math.max(0, dtMs) / halfLifeMs);
  return { yaw: velocity.yaw * factor, pitch: velocity.pitch * factor };
}

export function inertiaActive(velocity, threshold = INERTIA.stopBelow) {
  return Math.abs(velocity.yaw) > threshold || Math.abs(velocity.pitch) > threshold;
}

/** Advance an orbit camera by a velocity over `dtMs`, clamping pitch to the pole limit. */
export function applyVelocity(camera, velocity, dtMs, pitchLimit = PITCH_LIMIT) {
  return {
    ...cloneCamera(camera),
    yaw: camera.yaw + velocity.yaw * dtMs,
    pitch: clamp(camera.pitch + velocity.pitch * dtMs, -pitchLimit, pitchLimit),
  };
}

/** Bounding sphere (centre of the axis-aligned box, max distance to it) of [x, y, z] points. */
export function boundsOf(points) {
  if (!points.length) return { center: [0, 0, 0], radius: 0 };
  const minima = [Infinity, Infinity, Infinity], maxima = [-Infinity, -Infinity, -Infinity];
  for (const point of points) for (let axis = 0; axis < 3; axis += 1) {
    minima[axis] = Math.min(minima[axis], point[axis]); maxima[axis] = Math.max(maxima[axis], point[axis]);
  }
  const center = minima.map((value, axis) => (value + maxima[axis]) / 2);
  let radius = 0;
  for (const point of points) radius = Math.max(radius, Math.hypot(point[0] - center[0], point[1] - center[1], point[2] - center[2]));
  return { center, radius };
}

/**
 * Distance at which a sphere of `radius` fits inside a perspective frustum with vertical field of
 * view `fovY` and the given aspect ratio (the narrower of the two half-angles wins), with margin.
 */
export function fitDistance(radius, fovY = Math.PI / 4, aspect = 1, margin = 1.15) {
  const halfVertical = fovY / 2;
  const halfHorizontal = Math.atan(Math.tan(halfVertical) * Math.max(aspect, 1e-6));
  const half = Math.min(halfVertical, halfHorizontal);
  return Math.max(radius, 1e-9) * margin / Math.sin(half);
}

/** Apply vertical exaggeration to a scene point (the ecliptic plane z = 0 is invariant). */
export function exaggerate(point, factor) { return [point[0], point[1], point[2] * factor]; }

/** Re-express a camera target after the exaggeration factor changes so the framed point stays framed. */
export function rescaleTarget(camera, previousFactor, nextFactor) {
  const camera2 = cloneCamera(camera);
  camera2.target[2] = camera2.target[2] / previousFactor * nextFactor;
  return camera2;
}

/** Camera that keeps its orientation but looks at `position` from `distance` (follow-ship). */
export function followCamera(camera, position, distance = camera.distance) {
  return { ...cloneCamera(camera), target: [...position], distance };
}

/** Orthonormal camera basis {forward, right, up} for an orbit camera with world up [0, 0, 1]. */
export function cameraBasis(camera) {
  const cosine = Math.cos(camera.pitch);
  const forward = [-cosine * Math.cos(camera.yaw), -cosine * Math.sin(camera.yaw), -Math.sin(camera.pitch)];
  let right = [forward[1] * 1 - forward[2] * 0, forward[2] * 0 - forward[0] * 1, 0]; // forward x [0,0,1]
  const length = Math.hypot(right[0], right[1]) || 1;
  right = [right[0] / length, right[1] / length, 0];
  const up = [right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2], right[0] * forward[1] - right[1] * forward[0]];
  return { forward, right, up };
}

/**
 * World point on the focal plane (through the target, perpendicular to the view direction) under
 * a cursor at normalised device coordinates (ndcX, ndcY in [-1, 1], y up).
 */
export function cursorPointOnFocalPlane(camera, ndcX, ndcY, fovY = Math.PI / 4, aspect = 1) {
  const { right, up } = cameraBasis(camera);
  const halfHeight = Math.tan(fovY / 2) * camera.distance, halfWidth = halfHeight * aspect;
  return camera.target.map((value, axis) => value + right[axis] * ndcX * halfWidth + up[axis] * ndcY * halfHeight);
}

/**
 * Dolly the camera by `factor` (< 1 zooms in) towards `point`: the point stays under the cursor
 * because the target moves along the target->point vector by the same proportion.
 */
export function dollyTowards(camera, point, factor, zoom = null) {
  const distance = zoom ? clamp(camera.distance * factor, zoom.minimum, zoom.maximum) : camera.distance * factor;
  const applied = distance / camera.distance;
  return {
    ...cloneCamera(camera), distance,
    target: camera.target.map((value, axis) => value + (point[axis] - value) * (1 - applied)),
  };
}

export function clampCamera(camera, zoom, pitchLimit = PITCH_LIMIT) {
  return {
    ...cloneCamera(camera),
    pitch: clamp(camera.pitch, -pitchLimit, pitchLimit),
    distance: clamp(camera.distance, zoom.minimum, zoom.maximum),
  };
}
