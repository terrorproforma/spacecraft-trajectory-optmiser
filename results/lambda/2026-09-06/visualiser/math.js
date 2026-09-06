export const clamp = (value, minimum, maximum) =>
  Math.max(minimum, Math.min(maximum, value));

export function perspective(fieldOfView, aspect, near, far) {
  const f = 1 / Math.tan(fieldOfView / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * range, -1,
    0, 0, 2 * far * near * range, 0,
  ]);
}

export function lookAt(eye, center, up) {
  let [zx, zy, zz] = eye.map((value, index) => value - center[index]);
  const zLength = Math.hypot(zx, zy, zz) || 1;
  [zx, zy, zz] = [zx / zLength, zy / zLength, zz / zLength];
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  const xLength = Math.hypot(xx, xy, xz) || 1;
  [xx, xy, xz] = [xx / xLength, xy / xLength, xz / xLength];
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  return new Float32Array([
    xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1,
  ]);
}

export function multiply(left, right) {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      output[column * 4 + row] =
        left[row] * right[column * 4] +
        left[4 + row] * right[column * 4 + 1] +
        left[8 + row] * right[column * 4 + 2] +
        left[12 + row] * right[column * 4 + 3];
    }
  }
  return output;
}

export function orbitEye(camera) {
  const cosine = Math.cos(camera.pitch);
  return [
    camera.target[0] + camera.distance * cosine * Math.cos(camera.yaw),
    camera.target[1] + camera.distance * cosine * Math.sin(camera.yaw),
    camera.target[2] + camera.distance * Math.sin(camera.pitch),
  ];
}

export function normalizePoint(point, center, scale) {
  return point.map((value, index) => (value - center[index]) / scale);
}
