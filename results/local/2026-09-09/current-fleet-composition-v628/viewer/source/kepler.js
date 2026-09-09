// Heliocentric two-body helpers matching GTOC12 Appendix 6.1 (and the Python
// `spacepdhcg.gtoc12.ephemeris` module): classical elements at a reference epoch,
// mean anomaly advanced linearly, Kepler's equation solved by Newton iteration,
// position from the P/Q perifocal vectors. Pure functions; no DOM or WebGL.

export const AU_KM = 1.49597870691e8;
export const MU_SUN_KM3_S2 = 1.32712440018e11;
export const DAY_S = 86400;
export const MISSION_START_MJD = 64328; // 2035-01-01 00:00 TT
export const MISSION_END_MJD = 69807; // 2050-01-01 00:00 TT
export const ELEMENT_EPOCH_MJD = 64328;

// GTOC12_Problem.pdf Table 2, Earth at 64328 MJD (km, degrees).
export const EARTH_ELEMENTS_DEG = {
  name: "Earth",
  a_km: 1.49579151285e8,
  e: 1.65519129162e-2,
  i_deg: 4.6438915550e-3,
  node_deg: 1.98956406477e2,
  peri_deg: 2.629603647e2,
  m0_deg: 3.5803989947e2,
  epoch_mjd: ELEMENT_EPOCH_MJD,
};

const TWO_PI = Math.PI * 2;
const RAD = Math.PI / 180;

/** Solve M = E - e sin E for elliptic orbits (radians). */
export function solveKepler(meanAnomaly, eccentricity, tolerance = 1e-14) {
  if (!(eccentricity >= 0 && eccentricity < 1)) throw new RangeError("GTOC12 bodies are elliptic: 0 <= e < 1");
  const mean = ((meanAnomaly % TWO_PI) + TWO_PI) % TWO_PI;
  let eccentric = eccentricity > 0.8 ? Math.PI : mean;
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const step = (eccentric - eccentricity * Math.sin(eccentric) - mean) / (1 - eccentricity * Math.cos(eccentric));
    eccentric -= step;
    if (Math.abs(step) < tolerance) break;
  }
  return eccentric;
}

/** Convert a degree/AU-or-km element record into radians and km with cached perifocal vectors. */
export function prepareElements(record) {
  const a = record.a_km ?? record.a_au * AU_KM;
  const e = record.e;
  const i = record.i_deg * RAD, node = record.node_deg * RAD, peri = record.peri_deg * RAD;
  const cosNode = Math.cos(node), sinNode = Math.sin(node);
  const cosPeri = Math.cos(peri), sinPeri = Math.sin(peri);
  const cosInc = Math.cos(i), sinInc = Math.sin(i);
  return {
    a_km: a, e, m0_rad: record.m0_deg * RAD, epoch_mjd: record.epoch_mjd,
    meanMotion: Math.sqrt(MU_SUN_KM3_S2 / (a * a * a)), // rad/s
    p: [cosPeri * cosNode - sinPeri * sinNode * cosInc, cosPeri * sinNode + sinPeri * cosNode * cosInc, sinPeri * sinInc],
    q: [-sinPeri * cosNode - cosPeri * sinNode * cosInc, -sinPeri * sinNode + cosPeri * cosNode * cosInc, cosPeri * sinInc],
  };
}

/** Position (km) on the ellipse at eccentric anomaly E. */
export function positionAtEccentric(prepared, eccentric) {
  const { a_km: a, e, p, q } = prepared;
  const x = a * (Math.cos(eccentric) - e);
  const y = a * Math.sqrt(1 - e * e) * Math.sin(eccentric);
  return [x * p[0] + y * q[0], x * p[1] + y * q[1], x * p[2] + y * q[2]];
}

/** Heliocentric position (km) at an MJD epoch. */
export function positionAt(prepared, epochMjd) {
  const mean = prepared.m0_rad + prepared.meanMotion * (epochMjd - prepared.epoch_mjd) * DAY_S;
  return positionAtEccentric(prepared, solveKepler(mean, prepared.e));
}

/** Closed orbit polyline (km): `segments` points uniformly spaced in eccentric anomaly. */
export function orbitPoints(prepared, segments = 180) {
  const points = [];
  for (let index = 0; index < segments; index += 1) {
    points.push(positionAtEccentric(prepared, index / segments * TWO_PI));
  }
  return points;
}

/** Modified Julian Date -> proleptic Gregorian calendar date (UTC/TT day boundary). */
export function mjdToCalendar(mjd) {
  const dayNumber = Math.floor(mjd) + 2400001; // JDN of the civil day containing this MJD
  let l = dayNumber + 68569;
  const n = Math.floor(4 * l / 146097);
  l -= Math.floor((146097 * n + 3) / 4);
  const i = Math.floor(4000 * (l + 1) / 1461001);
  l = l - Math.floor(1461 * i / 4) + 31;
  const j = Math.floor(80 * l / 2447);
  const day = l - Math.floor(2447 * j / 80);
  l = Math.floor(j / 11);
  const month = j + 2 - 12 * l;
  const year = 100 * (n - 49) + i + l;
  return { year, month, day };
}

export function formatMjd(mjd) {
  const { year, month, day } = mjdToCalendar(mjd);
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

export function missionYears(mjd) {
  return (mjd - MISSION_START_MJD) / 365.25;
}

/** Largest index such that times[index] <= epoch, plus one (count of samples at or before epoch). */
export function countAtOrBefore(times, epoch) {
  let low = 0, high = times.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (times[middle] <= epoch) low = middle + 1; else high = middle;
  }
  return low;
}
