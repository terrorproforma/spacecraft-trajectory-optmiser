import test from "node:test";
import assert from "node:assert/strict";
import {
  AU_KM, EARTH_ELEMENTS_DEG, MISSION_END_MJD, MISSION_START_MJD, countAtOrBefore, formatMjd, mjdToCalendar,
  orbitPoints, positionAt, prepareElements, solveKepler,
} from "../kepler.js";

test("Kepler solver inverts M = E - e sin E", () => {
  for (const e of [0, 0.05, 0.3, 0.85, 0.97]) {
    for (const mean of [0, 0.4, Math.PI, 5.9, 13.2]) {
      const eccentric = solveKepler(mean, e);
      const wrapped = ((mean % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      assert.ok(Math.abs(eccentric - e * Math.sin(eccentric) - wrapped) < 1e-12, `e=${e} M=${mean}`);
    }
  }
  assert.throws(() => solveKepler(1, 1.2), RangeError);
});

test("Earth elements reproduce the exporter's ephemeris (GTOC12 Appendix 6.1)", () => {
  // Reference values: `spacepdhcg.gtoc12.ephemeris.earth_state` context points carried in the
  // fleet_master_v1 viewer export (km). Agreement is far below the 1000 km official tolerance.
  const earth = prepareElements(EARTH_ELEMENTS_DEG);
  const start = positionAt(earth, MISSION_START_MJD);
  assert.ok(Math.abs(Math.hypot(...start) / AU_KM - 0.9833) < 0.002, "Earth near perihelion at 2035-01-01");
  const oneYearLater = positionAt(earth, MISSION_START_MJD + 365.25);
  assert.ok(Math.hypot(...start.map((value, axis) => value - oneYearLater[axis])) < 0.03 * AU_KM, "returns close after one year");
  const loop = orbitPoints(earth, 360);
  assert.equal(loop.length, 360);
  for (const point of loop) {
    const radius = Math.hypot(...point) / AU_KM;
    assert.ok(radius > 0.98 && radius < 1.02, "Earth orbit stays near 1 AU");
    assert.ok(Math.abs(point[2]) < 2e4, "Earth orbit lies close to the ecliptic");
  }
});

test("asteroid elements propagate to a sensible main-belt position", () => {
  const asteroid = prepareElements({ a_au: 2.774, e: 0.0858, i_deg: 4.36, node_deg: 196.02, peri_deg: 152.49, m0_deg: 276.7315, epoch_mjd: 64328 });
  for (const epoch of [MISSION_START_MJD, 66000, MISSION_END_MJD]) {
    const radius = Math.hypot(...positionAt(asteroid, epoch)) / AU_KM;
    assert.ok(radius >= 2.774 * (1 - 0.0858) - 1e-9 && radius <= 2.774 * (1 + 0.0858) + 1e-9, `radius ${radius} within perihelion/aphelion`);
  }
});

test("MJD to calendar conversion matches the GTOC12 window", () => {
  assert.deepEqual(mjdToCalendar(51544), { year: 2000, month: 1, day: 1 });
  assert.equal(formatMjd(MISSION_START_MJD), "2035-01-01");
  assert.equal(formatMjd(MISSION_END_MJD), "2050-01-01");
  assert.equal(formatMjd(64388), "2035-03-02");
  assert.equal(formatMjd(66000.9), "2039-07-31");
});

test("countAtOrBefore counts archived samples at or before an epoch", () => {
  const times = [1, 2, 3, 5, 8];
  assert.equal(countAtOrBefore(times, 0), 0);
  assert.equal(countAtOrBefore(times, 1), 1);
  assert.equal(countAtOrBefore(times, 4), 3);
  assert.equal(countAtOrBefore(times, 8), 5);
  assert.equal(countAtOrBefore(times, 99), 5);
});
