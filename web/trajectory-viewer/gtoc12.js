// GTOC12 fleet view: Sun-centred J2000 ecliptic scene in AU, Keplerian Earth and asteroid
// orbits from the pinned catalogue elements, per-ship low-thrust arcs drawn as straight GPU
// segments between exact archived propagated samples (no interpolation), deploy/collect markers,
// a mission-epoch timeline, picking and the per-ship panels.

import { clamp, lookAt, multiply, orbitEye, perspective } from "./math.js";
import {
  AU_KM, MISSION_END_MJD, MISSION_START_MJD, countAtOrBefore, formatMjd, missionYears,
  orbitPoints, positionAt, prepareElements,
} from "./kepler.js";
import { GlResources, circleLines, flatten, hex, planeGrid, ribbonArrays, sphere } from "./webgl.js";
import { escapeHtml, metricRows } from "./dom.js";

// Mirrored by `.ship-colour-N` classes in styles.css (CSP forbids inline style attributes).
export const SHIP_COLOURS = [
  "#36d6ff", "#ff5d8f", "#ffd166", "#06d6a0", "#ff8c42", "#c77dff", "#7bf1a8", "#f95738",
  "#6fa8ff", "#f9c74f", "#e0aaff", "#43aa8b", "#ff9de2", "#b8f2e6", "#f4a261",
];
export const FLEET_CAMERA = { yaw: -0.72, pitch: 0.62, distance: 2.7, target: [0, 0, 0] };
export const FLEET_ZOOM = { minimum: 0.06, maximum: 12 };
export const ROLE_LABELS = {
  launch: "Launch from Earth", deploy: "Deploy miner", collect: "Collect mined mass",
  "earth-return": "Return to Earth (deliver)", flyby: "Flyby", rendezvous: "Rendezvous",
};
const MARKER = { disc: 1, ring: 2 };
const ROLE_STYLE = {
  launch: { marker: MARKER.ring, size: 12, colour: [0.27, 1, 0.61, 1] },
  deploy: { marker: MARKER.ring, size: 9, colour: null },
  collect: { marker: MARKER.disc, size: 8, colour: null },
  "earth-return": { marker: MARKER.ring, size: 12, colour: [1, 0.73, 0.31, 1] },
  flyby: { marker: MARKER.ring, size: 10, colour: [0.9, 0.9, 1, 1] },
  rendezvous: { marker: MARKER.ring, size: 8, colour: null },
};
const EARTH_COLOUR = [0.42, 0.68, 1, 1];

export function shipColour(index, alpha = 1) { return hex(SHIP_COLOURS[index % SHIP_COLOURS.length], alpha); }

/** Convert the imported fleet dataset into normalised scene geometry (1 unit = `scale` AU). */
export function buildFleetScene(fleet) {
  const earth = prepareElements(fleet.earth);
  const asteroids = fleet.asteroids.map((record) => ({ ...record, prepared: prepareElements(record) }));
  let maxRadiusAu = 1.1;
  for (const ship of fleet.ships) {
    for (const point of ship.replay.points_txyz) maxRadiusAu = Math.max(maxRadiusAu, Math.hypot(point[1], point[2], point[3]) / AU_KM);
  }
  for (const asteroid of asteroids) maxRadiusAu = Math.max(maxRadiusAu, asteroid.prepared.a_km * (1 + asteroid.e) / AU_KM);
  const scale = maxRadiusAu * 1.03; // AU per scene unit
  const unit = AU_KM * scale;
  const toScene = (km) => [km[0] / unit, km[1] / unit, km[2] / unit];
  const ships = fleet.ships.map((ship, index) => {
    const points = ship.replay.points_txyz.map((point) => toScene(point.slice(1, 4)));
    const times = Float64Array.from(ship.replay.points_txyz, (point) => point[0]);
    const events = ship.events.map((event) => ({ ...event, ship: index, scene: toScene(event.position_km) }));
    const byRole = {};
    for (const event of events) (byRole[event.role] ??= []).push(event);
    const roles = Object.fromEntries(Object.entries(byRole).map(([role, list]) => [role, {
      events: list, times: Float64Array.from(list, (event) => event.epoch_mjd), positions: flatten(list.map((event) => event.scene)),
    }]));
    return {
      index, id: ship.ship_id, colour: shipColour(index), points, times, path: flatten(points), ribbon: ribbonArrays(points),
      events, roles, launch: ship.launch_epoch_mjd, finish: ship.final_sample_epoch_mjd, record: ship,
    };
  });
  const asteroidOrbits = asteroids.map((asteroid) => {
    const loop = orbitPoints(asteroid.prepared, 180).map(toScene);
    const pairs = [];
    for (let index = 0; index < loop.length; index += 1) pairs.push(loop[index], loop[(index + 1) % loop.length]);
    return pairs;
  });
  const orbitVertexCount = asteroidOrbits[0]?.length ?? 0;
  const rings = [];
  for (let au = 1; au < scale; au += 1) rings.push(...circleLines(au / scale, 180));
  return {
    scale, unit, toScene, earth, asteroids, ships, orbitVertexCount,
    asteroidOrbitLines: flatten(asteroidOrbits.flat()),
    earthOrbit: flatten(orbitPoints(earth, 360).map(toScene)),
    rings: new Float32Array(rings), grid: planeGrid([0, 0, 0], 1, 0, 8),
    axes: flatten([[0, 0, 0], [1.08, 0, 0], [0, 0, 0], [0, 1.08, 0], [0, 0, 0], [0, 0, 0.35]]),
    sun: sphere(0.016, [0, 0, 0], 16, 32),
  };
}

export class FleetRenderer extends GlResources {
  constructor(canvasElement, scene) {
    super(canvasElement);
    this.scene = scene;
    const gl = this.gl;
    this.asteroidOrbitBuffer = this.makeBuffer(scene.asteroidOrbitLines);
    this.earthOrbitBuffer = this.makeBuffer(scene.earthOrbit);
    this.ringsBuffer = this.makeBuffer(scene.rings);
    this.gridBuffer = this.makeBuffer(scene.grid);
    this.axesBuffer = this.makeBuffer(scene.axes);
    this.sunPositions = this.makeBuffer(scene.sun.positions); this.sunNormals = this.makeBuffer(scene.sun.normals);
    this.shipBuffers = scene.ships.map((ship) => ({
      path: this.makeBuffer(ship.path),
      ribbon: {
        positions: this.makeBuffer(ship.ribbon.positions), previous: this.makeBuffer(ship.ribbon.previous),
        next: this.makeBuffer(ship.ribbon.next), sides: this.makeBuffer(ship.ribbon.sides),
      },
      roles: Object.fromEntries(Object.entries(ship.roles).map(([role, data]) => [role, this.makeBuffer(data.positions)])),
    }));
    this.dynamicPoint = this.makeBuffer(new Float32Array(3), gl.DYNAMIC_DRAW);
    this.asteroidPositions = new Float32Array(scene.asteroids.length * 3);
    this.asteroidMarkerBuffer = this.makeBuffer(this.asteroidPositions, gl.DYNAMIC_DRAW);
    this.earthPosition = [0, 0, 0];
  }
  matrix(camera) {
    return multiply(perspective(Math.PI / 4, this.canvas.width / this.canvas.height, 0.002, 60), lookAt(orbitEye(camera), camera.target, [0, 0, 1]));
  }
  /** Keplerian positions of Earth and every visited asteroid at the epoch (scene units). */
  updateEphemeris(epoch) {
    const { scene } = this;
    if (this.ephemerisEpoch === epoch) return;
    this.ephemerisEpoch = epoch;
    this.earthPosition = scene.toScene(positionAt(scene.earth, epoch));
    scene.asteroids.forEach((asteroid, index) => {
      const point = scene.toScene(positionAt(asteroid.prepared, epoch));
      this.asteroidPositions.set(point, index * 3);
    });
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.asteroidMarkerBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.asteroidPositions);
  }
  point(mvp, position, colour, size, marker) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamicPoint); gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(position));
    this.line(mvp, this.dynamicPoint, gl.POINTS, 1, colour, size, marker);
  }
  draw(view, camera) {
    const gl = this.gl, scene = this.scene; this.resize();
    this.updateEphemeris(view.epoch);
    const mvp = this.matrix(camera);
    gl.clearColor(0.018, 0.027, 0.055, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    this.line(mvp, this.gridBuffer, gl.LINES, scene.grid.length / 3, [0.3, 0.46, 0.66, 0.12]);
    this.line(mvp, this.ringsBuffer, gl.LINES, scene.rings.length / 3, [0.55, 0.68, 0.86, 0.22]);
    this.line(mvp, this.axesBuffer, gl.LINES, 6, [0.72, 0.8, 0.92, 0.5]);
    const selected = view.selected == null ? null : scene.ships[view.selected];
    this.line(mvp, this.asteroidOrbitBuffer, gl.LINES, scene.asteroidOrbitLines.length / 3, [0.66, 0.74, 0.9, selected ? 0.07 : 0.13]);
    const highlightedAsteroids = new Set(selected ? selected.record.asteroids : []);
    if (view.highlightAsteroid != null) highlightedAsteroids.add(view.highlightAsteroid);
    scene.asteroids.forEach((asteroid, index) => {
      if (!highlightedAsteroids.has(asteroid.id)) return;
      const colour = selected ? [selected.colour[0], selected.colour[1], selected.colour[2], 0.42] : [0.8, 0.86, 1, 0.5];
      this.line(mvp, this.asteroidOrbitBuffer, gl.LINES, scene.orbitVertexCount, colour, 1, 0, index * scene.orbitVertexCount);
    });
    this.line(mvp, this.earthOrbitBuffer, gl.LINE_LOOP, scene.earthOrbit.length / 3, [EARTH_COLOUR[0], EARTH_COLOUR[1], EARTH_COLOUR[2], 0.7]);
    this.surface(mvp, this.sunPositions, this.sunNormals, scene.sun.positions.length / 3, [1, 0.86, 0.45, 1]);
    scene.ships.forEach((ship, index) => {
      const visible = countAtOrBefore(ship.times, view.epoch);
      if (visible === 0) return;
      const dim = selected && selected !== ship;
      const alpha = dim ? 0.2 : 0.95;
      const colour = [ship.colour[0], ship.colour[1], ship.colour[2], alpha];
      const buffers = this.shipBuffers[index];
      if (visible > 1) {
        this.ribbon(mvp, buffers.ribbon, visible * 2, colour, dim ? 1.2 : selected === ship ? 2.6 : 1.7);
        this.line(mvp, buffers.path, gl.LINE_STRIP, visible, [colour[0], colour[1], colour[2], alpha * 0.85]);
      }
      if (!dim) {
        for (const [role, data] of Object.entries(ship.roles)) {
          const count = countAtOrBefore(data.times, view.epoch);
          const style = ROLE_STYLE[role] ?? ROLE_STYLE.rendezvous;
          this.line(mvp, buffers.roles[role], gl.POINTS, count, style.colour ?? ship.colour, style.size, style.marker);
        }
      }
      const current = ship.points[visible - 1];
      if (!dim) {
        this.point(mvp, current, [1, 1, 1, 0.95], 11, MARKER.disc);
        this.point(mvp, current, ship.colour, 7, MARKER.disc);
      } else this.point(mvp, current, [colour[0], colour[1], colour[2], 0.45], 6, MARKER.disc);
    });
    this.line(mvp, this.asteroidMarkerBuffer, gl.POINTS, scene.asteroids.length, [0.78, 0.84, 0.96, selected ? 0.25 : 0.5], 3.5, MARKER.disc);
    scene.asteroids.forEach((asteroid, index) => {
      if (!highlightedAsteroids.has(asteroid.id)) return;
      this.line(mvp, this.asteroidMarkerBuffer, gl.POINTS, 1, selected ? selected.colour : [0.9, 0.94, 1, 1], 6, MARKER.disc, index);
    });
    this.point(mvp, this.earthPosition, [1, 1, 1, 0.9], 12, MARKER.ring);
    this.point(mvp, this.earthPosition, EARTH_COLOUR, 8, MARKER.disc);
    const focus = view.hover ?? view.pinned;
    if (focus) this.point(mvp, focus.scene, [1, 1, 1, 0.9], 20, MARKER.ring);
  }
  /** Scene position -> CSS pixel position on the canvas (null when behind the camera). */
  project(mvp, p) {
    const w = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
    if (w <= 0) return null;
    const cx = (mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12]) / w;
    const cy = (mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13]) / w;
    return [(cx * 0.5 + 0.5) * this.canvas.clientWidth, (0.5 - cy * 0.5) * this.canvas.clientHeight];
  }
  /** Nearest pickable item within `radius` CSS pixels of (x, y) on the canvas. */
  pick(x, y, view, camera, radius = 14) {
    const scene = this.scene, mvp = this.matrix(camera);
    const project = (p) => this.project(mvp, p);
    let best = null, bestScore = radius;
    const consider = (position, item, bonus = 0) => {
      const screen = project(position); if (!screen) return;
      const distance = Math.hypot(screen[0] - x, screen[1] - y) - bonus;
      if (distance < bestScore) { bestScore = distance; best = { ...item, scene: position, screen }; }
    };
    const selected = view.selected == null ? null : scene.ships[view.selected];
    this.updateEphemeris(view.epoch);
    for (const ship of scene.ships) {
      const visible = countAtOrBefore(ship.times, view.epoch);
      if (visible === 0 || (selected && selected !== ship)) continue;
      for (const event of ship.events) {
        if (event.epoch_mjd <= view.epoch) consider(event.scene, { type: "event", ship: ship.index, event }, 4);
      }
      consider(ship.points[visible - 1], { type: "ship", ship: ship.index, sample: visible - 1, epoch: ship.times[visible - 1] }, 3);
    }
    consider(this.earthPosition, { type: "earth", epoch: view.epoch }, 3);
    scene.asteroids.forEach((asteroid, index) => {
      consider(Array.from(this.asteroidPositions.subarray(index * 3, index * 3 + 3)), { type: "asteroid", asteroid }, 1);
    });
    if (!best) {
      for (const ship of scene.ships) {
        const visible = countAtOrBefore(ship.times, view.epoch);
        if (selected && selected !== ship) continue;
        for (let index = 0; index < visible; index += 1) {
          consider(ship.points[index], { type: "sample", ship: ship.index, sample: index, epoch: ship.times[index] });
        }
      }
    }
    return best;
  }
}

export function createFleetView({ canvas, fleet, camera }) {
  const scene = buildFleetScene(fleet);
  const renderer = new FleetRenderer(canvas, scene);
  const view = { epoch: MISSION_END_MJD, selected: null, hover: null, pinned: null, highlightAsteroid: null };
  return {
    fleet, scene, renderer, view, camera, info: renderer.info,
    draw() { renderer.draw(view, camera); },
    setEpoch(mjd) { view.epoch = clamp(Math.round(mjd), MISSION_START_MJD, MISSION_END_MJD); },
    selectShip(index) {
      view.selected = index == null ? null : clamp(Number(index), 0, scene.ships.length - 1);
      view.highlightAsteroid = null; view.pinned = null;
    },
    focusShip(index) {
      const ship = scene.ships[index ?? view.selected ?? 0];
      if (!ship) return;
      const minima = [Infinity, Infinity, Infinity], maxima = [-Infinity, -Infinity, -Infinity];
      for (const point of ship.points) for (let axis = 0; axis < 3; axis += 1) {
        minima[axis] = Math.min(minima[axis], point[axis]); maxima[axis] = Math.max(maxima[axis], point[axis]);
      }
      const center = minima.map((value, axis) => (value + maxima[axis]) / 2);
      const radius = Math.max(...ship.points.map((point) => Math.hypot(...point.map((value, axis) => value - center[axis]))), 0.05);
      // 45° vertical field of view: distance 3 r keeps the whole bounding sphere inside the frame.
      Object.assign(camera, { target: center, distance: clamp(radius * 3.0, FLEET_ZOOM.minimum, FLEET_ZOOM.maximum), pitch: 1.3 });
    },
    /** CSS pixel position of an archived event marker (for tests and tooling). */
    eventScreenPosition(shipIndex, eventIndex) {
      const event = scene.ships[shipIndex]?.events[eventIndex];
      return event ? renderer.project(renderer.matrix(camera), event.scene) : null;
    },
    earthScreenPosition() {
      renderer.updateEphemeris(view.epoch);
      return renderer.project(renderer.matrix(camera), renderer.earthPosition);
    },
    hover(x, y) { view.hover = renderer.pick(x, y, view, camera); return view.hover; },
    click(x, y) {
      const item = renderer.pick(x, y, view, camera);
      if (item && item.ship != null) { view.selected = item.ship; view.pinned = item; view.highlightAsteroid = item.event?.event_id > 0 ? item.event.event_id : null; }
      else if (item?.type === "asteroid") { view.pinned = item; view.highlightAsteroid = item.asteroid.id; }
      else if (item?.type === "earth") view.pinned = item;
      else { view.pinned = null; view.highlightAsteroid = null; }
      return item;
    },
    dispose() { renderer.dispose(); },
  };
}

function epochLabel(mjd) { return `MJD ${mjd.toFixed(0)} · ${formatMjd(mjd)}`; }
function kg(value) { return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(1)} kg`; }

export function describeItem(item, fleetView) {
  if (!item) return "";
  const { scene } = fleetView;
  if (item.type === "event") {
    const ship = scene.ships[item.ship], event = item.event;
    return `<strong>Ship ${ship.id} · ${escapeHtml(ROLE_LABELS[event.role] ?? event.role)}</strong><span>${escapeHtml(event.body)} · ${epochLabel(event.epoch_mjd)}</span><span>Mass ${event.mass_before_kg.toFixed(1)} → ${event.mass_after_kg.toFixed(1)} kg (${kg(event.mass_delta_kg)})</span>`;
  }
  if (item.type === "ship" || item.type === "sample") {
    const ship = scene.ships[item.ship];
    const point = ship.record.replay.points_txyz[item.sample];
    const radius = Math.hypot(point[1], point[2], point[3]) / AU_KM;
    return `<strong>Ship ${ship.id}${item.type === "ship" ? " · current position" : ""}</strong><span>Archived sample ${item.sample + 1} / ${ship.record.replay.point_count} · ${epochLabel(point[0])}</span><span>r = ${radius.toFixed(3)} AU · ${ship.record.collected_kg.toFixed(1)} kg collected over the mission</span>`;
  }
  if (item.type === "asteroid") {
    const asteroid = item.asteroid;
    const visits = scene.ships.flatMap((ship) => ship.events.filter((event) => event.event_id === asteroid.id).map((event) => `ship ${ship.id} ${event.role} ${formatMjd(event.epoch_mjd)}`));
    return `<strong>Asteroid ${asteroid.id}</strong><span>a ${asteroid.a_au.toFixed(3)} AU · e ${asteroid.e.toFixed(3)} · i ${asteroid.i_deg.toFixed(2)}°</span><span>${escapeHtml(visits.join(" · ") || "not visited")}</span>`;
  }
  if (item.type === "earth") {
    const position = positionAt(scene.earth, item.epoch);
    return `<strong>Earth</strong><span>${epochLabel(item.epoch)} · r = ${(Math.hypot(...position) / AU_KM).toFixed(4)} AU</span><span>Keplerian position from GTOC12 Table 2 elements</span>`;
  }
  return "";
}

export function renderShipList(container, fleetView) {
  const { fleet, view } = fleetView;
  container.innerHTML = `<button type="button" class="ship-item ${view.selected == null ? "active" : ""}" data-ship="all" aria-pressed="${view.selected == null}">
      <span class="ship-swatch fleet-swatch" aria-hidden="true"></span><span><strong>Whole fleet</strong><small>${fleet.ships.length} ships · ${fleet.asteroids.length} asteroids · ${fleet.score.official_total_mass_kg} kg</small></span></button>${
    fleet.ships.map((ship, index) => `
    <button type="button" class="ship-item ${view.selected === index ? "active" : ""}" data-ship="${index}" aria-pressed="${view.selected === index}">
      <span class="ship-swatch ship-colour-${index + 1}" aria-hidden="true"></span>
      <span><strong>Ship ${ship.ship_id}</strong><small>${ship.collected_kg.toFixed(1)} kg · ${ship.asteroids.length} asteroids · launch ${formatMjd(ship.launch_epoch_mjd)}</small></span>
    </button>`).join("")}`;
}

export function renderShipDetail(container, fleetView) {
  const { fleet, view } = fleetView;
  if (view.selected == null) {
    container.innerHTML = `<p class="help-text">Select a ship to list its deploy/collect sequence. Fleet totals below are read from the verified export.</p><dl class="metric-list">${metricRows([
      ["Ships", String(fleet.score.ships)], ["Unique asteroids", String(fleet.score.unique_asteroids)],
      ["Official verifier mass", `${fleet.score.official_total_mass_kg} kg`], ["Independent verifier mass", `${fleet.score.independent_total_mass_kg?.toFixed(3)} kg`],
      ["Ship-count rule", `${fleet.score.ships} ≤ ${fleet.score.ship_limit?.toFixed(2)}`], ["Mission window", `${formatMjd(fleet.constants.mission_start_mjd)} → ${formatMjd(fleet.constants.mission_end_mjd)}`],
    ])}</dl>`;
    return;
  }
  const ship = fleet.ships[view.selected];
  const rows = ship.events.map((event) => `<tr class="${event.epoch_mjd <= view.epoch ? "passed" : "pending"} role-${event.role}">
      <td>${event.index + 1}</td><td>${formatMjd(event.epoch_mjd)}<small>MJD ${event.epoch_mjd}</small></td>
      <td>${escapeHtml(event.body)}</td><td>${escapeHtml(ROLE_LABELS[event.role] ?? event.role)}</td><td class="numeric">${kg(event.mass_delta_kg)}</td></tr>`).join("");
  container.innerHTML = `<dl class="metric-list">${metricRows([
    ["Collected", `${ship.collected_kg.toFixed(3)} kg`], ["Launch", `${formatMjd(ship.launch_epoch_mjd)} (MJD ${ship.launch_epoch_mjd})`],
    ["Return", `${formatMjd(ship.return_epoch_mjd)} (MJD ${ship.return_epoch_mjd})`], ["Initial → final mass", `${ship.initial_mass_kg.toFixed(1)} → ${ship.final_mass_kg.toFixed(1)} kg`],
    ["Miners deployed / collects", `${ship.miners_deployed} / ${ship.collects}`], ["Archived samples", `${ship.replay.point_count} of ${ship.replay.original_point_count}`],
  ])}</dl>
  <div class="event-table-wrap"><table class="event-table"><thead><tr><th>#</th><th>Epoch</th><th>Body</th><th>Event</th><th class="numeric">Δm</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

export function renderFleetSummary(container, fleetView) {
  const { fleet, view } = fleetView;
  const active = fleet.ships.filter((ship) => ship.launch_epoch_mjd <= view.epoch && ship.final_sample_epoch_mjd > view.epoch).length;
  const deployed = fleet.ships.reduce((sum, ship) => sum + ship.events.filter((event) => event.role === "deploy" && event.epoch_mjd <= view.epoch).length, 0);
  const collected = fleet.ships.reduce((sum, ship) => sum + ship.events.filter((event) => event.role === "collect" && event.epoch_mjd <= view.epoch).reduce((inner, event) => inner + event.mass_delta_kg, 0), 0);
  const delivered = fleet.ships.filter((ship) => ship.return_epoch_mjd <= view.epoch).reduce((sum, ship) => sum + ship.collected_kg, 0);
  container.innerHTML = metricRows([
    ["Epoch", `${epochLabel(view.epoch)} · T+${missionYears(view.epoch).toFixed(2)} yr`], ["Ships in flight", `${active} of ${fleet.ships.length}`],
    ["Miners deployed so far", String(deployed)], ["Mass collected so far", `${collected.toFixed(1)} kg`],
    ["Delivered to Earth so far", `${delivered.toFixed(1)} kg`], ["Final official score", `${fleet.score.official_total_mass_kg} kg`],
  ]);
}

export function renderFleetProvenance(container, fleetView) {
  const { fleet } = fleetView;
  container.innerHTML = `<p>Run <code>${escapeHtml(fleet.run_id)}</code> · export commit <code>${escapeHtml(fleet.generated_by_commit)}</code> · ${escapeHtml(fleet.source.generator)}.</p>
    <p>Official solution file <code>${escapeHtml(fleet.source.solution_basename)}</code> SHA-256 <code>${escapeHtml(fleet.source.solution_sha256)}</code> (${fleet.source.solution_bytes.toLocaleString()} bytes; official verifier ${fleet.source.official_verifier_ok === null ? "not recorded" : fleet.source.official_verifier_ok ? "pass" : "fail"}, independent verifier ${fleet.score.verifier_ok ? "pass" : "fail"}, max propagation error ${fleet.verification.max_position_error_km?.toFixed(2)} km).</p>
    <p>Export <code>trajectories.json</code> SHA-256 <code>${escapeHtml(fleet.source.export_trajectories_sha256)}</code>; asteroid catalogue <code>${escapeHtml(fleet.source.catalogue.name)}</code> SHA-256 <code>${escapeHtml(fleet.source.catalogue.sha256)}</code>.</p>
    <p>Ship arcs are straight GPU segments connecting ${fleet.ships.reduce((sum, ship) => sum + ship.replay.point_count, 0).toLocaleString()} exact archived propagated samples (≤ 512 per ship, every event epoch preserved) — connections between archived nodes, not interpolation. Earth and asteroid orbits and their epoch positions are two-body Keplerian curves from the pinned GTOC12 elements (Appendix 6.1); the viewer's propagation agrees with the exporter's context orbits to ${fleet.kepler_check.asteroid_max_error_km.toExponential(1)} km over ${fleet.kepler_check.context_points_checked.toLocaleString()} samples. The Sun marker is not to scale.</p>`;
}

export function renderTimelineOutput(output, fleetView) {
  const { view } = fleetView;
  output.textContent = `${epochLabel(view.epoch)} · T+${missionYears(view.epoch).toFixed(2)} yr`;
}

const LABEL_TEXT = { launch: (event) => `Launch ${formatMjd(event.epoch_mjd)}`, deploy: (event) => `deploy ${event.event_id}`, collect: (event) => `collect ${event.event_id}`, "earth-return": (event) => `Earth ${formatMjd(event.epoch_mjd)}` };

/** HTML labels over the selected ship's archived events, positioned with the same projection as the picking. */
export function renderEventLabels(container, fleetView) {
  const { scene, view, renderer, camera } = fleetView;
  if (view.selected == null) { container.replaceChildren(); container.dataset.ship = ""; return; }
  const ship = scene.ships[view.selected];
  if (container.dataset.ship !== String(ship.index)) {
    container.dataset.ship = String(ship.index);
    container.replaceChildren(...ship.events.map((event) => {
      const label = document.createElement("span");
      label.className = `event-label role-${event.role}`;
      label.textContent = (LABEL_TEXT[event.role] ?? ((item) => `${item.role} ${item.event_id}`))(event);
      return label;
    }));
  }
  const mvp = renderer.matrix(camera);
  const width = renderer.canvas.clientWidth, height = renderer.canvas.clientHeight;
  ship.events.forEach((event, index) => {
    const label = container.children[index];
    const screen = event.epoch_mjd <= view.epoch ? renderer.project(mvp, event.scene) : null;
    const visible = screen && screen[0] >= 0 && screen[1] >= 0 && screen[0] <= width && screen[1] <= height;
    label.hidden = !visible;
    if (visible) { label.style.left = `${screen[0]}px`; label.style.top = `${screen[1]}px`; }
  });
}
