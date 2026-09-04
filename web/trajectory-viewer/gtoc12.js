// GTOC12 fleet view: a perspective 3D Sun-centred J2000 ecliptic scene in AU. Keplerian Earth and
// asteroid orbits (pinned catalogue elements) are drawn as depth-faded 3D ribbons; the Sun, Earth,
// the visited asteroids and the ships are instanced lit spheres (Sun = point light, Lambert +
// Blinn-Phong, sky ambient, distance fog); each ship's low-thrust arc is a lit tube mesh whose
// vertices sit exactly on the archived propagated samples (straight connections, no
// interpolation) with a time-faded trail. Optional vertical (Z) exaggeration makes inclinations
// legible; it is a display transform only and is labelled as such in the UI.

import { clamp, lookAt, multiply, orbitEye, perspective } from "./math.js";
import {
  AU_KM, MISSION_END_MJD, MISSION_START_MJD, countAtOrBefore, formatMjd, missionYears,
  orbitPoints, positionAt, prepareElements,
} from "./kepler.js";
import {
  BACKGROUND_FRAGMENT, BACKGROUND_VERTEX, BODY_FRAGMENT, BODY_VERTEX, GlResources, STAR_FRAGMENT, STAR_VERTEX, TUBE_FRAGMENT,
  TUBE_VERTEX, circleLines, concatRibbons, discTriangles, flatten, hex, ribbonArrays, sphere, spokeLines, starField, tubeArrays,
} from "./webgl.js";
import { DEG, EXAGGERATION, boundsOf, exaggerate, fitDistance, followCamera } from "./camera.js";
import { escapeHtml, metricRows } from "./dom.js";

// Mirrored by `.ship-colour-N` classes in styles.css (CSP forbids inline style attributes).
export const SHIP_COLOURS = [
  "#36d6ff", "#ff5d8f", "#ffd166", "#06d6a0", "#ff8c42", "#c77dff", "#7bf1a8", "#f95738",
  "#6fa8ff", "#f9c74f", "#e0aaff", "#43aa8b", "#ff9de2", "#b8f2e6", "#f4a261",
  "#ff4fd8", "#b5ff3d", "#9d4edd", "#ffe8a3", "#c9184a",
  "#00b4d8", "#ef476f", "#8ac926", "#ffb703",
];
export const FLEET_CAMERA = { yaw: -0.72, pitch: 30 * DEG, distance: 2.9, target: [0, 0, 0] };
export const FLEET_ZOOM = { minimum: 0.03, maximum: 12 };
export const FOLLOW_DISTANCE = 0.34;
export const FIELD_OF_VIEW = Math.PI / 4;
/** Days after an archived event during which its marker and asteroid flash. */
export const FLASH_DAYS = 60;
/** Days of arc behind each ship drawn at full brightness; older arc fades to the base alpha. */
export const TRAIL_DAYS = 450;
export const ROLE_LABELS = {
  launch: "Launch from Earth", deploy: "Deploy miner", collect: "Collect mined mass",
  "earth-return": "Return to Earth (deliver)", flyby: "Flyby", rendezvous: "Rendezvous",
};
const MARKER = { disc: 1, ring: 2, glow: 3 };
const ROLE_STYLE = {
  launch: { marker: MARKER.ring, size: 12, colour: [0.27, 1, 0.61, 1] },
  deploy: { marker: MARKER.ring, size: 8, colour: null },
  collect: { marker: MARKER.disc, size: 7, colour: null },
  "earth-return": { marker: MARKER.ring, size: 12, colour: [1, 0.73, 0.31, 1] },
  flyby: { marker: MARKER.ring, size: 10, colour: [0.9, 0.9, 1, 1] },
  rendezvous: { marker: MARKER.ring, size: 8, colour: null },
};
const EARTH_COLOUR = [0.42, 0.68, 1, 1];
const SUN_COLOUR = [1, 0.86, 0.45, 1];
const PENDING_ASTEROID = [0.5, 0.56, 0.68, 1];
/** Sky colours: the fog colour matches the lower gradient so distant geometry sinks into the background. */
export const SKY = { top: [0.035, 0.05, 0.1], bottom: [0.006, 0.009, 0.02], fog: [0.012, 0.018, 0.04], ambient: [0.16, 0.18, 0.25] };
/** Tube cross-section sides; 6 keeps 19 ships x 512 samples under 60k vertices while reading as round at a few pixels. */
export const TUBE_SIDES = 6;
/** Floats per body instance: centre xyz, radius, rgba, emissive. */
const INSTANCE_FLOATS = 9;

export function shipColour(index, alpha = 1) { return hex(SHIP_COLOURS[index % SHIP_COLOURS.length], alpha); }
/** Headline mass: the summed archived collects (10700.48 kg for fleet_master_v4); the official verifier prints 6 significant digits. */
export function fleetMassLabel(fleet) { return (fleet.score.total_collected_kg ?? fleet.score.official_total_mass_kg).toFixed(2); }
export function massPerShip(fleet) { return (fleet.score.total_collected_kg ?? fleet.score.official_total_mass_kg) / fleet.ships.length; }

/** Mass collected by a ship at or before `epoch` (sum of archived collect events). */
export function collectedAt(ship, epoch) {
  const count = countAtOrBefore(ship.collects.times, epoch);
  return count === 0 ? 0 : ship.collects.cumulative[count - 1];
}
/** Flash intensity in [0, 1] for an event at `eventEpoch` seen at `epoch`. */
export function flashAt(eventEpoch, epoch, duration = FLASH_DAYS) {
  const age = epoch - eventEpoch;
  return age < 0 || age > duration ? 0 : 1 - age / duration;
}
/** Visual state of a visited asteroid at `epoch`: pending (not yet reached), deployed (miner on it) or collected. */
export function asteroidState(status, epoch) {
  if (epoch >= status.collect) return "collected";
  if (epoch >= status.deploy) return "deployed";
  return "pending";
}

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
  const asteroidIndex = new Map(asteroids.map((asteroid, index) => [asteroid.id, index]));
  const asteroidStatus = asteroids.map(() => ({ deploy: Infinity, collect: Infinity, ship: null }));
  const ships = fleet.ships.map((ship, index) => {
    const points = ship.replay.points_txyz.map((point) => toScene(point.slice(1, 4)));
    const times = Float64Array.from(ship.replay.points_txyz, (point) => point[0]);
    const events = ship.events.map((event) => ({ ...event, ship: index, scene: toScene(event.position_km) }));
    const byRole = {};
    for (const event of events) (byRole[event.role] ??= []).push(event);
    const roles = Object.fromEntries(Object.entries(byRole).map(([role, list]) => [role, {
      events: list, times: Float64Array.from(list, (event) => event.epoch_mjd), positions: flatten(list.map((event) => event.scene)),
    }]));
    const collectEvents = byRole.collect ?? [];
    let running = 0;
    const collects = {
      times: Float64Array.from(collectEvents, (event) => event.epoch_mjd),
      cumulative: Float64Array.from(collectEvents, (event) => { running += event.mass_delta_kg; return running; }),
    };
    for (const event of events) {
      const at = asteroidIndex.get(event.event_id);
      if (at == null) continue;
      const status = asteroidStatus[at];
      if (event.role === "deploy" && event.epoch_mjd < status.deploy) { status.deploy = event.epoch_mjd; status.ship = index; }
      if (event.role === "collect" && event.epoch_mjd < status.collect) status.collect = event.epoch_mjd;
    }
    return {
      index, id: ship.ship_id, colour: shipColour(index), points, times, tube: tubeArrays(points, Array.from(times), TUBE_SIDES),
      events, roles, collects, launch: ship.launch_epoch_mjd, finish: ship.final_sample_epoch_mjd, record: ship,
    };
  });
  const orbitRibbons = concatRibbons(asteroids.map((asteroid) => ribbonArrays(orbitPoints(asteroid.prepared, 240).map(toScene), null, true)));
  const rings = [];
  for (let au = 1; au < scale; au += 1) rings.push(...circleLines(au / scale, 240));
  return {
    scale, unit, toScene, earth, asteroids, asteroidIndex, asteroidStatus, ships, orbitRibbons,
    earthRibbon: ribbonArrays(orbitPoints(earth, 360).map(toScene), null, true),
    rings: new Float32Array(rings), disc: discTriangles(1.0, 144), spokes: spokeLines(0.03, 1.0, 30),
    axes: flatten([[0, 0, 0], [1.06, 0, 0], [0, 0, 0], [0, 1.06, 0], [0, 0, 0], [0, 0, 0.05]]),
    stars: starField(2600, 30, 12), unitSphere: sphere(1, [0, 0, 0], 18, 36),
  };
}

/** Build the per-frame body instance list (unexaggerated centres; the shader applies uZScale). */
export function bodyInstances(scene, view, radii, positions) {
  const selected = view.selected == null ? null : scene.ships[view.selected];
  const hoverShip = view.hover?.ship ?? view.pinned?.ship ?? null;
  const hoverAsteroid = view.hover?.type === "asteroid" ? view.hover.index : view.pinned?.type === "asteroid" ? view.pinned.index : null;
  const highlighted = new Set(selected ? selected.record.asteroids : []);
  if (view.highlightAsteroid != null) highlighted.add(view.highlightAsteroid);
  const instances = [], flashes = [];
  const push = (center, radius, colour, alpha, emissive) => instances.push(center[0], center[1], center[2], radius, colour[0], colour[1], colour[2], alpha, emissive);
  push([0, 0, 0], radii.sun, SUN_COLOUR, 1, 1);
  push(positions.earth, radii.earth, EARTH_COLOUR, 1, 0.05);
  scene.asteroids.forEach((asteroid, index) => {
    const status = scene.asteroidStatus[index], state = asteroidState(status, view.epoch);
    const owner = status.ship == null ? null : scene.ships[status.ship];
    const related = highlighted.has(asteroid.id), dim = selected && !related;
    const pulse = Math.max(flashAt(status.deploy, view.epoch), flashAt(status.collect, view.epoch));
    let colour = PENDING_ASTEROID, emissive = 0.08, radius = radii.asteroid;
    if (state === "deployed" && owner) { colour = owner.colour.map((value, axis) => (axis < 3 ? 0.55 + 0.45 * value : 1)); emissive = 0.4; radius *= 1.15; }
    if (state === "collected" && owner) { colour = owner.colour; emissive = 0.8; radius *= 1.35; }
    if (hoverAsteroid === index) { emissive = Math.max(emissive, 0.6); radius *= 1.5; }
    radius *= 1 + 1.4 * pulse;
    const position = positions.asteroid(index);
    push(position, radius, colour, dim ? 0.3 : 1, emissive);
    if (pulse > 0 && !dim) flashes.push({ position, colour, pulse });
  });
  scene.ships.forEach((ship) => {
    const visible = countAtOrBefore(ship.times, view.epoch);
    if (visible === 0) return;
    const dim = selected && selected !== ship, hot = hoverShip === ship.index;
    push(ship.points[visible - 1], radii.ship * (dim ? 0.7 : hot ? 1.35 : 1), ship.colour, dim ? 0.45 : 1, hot ? 0.55 : 0.3);
  });
  return { data: new Float32Array(instances), count: instances.length / INSTANCE_FLOATS, flashes, hoverShip, selected };
}

export class FleetRenderer extends GlResources {
  constructor(canvasElement, scene) {
    super(canvasElement);
    this.scene = scene;
    const gl = this.gl;
    this.bodyProgram = this.makeProgram(BODY_VERTEX, BODY_FRAGMENT);
    this.tubeProgram = this.makeProgram(TUBE_VERTEX, TUBE_FRAGMENT);
    this.starProgram = this.makeProgram(STAR_VERTEX, STAR_FRAGMENT);
    this.backgroundProgram = this.makeProgram(BACKGROUND_VERTEX, BACKGROUND_FRAGMENT);
    const ribbonBuffers = (ribbon) => ({
      positions: this.makeBuffer(ribbon.positions), previous: this.makeBuffer(ribbon.previous),
      next: this.makeBuffer(ribbon.next), sides: this.makeBuffer(ribbon.sides), times: ribbon.times ? this.makeBuffer(ribbon.times) : null,
    });
    this.orbitRibbon = ribbonBuffers(scene.orbitRibbons);
    this.earthRibbon = ribbonBuffers(scene.earthRibbon);
    this.ringsBuffer = this.makeBuffer(scene.rings);
    this.discBuffer = this.makeBuffer(scene.disc);
    this.spokesBuffer = this.makeBuffer(scene.spokes);
    this.axesBuffer = this.makeBuffer(scene.axes);
    this.starPositions = this.makeBuffer(scene.stars.positions); this.starMagnitudes = this.makeBuffer(scene.stars.magnitudes);
    this.unitSphere = this.makeBuffer(scene.unitSphere.normals);
    this.shipBuffers = scene.ships.map((ship) => ({
      tube: {
        positions: this.makeBuffer(ship.tube.positions), normals: this.makeBuffer(ship.tube.normals),
        times: this.makeBuffer(ship.tube.times), indices: this.makeIndexBuffer(ship.tube.indices),
      },
      roles: Object.fromEntries(Object.entries(ship.roles).map(([role, data]) => [role, this.makeBuffer(data.positions)])),
    }));
    this.dynamicPoint = this.makeBuffer(new Float32Array(3), gl.DYNAMIC_DRAW);
    this.instanceCapacity = 2 + scene.asteroids.length + scene.ships.length;
    this.instanceBuffer = this.makeBuffer(new Float32Array(this.instanceCapacity * INSTANCE_FLOATS), gl.DYNAMIC_DRAW);
    this.asteroidPositions = new Float32Array(scene.asteroids.length * 3);
    this.earthPosition = [0, 0, 0];
    this.exaggeration = EXAGGERATION.initial;
    this.lastInstanceCount = 0;
  }
  /** Scene point -> displayed point under the current vertical exaggeration. */
  place(point) { return exaggerate(point, this.exaggeration); }
  aspect() { return this.canvas.width / Math.max(1, this.canvas.height); }
  matrix(camera) {
    return multiply(perspective(FIELD_OF_VIEW, this.aspect(), 0.002, 80), lookAt(orbitEye(camera), camera.target, [0, 0, 1]));
  }
  /** Rotation-only matrix so the star sphere sits at infinity regardless of pan/zoom. */
  starMatrix(camera) {
    const eye = orbitEye(camera);
    return multiply(perspective(FIELD_OF_VIEW, this.aspect(), 0.5, 80), lookAt(eye.map((value, axis) => value - camera.target[axis]), [0, 0, 0], [0, 0, 1]));
  }
  /** Keplerian positions of Earth and every visited asteroid at the epoch (scene units, unexaggerated). */
  updateEphemeris(epoch) {
    const { scene } = this;
    if (this.ephemerisEpoch === epoch) return;
    this.ephemerisEpoch = epoch;
    this.earthPosition = scene.toScene(positionAt(scene.earth, epoch));
    scene.asteroids.forEach((asteroid, index) => this.asteroidPositions.set(scene.toScene(positionAt(asteroid.prepared, epoch)), index * 3));
  }
  asteroidPosition(index) { return Array.from(this.asteroidPositions.subarray(index * 3, index * 3 + 3)); }
  point(mvp, position, colour, size, marker) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamicPoint); gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(position));
    this.line(mvp, this.dynamicPoint, gl.POINTS, 1, colour, size, marker);
  }
  /** Fog density for this camera: geometry a full camera-distance behind the target is ~6% fogged, three distances ~40%. */
  fogDensity(camera) { return 0.25 / Math.max(camera.distance, 1e-3); }
  /** Procedural sky gradient: a full-screen triangle drawn first with depth testing off. */
  background() {
    const gl = this.gl, p = this.backgroundProgram; gl.useProgram(p);
    gl.uniform3fv(this.uniform(p, "uTop"), SKY.top); gl.uniform3fv(this.uniform(p, "uBottom"), SKY.bottom); gl.uniform1f(this.uniform(p, "uBand"), 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  /** Instanced lit spheres: one draw call for the Sun, Earth, every asteroid and every ship marker. */
  bodies(mvp, eye, camera, instances) {
    if (instances.count === 0) return;
    const gl = this.gl, p = this.bodyProgram; gl.useProgram(p);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, instances.data);
    this.attribute(p, "aNormal", this.unitSphere, 3);
    this.instancedAttribute(p, "aCenter", this.instanceBuffer, 3, INSTANCE_FLOATS, 0);
    this.instancedAttribute(p, "aRadius", this.instanceBuffer, 1, INSTANCE_FLOATS, 3);
    this.instancedAttribute(p, "aColor", this.instanceBuffer, 4, INSTANCE_FLOATS, 4);
    this.instancedAttribute(p, "aEmissive", this.instanceBuffer, 1, INSTANCE_FLOATS, 8);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform1f(this.uniform(p, "uZScale"), this.zScale);
    gl.uniform3fv(this.uniform(p, "uEye"), eye); gl.uniform3f(this.uniform(p, "uLight"), 0, 0, 0); gl.uniform3fv(this.uniform(p, "uAmbient"), SKY.ambient);
    this.fogUniforms(p, { fog: this.fogDensity(camera), fogColor: SKY.fog });
    gl.drawArraysInstanced(gl.TRIANGLES, 0, this.scene.unitSphere.normals.length / 3, instances.count);
    this.lastInstanceCount = instances.count;
  }
  /** Lit tube for one ship up to the archived sample `visible - 1` (never past an archived node). */
  tube(mvp, eye, camera, index, visible, colour, radius, options) {
    const ship = this.scene.ships[index], buffers = this.shipBuffers[index].tube;
    const segments = Math.min(visible, ship.points.length) - 1;
    if (segments < 1) return;
    const gl = this.gl, p = this.tubeProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", buffers.positions, 3); this.attribute(p, "aNormal", buffers.normals, 3); this.attribute(p, "aTime", buffers.times, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indices);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform1f(this.uniform(p, "uZScale"), this.zScale);
    gl.uniform1f(this.uniform(p, "uRadius"), radius); gl.uniform4fv(this.uniform(p, "uColor"), colour);
    gl.uniform3fv(this.uniform(p, "uEye"), eye); gl.uniform3f(this.uniform(p, "uLight"), 0, 0, 0); gl.uniform3fv(this.uniform(p, "uAmbient"), SKY.ambient);
    gl.uniform1f(this.uniform(p, "uEpoch"), options.epoch); gl.uniform1f(this.uniform(p, "uTrail"), options.trail ?? TRAIL_DAYS);
    gl.uniform1f(this.uniform(p, "uBaseAlpha"), options.baseAlpha ?? 0.4); gl.uniform1f(this.uniform(p, "uGlow"), options.glow ?? 0.35);
    this.fogUniforms(p, { fog: this.fogDensity(camera), fogColor: SKY.fog });
    gl.drawElements(gl.TRIANGLES, segments * ship.tube.indicesPerSegment, gl.UNSIGNED_SHORT, 0);
  }
  stars(mvp, alpha) {
    const gl = this.gl, p = this.starProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", this.starPositions, 3); this.attribute(p, "aMagnitude", this.starMagnitudes, 1);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform1f(this.uniform(p, "uAlpha"), alpha);
    gl.uniform1f(this.uniform(p, "uScale"), Math.min(devicePixelRatio || 1, 2));
    gl.drawArrays(gl.POINTS, 0, this.scene.stars.magnitudes.length);
  }
  additive(on) { const gl = this.gl; gl.blendFunc(gl.SRC_ALPHA, on ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA); }
  /** Body and tube radii in scene units, tied to camera distance so geometry keeps a legible on-screen size (not to scale). */
  radii(camera) {
    const d = camera.distance;
    return { sun: Math.max(0.02, 0.0075 * d), earth: 0.0042 * d, asteroid: 0.0028 * d, ship: 0.0042 * d, tube: 0.0022 * d };
  }
  draw(view, camera) {
    const gl = this.gl, scene = this.scene; this.resize();
    this.zScale = this.exaggeration;
    this.updateEphemeris(view.epoch);
    const mvp = this.matrix(camera), eye = orbitEye(camera), radii = this.radii(camera), fog = { fog: this.fogDensity(camera), fogColor: SKY.fog };
    const instances = bodyInstances(scene, view, radii, { earth: this.earthPosition, asteroid: (index) => this.asteroidPosition(index) });
    const { selected, hoverShip, flashes } = instances;
    gl.clearColor(SKY.fog[0], SKY.fog[1], SKY.fog[2], 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.BLEND); this.additive(false);
    // Sky and stars: no depth test or writes, they sit at infinity.
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false);
    this.background();
    this.stars(this.starMatrix(camera), 0.9);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    // Reference plane: depth-tested but not written so nothing is occluded by translucent helpers.
    this.line(mvp, this.discBuffer, gl.TRIANGLES, scene.disc.length / 3, [0.16, 0.26, 0.46, 0.14]);
    this.line(mvp, this.spokesBuffer, gl.LINES, scene.spokes.length / 3, [0.36, 0.5, 0.72, 0.13]);
    this.line(mvp, this.ringsBuffer, gl.LINES, scene.rings.length / 3, [0.55, 0.68, 0.86, 0.26]);
    this.line(mvp, this.axesBuffer, gl.LINES, 6, [0.72, 0.8, 0.92, 0.45]);
    const highlightedAsteroids = new Set(selected ? selected.record.asteroids : []);
    if (view.highlightAsteroid != null) highlightedAsteroids.add(view.highlightAsteroid);
    const orbitAlpha = selected ? 0.07 : 0.16;
    this.ribbon(mvp, this.orbitRibbon, scene.orbitRibbons.vertexCount, [0.66, 0.74, 0.9, orbitAlpha], 0.9, { shade: 0, ...fog });
    scene.asteroids.forEach((asteroid, index) => {
      if (!highlightedAsteroids.has(asteroid.id)) return;
      const colour = selected ? [selected.colour[0], selected.colour[1], selected.colour[2], 0.5] : [0.8, 0.86, 1, 0.55];
      this.ribbon(mvp, this.orbitRibbon, scene.orbitRibbons.counts[index], colour, 1.3, { first: scene.orbitRibbons.offsets[index], shade: 0, ...fog });
    });
    this.ribbon(mvp, this.earthRibbon, scene.earthRibbon.vertexCount, [EARTH_COLOUR[0], EARTH_COLOUR[1], EARTH_COLOUR[2], 0.65], 1.4, { shade: 0, ...fog });
    // Lit geometry writes depth so arcs pass behind bodies and each other correctly.
    gl.depthMask(true);
    this.bodies(mvp, eye, camera, instances);
    scene.ships.forEach((ship, index) => {
      const visible = countAtOrBefore(ship.times, view.epoch);
      if (visible < 2) return;
      const dim = selected && selected !== ship, hot = hoverShip === ship.index || selected === ship;
      const colour = [ship.colour[0], ship.colour[1], ship.colour[2], dim ? 0.4 : 1];
      this.tube(mvp, eye, camera, index, visible, colour, radii.tube * (dim ? 0.6 : hot ? 1.6 : 1),
        { epoch: view.epoch, trail: TRAIL_DAYS, baseAlpha: dim ? 0.5 : 0.42, glow: hot ? 0.6 : 0.3 });
    });
    // Event markers on the archived event states and the ship "now" dots (no depth writes).
    gl.depthMask(false);
    scene.ships.forEach((ship, index) => {
      const visible = countAtOrBefore(ship.times, view.epoch);
      if (visible === 0) return;
      const dim = selected && selected !== ship;
      if (dim) return;
      const buffers = this.shipBuffers[index];
      for (const [role, data] of Object.entries(ship.roles)) {
        const count = countAtOrBefore(data.times, view.epoch);
        const style = ROLE_STYLE[role] ?? ROLE_STYLE.rendezvous;
        this.line(mvp, buffers.roles[role], gl.POINTS, count, style.colour ?? ship.colour, style.size, style.marker);
      }
      for (const event of ship.events) {
        const pulse = flashAt(event.epoch_mjd, view.epoch);
        if (pulse > 0) flashes.push({ position: event.scene, colour: (ROLE_STYLE[event.role] ?? ROLE_STYLE.rendezvous).colour ?? ship.colour, pulse, ring: true });
      }
      this.point(mvp, ship.points[visible - 1], [1, 1, 1, 0.9], 4, MARKER.disc);
    });
    // Additive halos: layered Sun corona, Earth glow, ship glows, event/asteroid flashes.
    this.additive(true);
    const sunPixels = radii.sun / (2 * camera.distance * Math.tan(FIELD_OF_VIEW / 2)) * this.canvas.height;
    this.point(mvp, [0, 0, 0], [1, 0.84, 0.5, 0.7], clamp(sunPixels * 7, 64, 240), MARKER.glow);
    this.point(mvp, [0, 0, 0], [1, 0.7, 0.35, 0.3], clamp(sunPixels * 18, 140, 420), MARKER.glow);
    this.point(mvp, this.earthPosition, [EARTH_COLOUR[0], EARTH_COLOUR[1], EARTH_COLOUR[2], 0.35], 18, MARKER.glow);
    scene.ships.forEach((ship) => {
      const visible = countAtOrBefore(ship.times, view.epoch);
      if (visible === 0 || (selected && selected !== ship)) return;
      const hot = hoverShip === ship.index;
      this.point(mvp, ship.points[visible - 1], [ship.colour[0], ship.colour[1], ship.colour[2], hot ? 0.75 : 0.5], hot ? 36 : 24, MARKER.glow);
    });
    for (const flash of flashes) {
      this.point(mvp, flash.position, [flash.colour[0], flash.colour[1], flash.colour[2], 0.75 * flash.pulse], 18 + 40 * flash.pulse, MARKER.glow);
      if (flash.ring) this.point(mvp, flash.position, [1, 1, 1, 0.7 * flash.pulse], 12 + 30 * (1 - flash.pulse), MARKER.ring);
    }
    this.additive(false);
    const focus = view.hover ?? view.pinned;
    if (focus) this.point(mvp, focus.scene, [1, 1, 1, 0.9], 22, MARKER.ring);
    gl.depthMask(true);
  }
  /** Displayed (exaggerated) scene position -> CSS pixel position on the canvas (null when behind the camera). */
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
    const project = (p) => this.project(mvp, this.place(p));
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
    scene.asteroids.forEach((asteroid, index) => consider(this.asteroidPosition(index), { type: "asteroid", asteroid, index }, 1));
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
  const view = { epoch: MISSION_END_MJD, selected: null, hover: null, pinned: null, highlightAsteroid: null, follow: false };
  const currentPoint = (ship) => {
    const visible = countAtOrBefore(ship.times, view.epoch);
    return ship.points[Math.max(0, visible - 1)];
  };
  return {
    fleet, scene, renderer, view, camera, info: renderer.info,
    draw() { renderer.draw(view, camera); },
    setEpoch(mjd) { view.epoch = clamp(Math.round(mjd), MISSION_START_MJD, MISSION_END_MJD); },
    get exaggeration() { return renderer.exaggeration; },
    setExaggeration(factor) { renderer.exaggeration = clamp(Number(factor) || 1, EXAGGERATION.minimum, EXAGGERATION.maximum); return renderer.exaggeration; },
    place(point) { return renderer.place(point); },
    selectShip(index) {
      view.selected = index == null ? null : clamp(Number(index), 0, scene.ships.length - 1);
      view.highlightAsteroid = null; view.pinned = null;
      if (view.selected == null) view.follow = false;
    },
    /** Camera that frames the selected (or given) ship's whole arc from the current orientation; null without a ship. */
    focusCamera(index = view.selected ?? 0, base = camera) {
      const ship = scene.ships[index];
      if (!ship) return null;
      const { center, radius } = boundsOf(ship.points.map((point) => renderer.place(point)));
      const distance = clamp(fitDistance(Math.max(radius, 0.05), FIELD_OF_VIEW, renderer.aspect()), FLEET_ZOOM.minimum, FLEET_ZOOM.maximum);
      return { yaw: base.yaw, pitch: base.pitch, distance, target: center };
    },
    /** Displayed position of the selected ship at the current epoch (follow-ship target); null without a selection. */
    followTarget() {
      if (view.selected == null) return null;
      return renderer.place(currentPoint(scene.ships[view.selected]));
    },
    /** Camera looking at the selected ship from close range, keeping the current orientation. */
    followCamera(base = camera, distance = FOLLOW_DISTANCE) {
      const target = this.followTarget();
      return target ? followCamera(base, target, distance) : null;
    },
    /** CSS pixel position of an archived event marker (for tests and tooling). */
    eventScreenPosition(shipIndex, eventIndex) {
      const event = scene.ships[shipIndex]?.events[eventIndex];
      return event ? renderer.project(renderer.matrix(camera), renderer.place(event.scene)) : null;
    },
    shipScreenPosition(shipIndex) {
      const ship = scene.ships[shipIndex];
      return ship ? renderer.project(renderer.matrix(camera), renderer.place(currentPoint(ship))) : null;
    },
    earthScreenPosition() {
      renderer.updateEphemeris(view.epoch);
      return renderer.project(renderer.matrix(camera), renderer.place(renderer.earthPosition));
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
  const { scene, view } = fleetView;
  if (item.type === "event") {
    const ship = scene.ships[item.ship], event = item.event;
    return `<strong>Ship ${ship.id} · ${escapeHtml(ROLE_LABELS[event.role] ?? event.role)}</strong><span>${escapeHtml(event.body)} · ${epochLabel(event.epoch_mjd)}</span><span>Mass ${event.mass_before_kg.toFixed(1)} → ${event.mass_after_kg.toFixed(1)} kg (${kg(event.mass_delta_kg)})</span>`;
  }
  if (item.type === "ship" || item.type === "sample") {
    const ship = scene.ships[item.ship];
    const point = ship.record.replay.points_txyz[item.sample];
    const radius = Math.hypot(point[1], point[2], point[3]) / AU_KM;
    return `<strong>Ship ${ship.id}${item.type === "ship" ? " · current position" : ""}</strong><span>Archived sample ${item.sample + 1} / ${ship.record.replay.point_count} · ${epochLabel(point[0])}</span><span>r = ${radius.toFixed(3)} AU · z = ${(point[3] / AU_KM).toFixed(4)} AU · ${collectedAt(ship, view.epoch).toFixed(1)} of ${ship.record.collected_kg.toFixed(1)} kg collected so far</span>`;
  }
  if (item.type === "asteroid") {
    const asteroid = item.asteroid;
    const state = item.index == null ? null : asteroidState(scene.asteroidStatus[item.index], view.epoch);
    const visits = scene.ships.flatMap((ship) => ship.events.filter((event) => event.event_id === asteroid.id).map((event) => `ship ${ship.id} ${event.role} ${formatMjd(event.epoch_mjd)}`));
    return `<strong>Asteroid ${asteroid.id}${state ? ` · ${state}` : ""}</strong><span>a ${asteroid.a_au.toFixed(3)} AU · e ${asteroid.e.toFixed(3)} · i ${asteroid.i_deg.toFixed(2)}°</span><span>${escapeHtml(visits.join(" · ") || "not visited")}</span>`;
  }
  if (item.type === "earth") {
    const position = positionAt(scene.earth, item.epoch);
    return `<strong>Earth</strong><span>${epochLabel(item.epoch)} · r = ${(Math.hypot(...position) / AU_KM).toFixed(4)} AU</span><span>Keplerian position from GTOC12 Table 2 elements</span>`;
  }
  return "";
}

/**
 * Ship rows: colour swatch, name, running collected mass over the ship's total, a bar of that
 * fraction (filled by `renderShipCounters` as the epoch moves) and the asteroid count / launch date.
 */
export function renderShipList(container, fleetView) {
  const { fleet, view } = fleetView;
  const row = (index, colourClass, name, counterKey, totalKg, meta) => `
    <button type="button" class="ship-item ${view.selected === index ? "active" : ""}" data-ship="${index == null ? "all" : index}" aria-pressed="${view.selected === index}">
      <span class="ship-swatch ${colourClass}" aria-hidden="true"></span>
      <span class="ship-name">${name}</span>
      <span class="ship-mass"><span data-counter="${counterKey}">0.0</span><small> of ${totalKg} kg</small></span>
      <span class="mass-bar ${index == null ? "fleet-bar" : colourClass}" aria-hidden="true"><i data-bar="${counterKey}"></i></span>
      <span class="ship-meta">${meta}</span>
    </button>`;
  container.innerHTML = row(null, "fleet-swatch", "Whole fleet", "fleet", fleetMassLabel(fleet), `${fleet.ships.length} ships, ${fleet.asteroids.length} asteroids`)
    + fleet.ships.map((ship, index) => row(index, `ship-colour-${index % SHIP_COLOURS.length + 1}`, `Ship ${ship.ship_id}`, index, ship.collected_kg.toFixed(1),
      `${ship.asteroids.length} asteroids, launched ${formatMjd(ship.launch_epoch_mjd)}`)).join("");
  renderShipCounters(container, fleetView);
}

/** Running collected-mass counters and bars (per ship and fleet) at the current epoch, without re-rendering the list. */
export function renderShipCounters(container, fleetView) {
  const { fleet, scene, view } = fleetView;
  let total = 0;
  const paint = (key, collected, totalKg) => {
    const counter = container.querySelector(`[data-counter="${key}"]`);
    if (counter) counter.textContent = collected.toFixed(1);
    const bar = container.querySelector(`[data-bar="${key}"]`);
    if (bar) bar.style.transform = `scaleX(${totalKg > 0 ? clamp(collected / totalKg, 0, 1) : 0})`;
  };
  for (const ship of scene.ships) {
    const collected = collectedAt(ship, view.epoch); total += collected;
    paint(ship.index, collected, ship.record.collected_kg);
  }
  paint("fleet", total, fleet.score.total_collected_kg ?? fleet.score.official_total_mass_kg);
  return total;
}

/** Ship swatches in the on-canvas legend, one per ship in the loaded fleet. */
export function renderLegendShips(container, fleet) {
  container.innerHTML = fleet.ships.map((ship, index) => `<span><i class="ship-swatch ship-colour-${index % SHIP_COLOURS.length + 1}"></i>${ship.ship_id}</span>`).join("");
}

export function renderShipDetail(container, fleetView) {
  const { fleet, view } = fleetView;
  if (view.selected == null) {
    container.innerHTML = `<p class="help-text">Select a ship to list its deploy/collect sequence. Fleet totals below are read from the verified export.</p><dl class="metric-list">${metricRows([
      ["Ships", String(fleet.score.ships)], ["Unique asteroids", String(fleet.score.unique_asteroids)],
      ["Collected mass (archived events)", `${fleetMassLabel(fleet)} kg`], ["Mass per ship", `${massPerShip(fleet).toFixed(1)} kg`],
      ["Official verifier", `${fleet.source.official_verifier_ok === null ? "not recorded" : fleet.source.official_verifier_ok ? "pass" : "fail"} · ${fleet.score.official_total_mass_kg} kg`],
      ["Independent verifier", `${fleet.score.verifier_ok ? "pass" : "fail"} · ${fleet.score.independent_total_mass_kg?.toFixed(3)} kg`],
      ["Ship-count rule", `${fleet.score.ships} ≤ ${fleet.score.ship_limit?.toFixed(2)}`],
      ["Archived samples", `${fleet.ships.reduce((sum, ship) => sum + ship.replay.point_count, 0).toLocaleString()} (≤ 512 per ship)`],
      ["Mission window", `${formatMjd(fleet.constants.mission_start_mjd)} → ${formatMjd(fleet.constants.mission_end_mjd)}`],
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
  const { fleet, scene, view } = fleetView;
  const active = fleet.ships.filter((ship) => ship.launch_epoch_mjd <= view.epoch && ship.final_sample_epoch_mjd > view.epoch).length;
  const deployed = fleet.ships.reduce((sum, ship) => sum + ship.events.filter((event) => event.role === "deploy" && event.epoch_mjd <= view.epoch).length, 0);
  const collected = scene.ships.reduce((sum, ship) => sum + collectedAt(ship, view.epoch), 0);
  const delivered = fleet.ships.filter((ship) => ship.return_epoch_mjd <= view.epoch).reduce((sum, ship) => sum + ship.collected_kg, 0);
  const minedAsteroids = scene.asteroidStatus.filter((status) => status.collect <= view.epoch).length;
  container.innerHTML = metricRows([
    ["Epoch", `${epochLabel(view.epoch)} · T+${missionYears(view.epoch).toFixed(2)} yr`], ["Ships in flight", `${active} of ${fleet.ships.length}`],
    ["Miners deployed so far", String(deployed)], ["Asteroids collected so far", `${minedAsteroids} of ${fleet.asteroids.length}`],
    ["Mass collected so far", `${collected.toFixed(1)} of ${fleetMassLabel(fleet)} kg`],
    ["Delivered to Earth so far", `${delivered.toFixed(1)} kg`], ["Final official score", `${fleet.score.official_total_mass_kg} kg`],
  ]);
}

export function renderFleetProvenance(container, fleetView) {
  const { fleet } = fleetView;
  container.innerHTML = `<p>Run <code>${escapeHtml(fleet.run_id)}</code> · export commit <code>${escapeHtml(fleet.generated_by_commit)}</code> · ${escapeHtml(fleet.source.generator)}.</p>
    <p>Official solution file <code>${escapeHtml(fleet.source.solution_basename)}</code> SHA-256 <code>${escapeHtml(fleet.source.solution_sha256)}</code> (${fleet.source.solution_bytes.toLocaleString()} bytes; official verifier ${fleet.source.official_verifier_ok === null ? "not recorded" : fleet.source.official_verifier_ok ? "pass" : "fail"} at ${fleet.score.official_total_mass_kg} kg, independent verifier ${fleet.score.verifier_ok ? "pass" : "fail"} at ${fleet.score.independent_total_mass_kg?.toFixed(3)} kg, max propagation error ${fleet.verification.max_position_error_km?.toFixed(2)} km). The official verifier prints six significant digits; the headline ${fleetMassLabel(fleet)} kg is the sum of the archived collect events.</p>
    <p>Export <code>trajectories.json</code> SHA-256 <code>${escapeHtml(fleet.source.export_trajectories_sha256)}</code>; asteroid catalogue <code>${escapeHtml(fleet.source.catalogue.name)}</code> SHA-256 <code>${escapeHtml(fleet.source.catalogue.sha256)}</code>.</p>
    <p>Ship arcs are lit 3D tube meshes (${TUBE_SIDES}-sided, lit by the Sun with distance fog) whose straight segments connect ${fleet.ships.reduce((sum, ship) => sum + ship.replay.point_count, 0).toLocaleString()} exact archived propagated samples (≤ 512 per ship, every event epoch preserved) — connections between archived nodes, not interpolation; the bright trail marks the last ${TRAIL_DAYS} days of each arc. Earth and asteroid orbits and their epoch positions are two-body Keplerian curves from the pinned GTOC12 elements (Appendix 6.1); the viewer's propagation agrees with the exporter's context orbits to ${fleet.kepler_check.asteroid_max_error_km.toExponential(1)} km over ${fleet.kepler_check.context_points_checked.toLocaleString()} samples. Sun, Earth, asteroid and ship spheres are instanced lit display markers sized for legibility, not to scale; tube and sphere radii scale with camera distance. The vertical-exaggeration slider scales Z only for display and is not physical.</p>`;
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
    const screen = event.epoch_mjd <= view.epoch ? renderer.project(mvp, renderer.place(event.scene)) : null;
    const visible = screen && screen[0] >= 0 && screen[1] >= 0 && screen[0] <= width && screen[1] <= height;
    label.hidden = !visible;
    if (visible) { label.style.left = `${screen[0]}px`; label.style.top = `${screen[1]}px`; }
  });
}
