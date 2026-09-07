import { clamp, lookAt, multiply, normalizePoint, orbitEye, perspective } from "./math.js";
import { $, escapeHtml, format, metricRows } from "./dom.js";
import { GlResources, flatten, hex, planeGrid, ribbonArrays, sphere } from "./webgl.js";
import { MISSION_END_MJD, MISSION_START_MJD } from "./kepler.js";
import {
  CAMERA_PRESETS, EXAGGERATION, PITCH_LIMIT, applyVelocity, cameraEquals, clampCamera, createTransition, cursorPointOnFocalPlane,
  decayVelocity, dollyTowards, inertiaActive, presetCamera, rescaleTarget, sampleTransition,
} from "./camera.js";
import {
  FIELD_OF_VIEW, FLEET_CAMERA, FLEET_ZOOM, createFleetView, describeItem, fleetMassLabel, renderEventLabels, renderFleetProvenance,
  renderFleetSummary, renderLegendShips, renderShipCounters, renderShipDetail, renderShipList, renderTimelineOutput,
} from "./gtoc12.js";

const requiredIds = [
  "renderer-status", "renderer-status-text", "error-banner", "inventory-count",
  "trajectory-list", "data-mode", "mode-description", "play-button", "play-icon",
  "play-label", "reset-button", "timeline", "timeline-output", "sample-output",
  "trajectory-canvas", "family-label", "trajectory-title", "qualification-badge",
  "qualification-notice", "frame-overlay", "scene-overlay", "current-state",
  "frame-details", "validation-details", "gpu-details", "compute-details", "solver-progress", "provenance-content",
  "dataset-select", "dataset-help", "ship-list", "fleet-count", "mission-timeline",
  "mission-timeline-output", "mission-play-button", "mission-play-icon", "mission-play-label",
  "focus-ship-button", "fleet-reset-button", "fleet-summary", "ship-detail", "ship-detail-title",
  "fleet-legend", "hover-tooltip", "fleet-provenance-content", "legend-overlay", "event-labels",
  "speed-select", "camera-presets", "follow-ship-button", "exaggeration", "exaggeration-output", "legend-ships",
  "timeline-ticks", "mission-timeline-ticks",
];
for (const id of requiredIds) if (!$(id)) throw new Error(`Required DOM element #${id} is missing`);

/** Paint the elapsed fraction of a range input into `--fill` (the track is a strip chart, not a bare slider). */
function paintRange(input) {
  const min = Number(input.min), max = Number(input.max), value = Number(input.value);
  const fraction = max > min ? clamp((value - min) / (max - min), 0, 1) : 0;
  input.style.setProperty("--fill", `${(fraction * 100).toFixed(2)}%`);
}
/** Year ticks under the mission timeline: 1 January of each mission year at its MJD position. */
function renderMissionTicks(container) {
  const span = MISSION_END_MJD - MISSION_START_MJD;
  const ticks = [];
  for (let year = 2035; year <= 2050; year += 1) {
    const mjd = Date.UTC(year, 0, 1) / 86_400_000 + 40_587;
    if (mjd < MISSION_START_MJD || mjd > MISSION_END_MJD) continue;
    const tick = document.createElement("span");
    tick.textContent = String(year);
    if (year % 5 === 0) tick.classList.add("major");
    tick.style.left = `${((mjd - MISSION_START_MJD) / span * 100).toFixed(3)}%`;
    ticks.push(tick);
  }
  container.replaceChildren(...ticks);
}
renderMissionTicks($("mission-timeline-ticks"));
paintRange($("exaggeration"));

const canvas = $("trajectory-canvas");
const ARCHIVE_CAMERA = { yaw: -0.72, pitch: 0.48, distance: 3.25, target: [0, 0, 0] };
const ARCHIVE_ZOOM = { minimum: 1.35, maximum: 12 };
const FLEET_DATASETS = {
  gtoc12: { directory: "./data/gtoc12", label: "Incumbent GTOC12 fleet" },
  "gtoc12-v200": { directory: "./data/gtoc12-v200", label: "GPU campaign v200 (before fix)" },
  "gtoc12-v209": { directory: "./data/gtoc12-v209", label: "GPU campaign v209 (corrected solver)" },
  "gtoc12-v213": { directory: "./data/gtoc12-v213", label: "GPU retiming v213 (524 kg certified)" },
};
const fleetCache = new Map(), availableFleets = new Set();
let datasetRequest = 0;
const state = {
  dataset: "archive", data: null, fleet: null, fleetAvailable: false, fleetView: null,
  selected: 0, mode: "replay", progress: 100, playing: false, speedYearsPerSecond: 1,
  frame: 0, lastTime: 0, renderer: null,
  camera: { yaw: -0.72, pitch: 0.48, distance: 3.25, target: [0, 0, 0] },
  transition: null, velocity: { yaw: 0, pitch: 0 }, preset: null,
  pointer: null, contextLost: false,
};

function geometry(trajectory, mode) {
  const points = trajectory[mode].points_txyz;
  const spatial = points.map((point) => point.slice(1, 4));
  spatial.push(trajectory.terminal_target.slice(0, 3));
  const minima = [0, 1, 2].map((axis) => Math.min(...spatial.map((point) => point[axis])));
  const maxima = [0, 1, 2].map((axis) => Math.max(...spatial.map((point) => point[axis])));
  const central = trajectory.viewer.scene_kind === "central-body";
  const center = central ? [0, 0, 0] : minima.map((value, axis) => (value + maxima[axis]) / 2);
  const scale = Math.max(trajectory.viewer.body_radius || 0,
    ...spatial.map((point) => Math.hypot(...point.map((value, axis) => value - center[axis]))), 1);
  const pathPoints = spatial.slice(0, -1).map((point) => normalizePoint(point, center, scale));
  const path = flatten(pathPoints);
  const target = new Float32Array(normalizePoint(trajectory.terminal_target.slice(0, 3), center, scale));
  const origin = normalizePoint([0, 0, 0], center, scale);
  const axisLength = central ? 1.25 : 0.55;
  const axes = flatten([
    origin, [origin[0] + axisLength, origin[1], origin[2]],
    origin, [origin[0] - axisLength, origin[1], origin[2]],
    origin, [origin[0], origin[1] + axisLength, origin[2]],
    origin, [origin[0], origin[1], origin[2] + axisLength],
  ]);
  let surface, normals, grid, surfaceZ = null;
  if (central) {
    const radius = trajectory.viewer.body_radius / scale;
    ({ positions: surface, normals } = sphere(radius, origin));
    grid = new Float32Array([...planeGrid(origin, Math.max(radius * 1.45, 1.05), 0), ...sphereGrid(radius * 1.002, origin)]);
  } else {
    const z = (0 - center[2]) / scale;
    const planeCenter = [0, 0, z];
    surface = plane(planeCenter, 1.35, z);
    normals = new Float32Array(18);
    for (let i = 2; i < normals.length; i += 3) normals[i] = 1;
    grid = planeGrid(planeCenter, 1.35, z + 0.001);
    if (trajectory.viewer.scene_kind === "local-surface") surfaceZ = z;
  }
  const ribbon = ribbonArrays(pathPoints);
  return {
    center, scale, points, path, target, axes, surface, normals, grid, surfaceZ,
    start: path.slice(0, 3),
    ribbonPositions: ribbon.positions, ribbonPrevious: ribbon.previous, ribbonNext: ribbon.next, ribbonSides: ribbon.sides,
  };
}
function plane(center, extent, z) {
  return flatten([
    [center[0] - extent, center[1] - extent, z], [center[0] + extent, center[1] - extent, z],
    [center[0] + extent, center[1] + extent, z], [center[0] - extent, center[1] - extent, z],
    [center[0] + extent, center[1] + extent, z], [center[0] - extent, center[1] + extent, z],
  ]);
}
function sphereGrid(radius, center) {
  const result = [], segments = 72;
  const circle = (point) => {
    for (let i = 0; i < segments; i += 1) {
      result.push(point(i / segments * Math.PI * 2), point((i + 1) / segments * Math.PI * 2));
    }
  };
  for (let latitude = -60; latitude <= 60; latitude += 30) {
    const phi = latitude * Math.PI / 180;
    circle((theta) => [center[0] + radius * Math.cos(phi) * Math.cos(theta), center[1] + radius * Math.cos(phi) * Math.sin(theta), center[2] + radius * Math.sin(phi)]);
  }
  for (let longitude = 0; longitude < 180; longitude += 30) {
    const theta = longitude * Math.PI / 180;
    circle((phi) => [center[0] + radius * Math.cos(phi) * Math.cos(theta), center[1] + radius * Math.cos(phi) * Math.sin(theta), center[2] + radius * Math.sin(phi)]);
  }
  return flatten(result);
}

/** Archive renderer: one trajectory in its own physical frame. */
class Renderer extends GlResources {
  constructor(canvasElement, scene) {
    super(canvasElement);
    this.scene = scene;
    const gl = this.gl;
    const arrays = ["path", "target", "axes", "surface", "normals", "grid", "start", "ribbonPositions", "ribbonPrevious", "ribbonNext", "ribbonSides"];
    for (const name of arrays) this[`${name}Buffer`] = this.makeBuffer(scene[name]);
    this.currentBuffer = this.makeBuffer(new Float32Array(3), gl.DYNAMIC_DRAW);
    this.stemBuffer = this.makeBuffer(new Float32Array(6), gl.DYNAMIC_DRAW);
  }
  draw(visibleCount, camera) {
    const gl = this.gl, scene = this.scene; this.resize();
    const mvp = multiply(perspective(Math.PI / 4, this.canvas.width / this.canvas.height, 0.025, 80), lookAt(orbitEye(camera), camera.target, [0, 0, 1]));
    gl.clearColor(0.018, 0.027, 0.055, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA); gl.depthMask(true);
    this.surface(mvp, this.surfaceBuffer, this.normalsBuffer, scene.surface.length / 3, [0.10, 0.20, 0.34, scene.surfaceZ == null ? .96 : .48]);
    this.line(mvp, this.gridBuffer, gl.LINES, scene.grid.length / 3, [0.30, 0.46, 0.66, .32]);
    this.line(mvp, this.axesBuffer, gl.LINES, scene.axes.length / 3, [0.72, 0.8, 0.92, .72]);
    if (visibleCount > 1) {
      const buffers = { positions: this.ribbonPositionsBuffer, previous: this.ribbonPreviousBuffer, next: this.ribbonNextBuffer, sides: this.ribbonSidesBuffer };
      this.ribbon(mvp, buffers, visibleCount * 2, hex("#36d6ff"), 3.0, { shade: 0 });
      this.line(mvp, this.pathBuffer, gl.LINE_STRIP, visibleCount, [0.35, .9, 1, .95]);
    }
    if (scene.surfaceZ != null) {
      const offset = (visibleCount - 1) * 3, point = scene.path.slice(offset, offset + 3);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.stemBuffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...point, point[0], point[1], scene.surfaceZ]), gl.DYNAMIC_DRAW);
      this.line(mvp, this.stemBuffer, gl.LINES, 2, [.85, .9, 1, .65]);
    }
    this.line(mvp, this.pathBuffer, gl.POINTS, visibleCount, [0.22, .84, 1, .45], 4, 1);
    this.line(mvp, this.startBuffer, gl.POINTS, 1, [0.27, 1, .61, 1], 13, 2);
    this.line(mvp, this.targetBuffer, gl.POINTS, 1, [1, .73, .31, 1], 16, 2);
    const offset = (visibleCount - 1) * 3;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.currentBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, scene.path.slice(offset, offset + 3));
    this.line(mvp, this.currentBuffer, gl.POINTS, 1, [1, 1, 1, 1], 11, 1);
  }
}

const isFleet = () => Object.hasOwn(FLEET_DATASETS, state.dataset);
function trajectory() { return state.data.trajectories[state.selected]; }
function visibleCount() {
  const count = trajectory()[state.mode].point_count;
  return clamp(Math.ceil(count * state.progress / 100), 1, count);
}
function zoomBounds() { return isFleet() ? FLEET_ZOOM : ARCHIVE_ZOOM; }
function pitchLimit() { return isFleet() ? PITCH_LIMIT : 1.45; }
function setStatus(kind, text) {
  $("renderer-status").className = `renderer-status ${kind}`; $("renderer-status-text").textContent = text;
}
function renderGpuDetails(info) {
  $("gpu-details").innerHTML = metricRows([
    ["Status", "Active WebGL2"], ["Version", info.version],
    ["Renderer", info.renderer], ["Vendor", info.vendor],
    ["Max texture", `${info.maxTextureSize.toLocaleString()} px`],
    ["Vertex attributes", String(info.maxVertexAttribs)],
  ]);
}
function disposeRenderers() {
  state.renderer?.dispose(); state.renderer = null;
  state.fleetView?.dispose(); state.fleetView = null;
}
function createRenderer() {
  disposeRenderers();
  if (isFleet()) {
    state.fleetView = createFleetView({ canvas, fleet: state.fleet, camera: state.camera });
    state.fleetView.setEpoch(Number($("mission-timeline").value));
    state.fleetView.setExaggeration(Number($("exaggeration").value));
    renderGpuDetails(state.fleetView.info);
  } else {
    const scene = geometry(trajectory(), state.mode); state.scene = scene;
    state.renderer = new Renderer(canvas, scene);
    renderGpuDetails(state.renderer.info);
  }
  setStatus("ready", "WebGL2 GPU renderer");
}
function renderInventory() {
  $("inventory-count").textContent = String(state.data.trajectories.length);
  $("trajectory-list").innerHTML = state.data.trajectories.map((item, index) => `
    <button type="button" class="trajectory-item ${index === state.selected ? "active" : ""}" data-index="${index}" aria-pressed="${index === state.selected}">
      <span><strong>${escapeHtml(item.family)}</strong><small>${escapeHtml(item.physical_family)}</small></span>
      <i class="${item.qualification.qualified ? "qualified" : "warning"}">${item.qualification.qualified ? "Qualified" : "Diagnostic"}</i>
    </button>`).join("");
}
function updateDetails() {
  const item = trajectory(), source = item[state.mode], index = visibleCount() - 1, point = source.points_txyz[index];
  const radius = Math.hypot(point[1], point[2], point[3]);
  let vertical = `HCW radial X: ${format(point[1], item.position_units)}`;
  if (item.viewer.scene_kind === "local-surface") vertical = `Altitude: ${format(point[3], item.position_units)}`;
  if (item.viewer.scene_kind === "central-body") vertical = `${item.family === "P1-E" ? "Clearance above r_min" : "Altitude"}: ${format(radius - item.viewer.body_radius, item.position_units)}`;
  $("family-label").textContent = item.family; $("trajectory-title").textContent = item.physical_family;
  $("qualification-badge").className = `qualification-badge ${item.qualification.qualified ? "qualified" : "warning"}`;
  $("qualification-badge").textContent = item.qualification.qualified ? "Qualified scope" : "Diagnostic only";
  $("qualification-notice").className = `notice-panel ${item.qualification.qualified ? "qualified" : "warning"}`;
  $("qualification-notice").innerHTML = `<strong>${item.qualification.qualified ? "Qualified evidence scope" : "Unqualified trajectory warning"}</strong><p>${escapeHtml(item.qualification.label)}. Qualification applies only to the stated archived evidence scope.</p>`;
  $("frame-overlay").innerHTML = `<strong>${escapeHtml(item.frame)}</strong><span>${escapeHtml(item.viewer.axes.join(", "))}</span>`;
  $("scene-overlay").textContent = item.viewer.scene_kind === "hcw" ? "LVLH plane, Earthward −X" : item.viewer.radius_label;
  $("timeline-output").textContent = `${state.progress.toFixed(1)}%`;
  paintRange($("timeline"));
  $("sample-output").textContent = `Sample ${index + 1} / ${source.point_count} at ${format(point[0], item.time_units)}`;
  $("current-state").innerHTML = metricRows([
    ["Replay time", format(point[0], item.time_units)], ["Position", `[${point.slice(1).map((v) => format(v)).join(", ")}] ${item.position_units}`],
    ["Physical measure", vertical], ["Camera scale", format(state.camera.distance * state.scene.scale, item.position_units)],
  ]);
  $("frame-details").innerHTML = `<p>${escapeHtml(item.viewer.frame_choice)}</p><dl class="metric-list">${metricRows([["Body / surface", item.viewer.body_label], ["Radius rule", item.viewer.radius_label], ["Gravity", item.viewer.gravity_label]])}</dl>`;
  const terminal = item.validation.dense_replay_terminal_inf ?? item.validation.terminal_position_inf;
  const path = item.validation.dense_replay_physical_path_inf ?? item.validation.path_inf;
  $("validation-details").innerHTML = metricRows([
    ["Finite archive arrays", item.validation.finite ? "Passed" : "Failed"], ["Replay terminal ∞", format(terminal, item.position_units)],
    ["Path violation ∞", format(path, item.position_units)], ["Original replay", `${item.replay.original_point_count.toLocaleString()} points`],
  ]);
  $("provenance-content").innerHTML = `<p>Run <code>${escapeHtml(item.source.run_id)}</code> · commit <code>${escapeHtml(item.source.commit)}</code>.</p>
    <p>Raw evidence SHA-256: <code>${escapeHtml(item.raw_evidence_sha256)}</code></p>
    <p>Archive source SHA-256: <code>${escapeHtml(state.data.imported_source_sha256)}</code></p>
    <p>${source.point_count.toLocaleString()} exact selected archive points from ${source.original_point_count.toLocaleString()} original points. Selected indices and point hashes are retained. No visual interpolation is used.</p>`;
}

/** GTOC12 panels that depend on the selection (ship list, detail table, headings). */
function updateFleetSelection() {
  const fleetView = state.fleetView, fleet = state.fleet, selected = fleetView.view.selected;
  renderShipList($("ship-list"), fleetView);
  renderShipDetail($("ship-detail"), fleetView);
  renderLegendShips($("legend-ships"), fleet);
  $("fleet-count").textContent = String(fleet.ships.length);
  $("family-label").textContent = `GTOC12 run ${fleet.run_id}`;
  $("trajectory-title").textContent = selected == null
    ? `${fleet.title} — ${fleet.score.ships} ships, ${fleet.score.unique_asteroids} asteroids, ${fleetMassLabel(fleet)} kg`
    : `Ship ${fleet.ships[selected].ship_id} — ${fleet.ships[selected].collected_kg.toFixed(1)} kg from ${fleet.ships[selected].asteroids.length} asteroids`;
  $("ship-detail-title").textContent = selected == null ? "Fleet totals" : `Ship ${fleet.ships[selected].ship_id} event sequence`;
  $("focus-ship-button").disabled = selected == null;
  $("follow-ship-button").disabled = selected == null;
  const qualified = fleet.score.verifier_ok;
  $("qualification-badge").className = `qualification-badge ${qualified ? "qualified" : "warning"}`;
  $("qualification-badge").textContent = qualified ? "Verified fleet" : "Unverified";
  $("qualification-notice").className = `notice-panel ${qualified ? "qualified" : "warning"}`;
  $("qualification-notice").innerHTML = `<strong>${qualified ? "Verified GTOC12 fleet" : "Unverified GTOC12 fleet"}</strong><p>${escapeHtml(fleet.ships[0].qualification.label)}. ${fleetMassLabel(fleet)} kg collected by ${fleet.score.ships} ships from ${fleet.score.unique_asteroids} asteroids (official verifier ${fleet.score.official_total_mass_kg} kg; independent verifier ${fleet.score.independent_total_mass_kg?.toFixed(3)} kg; ship-count limit ${fleet.score.ship_limit?.toFixed(2)}). Arcs connect exact archived samples; Earth/asteroid orbits are Keplerian from the pinned catalogue.</p>`;
  $("frame-overlay").innerHTML = `<strong>${escapeHtml(fleet.frame)}</strong><span>Sun-centred; 1 scene unit = ${fleetView.scene.scale.toFixed(2)} AU; rings every 1 AU</span>`;
  updatePresetButtons();
  renderFleetProvenance($("fleet-provenance-content"), fleetView);
}
function updateFleetEpoch() {
  const fleetView = state.fleetView;
  $("mission-timeline").value = String(fleetView.view.epoch);
  paintRange($("mission-timeline"));
  renderTimelineOutput($("mission-timeline-output"), fleetView);
  renderFleetSummary($("fleet-summary"), fleetView);
  renderShipCounters($("ship-list"), fleetView);
  const passed = fleetView.view.selected == null ? -1 : state.fleet.ships[fleetView.view.selected].events.filter((event) => event.epoch_mjd <= fleetView.view.epoch).length;
  if (passed !== state.lastPassed) { state.lastPassed = passed; if (fleetView.view.selected != null) renderShipDetail($("ship-detail"), fleetView); }
}
function updateSceneOverlay() {
  const factor = state.fleetView?.exaggeration ?? 1;
  $("exaggeration-output").textContent = `${factor % 1 === 0 ? factor.toFixed(0) : factor.toFixed(1)}×`;
  paintRange($("exaggeration"));
  $("scene-overlay").textContent = factor > 1
    ? `Straight segments = connections between archived samples; Z exaggerated ${factor}× (not physical)`
    : "Straight segments = connections between archived samples";
}
function updatePresetButtons() {
  for (const button of $("camera-presets").querySelectorAll("[data-preset]")) {
    const active = button.dataset.preset === state.preset;
    button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active));
  }
}
function draw() {
  if (state.contextLost) return;
  if (isFleet()) {
    if (!state.fleetView) return;
    if (state.fleetView.view.follow) {
      const target = state.fleetView.followTarget();
      if (target && !state.transition) state.camera.target = target;
    }
    state.fleetView.draw(); renderEventLabels($("event-labels"), state.fleetView); updateFleetEpoch();
  } else {
    if (!state.renderer) return;
    state.renderer.draw(visibleCount(), state.camera); updateDetails();
  }
}

/* Single animation loop: camera transitions, pointer inertia and timeline playback. */
function needsFrames() { return state.playing || state.transition != null || inertiaActive(state.velocity); }
function requestFrame() { if (!state.frame) state.frame = requestAnimationFrame(tick); }
function tick(time) {
  state.frame = 0;
  const previous = state.lastTime || time; state.lastTime = time;
  const dt = Math.min(100, time - previous);
  if (state.transition) {
    const { camera, done } = sampleTransition(state.transition, time);
    Object.assign(state.camera, camera);
    if (done) state.transition = null;
  } else if (inertiaActive(state.velocity)) {
    Object.assign(state.camera, applyVelocity(state.camera, state.velocity, dt, pitchLimit()));
    state.velocity = decayVelocity(state.velocity, dt);
    if (!inertiaActive(state.velocity)) state.velocity = { yaw: 0, pitch: 0 };
  }
  if (state.playing) {
    if (isFleet()) {
      // Mission clock: `speedYearsPerSecond` mission years per wall-clock second; every frame lands on archived samples only.
      const epoch = state.fleetView.view.epoch + dt / 1000 * state.speedYearsPerSecond * 365.25;
      state.fleetView.setEpoch(Math.min(epoch, MISSION_END_MJD));
      if (state.fleetView.view.epoch >= MISSION_END_MJD) stopPlayback();
    } else {
      state.progress = Math.min(100, state.progress + dt / 120);
      $("timeline").value = String(state.progress);
      if (state.progress >= 100) stopPlayback();
    }
  }
  draw();
  if (needsFrames()) requestFrame(); else state.lastTime = 0;
}
function stopPlayback() {
  state.playing = false;
  $("play-icon").textContent = "▶"; $("play-label").textContent = "Play replay";
  $("mission-play-icon").textContent = "▶"; $("mission-play-label").textContent = "Play mission";
}
function togglePlayback() {
  if (state.playing) return stopPlayback();
  if (isFleet()) {
    if (state.fleetView.view.epoch >= MISSION_END_MJD) state.fleetView.setEpoch(MISSION_START_MJD);
    $("mission-play-icon").textContent = "❚❚"; $("mission-play-label").textContent = "Pause mission";
  } else {
    if (state.progress >= 100) state.progress = 0;
    $("play-icon").textContent = "❚❚"; $("play-label").textContent = "Pause replay";
  }
  state.playing = true; requestFrame();
}
/** Smoothly move the camera to `target` (instantly when the browser prefers reduced motion). */
function transitionTo(target, duration) {
  const next = clampCamera(target, zoomBounds(), pitchLimit());
  state.velocity = { yaw: 0, pitch: 0 };
  if (cameraEquals(state.camera, next, 1e-6) || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    state.transition = null; Object.assign(state.camera, next); draw(); return;
  }
  state.transition = createTransition(state.camera, next, performance.now(), duration);
  requestFrame();
}
function stopCameraMotion() { state.transition = null; state.velocity = { yaw: 0, pitch: 0 }; }
function setPreset(name) {
  if (!isFleet() || !state.fleetView) return;
  const fleetView = state.fleetView;
  if (name === "follow") {
    const camera = fleetView.followCamera(state.camera);
    if (!camera) return;
    fleetView.view.follow = true; state.preset = "follow";
    transitionTo(camera);
  } else {
    fleetView.view.follow = false; state.preset = name;
    transitionTo(presetCamera(name, state.camera));
  }
  updatePresetButtons();
}
function leaveFollow() { if (state.fleetView?.view.follow) { state.fleetView.view.follow = false; state.preset = null; updatePresetButtons(); } }
function resetCamera() {
  stopCameraMotion();
  const defaults = isFleet() ? FLEET_CAMERA : ARCHIVE_CAMERA;
  if (isFleet()) { leaveFollow(); state.preset = "oblique"; updatePresetButtons(); }
  transitionTo({ ...defaults, target: [...defaults.target] });
}
function focusShip() {
  const camera = state.fleetView?.focusCamera();
  if (!camera) return;
  leaveFollow(); transitionTo(camera);
}
function setExaggeration(value) {
  if (!state.fleetView) return;
  const previous = state.fleetView.exaggeration;
  const next = state.fleetView.setExaggeration(value);
  $("exaggeration").value = String(next);
  Object.assign(state.camera, rescaleTarget(state.camera, previous, next));
  if (state.transition) state.transition = createTransition(state.camera, rescaleTarget(state.transition.to, previous, next), performance.now(), 1);
  updateSceneOverlay(); draw();
}
function fatal(error) {
  const message = error instanceof Error ? error.message : String(error);
  $("error-banner").hidden = false; $("error-banner").textContent = message; setStatus("error", "Viewer unavailable");
  console.error(error);
}

/** Show the panels for the active dataset. */
function applyDatasetVisibility() {
  const fleet = isFleet();
  for (const element of document.querySelectorAll(".archive-only")) element.hidden = fleet;
  for (const element of document.querySelectorAll(".fleet-only")) element.hidden = !fleet;
  if (!fleet) { $("event-labels").replaceChildren(); $("event-labels").dataset.ship = ""; }
  hideTooltip();
}
async function loadFleet(dataset) {
  if (fleetCache.has(dataset)) return fleetCache.get(dataset);
  const response = await fetch(`${FLEET_DATASETS[dataset].directory}/fleet.json`, { cache: "no-store" });
  if (!response.ok) throw new Error(`GTOC12 fleet dataset request failed: HTTP ${response.status}`);
  const fleet = await response.json();
  if (fleet.dataset_kind !== "gtoc12-fleet" || !Array.isArray(fleet.ships) || fleet.ships.length === 0) throw new Error("GTOC12 fleet dataset has an unexpected shape");
  fleetCache.set(dataset, fleet);
  return fleet;
}
async function setDataset(dataset, options = {}) {
  const request = ++datasetRequest;
  stopPlayback(); stopCameraMotion();
  if (dataset !== "archive" && !availableFleets.has(dataset)) {
    $("dataset-select").value = state.dataset;
    $("dataset-help").textContent = "GTOC12 dataset not installed. Run `npm run import-gtoc12 -- --export <export dir> --catalogue <GTOC12_Asteroids_Data.txt>` (see README) and reload.";
    return;
  }
  try {
    const fleet = dataset === "archive" ? null : await loadFleet(dataset);
    if (request !== datasetRequest) return;
    state.fleet = fleet;
    state.dataset = dataset;
    $("dataset-select").value = dataset;
    applyDatasetVisibility();
    if (isFleet()) {
      const preset = options.preset && CAMERA_PRESETS[options.preset] ? options.preset : "oblique";
      Object.assign(state.camera, presetCamera(preset, { ...FLEET_CAMERA, target: [0, 0, 0] }));
      state.preset = preset;
      $("mission-timeline").min = String(MISSION_START_MJD); $("mission-timeline").max = String(MISSION_END_MJD);
      if (options.epoch == null) $("mission-timeline").value = String(MISSION_END_MJD);
      else $("mission-timeline").value = String(clamp(options.epoch, MISSION_START_MJD, MISSION_END_MJD));
      $("exaggeration").value = String(clamp(options.exaggeration ?? EXAGGERATION.initial, EXAGGERATION.minimum, EXAGGERATION.maximum));
      createRenderer();
      state.fleetView.selectShip(options.ship == null ? null : options.ship);
      state.lastPassed = null;
      if (state.fleetView.view.selected != null) {
        if (options.follow) { state.fleetView.view.follow = true; state.preset = "follow"; Object.assign(state.camera, state.fleetView.followCamera(state.camera)); }
        else if (options.focus) Object.assign(state.camera, state.fleetView.focusCamera());
      }
      updateFleetSelection(); updateSceneOverlay(); draw();
      $("dataset-help").textContent = `${state.fleet.title}: ${state.fleet.score.ships} ships, ${state.fleet.score.unique_asteroids} asteroids, ${fleetMassLabel(state.fleet)} kg (official verifier ${state.fleet.score.official_total_mass_kg} kg).`;
      void loadComputeDetails();
      void loadSolverProgress();
      void loadGpuRecovery();
      void loadGpuOuterValidation();
      void loadGpuVerification();
      void loadGpuScreening();
    } else {
      state.preset = null;
      Object.assign(state.camera, { ...ARCHIVE_CAMERA, target: [0, 0, 0] });
      state.progress = 100; $("timeline").value = "100";
      createRenderer(); renderInventory(); draw();
      $("dataset-help").textContent = "Verified archived P1/P2 evidence records, each in its own physical frame.";
    }
  } catch (error) { if (request === datasetRequest) fatal(error); }
}
function select(index) {
  stopPlayback(); stopCameraMotion(); state.selected = Number(index); state.progress = 100; $("timeline").value = "100";
  Object.assign(state.camera, { ...ARCHIVE_CAMERA, target: [0, 0, 0] });
  try { createRenderer(); renderInventory(); draw(); } catch (error) { fatal(error); }
}
function setMode(mode) {
  stopPlayback(); state.mode = mode; state.progress = 100; $("timeline").value = "100";
  $("mode-description").textContent = mode === "replay"
    ? "Independently integrated archived replay samples. No visual interpolation."
    : "Exact source solver/reference nodes. No visual interpolation.";
  try { createRenderer(); draw(); } catch (error) { fatal(error); }
}

function hideTooltip() { const tooltip = $("hover-tooltip"); tooltip.hidden = true; tooltip.innerHTML = ""; }
function showTooltip(item, x, y) {
  const tooltip = $("hover-tooltip"), wrap = canvas.parentElement;
  tooltip.innerHTML = describeItem(item, state.fleetView);
  tooltip.hidden = false;
  const bounds = wrap.getBoundingClientRect();
  const left = clamp(x + 14, 8, Math.max(8, bounds.width - tooltip.offsetWidth - 8));
  const top = y + 18 + tooltip.offsetHeight > bounds.height - 8 ? y - tooltip.offsetHeight - 10 : y + 18;
  tooltip.style.left = `${left}px`; tooltip.style.top = `${Math.max(8, top)}px`;
}
function canvasPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return [event.clientX - bounds.left, event.clientY - bounds.top];
}

$("dataset-select").addEventListener("change", (event) => setDataset(event.target.value));
$("trajectory-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]"); if (button) select(button.dataset.index);
});
$("ship-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-ship]"); if (!button || !state.fleetView) return;
  state.fleetView.selectShip(button.dataset.ship === "all" ? null : Number(button.dataset.ship));
  if (state.fleetView.view.selected == null) leaveFollow();
  state.lastPassed = null; updateFleetSelection(); draw();
});
$("camera-presets").addEventListener("click", (event) => {
  const button = event.target.closest("[data-preset]"); if (button && !button.disabled) setPreset(button.dataset.preset);
});
$("data-mode").addEventListener("change", (event) => setMode(event.target.value));
$("play-button").addEventListener("click", togglePlayback);
$("mission-play-button").addEventListener("click", togglePlayback);
$("speed-select").addEventListener("change", (event) => { state.speedYearsPerSecond = Number(event.target.value) || 1; });
$("reset-button").addEventListener("click", resetCamera);
$("fleet-reset-button").addEventListener("click", resetCamera);
$("focus-ship-button").addEventListener("click", focusShip);
$("exaggeration").addEventListener("input", (event) => setExaggeration(event.target.value));
$("timeline").addEventListener("input", (event) => {
  stopPlayback(); state.progress = Number(event.target.value); draw();
});
$("mission-timeline").addEventListener("input", (event) => {
  stopPlayback(); if (state.fleetView) { state.fleetView.setEpoch(Number(event.target.value)); draw(); }
});
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault(); canvas.focus(); stopCameraMotion();
  state.pointer = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, time: performance.now(), id: event.pointerId, pan: event.button === 2 || event.shiftKey, moved: false, velocity: { yaw: 0, pitch: 0 } };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.pointer) {
    if (isFleet() && state.fleetView && event.pointerType !== "touch") {
      const [x, y] = canvasPoint(event);
      const item = state.fleetView.hover(x, y);
      if (item) showTooltip(item, x, y); else hideTooltip();
      draw();
    }
    return;
  }
  const now = performance.now(), dtMs = Math.max(1, now - state.pointer.time);
  const dx = event.clientX - state.pointer.x, dy = event.clientY - state.pointer.y;
  state.pointer.x = event.clientX; state.pointer.y = event.clientY; state.pointer.time = now;
  if (Math.hypot(event.clientX - state.pointer.startX, event.clientY - state.pointer.startY) > 4) state.pointer.moved = true;
  if (state.pointer.pan || event.shiftKey) {
    leaveFollow();
    const speed = state.camera.distance * .0015;
    state.camera.target[0] -= dx * speed; state.camera.target[2] += dy * speed;
  } else {
    const yaw = -dx * .008, pitch = dy * .008;
    state.camera.yaw += yaw; state.camera.pitch = clamp(state.camera.pitch + pitch, -pitchLimit(), pitchLimit());
    // Angular velocity (rad/ms) for release inertia, smoothed over the last few samples.
    state.pointer.velocity = { yaw: state.pointer.velocity.yaw * 0.4 + (yaw / dtMs) * 0.6, pitch: state.pointer.velocity.pitch * 0.4 + (pitch / dtMs) * 0.6 };
  }
  draw();
});
canvas.addEventListener("pointerup", (event) => {
  const pointer = state.pointer;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  state.pointer = null;
  if (!pointer) return;
  if (!pointer.moved && isFleet() && state.fleetView) {
    const [x, y] = canvasPoint(event);
    const item = state.fleetView.click(x, y);
    if (item) showTooltip(item, x, y); else hideTooltip();
    state.lastPassed = null; updateFleetSelection(); draw();
  } else if (pointer.moved && !pointer.pan && performance.now() - pointer.time < 80 && isFleet()) {
    state.velocity = pointer.velocity; requestFrame();
  }
});
canvas.addEventListener("pointercancel", (event) => {
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); state.pointer = null;
});
canvas.addEventListener("pointerleave", () => { if (isFleet() && state.fleetView && !state.pointer) { state.fleetView.view.hover = null; hideTooltip(); draw(); } });
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault(); const bounds = zoomBounds(); state.transition = null;
  const factor = Math.exp(event.deltaY * .001);
  if (isFleet() && state.fleetView && !state.fleetView.view.follow) {
    // Dolly towards the cursor: the world point under the pointer (on the focal plane) stays put.
    const [x, y] = canvasPoint(event);
    const ndcX = (x / Math.max(1, canvas.clientWidth)) * 2 - 1, ndcY = 1 - (y / Math.max(1, canvas.clientHeight)) * 2;
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const point = cursorPointOnFocalPlane(state.camera, ndcX, ndcY, FIELD_OF_VIEW, aspect);
    Object.assign(state.camera, dollyTowards(state.camera, point, factor, bounds));
  } else {
    state.camera.distance = clamp(state.camera.distance * factor, bounds.minimum, bounds.maximum);
  }
  draw();
}, { passive: false });
canvas.addEventListener("keydown", (event) => {
  const pan = event.shiftKey, amount = pan ? .06 : .08, bounds = zoomBounds(), limit = pitchLimit();
  if (event.key === " ") togglePlayback();
  else if (event.key === "ArrowLeft") pan ? state.camera.target[0] -= amount : state.camera.yaw += amount;
  else if (event.key === "ArrowRight") pan ? state.camera.target[0] += amount : state.camera.yaw -= amount;
  else if (event.key === "ArrowUp") pan ? state.camera.target[2] += amount : state.camera.pitch = clamp(state.camera.pitch + amount, -limit, limit);
  else if (event.key === "ArrowDown") pan ? state.camera.target[2] -= amount : state.camera.pitch = clamp(state.camera.pitch - amount, -limit, limit);
  else if (["+", "="].includes(event.key)) state.camera.distance = clamp(state.camera.distance * .9, bounds.minimum, bounds.maximum);
  else if (["-", "_"].includes(event.key)) state.camera.distance = clamp(state.camera.distance * 1.1, bounds.minimum, bounds.maximum);
  else if (isFleet() && ["1", "2", "3", "4"].includes(event.key)) setPreset(["top", "oblique", "edge", "follow"][Number(event.key) - 1]);
  else return;
  if (pan) leaveFollow();
  event.preventDefault(); draw();
});
canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault(); state.contextLost = true; stopPlayback(); setStatus("error", "WebGL2 context lost — waiting for restore");
});
canvas.addEventListener("webglcontextrestored", () => {
  state.contextLost = false;
  try { createRenderer(); if (isFleet()) updateFleetSelection(); draw(); } catch (error) { fatal(error); }
});
new ResizeObserver(() => draw()).observe(canvas);
window.addEventListener("pagehide", () => { stopPlayback(); disposeRenderers(); }, { once: true });

/** Read-only hooks for browser tests and tooling; never used by the UI itself. */
window.viewerDebug = Object.freeze({
  get dataset() { return state.dataset; },
  get fleetAvailable() { return state.fleetAvailable; },
  get epoch() { return state.fleetView?.view.epoch ?? null; },
  get selectedShip() { return state.fleetView?.view.selected ?? null; },
  get hover() { return state.fleetView?.view.hover?.type ?? null; },
  get camera() { return { ...state.camera, target: [...state.camera.target] }; },
  get preset() { return state.preset; },
  get following() { return state.fleetView?.view.follow ?? false; },
  get exaggeration() { return state.fleetView?.exaggeration ?? null; },
  get transitioning() { return state.transition != null || inertiaActive(state.velocity); },
  /** Rendering facts for the browser check: context attributes, depth state, instanced body count, tube sides. */
  get glInfo() {
    const renderer = state.fleetView?.renderer ?? state.renderer;
    const gl = renderer?.gl;
    if (!gl) return null;
    const attributes = gl.getContextAttributes();
    return {
      antialias: attributes.antialias, depth: attributes.depth, depthTest: gl.isEnabled(gl.DEPTH_TEST),
      devicePixelRatio: Math.min(devicePixelRatio || 1, 2), drawingBuffer: [gl.drawingBufferWidth, gl.drawingBufferHeight],
      instances: state.fleetView?.renderer.lastInstanceCount ?? 0, tubeSides: state.fleetView?.scene.ships[0]?.tube.sides ?? 0,
    };
  },
  eventScreenPosition(ship, index) { return state.fleetView?.eventScreenPosition(ship, index) ?? null; },
  shipScreenPosition(ship) { return state.fleetView?.shipScreenPosition(ship) ?? null; },
  earthScreenPosition() { return state.fleetView?.earthScreenPosition() ?? null; },
  /** Finish any camera transition or inertia immediately and redraw (deterministic screenshots). */
  settle() {
    if (state.transition) { Object.assign(state.camera, state.transition.to); state.transition = null; }
    state.velocity = { yaw: 0, pitch: 0 }; draw();
  },
});


async function loadGpuRecovery() {
  const el = $("gpu-recovery");
  try {
    const response = await fetch("./data/gtoc12/gpu-recovery.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (result.schema_version !== 1 || result.kind !== "gpu-ipm-recovery-validation") {
      throw new Error("unrecognised recovery metadata");
    }
    el.innerHTML = metricRows([
      ["Runtime", `${result.runtime} · ${result.source_commit.slice(0, 8)}`],
      ["Hardware", result.hardware],
      ["Physics checks", `${result.qualified} / ${result.total} passed · unchanged 1e-6 threshold`],
      ["Outer result", `${result.converged} converged · ${result.trust_region_exhausted} trust-region limit`],
      ["CPU solver fallback", result.hidden_cpu_fallback],
      ...result.rows.map(row => [
        `${row.intervals} intervals · seed ${row.seed}`,
        `${row.scvx_seconds.toFixed(3)} s · ${row.status === 0 ? "converged" : "physics qualified; trust-region limit"}`,
      ]),
      ["Validation", result.tests],
      ["Old Lambda campaign", result.legacy_campaign],
      ["Scope", result.scope],
      ["Remaining", result.remaining],
    ]);
  } catch (error) {
    el.innerHTML = metricRows([["Status", `Recovery results unavailable (${String(error.message || error)})`]]);
  }
}

async function loadGpuOuterValidation() {
  const el = $("gpu-outer-validation");
  try {
    const response = await fetch("./data/gtoc12/gpu-outer-validation.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (result.schema_version !== 1 || result.kind !== "gpu-outer-graph-validation") {
      throw new Error("unrecognised validation metadata");
    }
    el.innerHTML = metricRows([
      ...(result.status ? [["Candidate status", result.status]] : []),
      ["Runtime", `${result.runtime} · ${result.source_commit.slice(0, 8)}`],
      ["Local regression", `${result.tests.local_regression_passed} passed${result.tests.local_regression_failed ? ` · ${result.tests.local_regression_failed} failed` : ""}`],
      ["H100 GTOC12 integration", `${result.tests.h100_integration_passed} passed${result.tests.h100_integration_failed ? ` · ${result.tests.h100_integration_failed} failed` : ""}`],
      ...(result.test_followup ? [["Test follow-up", result.test_followup]] : []),
      ...(result.repeated_qualification ? [["Repeated trajectory checks", result.repeated_qualification]] : []),
      ["Physical thrust", result.thrust_acceptance],
      ["GPU graph probes", result.graph_validation],
      ["H100 sanitizer", result.sanitizer_summary],
      ["Scope", result.scope],
      ["Remaining", result.remaining],
    ]);
  } catch (error) {
    el.innerHTML = metricRows([["Status", `Validation results unavailable (${String(error.message || error)})`]]);
  }
}

async function loadGpuScreening() {
  const el = $("gpu-screening");
  try {
    const response = await fetch("./data/gtoc12/gpu-screening.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (result.schema_version !== 1 || result.kind !== "gpu-candidate-screening") {
      throw new Error("unrecognised screening metadata");
    }
    const { local, h100 } = result.measurements;
    el.innerHTML = metricRows([
      ["Batch workload", `${result.transfers.toLocaleString()} candidate transfers · ${result.branches_per_batch.toLocaleString()} branches`],
      ["RTX 5090 screening", `${local.milliseconds.toFixed(2)} ms · ${Math.round(local.transfers_per_second).toLocaleString()} candidates/s`],
      ["H100 screening", `${h100.milliseconds.toFixed(2)} ms · ${Math.round(h100.transfers_per_second).toLocaleString()} candidates/s`],
      ["Screening versus NumPy", `${local.screening_speedup.toFixed(1)}× local · ${h100.screening_speedup.toFixed(1)}× H100`],
      ...(local.previous_gpu_speedup ? [["Additional gain over previous GPU path", `${local.previous_gpu_speedup.toFixed(2)}× local · ${h100.previous_gpu_speedup.toFixed(2)}× H100; paired comparison`]] : []),
      ["Small route search, warm", `${local.route_search_speedup.toFixed(2)}× local · ${h100.route_search_speedup.toFixed(2)}× H100; cold GPU startup can be slower`],
      ...(result.larger_search ? [
        ["1,000-asteroid search on H100", `${result.larger_search.cpu_seconds.toFixed(2)} s CPU → ${result.larger_search.gpu_seconds.toFixed(2)} s CUDA · ${result.larger_search.speedup.toFixed(2)}×`],
        ["Work per larger search", `${result.larger_search.branches.toLocaleString()} Lambert branches · ${result.larger_search.gpu_batches} GPU batches · ${result.larger_search.candidates} matching proxy candidates`],
      ] : []),
      ...(result.neighbour_selection ? [
        ["Additional gain from GPU neighbours", `${result.neighbour_selection.local.paired_speedup.toFixed(2)}× local · ${result.neighbour_selection.h100.paired_speedup.toFixed(2)}× H100; complete paired search`],
        ["60,000 asteroids, four warm queries", `${result.neighbour_selection.local.pool_gpu_ms.toFixed(2)} ms local · ${result.neighbour_selection.h100.pool_gpu_ms.toFixed(2)} ms H100`],
        ["Neighbour queries versus CPU", `${result.neighbour_selection.local.pool_speedup.toFixed(2)}× local · ${result.neighbour_selection.h100.pool_speedup.toFixed(2)}× H100; identical IDs/order`],
      ] : []),
      ...(result.collection_selection ? [
        ["Additional gain from GPU collection pricing", `${result.collection_selection.local_speedup.toFixed(2)}× local · ${result.collection_selection.h100_speedup.toFixed(2)}× H100; complete paired search`],
        ["Collection/return options priced on GPU", `${result.collection_selection.options.toLocaleString()} per search; checks on existing transfers`],
      ] : []),
      ["Timing scope", result.timing_scope],
      ["Candidate agreement", "Same top 100 on CPU and both GPUs; fleet score unchanged"],
      ["Accuracy checks", result.accuracy],
      ["Remaining CPU work", result.remaining],
    ]);
  } catch (error) {
    el.innerHTML = metricRows([["Status", `Screening results unavailable (${String(error.message || error)})`]]);
  }
}

async function loadGpuVerification() {
  const el = $("gpu-verification");
  try {
    const response = await fetch("./data/gtoc12/gpu-verification.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (result.schema_version !== 1 || result.kind !== "gpu-batched-verification") {
      throw new Error("unrecognised propagation metadata");
    }
    el.innerHTML = metricRows([
      ["Completed trajectories", `${result.legs} / ${result.legs} on both GPUs`],
      ["Local propagation batch", `${result.local_median_ms.toFixed(2)} ms`],
      ["H100 propagation batch", `${result.h100_median_ms.toFixed(2)} ms`],
      ["H100 propagation throughput", `${Math.round(result.legs * 1000 / result.h100_median_ms).toLocaleString()} trajectory legs/s`],
      ["Timing scope", result.timing_scope],
      ["Complete verification, local", `${result.mission_cpu_seconds.toFixed(2)} s CPU → ${result.mission_cuda_seconds.toFixed(3)} s CUDA · ${result.mission_speedup.toFixed(1)}×`],
      ["Complete-check scope", "From parsed inputs, including packing and mission rules; one comparison."],
      ["Largest CPU position difference", `${result.max_position_difference_m.toFixed(3)} m across all legs`],
      ["Accuracy checks", result.accuracy],
      ["Remaining CPU work", result.remaining],
    ]);
  } catch (error) {
    el.innerHTML = metricRows([["Status", `Propagation results unavailable (${String(error.message || error)})`]]);
  }
}

async function loadSolverProgress() {
  const el = $("solver-progress");
  try {
    const response = await fetch("./data/gtoc12/solver-progress.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const benchmark = await response.json();
    if (benchmark.schema_version !== 1 || benchmark.kind !== "gtoc12-synthetic-transfer-benchmark") {
      throw new Error("unrecognised benchmark metadata");
    }
    const milliseconds = Number.isFinite(benchmark.median_seconds)
      ? `${(benchmark.median_seconds * 1000).toFixed(1)} ms` : "Unavailable";
    el.innerHTML = metricRows([
      ["Runtime", benchmark.runtime],
      ["Hardware", benchmark.hardware],
      ["Physics-qualified transfers", `${benchmark.qualified_transfers} / ${benchmark.total_transfers}`],
      ["Median complete transfer", `${milliseconds} · warmup excluded`],
      ["Speedup finding", benchmark.speedup_finding],
      ["Outer command download", benchmark.control_download],
      ["Report collection", benchmark.report_collection || "Synchronous per solve"],
      ["Regression tests", `${benchmark.regression_passed} passed${benchmark.regression_failed ? ` · ${benchmark.regression_failed} failed` : ""}`],
      ["Remaining CPU work", benchmark.remaining_cpu_work],
      ["Accuracy follow-up", benchmark.accuracy_followup],
      ...(benchmark.lambda_snapshot ? [
        ["Lambda campaign snapshot", `${benchmark.lambda_snapshot.completed_groups} / ${benchmark.lambda_snapshot.total_groups} groups complete · ${benchmark.lambda_snapshot.recorded_utc}`],
        ["Lambda attempt outcomes", `${benchmark.lambda_snapshot.timeouts} timeouts · ${benchmark.lambda_snapshot.numerical_failures} numerical failures · ${benchmark.lambda_snapshot.successes} successes`],
        ["Lambda scope", "Older G4 benchmark campaign; separate from this solver and the fleet score"],
      ] : []),
      ["Checkpoint", benchmark.checkpoint_sha256.slice(0, 16)],
    ]);
  } catch (error) {
    el.innerHTML = metricRows([["Status", `Solver progress unavailable (${String(error.message || error)})`]]);
  }
}

async function loadComputeDetails() {
  const el = $("compute-details");
  if (!el) return;
  const dataset = state.dataset, fleet = state.fleet, request = datasetRequest;
  const current = () => request === datasetRequest && dataset === state.dataset;
  el.innerHTML = metricRows([["Status", "Loading compute metadata…"]]);
  try {
    const response = await fetch(`${FLEET_DATASETS[dataset].directory}/compute.json`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const meta = await response.json();
    if (!current()) return;
    if ((meta.fleet_run_id ?? `${meta.run_id}_fleet`) !== fleet.run_id || meta.commit !== fleet.generated_by_commit) {
      throw new Error("metadata does not match the displayed fleet");
    }
    const minutes = (value) => Number.isFinite(value) ? `${(value / 60).toFixed(1)} min` : "—";
    const rows = [
      ["Weighted score (fixed bonus)", Number.isFinite(meta.weighted_score_fixed_bonus_kg) ? `${meta.weighted_score_fixed_bonus_kg.toFixed(3)} kg` : "—"],
      ["Raw returned mass / ship", Number.isFinite(meta.raw_kg_per_ship) ? `${meta.raw_kg_per_ship.toFixed(1)} kg` : "—"],
      ["Historical comparison", meta.retrospective_competition_position ? `${meta.retrospective_competition_position}th if inserted · not an official rank` : "—"],
      ["GPU / hardware", meta.hardware?.gpu || meta.hardware?.fleet_assembly || "—"],
      ["Upstream search GPUs", meta.hardware?.upstream_search || "—"],
      ["Wall time", meta.timing?.wall_human || (meta.timing?.wall_seconds_total != null ? `${(meta.timing.wall_seconds_total / 3600).toFixed(2)} h` : "—")],
      ["Master / recert", `${minutes(meta.timing?.master_wall_seconds)} master · ${minutes(meta.timing?.recertification_wall_seconds)} recert`],
      ["Dynamics model", meta.model?.dynamics || "—"],
      ["Local refine", meta.model?.local_refine || "—"],
      ["Optimisation", meta.optimisation?.strategy || "—"],
      ["Proven optimal", meta.optimisation?.proven_optimal === false ? "No" : meta.optimisation?.proven_optimal ? "Yes (candidate pool)" : "—"],
      ["Run", `${meta.run_id || "—"} · commit ${(meta.commit || "").slice(0, 12)}`],
    ];
    el.innerHTML = metricRows(rows);
  } catch (error) {
    if (!current()) return;
    el.innerHTML = metricRows([["Status", `Compute metadata unavailable (${String(error.message || error)})`]]);
  }
}

/** Optional GTOC12 dataset: present only after `npm run import-gtoc12` (data/gtoc12 is ignored by git). */
async function probeFleetDataset() {
  await Promise.all(Object.entries(FLEET_DATASETS).map(async ([dataset, config]) => {
    let manifest = null;
    try {
      const response = await fetch(`${config.directory}/manifest.json`, { cache: "no-store" });
      if (response.ok) manifest = await response.json();
    } catch { /* Optional dataset remains unavailable. */ }
    const available = manifest?.dataset_kind === "gtoc12-fleet";
    if (available) availableFleets.add(dataset);
    const option = $("dataset-select").querySelector(`option[value="${dataset}"]`);
    option.disabled = !available;
    const summary = manifest?.summary;
    option.textContent = available
      ? `${config.label}${summary ? ` (${summary.ships} ships, ${summary.unique_asteroids} asteroids, ${summary.official_total_mass_kg} kg)` : ""}`
      : `${config.label} — not installed`;
  }));
  state.fleetAvailable = availableFleets.has("gtoc12");
}

try {
  const response = await fetch("./data/trajectories.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Trajectory archive request failed: HTTP ${response.status}`);
  state.data = await response.json();
  await probeFleetDataset();
  const params = new URLSearchParams(location.search);
  const datasetParam = params.get("dataset");
  // Prefer GTOC12 fleet when installed; only stay on archive if explicitly requested.
  const initialDataset = availableFleets.has(datasetParam) ? datasetParam : "gtoc12";
  const wantsFleet = availableFleets.has(initialDataset) && datasetParam !== "archive";
  if (wantsFleet) {
    const ship = params.has("ship") ? Number(params.get("ship")) - 1 : null;
    await setDataset(initialDataset, {
      ship: Number.isInteger(ship) && ship >= 0 ? ship : null, epoch: params.has("epoch") ? Number(params.get("epoch")) : null,
      focus: params.get("focus") === "1", follow: params.get("follow") === "1",
      preset: params.get("preset") || "oblique",
      exaggeration: params.has("z") ? Number(params.get("z")) : null,
    });
  } else {
    applyDatasetVisibility(); createRenderer(); renderInventory(); draw();
    if (state.fleetAvailable) $("dataset-help").textContent = "Verified archived P1/P2 evidence records, each in its own physical frame.";
  }
} catch (error) { fatal(error); }
