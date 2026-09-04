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
  "frame-details", "validation-details", "gpu-details", "provenance-content",
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

const isFleet = () => state.dataset === "gtoc12";
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
async function loadFleet() {
  if (state.fleet) return state.fleet;
  const response = await fetch("./data/gtoc12/fleet.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`GTOC12 fleet dataset request failed: HTTP ${response.status}`);
  const fleet = await response.json();
  if (fleet.dataset_kind !== "gtoc12-fleet" || !Array.isArray(fleet.ships) || fleet.ships.length === 0) throw new Error("GTOC12 fleet dataset has an unexpected shape");
  state.fleet = fleet;
  return fleet;
}
async function setDataset(dataset, options = {}) {
  stopPlayback(); stopCameraMotion();
  if (dataset === "gtoc12" && !state.fleetAvailable) {
    $("dataset-select").value = state.dataset;
    $("dataset-help").textContent = "GTOC12 dataset not installed. Run `npm run import-gtoc12 -- --export <export dir> --catalogue <GTOC12_Asteroids_Data.txt>` (see README) and reload.";
    return;
  }
  try {
    state.dataset = dataset;
    $("dataset-select").value = dataset;
    applyDatasetVisibility();
    if (dataset === "gtoc12") {
      await loadFleet();
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
    } else {
      state.preset = null;
      Object.assign(state.camera, { ...ARCHIVE_CAMERA, target: [0, 0, 0] });
      state.progress = 100; $("timeline").value = "100";
      createRenderer(); renderInventory(); draw();
      $("dataset-help").textContent = "Verified archived P1/P2 evidence records, each in its own physical frame.";
    }
  } catch (error) { fatal(error); }
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

/** Optional GTOC12 dataset: present only after `npm run import-gtoc12` (data/gtoc12 is ignored by git). */
async function probeFleetDataset() {
  let manifest = null;
  try {
    const response = await fetch("./data/gtoc12/manifest.json", { cache: "no-store" });
    if (response.ok) manifest = await response.json();
    state.fleetAvailable = manifest?.dataset_kind === "gtoc12-fleet";
  } catch { state.fleetAvailable = false; }
  const option = $("dataset-select").querySelector('option[value="gtoc12"]');
  option.disabled = !state.fleetAvailable;
  const summary = manifest?.summary;
  option.textContent = state.fleetAvailable
    ? `GTOC12 fleet${summary ? ` (${summary.ships} ships, ${summary.unique_asteroids} asteroids, ${summary.official_total_mass_kg} kg)` : ""}`
    : "GTOC12 fleet — not installed";
  if (!state.fleetAvailable) $("dataset-help").textContent = "GTOC12 dataset not installed: run `npm run import-gtoc12` (see README) to add data/gtoc12/.";
}

try {
  const response = await fetch("./data/trajectories.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Trajectory archive request failed: HTTP ${response.status}`);
  state.data = await response.json();
  await probeFleetDataset();
  const params = new URLSearchParams(location.search);
  const wantsFleet = params.get("dataset") === "gtoc12" && state.fleetAvailable;
  if (wantsFleet) {
    const ship = params.has("ship") ? Number(params.get("ship")) - 1 : null;
    await setDataset("gtoc12", {
      ship: Number.isInteger(ship) && ship >= 0 ? ship : null, epoch: params.has("epoch") ? Number(params.get("epoch")) : null,
      focus: params.get("focus") === "1", follow: params.get("follow") === "1", preset: params.get("preset"),
      exaggeration: params.has("z") ? Number(params.get("z")) : null,
    });
  } else {
    applyDatasetVisibility(); createRenderer(); renderInventory(); draw();
    if (state.fleetAvailable) $("dataset-help").textContent = "Verified archived P1/P2 evidence records, each in its own physical frame.";
  }
} catch (error) { fatal(error); }
