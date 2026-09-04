import { clamp, lookAt, multiply, normalizePoint, orbitEye, perspective } from "./math.js";

const $ = (id) => document.getElementById(id);
const requiredIds = [
  "renderer-status", "renderer-status-text", "error-banner", "inventory-count",
  "trajectory-list", "data-mode", "mode-description", "play-button", "play-icon",
  "play-label", "reset-button", "timeline", "timeline-output", "sample-output",
  "trajectory-canvas", "family-label", "trajectory-title", "qualification-badge",
  "qualification-notice", "frame-overlay", "scene-overlay", "current-state",
  "frame-details", "validation-details", "gpu-details", "provenance-content",
];
for (const id of requiredIds) if (!$(id)) throw new Error(`Required DOM element #${id} is missing`);

const canvas = $("trajectory-canvas");
const state = {
  data: null, selected: 0, mode: "replay", progress: 100, playing: false,
  animation: 0, lastTime: 0, renderer: null,
  camera: { yaw: -0.72, pitch: 0.48, distance: 3.25, target: [0, 0, 0] },
  pointer: null, contextLost: false,
};

function flatten(points) { return new Float32Array(points.flat()); }
function format(value, unit = "") {
  if (value == null) return "Not applicable";
  if (!Number.isFinite(value)) return "Invalid";
  const text = value === 0 ? "0" : Math.abs(value) < 0.001 || Math.abs(value) >= 10000
    ? value.toExponential(3) : value.toFixed(4);
  return `${text}${unit ? ` ${unit}` : ""}`;
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}
function metricRows(rows) {
  return rows.map(([name, value]) =>
    `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
}

function sphere(radius, center) {
  const positions = [], normals = [], latitudes = 28, longitudes = 48;
  const unit = (phi, theta) => [
    Math.cos(phi) * Math.cos(theta), Math.cos(phi) * Math.sin(theta), Math.sin(phi),
  ];
  for (let latitude = 0; latitude < latitudes; latitude += 1) {
    const p0 = -Math.PI / 2 + latitude / latitudes * Math.PI;
    const p1 = -Math.PI / 2 + (latitude + 1) / latitudes * Math.PI;
    for (let longitude = 0; longitude < longitudes; longitude += 1) {
      const t0 = longitude / longitudes * Math.PI * 2;
      const t1 = (longitude + 1) / longitudes * Math.PI * 2;
      for (const normal of [unit(p0, t0), unit(p1, t0), unit(p1, t1), unit(p0, t0), unit(p1, t1), unit(p0, t1)]) {
        normals.push(normal);
        positions.push(normal.map((value, axis) => center[axis] + value * radius));
      }
    }
  }
  return { positions: flatten(positions), normals: flatten(normals) };
}
function plane(center, extent, z) {
  return flatten([
    [center[0] - extent, center[1] - extent, z], [center[0] + extent, center[1] - extent, z],
    [center[0] + extent, center[1] + extent, z], [center[0] - extent, center[1] - extent, z],
    [center[0] + extent, center[1] + extent, z], [center[0] - extent, center[1] + extent, z],
  ]);
}
function planeGrid(center, extent, z) {
  const result = [], divisions = 12;
  for (let index = -divisions; index <= divisions; index += 1) {
    const offset = index / divisions * extent;
    result.push(
      [center[0] - extent, center[1] + offset, z], [center[0] + extent, center[1] + offset, z],
      [center[0] + offset, center[1] - extent, z], [center[0] + offset, center[1] + extent, z],
    );
  }
  return flatten(result);
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
  const ribbonPositions = [], ribbonPrevious = [], ribbonNext = [], ribbonSides = [];
  pathPoints.forEach((point, index) => {
    const previous = pathPoints[Math.max(0, index - 1)];
    const next = pathPoints[Math.min(pathPoints.length - 1, index + 1)];
    for (const side of [-1, 1]) {
      ribbonPositions.push(...point); ribbonPrevious.push(...previous); ribbonNext.push(...next); ribbonSides.push(side);
    }
  });
  return {
    center, scale, points, path, target, axes, surface, normals, grid, surfaceZ,
    start: path.slice(0, 3),
    ribbonPositions: new Float32Array(ribbonPositions), ribbonPrevious: new Float32Array(ribbonPrevious),
    ribbonNext: new Float32Array(ribbonNext), ribbonSides: new Float32Array(ribbonSides),
  };
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Unable to allocate WebGL shader");
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw new Error(message);
  }
  return shader;
}
function link(gl, vertexSource, fragmentSource) {
  const vertex = compile(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const output = gl.createProgram();
  if (!output) throw new Error("Unable to allocate WebGL program");
  gl.attachShader(output, vertex); gl.attachShader(output, fragment); gl.linkProgram(output);
  gl.deleteShader(vertex); gl.deleteShader(fragment);
  if (!gl.getProgramParameter(output, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(output); gl.deleteProgram(output); throw new Error(message);
  }
  return output;
}
function hex(value) {
  const text = value.replace("#", "");
  return [0, 2, 4].map((index) => parseInt(text.slice(index, index + 2), 16) / 255).concat(1);
}

class Renderer {
  constructor(canvasElement, scene) {
    this.canvas = canvasElement; this.scene = scene; this.buffers = []; this.programs = [];
    this.gl = canvasElement.getContext("webgl2", { antialias: true, alpha: false, depth: true, powerPreference: "high-performance" });
    if (!this.gl) throw new Error("WebGL2 is unavailable. Enable hardware acceleration or use a WebGL2-capable browser.");
    const gl = this.gl;
    this.lineProgram = this.makeProgram(`#version 300 es
      in vec3 aPosition; uniform mat4 uMvp; uniform float uSize;
      void main(){gl_Position=uMvp*vec4(aPosition,1.0);gl_PointSize=uSize;}`,
      `#version 300 es
      precision highp float; uniform vec4 uColor; uniform float uMarker; out vec4 color;
      void main(){if(uMarker>0.5){float r=distance(gl_PointCoord,vec2(.5));if(r>.5)discard;if(uMarker>1.5&&r<.28)discard;}color=uColor;}`);
    this.surfaceProgram = this.makeProgram(`#version 300 es
      in vec3 aPosition;in vec3 aNormal;uniform mat4 uMvp;out vec3 normal;
      void main(){normal=aNormal;gl_Position=uMvp*vec4(aPosition,1.0);}`,
      `#version 300 es
      precision highp float;in vec3 normal;uniform vec4 uColor;out vec4 color;
      void main(){float d=max(dot(normalize(normal),normalize(vec3(-.45,.65,.8))),0.);color=vec4(uColor.rgb*(.2+.8*d),uColor.a);}`);
    this.ribbonProgram = this.makeProgram(`#version 300 es
      in vec3 aPosition;in vec3 aPrevious;in vec3 aNext;in float aSide;uniform mat4 uMvp;uniform vec2 uViewport;out float edge;
      void main(){vec4 c=uMvp*vec4(aPosition,1.);vec4 p=uMvp*vec4(aPrevious,1.);vec4 n=uMvp*vec4(aNext,1.);
      vec2 dir=normalize((n.xy/max(n.w,.001)-p.xy/max(p.w,.001))*uViewport);vec2 normal=vec2(-dir.y,dir.x);
      c.xy+=normal*aSide*3.0/uViewport*c.w;gl_Position=c;edge=aSide;}`,
      `#version 300 es
      precision highp float;in float edge;uniform vec4 uColor;out vec4 color;
      void main(){float alpha=1.-smoothstep(.72,1.,abs(edge));color=vec4(uColor.rgb,uColor.a*alpha);}`);
    const arrays = ["path", "target", "axes", "surface", "normals", "grid", "start", "ribbonPositions", "ribbonPrevious", "ribbonNext", "ribbonSides"];
    for (const name of arrays) this[`${name}Buffer`] = this.makeBuffer(scene[name]);
    this.currentBuffer = this.makeBuffer(new Float32Array(3), gl.DYNAMIC_DRAW);
    this.stemBuffer = this.makeBuffer(new Float32Array(6), gl.DYNAMIC_DRAW);
    const debug = gl.getExtension("WEBGL_debug_renderer_info");
    this.info = {
      version: gl.getParameter(gl.VERSION),
      renderer: gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
      vendor: gl.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
      maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
    };
  }
  makeProgram(vertex, fragment) { const value = link(this.gl, vertex, fragment); this.programs.push(value); return value; }
  makeBuffer(data, usage = this.gl.STATIC_DRAW) {
    const value = this.gl.createBuffer(); if (!value) throw new Error("Unable to allocate WebGL buffer");
    this.buffers.push(value); this.gl.bindBuffer(this.gl.ARRAY_BUFFER, value); this.gl.bufferData(this.gl.ARRAY_BUFFER, data, usage); return value;
  }
  resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    this.gl.viewport(0, 0, width, height);
  }
  attribute(program, name, buffer, size) {
    const gl = this.gl, location = gl.getAttribLocation(program, name);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
  }
  line(mvp, buffer, mode, count, color, size = 1, marker = 0) {
    const gl = this.gl, p = this.lineProgram; gl.useProgram(p); this.attribute(p, "aPosition", buffer, 3);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "uMvp"), false, mvp); gl.uniform4fv(gl.getUniformLocation(p, "uColor"), color);
    gl.uniform1f(gl.getUniformLocation(p, "uSize"), size); gl.uniform1f(gl.getUniformLocation(p, "uMarker"), marker); gl.drawArrays(mode, 0, count);
  }
  draw(visibleCount, camera) {
    const gl = this.gl, scene = this.scene; this.resize();
    const mvp = multiply(perspective(Math.PI / 4, this.canvas.width / this.canvas.height, 0.025, 80), lookAt(orbitEye(camera), camera.target, [0, 0, 1]));
    gl.clearColor(0.018, 0.027, 0.055, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const p = this.surfaceProgram; gl.useProgram(p); this.attribute(p, "aPosition", this.surfaceBuffer, 3); this.attribute(p, "aNormal", this.normalsBuffer, 3);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "uMvp"), false, mvp); gl.uniform4fv(gl.getUniformLocation(p, "uColor"), [0.10, 0.20, 0.34, scene.surfaceZ == null ? .96 : .48]);
    gl.drawArrays(gl.TRIANGLES, 0, scene.surface.length / 3);
    this.line(mvp, this.gridBuffer, gl.LINES, scene.grid.length / 3, [0.30, 0.46, 0.66, .32]);
    this.line(mvp, this.axesBuffer, gl.LINES, scene.axes.length / 3, [0.72, 0.8, 0.92, .72]);
    if (visibleCount > 1) {
      const ribbon = this.ribbonProgram; gl.useProgram(ribbon);
      this.attribute(ribbon, "aPosition", this.ribbonPositionsBuffer, 3); this.attribute(ribbon, "aPrevious", this.ribbonPreviousBuffer, 3);
      this.attribute(ribbon, "aNext", this.ribbonNextBuffer, 3); this.attribute(ribbon, "aSide", this.ribbonSidesBuffer, 1);
      gl.uniformMatrix4fv(gl.getUniformLocation(ribbon, "uMvp"), false, mvp); gl.uniform2f(gl.getUniformLocation(ribbon, "uViewport"), this.canvas.width, this.canvas.height);
      gl.uniform4fv(gl.getUniformLocation(ribbon, "uColor"), hex("#36d6ff")); gl.drawArrays(gl.TRIANGLE_STRIP, 0, visibleCount * 2);
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
  dispose() { for (const buffer of this.buffers) this.gl.deleteBuffer(buffer); for (const program of this.programs) this.gl.deleteProgram(program); this.buffers = []; this.programs = []; }
}

function trajectory() { return state.data.trajectories[state.selected]; }
function visibleCount() {
  const count = trajectory()[state.mode].point_count;
  return clamp(Math.ceil(count * state.progress / 100), 1, count);
}
function stopPlayback() {
  state.playing = false; state.lastTime = 0;
  if (state.animation) cancelAnimationFrame(state.animation);
  state.animation = 0; $("play-icon").textContent = "▶"; $("play-label").textContent = "Play replay";
}
function resetCamera() {
  Object.assign(state.camera, { yaw: -.72, pitch: .48, distance: 3.25, target: [0, 0, 0] }); draw();
}
function setStatus(kind, text) {
  $("renderer-status").className = `renderer-status ${kind}`; $("renderer-status-text").textContent = text;
}
function createRenderer() {
  state.renderer?.dispose(); state.renderer = null;
  const scene = geometry(trajectory(), state.mode); state.scene = scene;
  state.renderer = new Renderer(canvas, scene); setStatus("ready", "WebGL2 GPU renderer");
  $("gpu-details").innerHTML = metricRows([
    ["Status", "Active WebGL2"], ["Version", state.renderer.info.version],
    ["Renderer", state.renderer.info.renderer], ["Vendor", state.renderer.info.vendor],
    ["Max texture", `${state.renderer.info.maxTextureSize.toLocaleString()} px`],
    ["Vertex attributes", String(state.renderer.info.maxVertexAttribs)],
  ]);
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
  $("frame-overlay").innerHTML = `<strong>${escapeHtml(item.frame)}</strong><span>${escapeHtml(item.viewer.axes.join(" · "))}</span>`;
  $("scene-overlay").textContent = item.viewer.scene_kind === "hcw" ? "LVLH plane · Earthward −X" : item.viewer.radius_label;
  $("timeline-output").textContent = `${state.progress.toFixed(1)}%`;
  $("sample-output").textContent = `Sample ${index + 1} / ${source.point_count} · ${format(point[0], item.time_units)}`;
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
function draw() {
  if (!state.renderer || state.contextLost) return;
  state.renderer.draw(visibleCount(), state.camera); updateDetails();
}
function select(index) {
  stopPlayback(); state.selected = Number(index); state.progress = 100; $("timeline").value = "100";
  Object.assign(state.camera, { yaw: -.72, pitch: .48, distance: 3.25, target: [0, 0, 0] });
  try { createRenderer(); renderInventory(); draw(); } catch (error) { fatal(error); }
}
function setMode(mode) {
  stopPlayback(); state.mode = mode; state.progress = 100; $("timeline").value = "100";
  $("mode-description").textContent = mode === "replay"
    ? "Independently integrated archived replay samples. No visual interpolation."
    : "Exact source solver/reference nodes. No visual interpolation.";
  try { createRenderer(); draw(); } catch (error) { fatal(error); }
}
function animate(time) {
  if (!state.playing) return;
  const previous = state.lastTime || time; state.lastTime = time;
  state.progress = Math.min(100, state.progress + (time - previous) / 120);
  $("timeline").value = String(state.progress); draw();
  if (state.progress >= 100) stopPlayback(); else state.animation = requestAnimationFrame(animate);
}
function togglePlayback() {
  if (state.playing) return stopPlayback();
  if (state.progress >= 100) state.progress = 0;
  state.playing = true; $("play-icon").textContent = "❚❚"; $("play-label").textContent = "Pause replay";
  state.animation = requestAnimationFrame(animate);
}
function fatal(error) {
  const message = error instanceof Error ? error.message : String(error);
  $("error-banner").hidden = false; $("error-banner").textContent = message; setStatus("error", "Viewer unavailable");
  console.error(error);
}

$("trajectory-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]"); if (button) select(button.dataset.index);
});
$("data-mode").addEventListener("change", (event) => setMode(event.target.value));
$("play-button").addEventListener("click", togglePlayback);
$("reset-button").addEventListener("click", resetCamera);
$("timeline").addEventListener("input", (event) => {
  stopPlayback(); state.progress = Number(event.target.value); draw();
});
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault(); canvas.focus(); state.pointer = { x: event.clientX, y: event.clientY, id: event.pointerId, pan: event.button === 2 || event.shiftKey };
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove", (event) => {
  if (!state.pointer) return;
  const dx = event.clientX - state.pointer.x, dy = event.clientY - state.pointer.y;
  state.pointer.x = event.clientX; state.pointer.y = event.clientY;
  if (state.pointer.pan || event.shiftKey) {
    const speed = state.camera.distance * .0015;
    state.camera.target[0] -= dx * speed; state.camera.target[2] += dy * speed;
  } else {
    state.camera.yaw -= dx * .008; state.camera.pitch = clamp(state.camera.pitch + dy * .008, -1.45, 1.45);
  }
  draw();
});
for (const name of ["pointerup", "pointercancel"]) canvas.addEventListener(name, (event) => {
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); state.pointer = null;
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault(); state.camera.distance = clamp(state.camera.distance * Math.exp(event.deltaY * .001), 1.35, 12); draw();
}, { passive: false });
canvas.addEventListener("keydown", (event) => {
  const pan = event.shiftKey, amount = pan ? .06 : .08;
  if (event.key === " ") togglePlayback();
  else if (event.key === "ArrowLeft") pan ? state.camera.target[0] -= amount : state.camera.yaw += amount;
  else if (event.key === "ArrowRight") pan ? state.camera.target[0] += amount : state.camera.yaw -= amount;
  else if (event.key === "ArrowUp") pan ? state.camera.target[2] += amount : state.camera.pitch = clamp(state.camera.pitch + amount, -1.45, 1.45);
  else if (event.key === "ArrowDown") pan ? state.camera.target[2] -= amount : state.camera.pitch = clamp(state.camera.pitch - amount, -1.45, 1.45);
  else if (["+", "="].includes(event.key)) state.camera.distance = clamp(state.camera.distance * .9, 1.35, 12);
  else if (["-", "_"].includes(event.key)) state.camera.distance = clamp(state.camera.distance * 1.1, 1.35, 12);
  else return;
  event.preventDefault(); draw();
});
canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault(); state.contextLost = true; stopPlayback(); setStatus("error", "WebGL2 context lost — waiting for restore");
});
canvas.addEventListener("webglcontextrestored", () => {
  state.contextLost = false;
  try { createRenderer(); draw(); } catch (error) { fatal(error); }
});
new ResizeObserver(() => draw()).observe(canvas);
window.addEventListener("pagehide", () => { stopPlayback(); state.renderer?.dispose(); state.renderer = null; }, { once: true });

try {
  const response = await fetch("./data/trajectories.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Trajectory archive request failed: HTTP ${response.status}`);
  state.data = await response.json();
  createRenderer(); renderInventory(); draw();
} catch (error) { fatal(error); }
