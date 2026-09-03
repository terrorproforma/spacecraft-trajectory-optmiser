// WebGL2 helpers shared by the archive renderer (app.js) and the GTOC12 fleet renderer
// (gtoc12.js): geometry primitives, shader compilation and the three core programs.

export function flatten(points) { return new Float32Array(points.flat()); }

export function hex(value, alpha = 1) {
  const text = value.replace("#", "");
  return [0, 2, 4].map((index) => parseInt(text.slice(index, index + 2), 16) / 255).concat(alpha);
}

export function sphere(radius, center, latitudes = 28, longitudes = 48) {
  const positions = [], normals = [];
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

export function planeGrid(center, extent, z, divisions = 12) {
  const result = [];
  for (let index = -divisions; index <= divisions; index += 1) {
    const offset = index / divisions * extent;
    result.push(
      [center[0] - extent, center[1] + offset, z], [center[0] + extent, center[1] + offset, z],
      [center[0] + offset, center[1] - extent, z], [center[0] + offset, center[1] + extent, z],
    );
  }
  return flatten(result);
}

/** Circle in the z = 0 plane as LINES pairs. */
export function circleLines(radius, segments = 180, z = 0) {
  const result = [];
  for (let index = 0; index < segments; index += 1) {
    const a = index / segments * Math.PI * 2, b = (index + 1) / segments * Math.PI * 2;
    result.push([radius * Math.cos(a), radius * Math.sin(a), z], [radius * Math.cos(b), radius * Math.sin(b), z]);
  }
  return flatten(result);
}

/** Ribbon attribute arrays (position / previous / next / side) for a polyline of [x, y, z] points. */
export function ribbonArrays(points) {
  const positions = [], previous = [], next = [], sides = [];
  points.forEach((point, index) => {
    const before = points[Math.max(0, index - 1)];
    const after = points[Math.min(points.length - 1, index + 1)];
    for (const side of [-1, 1]) {
      positions.push(...point); previous.push(...before); next.push(...after); sides.push(side);
    }
  });
  return {
    positions: new Float32Array(positions), previous: new Float32Array(previous),
    next: new Float32Array(next), sides: new Float32Array(sides),
  };
}

export function compile(gl, type, source) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Unable to allocate WebGL shader");
  gl.shaderSource(shader, source); gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader); gl.deleteShader(shader); throw new Error(message);
  }
  return shader;
}

export function link(gl, vertexSource, fragmentSource) {
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

export const LINE_VERTEX = `#version 300 es
  in vec3 aPosition; uniform mat4 uMvp; uniform float uSize;
  void main(){gl_Position=uMvp*vec4(aPosition,1.0);gl_PointSize=uSize;}`;
// uMarker 0 = plain, 1 = filled disc, 2 = ring.
export const LINE_FRAGMENT = `#version 300 es
  precision highp float; uniform vec4 uColor; uniform float uMarker; out vec4 color;
  void main(){if(uMarker>0.5){float r=distance(gl_PointCoord,vec2(.5));if(r>.5)discard;if(uMarker>1.5&&r<.28)discard;}color=uColor;}`;
export const SURFACE_VERTEX = `#version 300 es
  in vec3 aPosition;in vec3 aNormal;uniform mat4 uMvp;out vec3 normal;
  void main(){normal=aNormal;gl_Position=uMvp*vec4(aPosition,1.0);}`;
export const SURFACE_FRAGMENT = `#version 300 es
  precision highp float;in vec3 normal;uniform vec4 uColor;out vec4 color;
  void main(){float d=max(dot(normalize(normal),normalize(vec3(-.45,.65,.8))),0.);color=vec4(uColor.rgb*(.2+.8*d),uColor.a);}`;
// Screen-space ribbon: uWidth is the half-width in device pixels.
export const RIBBON_VERTEX = `#version 300 es
  in vec3 aPosition;in vec3 aPrevious;in vec3 aNext;in float aSide;uniform mat4 uMvp;uniform vec2 uViewport;uniform float uWidth;out float edge;
  void main(){vec4 c=uMvp*vec4(aPosition,1.);vec4 p=uMvp*vec4(aPrevious,1.);vec4 n=uMvp*vec4(aNext,1.);
  vec2 dir=normalize((n.xy/max(n.w,.001)-p.xy/max(p.w,.001))*uViewport);vec2 normal=vec2(-dir.y,dir.x);
  c.xy+=normal*aSide*uWidth/uViewport*c.w;gl_Position=c;edge=aSide;}`;
export const RIBBON_FRAGMENT = `#version 300 es
  precision highp float;in float edge;uniform vec4 uColor;out vec4 color;
  void main(){float alpha=1.-smoothstep(.72,1.,abs(edge));color=vec4(uColor.rgb,uColor.a*alpha);}`;

export function createContext(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: true, alpha: false, depth: true, powerPreference: "high-performance" });
  if (!gl) throw new Error("WebGL2 is unavailable. Enable hardware acceleration or use a WebGL2-capable browser.");
  return gl;
}

export function contextInfo(gl) {
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    version: gl.getParameter(gl.VERSION),
    renderer: gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER),
    vendor: gl.getParameter(debug ? debug.UNMASKED_VENDOR_WEBGL : gl.VENDOR),
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxVertexAttribs: gl.getParameter(gl.MAX_VERTEX_ATTRIBS),
  };
}

/** Base class owning GL resources and the three core programs. */
export class GlResources {
  constructor(canvasElement) {
    this.canvas = canvasElement; this.buffers = []; this.programs = [];
    this.gl = createContext(canvasElement);
    this.lineProgram = this.makeProgram(LINE_VERTEX, LINE_FRAGMENT);
    this.surfaceProgram = this.makeProgram(SURFACE_VERTEX, SURFACE_FRAGMENT);
    this.ribbonProgram = this.makeProgram(RIBBON_VERTEX, RIBBON_FRAGMENT);
    this.info = contextInfo(this.gl);
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
  line(mvp, buffer, mode, count, color, size = 1, marker = 0, first = 0) {
    if (count <= 0) return;
    const gl = this.gl, p = this.lineProgram; gl.useProgram(p); this.attribute(p, "aPosition", buffer, 3);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "uMvp"), false, mvp); gl.uniform4fv(gl.getUniformLocation(p, "uColor"), color);
    gl.uniform1f(gl.getUniformLocation(p, "uSize"), size); gl.uniform1f(gl.getUniformLocation(p, "uMarker"), marker); gl.drawArrays(mode, first, count);
  }
  surface(mvp, positions, normals, count, color) {
    const gl = this.gl, p = this.surfaceProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", positions, 3); this.attribute(p, "aNormal", normals, 3);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "uMvp"), false, mvp); gl.uniform4fv(gl.getUniformLocation(p, "uColor"), color);
    gl.drawArrays(gl.TRIANGLES, 0, count);
  }
  ribbon(mvp, buffers, vertexCount, color, width = 3) {
    if (vertexCount <= 0) return;
    const gl = this.gl, p = this.ribbonProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", buffers.positions, 3); this.attribute(p, "aPrevious", buffers.previous, 3);
    this.attribute(p, "aNext", buffers.next, 3); this.attribute(p, "aSide", buffers.sides, 1);
    gl.uniformMatrix4fv(gl.getUniformLocation(p, "uMvp"), false, mvp); gl.uniform2f(gl.getUniformLocation(p, "uViewport"), this.canvas.width, this.canvas.height);
    gl.uniform1f(gl.getUniformLocation(p, "uWidth"), width); gl.uniform4fv(gl.getUniformLocation(p, "uColor"), color);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexCount);
  }
  dispose() {
    const gl = this.gl;
    // Detach every vertex attribute first so a later renderer on the same context never draws
    // with an enabled attribute that still points at a deleted buffer.
    for (let location = 0; location < this.info.maxVertexAttribs; location += 1) gl.disableVertexAttribArray(location);
    gl.bindBuffer(gl.ARRAY_BUFFER, null); gl.useProgram(null);
    for (const buffer of this.buffers) gl.deleteBuffer(buffer);
    for (const program of this.programs) gl.deleteProgram(program);
    this.buffers = []; this.programs = [];
  }
}
