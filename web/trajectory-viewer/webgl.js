// WebGL2 helpers shared by the archive renderer (app.js) and the GTOC12 fleet renderer
// (gtoc12.js): geometry primitives, shader compilation and the core programs. Every program
// takes a `uZScale` uniform so the fleet view can exaggerate the vertical (Z) axis without
// rebuilding geometry; the archive renderer leaves it at 1.

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

/** Filled disc in the z = 0 plane as TRIANGLES (fan unrolled), `segments` wedges. */
export function discTriangles(radius, segments = 96, z = 0) {
  const result = [];
  for (let index = 0; index < segments; index += 1) {
    const a = index / segments * Math.PI * 2, b = (index + 1) / segments * Math.PI * 2;
    result.push([0, 0, z], [radius * Math.cos(a), radius * Math.sin(a), z], [radius * Math.cos(b), radius * Math.sin(b), z]);
  }
  return flatten(result);
}

/** Radial spokes in the z = 0 plane every `stepDegrees`, from `inner` to `outer` radius, as LINES. */
export function spokeLines(inner, outer, stepDegrees = 30, z = 0) {
  const result = [];
  for (let degrees = 0; degrees < 360; degrees += stepDegrees) {
    const angle = degrees * Math.PI / 180;
    result.push([inner * Math.cos(angle), inner * Math.sin(angle), z], [outer * Math.cos(angle), outer * Math.sin(angle), z]);
  }
  return flatten(result);
}

/** Deterministic 32-bit PRNG (mulberry32) so the procedural star field is reproducible. */
export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Procedural star field: `count` points uniformly distributed on a sphere of `radius` with a
 * magnitude in (0, 1] skewed towards faint stars. Purely decorative and reproducible from `seed`.
 */
export function starField(count, radius, seed = 12) {
  const random = mulberry32(seed);
  const positions = new Float32Array(count * 3), magnitudes = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const z = random() * 2 - 1, theta = random() * Math.PI * 2, ring = Math.sqrt(Math.max(0, 1 - z * z));
    positions[index * 3] = radius * ring * Math.cos(theta);
    positions[index * 3 + 1] = radius * ring * Math.sin(theta);
    positions[index * 3 + 2] = radius * z;
    magnitudes[index] = Math.pow(random(), 2.2) * 0.85 + 0.15;
  }
  return { positions, magnitudes };
}

/**
 * Ribbon attribute arrays (position / previous / next / side, optionally per-vertex time) for a
 * polyline of [x, y, z] points. With `closed`, the loop is sealed by repeating the first point and
 * the tangents wrap around, so a Keplerian orbit renders without a seam.
 */
export function ribbonArrays(points, times = null, closed = false) {
  const positions = [], previous = [], next = [], sides = [], stamps = [];
  const count = points.length;
  const total = closed ? count + 1 : count;
  for (let index = 0; index < total; index += 1) {
    const at = closed ? index % count : index;
    const point = points[at];
    const before = closed ? points[(at - 1 + count) % count] : points[Math.max(0, index - 1)];
    const after = closed ? points[(at + 1) % count] : points[Math.min(count - 1, index + 1)];
    for (const side of [-1, 1]) {
      positions.push(...point); previous.push(...before); next.push(...after); sides.push(side);
      if (times) stamps.push(times[at]);
    }
  }
  const result = {
    positions: new Float32Array(positions), previous: new Float32Array(previous),
    next: new Float32Array(next), sides: new Float32Array(sides), vertexCount: total * 2,
  };
  if (times) result.times = new Float32Array(stamps);
  return result;
}

/**
 * Lit tube mesh along a polyline: `sides` vertices per archived sample, each carrying the sample
 * position (the tube axis), a unit radial normal from a parallel-transported frame, and the
 * sample time. The vertex shader displaces along the normal by a uniform radius, so one mesh
 * serves every zoom level. Segment i uses indices [i * sides * 6, (i + 1) * sides * 6), so drawing
 * the first (visible - 1) segments shows the arc up to an archived sample, never beyond it.
 */
export function tubeArrays(points, times = null, sides = 6) {
  const count = points.length;
  if (count < 2) throw new RangeError("A tube needs at least two points");
  if (count * sides > 65535) throw new RangeError("Tube exceeds 16-bit index range");
  const positions = new Float32Array(count * sides * 3), normals = new Float32Array(count * sides * 3);
  const stamps = times ? new Float32Array(count * sides) : null;
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const unit = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  let normal = null;
  for (let index = 0; index < count; index += 1) {
    const before = points[Math.max(0, index - 1)], after = points[Math.min(count - 1, index + 1)];
    let tangent = unit(sub(after, before));
    if (Math.hypot(...tangent) < 1e-9) tangent = [1, 0, 0];
    if (!normal) {
      const seed = Math.abs(tangent[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
      normal = unit(cross(cross(tangent, seed), tangent));
    } else {
      // Parallel transport: remove the component along the new tangent, keeping the frame continuous.
      const along = dot(normal, tangent);
      normal = unit([normal[0] - tangent[0] * along, normal[1] - tangent[1] * along, normal[2] - tangent[2] * along]);
    }
    const binormal = cross(tangent, normal);
    for (let side = 0; side < sides; side += 1) {
      const angle = side / sides * Math.PI * 2, c = Math.cos(angle), s = Math.sin(angle);
      const vertex = (index * sides + side) * 3;
      positions.set(points[index], vertex);
      normals.set([normal[0] * c + binormal[0] * s, normal[1] * c + binormal[1] * s, normal[2] * c + binormal[2] * s], vertex);
      if (stamps) stamps[index * sides + side] = times[index];
    }
  }
  const indices = new Uint16Array((count - 1) * sides * 6);
  let cursor = 0;
  for (let segment = 0; segment < count - 1; segment += 1) {
    for (let side = 0; side < sides; side += 1) {
      const a = segment * sides + side, b = segment * sides + (side + 1) % sides;
      const c = a + sides, d = b + sides;
      indices.set([a, c, d, a, d, b], cursor); cursor += 6;
    }
  }
  return { positions, normals, times: stamps, indices, sides, indicesPerSegment: sides * 6, vertexCount: count * sides };
}

/** Concatenate several ribbons into one attribute set with per-ribbon vertex offsets. */
export function concatRibbons(ribbons) {
  const offsets = [], counts = [];
  const parts = { positions: [], previous: [], next: [], sides: [] };
  let vertex = 0;
  for (const ribbon of ribbons) {
    offsets.push(vertex); counts.push(ribbon.vertexCount); vertex += ribbon.vertexCount;
    for (const key of Object.keys(parts)) parts[key].push(ribbon[key]);
  }
  const join = (chunks, width) => {
    const output = new Float32Array(vertex * width);
    let cursor = 0;
    for (const chunk of chunks) { output.set(chunk, cursor); cursor += chunk.length; }
    return output;
  };
  return { positions: join(parts.positions, 3), previous: join(parts.previous, 3), next: join(parts.next, 3), sides: join(parts.sides, 1), offsets, counts, vertexCount: vertex };
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
  in vec3 aPosition; uniform mat4 uMvp; uniform float uSize; uniform float uZScale;
  void main(){gl_Position=uMvp*vec4(aPosition*vec3(1.,1.,uZScale),1.0);gl_PointSize=uSize;}`;
// uMarker 0 = plain, 1 = filled disc, 2 = ring, 3 = soft glow (quadratic falloff, for additive halos).
export const LINE_FRAGMENT = `#version 300 es
  precision highp float; uniform vec4 uColor; uniform float uMarker; out vec4 color;
  void main(){float a=uColor.a;
  if(uMarker>2.5){float r=distance(gl_PointCoord,vec2(.5))*2.;if(r>1.)discard;a*=(1.-r)*(1.-r);}
  else if(uMarker>0.5){float r=distance(gl_PointCoord,vec2(.5));if(r>.5)discard;if(uMarker>1.5&&r<.28)discard;}
  color=vec4(uColor.rgb,a);}`;
// Fixed-light surface shading for arbitrary geometry (archive body/plane).
export const SURFACE_VERTEX = `#version 300 es
  in vec3 aPosition;in vec3 aNormal;uniform mat4 uMvp;uniform float uZScale;out vec3 normal;
  void main(){normal=aNormal;gl_Position=uMvp*vec4(aPosition*vec3(1.,1.,uZScale),1.0);}`;
export const SURFACE_FRAGMENT = `#version 300 es
  precision highp float;in vec3 normal;uniform vec4 uColor;out vec4 color;
  void main(){float d=max(dot(normalize(normal),normalize(vec3(-.45,.65,.8))),0.);color=vec4(uColor.rgb*(.2+.8*d),uColor.a);}`;
// Shared GLSL: exponential-squared distance fog towards the background colour (uFog = density,
// 0 disables) applied to both colour and, for translucent helpers, alpha.
const FOG_GLSL = `uniform float uFog;uniform vec3 uFogColor;
  float fogAmount(float depth){float f=depth*uFog;return uFog>0.?1.-exp(-f*f):0.;}`;
// Screen-space ribbon: uWidth is the half-width in device pixels. aTime is an optional per-vertex
// epoch; with uTrail > 0 the ribbon fades from full alpha (recent) to uBaseAlpha (older than
// uTrail before uEpoch). The edge profile can be shaded like a lit tube (uShade = 1) and the
// ribbon fades with depth when fog is enabled (orbit ribbons).
export const RIBBON_VERTEX = `#version 300 es
  in vec3 aPosition;in vec3 aPrevious;in vec3 aNext;in float aSide;in float aTime;
  uniform mat4 uMvp;uniform vec2 uViewport;uniform float uWidth;uniform float uZScale;out float edge;out float vTime;out float vDepth;
  void main(){vec3 s=vec3(1.,1.,uZScale);vec4 c=uMvp*vec4(aPosition*s,1.);vec4 p=uMvp*vec4(aPrevious*s,1.);vec4 n=uMvp*vec4(aNext*s,1.);
  vec2 d=(n.xy/max(n.w,.001)-p.xy/max(p.w,.001))*uViewport;vec2 dir=length(d)>1e-6?normalize(d):vec2(1.,0.);vec2 normal=vec2(-dir.y,dir.x);
  c.xy+=normal*aSide*uWidth/uViewport*c.w;gl_Position=c;edge=aSide;vTime=aTime;vDepth=c.w;}`;
export const RIBBON_FRAGMENT = `#version 300 es
  precision highp float;in float edge;in float vTime;in float vDepth;uniform vec4 uColor;uniform float uEpoch;uniform float uTrail;uniform float uBaseAlpha;uniform float uShade;
  ${FOG_GLSL}out vec4 color;
  void main(){float aa=1.-smoothstep(.72,1.,abs(edge));float tube=mix(1.,.55+.6*sqrt(max(0.,1.-edge*edge)),uShade);
  float trail=uTrail>0.?clamp(1.-(uEpoch-vTime)/uTrail,0.,1.):1.;float fog=fogAmount(vDepth);
  float a=uColor.a*mix(uBaseAlpha,1.,trail)*aa*(1.-.85*fog);
  color=vec4(mix(uColor.rgb*tube,uFogColor,fog),a);}`;
// Lit tube along an archived arc: aPosition is the sample (axis) position, aNormal the radial
// unit normal, displaced by uRadius after vertical exaggeration so the cross-section stays round.
// Lambert + Blinn-Phong from the Sun at uLight, ambient from the sky, time trail as the ribbon.
export const TUBE_VERTEX = `#version 300 es
  in vec3 aPosition;in vec3 aNormal;in float aTime;uniform mat4 uMvp;uniform float uZScale;uniform float uRadius;
  out vec3 vNormal;out vec3 vWorld;out float vTime;out float vDepth;
  void main(){vec3 world=aPosition*vec3(1.,1.,uZScale)+aNormal*uRadius;vNormal=aNormal;vWorld=world;vTime=aTime;
  gl_Position=uMvp*vec4(world,1.);vDepth=gl_Position.w;}`;
export const TUBE_FRAGMENT = `#version 300 es
  precision highp float;in vec3 vNormal;in vec3 vWorld;in float vTime;in float vDepth;
  uniform vec4 uColor;uniform vec3 uEye;uniform vec3 uLight;uniform vec3 uAmbient;uniform float uEpoch;uniform float uTrail;uniform float uBaseAlpha;uniform float uGlow;
  ${FOG_GLSL}out vec4 color;
  void main(){vec3 n=normalize(vNormal);vec3 l=normalize(uLight-vWorld);vec3 v=normalize(uEye-vWorld);
  float diff=max(dot(n,l),0.);vec3 h=normalize(l+v);float spec=pow(max(dot(n,h),0.),30.)*.4;
  float trail=uTrail>0.?clamp(1.-(uEpoch-vTime)/uTrail,0.,1.):1.;
  vec3 lit=uColor.rgb*(uAmbient+vec3(.85*diff))+vec3(spec)*(.4+.6*trail);
  vec3 rgb=mix(lit,uColor.rgb*1.15,uGlow*trail);float fog=fogAmount(vDepth);
  color=vec4(mix(rgb,uFogColor,fog),uColor.a*mix(uBaseAlpha,1.,trail)*(1.-.5*fog));}`;
// Instanced lit bodies: one unit sphere mesh (aNormal doubles as the position), per-instance
// centre / radius / colour / emissive attributes (divisor 1). Lit by a point light at uLight (the
// Sun) with Blinn-Phong specular and a soft sky ambient; uEmissive blends towards a limb-darkened
// self-luminous look (1 = the Sun, ~0.8 = a mined asteroid flagged bright).
export const BODY_VERTEX = `#version 300 es
  in vec3 aNormal;in vec3 aCenter;in float aRadius;in vec4 aColor;in float aEmissive;uniform mat4 uMvp;uniform float uZScale;
  out vec3 vNormal;out vec3 vWorld;out vec4 vColor;out float vEmissive;out float vDepth;
  void main(){vec3 world=aCenter*vec3(1.,1.,uZScale)+aNormal*aRadius;vNormal=aNormal;vWorld=world;vColor=aColor;vEmissive=aEmissive;
  gl_Position=uMvp*vec4(world,1.);vDepth=gl_Position.w;}`;
export const BODY_FRAGMENT = `#version 300 es
  precision highp float;in vec3 vNormal;in vec3 vWorld;in vec4 vColor;in float vEmissive;in float vDepth;
  uniform vec3 uEye;uniform vec3 uLight;uniform vec3 uAmbient;${FOG_GLSL}out vec4 color;
  void main(){vec3 n=normalize(vNormal);vec3 l=normalize(uLight-vWorld);vec3 v=normalize(uEye-vWorld);
  float diff=max(dot(n,l),0.);vec3 h=normalize(l+v);float spec=pow(max(dot(n,h),0.),48.)*.45*step(.01,diff);
  float rim=pow(1.-max(dot(n,v),0.),3.)*.25;
  vec3 lit=vColor.rgb*(uAmbient+vec3(.95*diff)+vec3(rim))+vec3(spec);float limb=.58+.42*pow(max(dot(n,v),0.),.6);vec3 glow=vColor.rgb*limb*1.25;
  float fog=fogAmount(vDepth)*(1.-vEmissive*.7);color=vec4(mix(mix(lit,glow,vEmissive),uFogColor,fog),vColor.a);}`;
// Procedural stars: point sprites sized and brightened by magnitude, with a soft radial profile.
export const STAR_VERTEX = `#version 300 es
  in vec3 aPosition;in float aMagnitude;uniform mat4 uMvp;uniform float uScale;out float vMag;
  void main(){vMag=aMagnitude;gl_Position=uMvp*vec4(aPosition,1.);gl_PointSize=(1.+2.6*aMagnitude)*uScale;}`;
export const STAR_FRAGMENT = `#version 300 es
  precision highp float;in float vMag;uniform float uAlpha;out vec4 color;
  void main(){float r=distance(gl_PointCoord,vec2(.5))*2.;if(r>1.)discard;float a=(1.-smoothstep(.2,1.,r))*(.35+.65*vMag)*uAlpha;
  color=vec4(mix(vec3(.74,.82,1.),vec3(1.,.97,.9),vMag),a);}`;
// Procedural sky: a full-screen triangle shaded with a vertical gradient, a faint cool band and a
// vignette; it provides the soft ambient look behind the star field (no textures).
export const BACKGROUND_VERTEX = `#version 300 es
  const vec2 corners[3]=vec2[3](vec2(-1.,-1.),vec2(3.,-1.),vec2(-1.,3.));out vec2 vUv;
  void main(){vec2 c=corners[gl_VertexID];vUv=c*.5+.5;gl_Position=vec4(c,0.,1.);}`;
export const BACKGROUND_FRAGMENT = `#version 300 es
  precision highp float;in vec2 vUv;uniform vec3 uTop;uniform vec3 uBottom;uniform float uBand;out vec4 color;
  float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
  void main(){vec3 c=mix(uBottom,uTop,smoothstep(0.,1.,vUv.y));
  float band=exp(-pow((vUv.y-.42+.12*sin(vUv.x*3.1))*3.4,2.))*uBand;c+=vec3(.05,.07,.12)*band;
  float vignette=1.-.55*pow(length(vUv-.5)*1.15,2.2);c*=vignette;
  c+=(hash(vUv*vec2(1917.,1043.))-.5)*(1./255.);color=vec4(c,1.);}`;

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

/** Base class owning GL resources and the three core programs (line, surface, ribbon). */
export class GlResources {
  constructor(canvasElement) {
    this.canvas = canvasElement; this.buffers = []; this.programs = [];
    this.gl = createContext(canvasElement);
    this.lineProgram = this.makeProgram(LINE_VERTEX, LINE_FRAGMENT);
    this.surfaceProgram = this.makeProgram(SURFACE_VERTEX, SURFACE_FRAGMENT);
    this.ribbonProgram = this.makeProgram(RIBBON_VERTEX, RIBBON_FRAGMENT);
    this.info = contextInfo(this.gl);
    /** Vertical exaggeration applied by every program this frame (1 = physical). */
    this.zScale = 1;
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
    if (location < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    gl.vertexAttribDivisor(location, 0);
  }
  /** Give an attribute a constant value (used when a ribbon has no per-vertex time). */
  constantAttribute(program, name, value) {
    const gl = this.gl, location = gl.getAttribLocation(program, name);
    if (location < 0) return;
    gl.disableVertexAttribArray(location); gl.vertexAttribDivisor(location, 0); gl.vertexAttrib1f(location, value);
  }
  /** Per-instance attribute (divisor 1) read from an interleaved float buffer. */
  instancedAttribute(program, name, buffer, size, strideFloats, offsetFloats) {
    const gl = this.gl, location = gl.getAttribLocation(program, name);
    if (location < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, strideFloats * 4, offsetFloats * 4); gl.vertexAttribDivisor(location, 1);
  }
  makeIndexBuffer(data) {
    const value = this.gl.createBuffer(); if (!value) throw new Error("Unable to allocate WebGL index buffer");
    this.buffers.push(value); this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER, value); this.gl.bufferData(this.gl.ELEMENT_ARRAY_BUFFER, data, this.gl.STATIC_DRAW); return value;
  }
  fogUniforms(program, options) {
    const gl = this.gl;
    gl.uniform1f(this.uniform(program, "uFog"), options.fog ?? 0);
    gl.uniform3fv(this.uniform(program, "uFogColor"), options.fogColor ?? [0.012, 0.018, 0.04]);
  }
  uniform(program, name) { return this.gl.getUniformLocation(program, name); }
  line(mvp, buffer, mode, count, color, size = 1, marker = 0, first = 0) {
    if (count <= 0) return;
    const gl = this.gl, p = this.lineProgram; gl.useProgram(p); this.attribute(p, "aPosition", buffer, 3);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform4fv(this.uniform(p, "uColor"), color);
    gl.uniform1f(this.uniform(p, "uSize"), size); gl.uniform1f(this.uniform(p, "uMarker"), marker);
    gl.uniform1f(this.uniform(p, "uZScale"), this.zScale); gl.drawArrays(mode, first, count);
  }
  surface(mvp, positions, normals, count, color) {
    const gl = this.gl, p = this.surfaceProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", positions, 3); this.attribute(p, "aNormal", normals, 3);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform4fv(this.uniform(p, "uColor"), color);
    gl.uniform1f(this.uniform(p, "uZScale"), this.zScale); gl.drawArrays(gl.TRIANGLES, 0, count);
  }
  /**
   * Screen-space ribbon. `options`: first (vertex offset), epoch/trail/baseAlpha (time fade when
   * `buffers.times` exists), shade (1 = tube shading, 0 = flat).
   */
  ribbon(mvp, buffers, vertexCount, color, width = 3, options = {}) {
    if (vertexCount <= 0) return;
    const gl = this.gl, p = this.ribbonProgram; gl.useProgram(p);
    this.attribute(p, "aPosition", buffers.positions, 3); this.attribute(p, "aPrevious", buffers.previous, 3);
    this.attribute(p, "aNext", buffers.next, 3); this.attribute(p, "aSide", buffers.sides, 1);
    if (buffers.times) this.attribute(p, "aTime", buffers.times, 1); else this.constantAttribute(p, "aTime", 0);
    gl.uniformMatrix4fv(this.uniform(p, "uMvp"), false, mvp); gl.uniform2f(this.uniform(p, "uViewport"), this.canvas.width, this.canvas.height);
    gl.uniform1f(this.uniform(p, "uWidth"), width); gl.uniform4fv(this.uniform(p, "uColor"), color);
    gl.uniform1f(this.uniform(p, "uZScale"), this.zScale);
    gl.uniform1f(this.uniform(p, "uEpoch"), options.epoch ?? 0); gl.uniform1f(this.uniform(p, "uTrail"), buffers.times ? options.trail ?? 0 : 0);
    gl.uniform1f(this.uniform(p, "uBaseAlpha"), options.baseAlpha ?? 1); gl.uniform1f(this.uniform(p, "uShade"), options.shade ?? 1);
    this.fogUniforms(p, options);
    gl.drawArrays(gl.TRIANGLE_STRIP, options.first ?? 0, vertexCount);
  }
  dispose() {
    const gl = this.gl;
    // Detach every vertex attribute first so a later renderer on the same context never draws
    // with an enabled attribute that still points at a deleted buffer.
    for (let location = 0; location < this.info.maxVertexAttribs; location += 1) { gl.disableVertexAttribArray(location); gl.vertexAttribDivisor(location, 0); }
    gl.bindBuffer(gl.ARRAY_BUFFER, null); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); gl.useProgram(null);
    for (const buffer of this.buffers) gl.deleteBuffer(buffer);
    for (const program of this.programs) gl.deleteProgram(program);
    this.buffers = []; this.programs = [];
  }
}
