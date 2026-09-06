// Ship palette generator and colour-difference helpers for the GTOC12 fleet view.
//
//   node scripts/palette.mjs            # print the 40 colours (JS / CSS / Python) and the metrics
//   node scripts/palette.mjs --json     # machine-readable palette + metrics
//
// `SHIP_COLOURS` in gtoc12.js, the `.ship-colour-N` classes in styles.css and `SHIP_COLOURS` in
// scripts/plot_gtoc12_fleet.py are the committed output of `generateShipPalette(SHIP_PALETTE_SPEC)`;
// scripts/check.mjs regenerates the palette from the spec and asserts the committed copies match it
// (per channel, within rounding) and satisfy the stated colour-difference floors.
//
// Method (deterministic, no randomness):
//   1. Two OKLCH lightness bands (light 0.74, dark 0.62). For each band the hue circle is sampled
//      at 0.2 deg; every sample takes the largest sRGB-in-gamut chroma at that lightness and hue
//      (bisection), scaled by 0.98 to stay inside the gamut after 8-bit rounding and capped per band.
//   2. Samples closer than `reservedMinDeltaE` (CIE76 dE*ab) to any reserved scene/UI colour (Sun,
//      Earth, pending asteroid, launch ring, Earth-return ring, verified/caution/alert/focus/bone
//      from styles.css) are forbidden so no ship reads as one of those.
//   3. Twenty hues per band are placed at equal CIELAB arc length along the allowed part of the
//      max-chroma curve (the perceptual step between adjacent hues of a band is uniform; sRGB's
//      narrow blue region therefore gets wider hue spacing than yellow-green).
//   4. Ship order interleaves the bands and strides through the hue slots: ship k (0-based) uses
//      band k mod 2 and hue slot (13 * floor(k / 2) + 7 * band) mod 20, so consecutive ships differ
//      in lightness and by >= 5 hue slots; neighbours in ship order are >= 25 dE apart (actual >= 60).
// Colour difference is CIE76 dE*ab in CIELAB (D65), computed from the rounded sRGB hex values.

import { pathToFileURL } from "node:url";

export const SHIP_PALETTE_SPEC = Object.freeze({
  hues: 20,
  bands: Object.freeze([
    Object.freeze({ lightness: 0.74, chromaCap: 0.22 }), // light band: even ship indices (ships 1, 3, 5, ...)
    Object.freeze({ lightness: 0.62, chromaCap: 0.26 }), // dark band: odd ship indices (ships 2, 4, 6, ...)
  ]),
  hueSamples: 1800,
  huePhaseDeg: 6,
  chromaMargin: 0.98,
  reservedMinDeltaE: 14,
  reserved: Object.freeze({
    sun: "#ffdb73", // SUN_COLOUR in gtoc12.js and .legend-mark.sun
    earth: "#6badff", // EARTH_COLOUR in gtoc12.js and .legend-line.earth
    asteroidPending: "#808fad", // PENDING_ASTEROID in gtoc12.js
    launch: "#45ff9c", // ROLE_STYLE.launch ring
    earthReturn: "#ffba4f", // ROLE_STYLE["earth-return"] ring
    verified: "#5fd3a0", // styles.css --verified
    caution: "#f1b866", // styles.css --caution
    alert: "#ff6f80", // styles.css --alert
    focus: "#8fc6ff", // styles.css --focus
    bone: "#ebe8e1", // styles.css --bone
  }),
  order: Object.freeze({ stride: 13, offset: 7 }),
});

/** Floors asserted by scripts/check.mjs on the committed palette (CIE76 dE*ab). */
export const PALETTE_FLOORS = Object.freeze({ neighbour: 25, pairwise: 14, reserved: 14 });

const clamp01 = (value) => Math.min(1, Math.max(0, value));

/** OKLab -> linear sRGB (Björn Ottosson's matrices). */
export function oklabToLinearSrgb(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ ** 3, m = m_ ** 3, s = s_ ** 3;
  return [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}
export function oklchToLinearSrgb(L, C, hueDeg) {
  const h = (hueDeg * Math.PI) / 180;
  return oklabToLinearSrgb(L, C * Math.cos(h), C * Math.sin(h));
}
const encodeGamma = (x) => (x <= 0.0031308 ? 12.92 * x : 1.055 * x ** (1 / 2.4) - 0.055);
const decodeGamma = (x) => (x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4);
export const isInGamut = (rgb) => rgb.every((value) => value >= -1e-6 && value <= 1 + 1e-6);

/** Largest OKLCH chroma at (L, hue) whose linear sRGB stays inside [0, 1] (40 bisection steps). */
export function maxInGamutChroma(L, hueDeg) {
  let low = 0, high = 0.5;
  for (let step = 0; step < 40; step += 1) {
    const mid = (low + high) / 2;
    if (isInGamut(oklchToLinearSrgb(L, mid, hueDeg))) low = mid; else high = mid;
  }
  return low;
}
export function linearSrgbToHex(rgb) {
  return `#${rgb.map((value) => Math.round(clamp01(encodeGamma(clamp01(value))) * 255).toString(16).padStart(2, "0")).join("")}`;
}
export function hexToRgb255(hex) {
  const match = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!match) throw new Error(`not a #rrggbb colour: ${hex}`);
  return [0, 2, 4].map((offset) => parseInt(match[1].slice(offset, offset + 2), 16));
}
export const hexToLinearSrgb = (hex) => hexToRgb255(hex).map((value) => decodeGamma(value / 255));

/** Linear sRGB (D65) -> CIELAB. */
export function linearSrgbToLab([r, g, b]) {
  const X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  const Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : t / (3 * (6 / 29) ** 2) + 4 / 29);
  const fx = f(X / 0.95047), fy = f(Y / 1), fz = f(Z / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}
export const hexToLab = (hex) => linearSrgbToLab(hexToLinearSrgb(hex));
export const labDistance = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
/** CIE76 colour difference dE*ab between two #rrggbb colours. */
export const deltaE = (hexA, hexB) => labDistance(hexToLab(hexA), hexToLab(hexB));

/** Twenty (L, C, h) picks for one band: equal CIELAB arc length along the allowed max-chroma curve. */
export function bandHues(band, spec = SHIP_PALETTE_SPEC) {
  const { hues, hueSamples, huePhaseDeg, chromaMargin, reservedMinDeltaE } = spec;
  const reserved = Object.values(spec.reserved).map(hexToLab);
  const samples = [];
  for (let index = 0; index < hueSamples; index += 1) {
    const hue = (huePhaseDeg + (360 * index) / hueSamples) % 360;
    const chroma = Math.min(band.chromaCap, chromaMargin * maxInGamutChroma(band.lightness, hue));
    const lab = linearSrgbToLab(oklchToLinearSrgb(band.lightness, chroma, hue));
    const allowed = reserved.every((colour) => labDistance(lab, colour) >= reservedMinDeltaE);
    samples.push({ hue, chroma, lab, allowed });
  }
  const cumulative = [0];
  for (let index = 1; index < hueSamples; index += 1) {
    const previous = samples[index - 1], current = samples[index];
    cumulative.push(cumulative[index - 1] + (previous.allowed && current.allowed ? labDistance(previous.lab, current.lab) : 0));
  }
  const total = cumulative[hueSamples - 1];
  const picks = [];
  let index = 0;
  for (let slot = 0; slot < hues; slot += 1) {
    const target = ((slot + 0.5) * total) / hues;
    while (index < hueSamples - 1 && cumulative[index + 1] < target) index += 1;
    let chosen = null;
    for (let distance = 0; chosen == null && distance < hueSamples; distance += 1) {
      for (const candidate of [index + distance, index - distance]) {
        if (candidate >= 0 && candidate < hueSamples && samples[candidate].allowed) { chosen = candidate; break; }
      }
    }
    picks.push({ lightness: band.lightness, chroma: samples[chosen].chroma, hue: samples[chosen].hue });
  }
  return { picks, allowedFraction: samples.filter((sample) => sample.allowed).length / hueSamples, arcLength: total };
}

/** The ship palette in ship order: `spec.hues * spec.bands.length` #rrggbb strings. */
export function generateShipPalette(spec = SHIP_PALETTE_SPEC, detailed = false) {
  const bands = spec.bands.map((band) => bandHues(band, spec));
  const colours = [];
  for (let ship = 0; ship < spec.hues * spec.bands.length; ship += 1) {
    const band = ship % spec.bands.length, row = Math.floor(ship / spec.bands.length);
    const slot = (row * spec.order.stride + band * spec.order.offset) % spec.hues;
    const pick = bands[band].picks[slot];
    const hex = linearSrgbToHex(oklchToLinearSrgb(pick.lightness, pick.chroma, pick.hue));
    colours.push(detailed ? { ship: ship + 1, band, slot, ...pick, hex } : hex);
  }
  return colours;
}

/** Minimum colour differences of a palette: consecutive ships, all pairs, and against reserved colours. */
export function paletteMetrics(colours, reserved = SHIP_PALETTE_SPEC.reserved) {
  const labs = colours.map(hexToLab);
  const reservedLabs = Object.entries(reserved).map(([name, hex]) => [name, hexToLab(hex)]);
  let neighbour = Infinity, neighbourPair = null, pairwise = Infinity, pairwisePair = null, nearestReserved = Infinity, reservedPair = null;
  for (let i = 0; i < labs.length; i += 1) {
    if (i + 1 < labs.length) {
      const distance = labDistance(labs[i], labs[i + 1]);
      if (distance < neighbour) { neighbour = distance; neighbourPair = [i + 1, i + 2]; }
    }
    for (let j = i + 1; j < labs.length; j += 1) {
      const distance = labDistance(labs[i], labs[j]);
      if (distance < pairwise) { pairwise = distance; pairwisePair = [i + 1, j + 1]; }
    }
    for (const [name, lab] of reservedLabs) {
      const distance = labDistance(labs[i], lab);
      if (distance < nearestReserved) { nearestReserved = distance; reservedPair = [i + 1, name]; }
    }
  }
  return { count: colours.length, distinct: new Set(colours).size, neighbour, neighbourPair, pairwise, pairwisePair, reserved: nearestReserved, reservedPair };
}

/** Largest per-channel 8-bit difference between two equally long colour lists (for the regeneration check). */
export function maxChannelDifference(a, b) {
  if (a.length !== b.length) return Infinity;
  let worst = 0;
  a.forEach((hex, index) => {
    const p = hexToRgb255(hex), q = hexToRgb255(b[index]);
    worst = Math.max(worst, ...p.map((value, channel) => Math.abs(value - q[channel])));
  });
  return worst;
}

export function formatJs(colours, perLine = 8) {
  const lines = [];
  for (let start = 0; start < colours.length; start += perLine) lines.push(`  ${colours.slice(start, start + perLine).map((hex) => `"${hex}"`).join(", ")},`);
  return lines.join("\n");
}
export function formatCss(colours, perLine = 3) {
  const lines = [];
  for (let start = 0; start < colours.length; start += perLine) {
    lines.push(colours.slice(start, start + perLine).map((hex, offset) => `.ship-colour-${start + offset + 1} { color: ${hex}; }`).join(" "));
  }
  return lines.join("\n");
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const detailed = generateShipPalette(SHIP_PALETTE_SPEC, true);
  const colours = detailed.map((entry) => entry.hex);
  const metrics = paletteMetrics(colours);
  if (process.argv.includes("--json")) {
    console.log(JSON.stringify({ spec: SHIP_PALETTE_SPEC, colours: detailed, metrics }, null, 2));
  } else {
    console.log("// gtoc12.js SHIP_COLOURS\n" + formatJs(colours));
    console.log("\n/* styles.css */\n" + formatCss(colours));
    console.log("\n# plot_gtoc12_fleet.py\n" + formatJs(colours));
    console.log("\n// per ship: band, hue slot, OKLCH");
    for (const entry of detailed) console.log(`${String(entry.ship).padStart(2)} ${entry.hex} band ${entry.band} slot ${String(entry.slot).padStart(2)} L ${entry.lightness.toFixed(2)} C ${entry.chroma.toFixed(3)} h ${entry.hue.toFixed(1)}`);
    console.log(`\nmetrics: ${metrics.count} colours, ${metrics.distinct} distinct; min neighbour dE ${metrics.neighbour.toFixed(1)} (ships ${metrics.neighbourPair}); min pairwise dE ${metrics.pairwise.toFixed(1)} (ships ${metrics.pairwisePair}); min dE to reserved ${metrics.reserved.toFixed(1)} (ship ${metrics.reservedPair[0]} vs ${metrics.reservedPair[1]})`);
  }
}
