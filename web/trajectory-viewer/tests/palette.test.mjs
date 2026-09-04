import test from "node:test";
import assert from "node:assert/strict";
import { DENSE_SHIP_THRESHOLD, MAX_SHIPS, SHIP_COLOURS, isDenseFleet, shipColour, shipColourClass } from "../gtoc12.js";
import {
  PALETTE_FLOORS, SHIP_PALETTE_SPEC, bandHues, deltaE, generateShipPalette, hexToLab, hexToRgb255, isInGamut, linearSrgbToHex,
  maxChannelDifference, maxInGamutChroma, oklchToLinearSrgb, paletteMetrics,
} from "../scripts/palette.mjs";

test("colour maths: sRGB grey axis, CIELAB white/black and CIE76 distances", () => {
  assert.deepEqual(hexToRgb255("#ff8000"), [255, 128, 0]);
  const white = hexToLab("#ffffff"), black = hexToLab("#000000");
  assert.ok(Math.abs(white[0] - 100) < 0.05 && Math.abs(white[1]) < 0.05 && Math.abs(white[2]) < 0.05, `white is L* 100 (${white})`);
  assert.deepEqual(black, [0, 0, 0]);
  assert.ok(Math.abs(deltaE("#ffffff", "#000000") - 100) < 0.05, "white to black is 100 dE");
  assert.equal(deltaE("#15baf2", "#15baf2"), 0);
  assert.ok(deltaE("#ff0000", "#00ff00") > 100, "red and green are far apart");
  // OKLCH at zero chroma is a grey whose linear channels are equal; encoding it round-trips.
  const grey = oklchToLinearSrgb(0.7, 0, 123);
  assert.ok(Math.abs(grey[0] - grey[1]) < 1e-9 && Math.abs(grey[1] - grey[2]) < 1e-9);
  assert.match(linearSrgbToHex(grey), /^#([0-9a-f]{2})\1\1$/);
  assert.ok(isInGamut([0, 0.5, 1]) && !isInGamut([0, 1.01, 0.5]) && !isInGamut([-0.01, 0, 0]));
});

test("max in-gamut chroma is on the gamut boundary and larger for saturated hues than for pastel blue", () => {
  for (const [L, h] of [[0.74, 34], [0.74, 285], [0.62, 12], [0.62, 264]]) {
    const chroma = maxInGamutChroma(L, h);
    assert.ok(isInGamut(oklchToLinearSrgb(L, chroma, h)), `C=${chroma} at L=${L} h=${h} inside sRGB`);
    assert.ok(!isInGamut(oklchToLinearSrgb(L, chroma + 1e-6, h)), `C=${chroma} at L=${L} h=${h} is maximal`);
  }
  assert.ok(maxInGamutChroma(0.62, 12) > maxInGamutChroma(0.74, 285), "sRGB allows more chroma for a mid-light red than a light violet");
});

test("each band places 20 hues on allowed, monotone hue slots at max chroma", () => {
  for (const band of SHIP_PALETTE_SPEC.bands) {
    const { picks, allowedFraction, arcLength } = bandHues(band, SHIP_PALETTE_SPEC);
    assert.equal(picks.length, SHIP_PALETTE_SPEC.hues);
    assert.ok(allowedFraction > 0.7 && allowedFraction <= 1, `reserved colours forbid < 30% of the hue circle (${allowedFraction})`);
    assert.ok(arcLength > 300, `the max-chroma curve is long enough for 20 distinct hues (${arcLength})`);
    const unwrapped = picks.map((pick) => (pick.hue - SHIP_PALETTE_SPEC.huePhaseDeg + 360) % 360);
    assert.ok(unwrapped.every((hue, index) => index === 0 || hue > unwrapped[index - 1]), "hue slots are in increasing hue order");
    for (const pick of picks) {
      assert.equal(pick.lightness, band.lightness);
      assert.ok(pick.chroma <= band.chromaCap + 1e-12 && pick.chroma > 0.09, `chroma ${pick.chroma} within (0.09, cap]`);
      assert.ok(isInGamut(oklchToLinearSrgb(pick.lightness, pick.chroma, pick.hue)));
      const hex = linearSrgbToHex(oklchToLinearSrgb(pick.lightness, pick.chroma, pick.hue));
      for (const reserved of Object.values(SHIP_PALETTE_SPEC.reserved)) {
        assert.ok(deltaE(hex, reserved) >= SHIP_PALETTE_SPEC.reservedMinDeltaE - 1, `${hex} keeps clear of reserved ${reserved}`);
      }
    }
  }
});

test("the committed SHIP_COLOURS are the spec's output and satisfy the distinctness floors", () => {
  const regenerated = generateShipPalette(SHIP_PALETTE_SPEC);
  assert.equal(regenerated.length, 40);
  assert.equal(SHIP_COLOURS.length, 40);
  assert.equal(MAX_SHIPS, 40);
  assert.ok(maxChannelDifference(SHIP_COLOURS, regenerated) <= 2, "gtoc12.js SHIP_COLOURS regenerate from the spec");
  assert.equal(new Set(SHIP_COLOURS).size, 40, "all distinct");
  const metrics = paletteMetrics(SHIP_COLOURS);
  assert.ok(metrics.neighbour >= PALETTE_FLOORS.neighbour, `consecutive ships >= ${PALETTE_FLOORS.neighbour} dE (${metrics.neighbour})`);
  assert.ok(metrics.pairwise >= PALETTE_FLOORS.pairwise, `all pairs >= ${PALETTE_FLOORS.pairwise} dE (${metrics.pairwise})`);
  assert.ok(metrics.reserved >= PALETTE_FLOORS.reserved, `reserved colours >= ${PALETTE_FLOORS.reserved} dE (${metrics.reserved})`);
  // Ship order interleaves the bands: odd ships light, even ships dark.
  for (let index = 0; index < 40; index += 1) {
    const L = hexToLab(SHIP_COLOURS[index])[0];
    if (index % 2 === 0) assert.ok(L > 65, `ship ${index + 1} is in the light band (L* ${L})`);
    else assert.ok(L < 65 && L > 45, `ship ${index + 1} is in the dark band (L* ${L})`);
  }
  const detailed = generateShipPalette(SHIP_PALETTE_SPEC, true);
  const slots = detailed.map((entry) => `${entry.band}/${entry.slot}`);
  assert.equal(new Set(slots).size, 40, "every (band, hue slot) pair is used exactly once");
  assert.ok(detailed.every((entry, index) => entry.band === index % 2), "bands alternate along ship order");
});

test("ship colour helpers cover 40 ships without wrapping and switch layouts above 20", () => {
  const classes = new Set(), colours = new Set();
  for (let index = 0; index < MAX_SHIPS; index += 1) {
    classes.add(shipColourClass(index));
    colours.add(JSON.stringify(shipColour(index)));
  }
  assert.equal(classes.size, 40, "40 distinct CSS classes");
  assert.equal(colours.size, 40, "40 distinct GL colours");
  assert.equal(shipColourClass(0), "ship-colour-1");
  assert.equal(shipColourClass(20), "ship-colour-21", "ship 21 does not reuse ship 1's class");
  assert.equal(shipColourClass(39), "ship-colour-40");
  assert.equal(shipColourClass(40), "ship-colour-1", "wraps only beyond MAX_SHIPS (refused by check.mjs)");
  assert.deepEqual(shipColour(0).map((value) => Math.round(value * 255)), [...hexToRgb255(SHIP_COLOURS[0]), 255]);
  assert.equal(DENSE_SHIP_THRESHOLD, 20);
  assert.equal(isDenseFleet(20), false);
  assert.equal(isDenseFleet(21), true);
  assert.equal(isDenseFleet(39), true);
});
