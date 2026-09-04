const assert = require("node:assert/strict");
const { mkdir, writeFile } = require("node:fs/promises");
const { join, resolve } = require("node:path");

async function main() {
  const modulePath = process.env.PLAYWRIGHT_PATH;
  if (!modulePath) throw new Error("Set PLAYWRIGHT_PATH to an installed Playwright module");
  const { chromium } = require(modulePath);
  const root = resolve(__dirname, "..");
  // BROWSER_CHECK_ARTIFACTS redirects the screenshots/report (e.g. to /tmp) so a run against a
  // different imported fleet does not overwrite the committed artefact set.
  const artifacts = process.env.BROWSER_CHECK_ARTIFACTS ? resolve(process.env.BROWSER_CHECK_ARTIFACTS) : join(root, "test-artifacts");
  await mkdir(artifacts, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    args: ["--enable-webgl", "--enable-unsafe-swiftshader", "--use-angle=swiftshader"],
  });
  try {
    await run(browser, artifacts);
  } finally {
    await browser.close();
  }
}

async function run(browser, artifacts) {
  const errors = [];
  const requests = [];
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  page.on("console", (message) => { if (message.type() === "error") errors.push(`console: ${message.text()}`); });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  page.on("requestfailed", (request) => errors.push(`request: ${request.url()} ${request.failure()?.errorText}`));
  page.on("response", (response) => requests.push({ status: response.status(), url: response.url() }));

  await page.goto("http://127.0.0.1:4173/", { waitUntil: "networkidle" });
  await page.locator("#renderer-status-text").waitFor({ state: "visible" });
  assert.equal(await page.locator("#renderer-status-text").textContent(), "WebGL2 GPU renderer");
  assert.match(await page.locator("#gpu-details").textContent(), /WebGL 2/i);
  assert.match(await page.locator("#sample-output").textContent(), /251 \/ 251/);
  assert.equal(await page.locator("#error-banner").isHidden(), true);
  const canvasBox = await page.locator("#trajectory-canvas").boundingBox();
  assert.ok(canvasBox && canvasBox.width > 800 && canvasBox.height >= 500);

  await page.getByRole("button", { name: /P1-E/ }).click();
  assert.match(await page.locator("#sample-output").textContent(), /512 \/ 512/);
  assert.match(await page.locator("#scene-overlay").textContent(), /6,500 km constraint/);
  assert.match(await page.locator("#qualification-notice").textContent(), /Unqualified/i);
  await page.getByText("Transcription nodes", { exact: true }).click();
  assert.match(await page.locator("#sample-output").textContent(), /256 \/ 256/);
  await page.getByText("Dense replay", { exact: true }).click();

  await page.getByRole("button", { name: /^P2/ }).click();
  assert.match(await page.locator("#sample-output").textContent(), /512 \/ 512/);
  assert.match(await page.locator("#scene-overlay").textContent(), /Earth equatorial radius/);
  await page.locator("#timeline").fill("30");
  assert.match(await page.locator("#sample-output").textContent(), /Sample 154 \/ 512/);
  await page.locator("#play-button").click();
  await page.waitForTimeout(300);
  assert.ok(Number(await page.locator("#timeline").inputValue()) > 30);
  await page.locator("#play-button").click();

  const canvas = page.locator("#trajectory-canvas");
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 100, box.y + box.height / 2 + 40, { steps: 8 });
  await page.mouse.up();
  await page.mouse.wheel(0, -240);
  await canvas.focus();
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("Shift+ArrowUp");
  await page.keyboard.press("+");
  await page.screenshot({ path: join(artifacts, "desktop-p2-earth.png"), fullPage: true });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("button", { name: /P1-C/ }).click();
  assert.match(await page.locator("#sample-output").textContent(), /201 \/ 201/);
  assert.match(await page.locator("#scene-overlay").textContent(), /Z = 0/);
  assert.match(await page.locator("#qualification-notice").textContent(), /Unqualified/i);
  assert.ok((await page.locator("#trajectory-canvas").boundingBox()).width <= 370);
  await page.screenshot({ path: join(artifacts, "mobile-p1c-local-surface.png"), fullPage: true });

  assert.deepEqual(errors, []);
  assert.ok(requests.length >= 5);
  assert.ok(requests.every((entry) => entry.status < 400), JSON.stringify(requests));
  assert.ok(requests.every((entry) => entry.url.startsWith("http://127.0.0.1:4173/")));
  await context.close();

  const gtoc12 = await checkGtoc12(browser, artifacts);
  const report = {
    browser: await browser.version(),
    desktop: { viewport: "1440x1000", trajectory: "P2", replay_points: 512, interactions: ["switch", "timeline", "play", "drag", "wheel", "keyboard"] },
    mobile: { viewport: "390x844", trajectory: "P1-C", replay_points: 201 },
    gtoc12,
    requests,
    errors,
    webgl2: true,
  };
  await writeFile(join(artifacts, "browser-report.json"), `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify(report, null, 2));
}

const near = (a, b, tolerance) => Math.abs(a - b) <= tolerance;
const regexEscape = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * GTOC12 fleet dataset: graceful absence, then (when installed) the perspective 3D scene — camera
 * presets, vertical exaggeration, follow-ship, timeline playback with speed, flashes/counters,
 * picking — with screenshots. Expectations are derived from the installed fleet.json so the check
 * follows whichever verified fleet is imported.
 */
async function checkGtoc12(browser, artifacts) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on("console", (message) => { if (message.type() === "error" || message.type() === "warning") errors.push(`console ${message.type()}: ${message.text()}`); });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));
  const debug = (expression) => page.evaluate(`(() => { const d = window.viewerDebug; return ${expression}; })()`);
  const settle = () => page.evaluate(() => window.viewerDebug.settle());
  const shot = (name) => page.locator(".viewport-card").screenshot({ path: join(artifacts, name) });

  // 1. Dataset absent: the selector degrades gracefully and the archive keeps working.
  await page.route("**/data/gtoc12/**", (route) => route.fulfill({ status: 404, contentType: "text/plain", body: "Not found" }));
  await page.goto("http://127.0.0.1:4173/?dataset=gtoc12", { waitUntil: "networkidle" });
  assert.equal(await page.locator("#renderer-status-text").textContent(), "WebGL2 GPU renderer");
  assert.equal(await page.locator("#dataset-select").inputValue(), "archive");
  assert.equal(await page.locator('#dataset-select option[value="gtoc12"]').isDisabled(), true);
  assert.match(await page.locator("#dataset-help").textContent(), /not installed/i);
  assert.match(await page.locator("#sample-output").textContent(), /251 \/ 251/);
  assert.equal(await page.locator("#error-banner").isHidden(), true);
  await page.unroute("**/data/gtoc12/**");
  // The simulated absence produces exactly one console 404 for the probed manifest; nothing else may be logged.
  assert.deepEqual(errors.map((entry) => entry.replace(/^console error: /, "")), ["Failed to load resource: the server responded with a status of 404 (Not Found)"]);
  errors.length = 0;

  const manifestResponse = await page.request.get("http://127.0.0.1:4173/data/gtoc12/manifest.json");
  if (!manifestResponse.ok()) {
    assert.deepEqual(errors, []);
    await context.close();
    return { installed: false, note: "data/gtoc12 absent; run npm run import-gtoc12 to exercise the fleet view" };
  }
  const fleet = await (await page.request.get("http://127.0.0.1:4173/data/gtoc12/fleet.json")).json();
  const ships = fleet.ships.length, asteroids = fleet.asteroids.length;
  const massLabel = fleet.score.total_collected_kg.toFixed(2);
  const richest = fleet.ships.reduce((best, ship, index) => (ship.collected_kg > fleet.ships[best].collected_kg ? index : best), 0);
  const richShip = fleet.ships[richest];
  const collects = richShip.events.filter((event) => event.role === "collect");
  const midCollect = collects[Math.floor(collects.length / 2)];
  const followEpoch = midCollect.epoch_mjd + 20; // inside the flash window of that collect
  const firstDeploy = richShip.events.findIndex((event) => event.role === "deploy");

  // 2. Whole fleet, 30° oblique preset (default), end of mission, vertical exaggeration 6×.
  await page.goto("http://127.0.0.1:4173/?dataset=gtoc12", { waitUntil: "networkidle" });
  await page.locator("#trajectory-title").filter({ hasText: /ships/ }).waitFor();
  assert.equal(await page.locator("#renderer-status-text").textContent(), "WebGL2 GPU renderer");
  assert.equal(await page.locator("#dataset-select").inputValue(), "gtoc12");
  const title = await page.locator("#trajectory-title").textContent();
  assert.match(title, new RegExp(`${ships} ships, ${asteroids} asteroids, ${regexEscape(massLabel)} kg`));
  assert.equal(await page.locator("#ship-list button").count(), ships + 1);
  assert.equal(await page.locator("#legend-ships span").count(), ships, "one legend swatch per ship");
  assert.match(await page.locator("#mission-timeline-output").textContent(), /MJD 69807 · 2050-01-01/);
  assert.match(await page.locator("#fleet-summary").textContent(), new RegExp(`Delivered to Earth so far\\s*${regexEscape(fleet.score.total_collected_kg.toFixed(1))} kg`));
  assert.equal(await page.locator('#ship-list [data-counter="fleet"]').textContent(), fleet.score.total_collected_kg.toFixed(1), "fleet counter reaches the total at mission end");
  assert.match(await page.locator("#ship-detail").textContent(), /Mass per ship/);
  assert.match(await page.locator("#ship-detail").textContent(), new RegExp(`${regexEscape((fleet.score.total_collected_kg / ships).toFixed(1))} kg`));
  assert.match(await page.locator("#ship-detail").textContent(), /Official verifier\s*pass/);
  assert.match(await page.locator("#fleet-legend").textContent(), /segments connect exact archived samples — no interpolation/i);
  assert.match(await page.locator("#scene-overlay").textContent(), /connections between archived samples/);
  assert.equal(await page.locator("#qualification-badge").textContent(), "Verified fleet");
  assert.equal(await page.locator(".archive-only:visible").count(), 0, "archive panels hidden in fleet mode");
  // Timeline strip and ship rows: year ticks 2035..2050 under the mission clock; mass bars full at mission end.
  assert.equal(await page.locator("#mission-timeline-ticks span").count(), 16, "one tick per mission year");
  assert.match(await page.locator('#ship-list [data-bar="fleet"]').evaluate((element) => element.style.transform), /scaleX\(1\)/, "fleet mass bar is full at mission end");
  assert.equal(await page.locator("#ship-list .mass-bar").count(), ships + 1, "one mass bar per ship row");
  // Opens in real 3D: 30° oblique preset at 6x vertical exaggeration (labelled), full-bleed canvas.
  assert.equal(await debug("d.preset"), "oblique");
  assert.ok(near(await debug("d.camera.pitch"), Math.PI / 6, 1e-9), "oblique preset is 30 degrees");
  assert.equal(await debug("d.exaggeration"), 6, "opens at 6x vertical exaggeration");
  assert.equal(await page.locator("#exaggeration-output").textContent(), "6×");
  assert.match(await page.locator("#scene-overlay").textContent(), /Z exaggerated 6× \(not physical\)/);
  assert.match(await page.locator(".camera-panel").textContent(), /Vertical exaggeration — not physical/);
  const openingCanvas = await page.locator("#trajectory-canvas").boundingBox();
  assert.ok(openingCanvas.height >= 0.7 * 900, `canvas fills >= 70% of the window height (got ${openingCanvas.height}px of 900)`);
  const viewerColumn = await page.locator(".viewer-column").boundingBox();
  assert.ok(near(openingCanvas.width, viewerColumn.width - 2, 3), "canvas fills the main column");
  const glInfo = await debug("d.glInfo");
  assert.equal(glInfo.antialias, true, "MSAA/antialiasing requested on the WebGL2 context");
  assert.ok(glInfo.depthTest, "depth testing enabled");
  assert.ok(glInfo.instances >= ships + asteroids + 2, `one instanced sphere per body (${glInfo.instances})`);
  assert.equal(glInfo.tubeSides, 6, "ship arcs are tube meshes");
  await page.evaluate(() => window.scrollTo(0, 0));
  await shot("gtoc12-3d-oblique-fleet.png");
  await page.screenshot({ path: join(artifacts, "gtoc12-3d-desktop-window.png") });
  await page.screenshot({ path: join(artifacts, "gtoc12-3d-desktop-fullpage.png"), fullPage: true });

  // 2b. Ship palette and per-ship UI at 1440x900 and 1920x1080: every ship has its own colour in the
  // legend and the rail (no palette wrap), the > 20-ship fleets use the dense layouts, and neither
  // the rail list, its rows, the legend nor the toolbar overflow their boxes.
  const palette = await checkShipPalette(page, fleet, artifacts, "1440x900");
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.evaluate(() => window.scrollTo(0, 0));
  const paletteWide = await checkShipPalette(page, fleet, artifacts, "1920x1080");
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.evaluate(() => window.scrollTo(0, 0));

  // Mouse-wheel dolly towards the cursor: the distance shrinks and the target slides towards the pointed-at region.
  const before = await debug("d.camera");
  await page.mouse.move(openingCanvas.x + openingCanvas.width * 0.72, openingCanvas.y + openingCanvas.height * 0.35);
  await page.mouse.wheel(0, -600);
  await page.waitForTimeout(50);
  const after = await debug("d.camera");
  assert.ok(after.distance < before.distance * 0.7, `wheel zooms in (${before.distance} -> ${after.distance})`);
  assert.ok(Math.hypot(...after.target.map((value, axis) => value - before.target[axis])) > 0.05, "dolly moves the target towards the cursor");
  await page.locator("#fleet-reset-button").click(); await settle();
  assert.ok(near(await debug("d.camera.distance"), before.distance, 1e-6) && (await debug("d.camera.target")).every((value) => near(value, 0, 1e-9)), "reset view restores the Sun-centred oblique framing");

  // 3. Edge-on preset with a smooth transition: inclinations read as a band around the ecliptic.
  await page.locator('[data-preset="edge"]').click();
  assert.equal(await debug("d.transitioning"), true, "preset change animates");
  await page.locator("#exaggeration").fill("10");
  await settle();
  assert.equal(await debug("d.transitioning"), false);
  assert.equal(await debug("d.preset"), "edge");
  assert.ok(near(await debug("d.camera.pitch"), 1.5 * Math.PI / 180, 1e-9), "edge-on preset pitch");
  assert.equal(await page.locator('[data-preset="edge"]').getAttribute("aria-pressed"), "true");
  await shot("gtoc12-3d-edge-on.png");
  await page.locator('[data-preset="top"]').click(); await settle();
  assert.ok((await debug("d.camera.pitch")) > 1.5, "top-down preset looks straight down");

  // 4. Mid-mission timeline frame with flashes/counters (oblique, 4×).
  await page.locator('[data-preset="oblique"]').click(); await settle();
  await page.locator("#exaggeration").fill("4");
  await page.locator("#mission-timeline").fill("67000");
  assert.match(await page.locator("#mission-timeline-output").textContent(), /MJD 67000 · 2042-04-26 · T\+7\.32 yr/);
  const inFlight = fleet.ships.filter((ship) => ship.launch_epoch_mjd <= 67000 && ship.final_sample_epoch_mjd > 67000).length;
  assert.match(await page.locator("#fleet-summary").textContent(), new RegExp(`Ships in flight\\s*${inFlight} of ${ships}`));
  const collectedMid = fleet.ships.reduce((sum, ship) => sum + ship.events.filter((event) => event.role === "collect" && event.epoch_mjd <= 67000).reduce((inner, event) => inner + event.mass_delta_kg, 0), 0);
  assert.equal(await page.locator('#ship-list [data-counter="fleet"]').textContent(), collectedMid.toFixed(1), "running fleet counter matches archived collects");
  await shot("gtoc12-3d-timeline-mid-mission.png");

  // 5. Follow-ship close-up on the richest ship at a collect flash, then hover an event marker.
  await page.locator(`#ship-list button[data-ship="${richest}"]`).click();
  assert.match(await page.locator("#trajectory-title").textContent(), new RegExp(`^Ship ${richShip.ship_id} — ${regexEscape(richShip.collected_kg.toFixed(1))} kg from ${richShip.asteroids.length} asteroids`));
  assert.equal(await page.locator("#ship-detail tbody tr").count(), richShip.events.length);
  await page.locator("#mission-timeline").fill(String(followEpoch));
  await page.locator('[data-preset="follow"]').click(); await settle();
  assert.equal(await debug("d.following"), true);
  const canvasBox = await page.locator("#trajectory-canvas").boundingBox();
  const shipScreen = await debug(`d.shipScreenPosition(${richest})`);
  assert.ok(near(shipScreen[0], canvasBox.width / 2, 2) && near(shipScreen[1], canvasBox.height / 2, 2), `followed ship is centred: ${JSON.stringify(shipScreen)}`);
  await page.evaluate(() => window.scrollTo(0, 0));
  await shot("gtoc12-3d-follow-ship.png");
  // Playback keeps the followed ship centred while the epoch advances.
  await page.locator("#mission-play-button").click();
  await page.waitForTimeout(350);
  await page.locator("#mission-play-button").click();
  assert.ok(Number(await page.locator("#mission-timeline").inputValue()) > followEpoch);
  const followed = await debug(`d.shipScreenPosition(${richest})`);
  assert.ok(near(followed[0], canvasBox.width / 2, 2) && near(followed[1], canvasBox.height / 2, 2), "follow-ship tracks the archived samples during playback");

  // Frame the whole arc and hover its first deploy marker.
  await page.locator("#focus-ship-button").click(); await settle();
  assert.equal(await debug("d.following"), false, "framing the arc leaves follow mode");
  await page.locator("#mission-timeline").fill(String(richShip.return_epoch_mjd));
  // Filling sidebar controls can scroll the page; re-measure the canvas after scrolling back to the top.
  await page.evaluate(() => window.scrollTo(0, 0));
  Object.assign(canvasBox, await page.locator("#trajectory-canvas").boundingBox());
  const marker = await debug(`d.eventScreenPosition(${richest}, ${firstDeploy})`);
  assert.ok(marker && marker[0] > 0 && marker[1] > 0 && marker[0] < canvasBox.width && marker[1] < canvasBox.height, `first deploy marker inside the canvas: ${JSON.stringify(marker)}`);
  await page.mouse.move(canvasBox.x + marker[0], canvasBox.y + marker[1]);
  await page.waitForTimeout(100);
  const tooltip = await page.locator("#hover-tooltip").textContent();
  assert.match(tooltip, new RegExp(`Ship ${richShip.ship_id} · Deploy miner`));
  assert.match(tooltip, new RegExp(regexEscape(richShip.events[firstDeploy].body)));
  assert.equal(await debug("d.hover"), "event");
  await shot("gtoc12-3d-ship-arc-framed.png");
  // Hovering the ship's current marker highlights it (bigger, more emissive instance).
  const shipNow = await debug(`d.shipScreenPosition(${richest})`);
  if (shipNow && shipNow[0] > 0 && shipNow[1] > 0 && shipNow[0] < canvasBox.width && shipNow[1] < canvasBox.height) {
    await page.mouse.move(canvasBox.x + shipNow[0], canvasBox.y + shipNow[1]);
    await page.waitForTimeout(80);
    // At the return epoch the ship sits on its Earth-return event, so either pick names this ship.
    assert.ok(["ship", "event"].includes(await debug("d.hover")), "hovering the ship marker picks the ship (or its coincident event)");
    assert.match(await page.locator("#hover-tooltip").textContent(), new RegExp(`Ship ${richShip.ship_id}`));
    await page.mouse.move(canvasBox.x + marker[0], canvasBox.y + marker[1]);
    await page.waitForTimeout(80);
  }
  await page.mouse.click(canvasBox.x + marker[0], canvasBox.y + marker[1]);
  assert.equal(await debug("d.selectedShip"), richest);

  // 6. Drag with release inertia, then playback speed control.
  await page.mouse.move(canvasBox.x + canvasBox.width * 0.3, canvasBox.y + canvasBox.height * 0.6);
  await page.mouse.down();
  await page.mouse.move(canvasBox.x + canvasBox.width * 0.6, canvasBox.y + canvasBox.height * 0.55, { steps: 4 });
  await page.mouse.up();
  assert.equal(await debug("d.transitioning"), true, "a quick drag releases with inertia");
  await settle();
  await page.locator("#mission-timeline").fill("64328");
  await page.selectOption("#speed-select", "4");
  await page.locator("#mission-play-button").click();
  await page.waitForTimeout(600);
  await page.locator("#mission-play-button").click();
  const fast = Number(await page.locator("#mission-timeline").inputValue()) - 64328;
  // Frame deltas are clamped to 100 ms, so software rendering under load still advances at least ~0.5 yr here.
  assert.ok(fast > 150, `4 yr/s advances well over a season in 0.6 s (got ${fast} days)`);

  // 7. Ten animated preview frames (whole fleet, oblique, 6×) spanning the timeline, for the GIF built by scripts/build_gif.py.
  await page.locator('#ship-list button[data-ship="all"]').click();
  await page.locator('[data-preset="oblique"]').click(); await settle();
  await page.locator("#exaggeration").fill("6");
  const frameEpochs = [64500, 65090, 65680, 66270, 66860, 67450, 68040, 68630, 69220, 69807];
  for (const [index, epoch] of frameEpochs.entries()) {
    await page.locator("#mission-timeline").fill(String(epoch));
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot(`gtoc12-3d-frame-${String(index + 1).padStart(2, "0")}.png`);
  }

  // 8. Switching back to the archive restores it; then mobile width.
  await page.selectOption("#dataset-select", "archive");
  assert.match(await page.locator("#sample-output").textContent(), /251 \/ 251/);
  assert.equal(await page.locator(".fleet-only:visible").count(), 0, "fleet panels hidden in archive mode");
  await page.selectOption("#dataset-select", "gtoc12");
  await page.locator("#trajectory-title").filter({ hasText: /ships/ }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#mission-timeline").fill("67500");
  await page.locator("#exaggeration").fill("5");
  await page.evaluate(() => window.scrollTo(0, 0));
  assert.ok((await page.locator("#trajectory-canvas").boundingBox()).width <= 370);
  assert.equal(await page.locator("#error-banner").isHidden(), true);
  await page.screenshot({ path: join(artifacts, "gtoc12-3d-mobile.png"), fullPage: true });

  assert.deepEqual(errors, []);
  await context.close();
  return {
    installed: true, title, run_id: fleet.run_id, ships, asteroids, collected_kg: Number(massLabel), official_kg: fleet.score.official_total_mass_kg,
    followed_ship: richShip.ship_id, follow_epoch: followEpoch,
    screenshots: [
      "gtoc12-3d-oblique-fleet.png", "gtoc12-3d-desktop-window.png", "gtoc12-3d-desktop-fullpage.png", "gtoc12-3d-edge-on.png", "gtoc12-3d-timeline-mid-mission.png",
      "gtoc12-3d-follow-ship.png", "gtoc12-3d-ship-arc-framed.png", ...frameEpochs.map((_, index) => `gtoc12-3d-frame-${String(index + 1).padStart(2, "0")}.png`), "gtoc12-3d-mobile.png",
      ...palette.screenshots, ...paletteWide.screenshots,
    ],
    palette: { "1440x900": palette, "1920x1080": paletteWide },
    gl: glInfo,
    interactions: ["absent-dataset-degrade", "presets", "exaggeration-default-6x", "full-bleed", "wheel-dolly-to-cursor", "transition", "timeline", "counters", "follow", "play", "frame-arc", "hover-tooltip", "hover-highlight", "click-select", "inertia", "speed", "dataset-switch", "mobile", "palette-distinct", "dense-layout", "no-overflow-1440x900", "no-overflow-1920x1080"],
  };
}

/**
 * Palette / layout contract for the loaded fleet at the current viewport: one distinct swatch colour
 * per ship in the legend and the rail (legend and rail agree), the dense layouts engage above 20
 * ships, nothing overflows, and the legend stays inside the porthole. Writes viewer40-*-<size>.png.
 */
async function checkShipPalette(page, fleet, artifacts, size) {
  const ships = fleet.ships.length, dense = ships > 20;
  const legendColours = await page.locator("#legend-ships .ship-swatch").evaluateAll((nodes) => nodes.map((node) => getComputedStyle(node).backgroundColor));
  const railColours = await page.locator('#ship-list .ship-item:not([data-ship="all"]) .ship-swatch').evaluateAll((nodes) => nodes.map((node) => getComputedStyle(node).backgroundColor));
  assert.equal(legendColours.length, ships, `${size}: one legend swatch per ship`);
  assert.deepEqual(railColours, legendColours, `${size}: rail swatches carry the legend colours in ship order`);
  assert.equal(new Set(legendColours).size, ships, `${size}: no two ships share a colour (${ships} ships)`);
  const barColours = await page.locator('#ship-list .ship-item:not([data-ship="all"]) .mass-bar').evaluateAll((nodes) => nodes.map((node) => getComputedStyle(node).color));
  assert.deepEqual(barColours, legendColours, `${size}: each mass bar is painted in its ship's colour`);
  assert.equal(await page.locator("#ship-list").evaluate((node) => node.classList.contains("dense")), dense, `${size}: dense rail layout iff > 20 ships`);
  assert.equal(await page.locator("#legend-ships").evaluate((node) => node.classList.contains("dense")), dense, `${size}: dense legend layout iff > 20 ships`);
  const overflow = await page.evaluate(() => {
    const box = (selector) => document.querySelector(selector).getBoundingClientRect();
    const horizontal = (node) => node.scrollWidth - node.clientWidth;
    const list = document.querySelector("#ship-list");
    const rows = [...list.querySelectorAll(".ship-item")];
    const masses = [...list.querySelectorAll(".ship-mass")];
    const names = [...list.querySelectorAll(".ship-name")];
    const legend = document.querySelector("#fleet-legend"), legendShips = document.querySelector("#legend-ships"), canvas = box("#trajectory-canvas"), legendBox = legend.getBoundingClientRect();
    return {
      listHorizontal: horizontal(list), rowHorizontal: Math.max(...rows.map(horizontal)), massClipped: Math.max(...masses.map(horizontal)), nameClipped: Math.max(...names.map(horizontal)),
      legendHorizontal: horizontal(legendShips), legendRows: Math.round(legendShips.getBoundingClientRect().height / 14),
      legendInsideCanvas: legendBox.left >= canvas.left && legendBox.right <= canvas.right && legendBox.top >= canvas.top && legendBox.bottom <= canvas.bottom,
      legendHeightFraction: legendBox.height / canvas.height,
      toolbarHorizontal: horizontal(document.querySelector(".viewport-toolbar")), pageHorizontal: document.documentElement.scrollWidth - document.documentElement.clientWidth,
      listVisibleRows: rows.filter((row) => { const r = row.getBoundingClientRect(), l = list.getBoundingClientRect(); return r.top >= l.top - 1 && r.bottom <= l.bottom + 1; }).length,
    };
  });
  assert.equal(overflow.listHorizontal, 0, `${size}: ship list has no horizontal overflow`);
  assert.equal(overflow.rowHorizontal, 0, `${size}: ship rows have no horizontal overflow`);
  assert.equal(overflow.massClipped, 0, `${size}: running-mass text is not clipped`);
  assert.equal(overflow.nameClipped, 0, `${size}: ship names are not clipped`);
  assert.equal(overflow.legendHorizontal, 0, `${size}: legend swatches wrap instead of overflowing`);
  assert.ok(overflow.legendInsideCanvas, `${size}: legend stays inside the porthole`);
  assert.ok(overflow.legendHeightFraction <= 0.3, `${size}: legend covers <= 30% of the porthole height (${overflow.legendHeightFraction.toFixed(2)})`);
  assert.equal(overflow.toolbarHorizontal, 0, `${size}: scene toolbar has no horizontal overflow`);
  assert.equal(overflow.pageHorizontal, 0, `${size}: no horizontal page scrollbar`);
  assert.ok(overflow.listVisibleRows >= (dense ? 8 : 4), `${size}: at least ${dense ? 8 : 4} ship rows visible without scrolling (${overflow.listVisibleRows})`);
  const screenshots = [`viewer40-fleet-${size}.png`, `viewer40-rail-${size}.png`, `viewer40-legend-${size}.png`];
  await page.screenshot({ path: join(artifacts, screenshots[0]) });
  await page.locator(".sidebar").screenshot({ path: join(artifacts, screenshots[1]) });
  await page.locator("#fleet-legend").screenshot({ path: join(artifacts, screenshots[2]) });
  return { ships, dense, distinct_colours: new Set(legendColours).size, legend_rows: overflow.legendRows, visible_rows: overflow.listVisibleRows, legend_height_fraction: Number(overflow.legendHeightFraction.toFixed(3)), screenshots };
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
