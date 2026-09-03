const assert = require("node:assert/strict");
const { mkdir, writeFile } = require("node:fs/promises");
const { join, resolve } = require("node:path");

async function main() {
  const modulePath = process.env.PLAYWRIGHT_PATH;
  if (!modulePath) throw new Error("Set PLAYWRIGHT_PATH to an installed Playwright module");
  const { chromium } = require(modulePath);
  const root = resolve(__dirname, "..");
  const artifacts = join(root, "test-artifacts");
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

/** GTOC12 fleet dataset: graceful absence, then (when installed) the heliocentric scene, timeline, picking and screenshots. */
async function checkGtoc12(browser, artifacts) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on("console", (message) => { if (message.type() === "error" || message.type() === "warning") errors.push(`console ${message.type()}: ${message.text()}`); });
  page.on("pageerror", (error) => errors.push(`page: ${error.message}`));

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

  const installed = (await page.request.get("http://127.0.0.1:4173/data/gtoc12/manifest.json")).ok();
  if (!installed) {
    assert.deepEqual(errors, []);
    await context.close();
    return { installed: false, note: "data/gtoc12 absent; run npm run import-gtoc12 to exercise the fleet view" };
  }

  // 2. Whole-fleet heliocentric view at the end of the mission.
  await page.goto("http://127.0.0.1:4173/?dataset=gtoc12", { waitUntil: "networkidle" });
  await page.locator("#trajectory-title").filter({ hasText: /ships/ }).waitFor();
  assert.equal(await page.locator("#renderer-status-text").textContent(), "WebGL2 GPU renderer");
  assert.equal(await page.locator("#dataset-select").inputValue(), "gtoc12");
  const title = await page.locator("#trajectory-title").textContent();
  assert.match(title, /15 ships, 109 asteroids, 7575\.58 kg/);
  assert.equal(await page.locator("#ship-list button").count(), 16);
  assert.match(await page.locator("#mission-timeline-output").textContent(), /MJD 69807 · 2050-01-01/);
  assert.match(await page.locator("#fleet-summary").textContent(), /Delivered to Earth so far\s*7575\.6 kg/);
  assert.match(await page.locator("#fleet-legend").textContent(), /straight segments between exact archived samples/i);
  assert.match(await page.locator("#scene-overlay").textContent(), /connections between archived samples/);
  assert.equal(await page.locator("#qualification-badge").textContent(), "Verified fleet");
  assert.equal(await page.locator(".archive-only:visible").count(), 0, "archive panels hidden in fleet mode");
  await page.locator(".viewport-card").screenshot({ path: join(artifacts, "gtoc12-fleet-heliocentric.png") });
  await page.screenshot({ path: join(artifacts, "gtoc12-desktop-fullpage.png"), fullPage: true });

  // 3. Mid-mission timeline frame (deployment phase complete, collection about to start).
  await page.locator("#mission-timeline").fill("67000");
  assert.match(await page.locator("#mission-timeline-output").textContent(), /MJD 67000 · 2042-04-26 · T\+7\.32 yr/);
  assert.match(await page.locator("#fleet-summary").textContent(), /Ships in flight\s*15 of 15/);
  await page.locator(".viewport-card").screenshot({ path: join(artifacts, "gtoc12-timeline-mid-mission.png") });

  // 4. One ship's hop sequence: select ship 4, focus, and land the epoch inside its collection phase.
  await page.locator('#ship-list button[data-ship="3"]').click();
  assert.match(await page.locator("#trajectory-title").textContent(), /^Ship 4 — 541\.3 kg from 8 asteroids/);
  assert.equal(await page.locator("#ship-detail tbody tr").count(), 18);
  assert.match(await page.locator("#ship-detail").textContent(), /Deploy miner/);
  assert.match(await page.locator("#ship-detail").textContent(), /Collect mined mass/);
  await page.locator("#focus-ship-button").click();
  await page.locator("#mission-timeline").fill("68900");
  await page.evaluate(() => window.scrollTo(0, 0));
  const marker = await page.evaluate(() => window.viewerDebug.eventScreenPosition(3, 1));
  const canvasBox = await page.locator("#trajectory-canvas").boundingBox();
  assert.ok(marker && marker[0] > 0 && marker[1] > 0 && marker[0] < canvasBox.width && marker[1] < canvasBox.height, `first deploy marker inside the canvas: ${JSON.stringify(marker)}`);
  await page.mouse.move(canvasBox.x + marker[0], canvasBox.y + marker[1]);
  await page.waitForTimeout(100);
  const tooltip = await page.locator("#hover-tooltip").textContent();
  assert.match(tooltip, /Ship 4 · Deploy miner/);
  assert.match(tooltip, /asteroid 24684/);
  assert.equal(await page.evaluate(() => window.viewerDebug.hover), "event");
  await page.locator(".viewport-card").screenshot({ path: join(artifacts, "gtoc12-ship4-hop-sequence.png") });
  await page.mouse.click(canvasBox.x + marker[0], canvasBox.y + marker[1]);
  assert.equal(await page.evaluate(() => window.viewerDebug.selectedShip), 3);

  // 5. Playback advances the epoch; switching back to the archive restores it.
  await page.locator("#mission-timeline").fill("64328");
  await page.locator("#mission-play-button").click();
  await page.waitForTimeout(400);
  assert.ok(Number(await page.locator("#mission-timeline").inputValue()) > 64328);
  await page.locator("#mission-play-button").click();
  await page.selectOption("#dataset-select", "archive");
  assert.match(await page.locator("#sample-output").textContent(), /251 \/ 251/);
  assert.equal(await page.locator(".fleet-only:visible").count(), 0, "fleet panels hidden in archive mode");
  await page.selectOption("#dataset-select", "gtoc12");
  await page.locator("#trajectory-title").filter({ hasText: /ships/ }).waitFor();

  // 6. Mobile width.
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator("#mission-timeline").fill("67500");
  await page.evaluate(() => window.scrollTo(0, 0));
  assert.ok((await page.locator("#trajectory-canvas").boundingBox()).width <= 370);
  assert.equal(await page.locator("#error-banner").isHidden(), true);
  await page.screenshot({ path: join(artifacts, "gtoc12-mobile.png"), fullPage: true });

  assert.deepEqual(errors, []);
  await context.close();
  return {
    installed: true, title, ships: 15, asteroids: 109, official_kg: 7575.58,
    screenshots: ["gtoc12-fleet-heliocentric.png", "gtoc12-desktop-fullpage.png", "gtoc12-timeline-mid-mission.png", "gtoc12-ship4-hop-sequence.png", "gtoc12-mobile.png"],
    interactions: ["absent-dataset-degrade", "select", "timeline", "focus", "hover-tooltip", "click-select", "play", "dataset-switch", "mobile"],
  };
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
