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
  const errors = [];
  const requests = [];
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
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
  const report = {
    browser: await browser.version(),
    desktop: { viewport: "1440x1000", trajectory: "P2", replay_points: 512, interactions: ["switch", "timeline", "play", "drag", "wheel", "keyboard"] },
    mobile: { viewport: "390x844", trajectory: "P1-C", replay_points: 201 },
    requests,
    errors,
    webgl2: true,
  };
  await writeFile(join(artifacts, "browser-report.json"), `${JSON.stringify(report, null, 2)}\n`);
  await context.close();
  await browser.close();
  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
