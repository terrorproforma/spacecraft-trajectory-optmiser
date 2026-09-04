import test from "node:test";
import assert from "node:assert/strict";
import { once } from "node:events";
import { request } from "node:http";
import { createStaticServer } from "../scripts/serve.mjs";

async function withServer(run) {
  const server = createStaticServer(new URL("..", import.meta.url).pathname.slice(1));
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  try {
    const { port } = server.address();
    await run(`http://127.0.0.1:${port}`);
  } finally {
    server.close();
    await once(server, "close");
  }
}

function rawStatus(base, path, method = "GET") {
  const url = new URL(base);
  return new Promise((resolve, reject) => {
    const outgoing = request({
      hostname: url.hostname, port: url.port, path, method,
    }, (response) => {
      response.resume();
      response.once("end", () => resolve(response.statusCode));
    });
    outgoing.once("error", reject);
    outgoing.end();
  });
}

test("server sends MIME and security headers", async () => withServer(async (base) => {
  const html = await fetch(`${base}/`);
  assert.equal(html.status, 200);
  assert.match(html.headers.get("content-type"), /^text\/html/);
  assert.equal(html.headers.get("x-content-type-options"), "nosniff");
  assert.match(html.headers.get("content-security-policy"), /default-src 'self'/);
  const module = await fetch(`${base}/app.js`);
  assert.match(module.headers.get("content-type"), /^text\/javascript/);
  const data = await fetch(`${base}/data/trajectories.json`);
  assert.equal(data.headers.get("cache-control"), "no-store");
}));

test("server rejects traversal and unsupported methods", async () => withServer(async (base) => {
  assert.equal(await rawStatus(base, "/%2e%2e/package.json"), 400);
  assert.equal(await rawStatus(base, "/..%5cpackage.json"), 400);
  assert.equal((await fetch(`${base}/missing.file`)).status, 404);
  assert.equal(await rawStatus(base, "/", "POST"), 405);
}));
