import { createServer as createHttpServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));
const MIME = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
};

function headers(pathname, type) {
  return {
    "Cache-Control": pathname.startsWith("/data/") ? "no-store" : "no-cache",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    "Content-Type": type,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
  };
}

export function createStaticServer(root = ROOT) {
  const safeRoot = resolve(root);
  return createHttpServer(async (request, response) => {
    try {
      const rawTarget = request.url || "/";
      if (request.method !== "GET" && request.method !== "HEAD") {
        response.writeHead(405, { Allow: "GET, HEAD" }); response.end("Method not allowed"); return;
      }
      if (/%2e|%2f|%5c/i.test(rawTarget)) {
        response.writeHead(400, headers(rawTarget, "text/plain; charset=utf-8")); response.end("Invalid path"); return;
      }
      const url = new URL(rawTarget, "http://localhost");
      let pathname;
      try { pathname = decodeURIComponent(url.pathname); } catch {
        response.writeHead(400, headers("/", "text/plain; charset=utf-8")); response.end("Invalid path"); return;
      }
      if (pathname.includes("\0") || pathname.split(/[\\/]/).includes("..")) {
        response.writeHead(400, headers(pathname, "text/plain; charset=utf-8")); response.end("Invalid path"); return;
      }
      if (pathname === "/") pathname = "/index.html";
      const filePath = resolve(safeRoot, `.${pathname}`);
      if (filePath !== safeRoot && !filePath.startsWith(`${safeRoot}${sep}`)) {
        response.writeHead(403, headers(pathname, "text/plain; charset=utf-8")); response.end("Forbidden"); return;
      }
      const fileStat = await stat(filePath);
      if (!fileStat.isFile()) throw Object.assign(new Error("Not found"), { code: "ENOENT" });
      const body = await readFile(filePath);
      response.writeHead(200, { ...headers(pathname, MIME[extname(filePath).toLowerCase()] || "application/octet-stream"), "Content-Length": body.length });
      response.end(request.method === "HEAD" ? undefined : body);
    } catch (error) {
      const status = error?.code === "ENOENT" ? 404 : 500;
      response.writeHead(status, headers("/", "text/plain; charset=utf-8"));
      response.end(status === 404 ? "Not found" : "Internal server error");
    }
  });
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const portArgument = process.argv.find((value) => value.startsWith("--port="))?.split("=")[1];
  const port = Number(portArgument || process.env.PORT || 4173);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`Invalid port: ${port}`);
  const host = process.env.HOST || "127.0.0.1";
  const server = createStaticServer();
  server.listen(port, host, () => {
    const address = server.address();
    console.log(`Trajectory viewer: http://${host}:${address.port}/`);
  });
  const shutdown = (signal) => {
    console.log(`${signal}: closing server`);
    server.close((error) => process.exitCode = error ? 1 : 0);
  };
  process.once("SIGINT", () => shutdown("SIGINT"));
  process.once("SIGTERM", () => shutdown("SIGTERM"));
}
