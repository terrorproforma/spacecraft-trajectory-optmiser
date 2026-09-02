# SpacePDHCG trajectory evidence viewer

A dependency-free, standalone WebGL2 viewer for the verified archived SpacePDHCG trajectory dataset. It renders each record in its own physical frame and scale; it does not compare unrelated coordinates in one scene.

## Launch

Node.js 18 or newer is required for the validation and safe development server. From `web/trajectory-viewer`:

```text
npm run import-data
npm run check
npm test
npm run serve
```

Open `http://127.0.0.1:4173/`. To choose another port:

```text
npm run serve -- --port=8080
```

The checked-in generated data can alternatively be served with Python 3:

```text
python -m http.server 4173 --bind 127.0.0.1 --directory .
```

Then open `http://127.0.0.1:4173/`. The Node server is preferred because it adds strict MIME, cache, CSP, anti-framing and traversal protections.

## Controls

- Left-drag or use arrow keys to orbit.
- Right-drag, Shift+drag or Shift+arrow keys to pan.
- Wheel or `+`/`-` to zoom.
- Space or **Play replay** starts and pauses timeline playback.
- The timeline selects an exact archived sample.
- **Dense replay** is the default. **Transcription nodes** shows source solver/reference nodes.
- **Reset view** restores the camera without changing the selected evidence.

The canvas is focusable and all controls have keyboard focus indicators. Layouts adapt to narrow screens. Reduced-motion preferences suppress decorative motion; playback only starts on explicit input.

## Evidence and smoothness

The display defaults to the compact dataset's real dense replay arrays: 201 points for P1-C, 251 for P1-B, and 512 deterministically selected exact archived replay points for P1-D, P1-E and P2. Smooth paths come from those dense physical samples plus a GPU antialiased ribbon and line. No visual interpolation is performed. Switching to transcription mode shows the exact archived nodes, including the two P2 Lambert endpoints. Display geometry never changes validation metrics.

The importer verifies authoritative source SHA-256 `83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2` before conversion. It preserves the compact source fields, selected indices, original point hashes, replay/transcription arrays, source metadata, qualification and validation records. Stable key ordering produces deterministic `data/trajectories.json` and a SHA-256 manifest.

Qualification is record-specific:

- P1-B is qualified CPU solver and replay evidence in Hill/LVLH relative coordinates. It has an LVLH plane and no globe.
- P1-C and P1-D are unqualified diagnostic powered-descent records. Their generic local surface is physical `Z = 0`; no planetary identity or radius is inferred.
- P1-E is unqualified because the archived host optimizer dual is absent. Its unnamed 6,500 km sphere is the minimum-radius constraint, not a claimed planetary surface.
- P2 has GPU Lambert component parity evidence, while the plotted dense history is explicitly a CPU replay. Its Earth ECI sphere uses radius 6,378,136.3 m.
- P1-A is non-trajectory evidence. Other absent, unsupported or synthetic histories remain excluded as documented in the embedded validation report.

## Package structure

- `index.html` — semantic application shell.
- `app.js` — UI state, interaction and WebGL2 renderer.
- `math.js` — pure camera and matrix helpers.
- `styles.css` — responsive local styling; no fonts or external assets.
- `data/trajectories.json` — deterministic transformed evidence.
- `data/manifest.json` — source and output hashes.
- `scripts/import-data.mjs` — verified lossless source transform.
- `scripts/check.mjs` — schema, data, physical-rule, DOM and asset checks.
- `scripts/serve.mjs` — localhost static server.
- `tests/` — Node standard-runner tests for data, import determinism, math and server behavior.
- `test-artifacts/` — browser-verification screenshots.

WebGL2 context loss pauses playback and reports status; restoration recreates shaders and buffers. Page shutdown deletes all allocated buffers and programs.
