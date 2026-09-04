# SpacePDHCG trajectory evidence viewer

A dependency-free, standalone WebGL2 viewer for verified SpacePDHCG trajectory evidence. It ships two dataset kinds behind one **Evidence source** selector:

- **Archived P1/P2 evidence** (default, checked in): each record in its own physical frame and scale; unrelated coordinates are never compared in one scene.
- **GTOC12 fleet** (optional, generated locally): the verified multi-ship asteroid-mining fleet in a Sun-centred J2000 ecliptic frame in AU, with Earth and asteroid orbits, per-ship low-thrust arcs, deploy/collect markers and a 2035–2050 mission timeline.

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

## GTOC12 fleet dataset

The GTOC12 data is multi-megabyte and regenerable, so it is **not committed**; `data/gtoc12/` is ignored by git. Without it the selector shows "GTOC12 fleet — not installed" (disabled), the help text names the import command, `npm run check` prints a notice, and the archive view is unaffected.

1. Export the verified fleet from the GTOC12 worktree with the documented CLI (propagates the official `Result.txt` through the independent verifier model; ~16 s, CPU only). The output directory must be under the ignored `results/` tree:

   ```text
   cd /home/angus/worktrees/spacepdhcg-gtoc12
   PYTHONPATH=src .venv/bin/python -m spacepdhcg gtoc12 export-viewer \
     results/gtoc12/runs/fleet_master_v1/fleet/Result.txt \
     --output results/gtoc12/viewer-exports/fleet_master_v1 --run-id fleet_master_v1_fleet
   ```

2. Import it into the viewer (from `web/trajectory-viewer`; on Windows the WSL paths are reachable as `\\wsl.localhost\Ubuntu-22.04\home\angus\...`):

   ```text
   npm run import-gtoc12 -- \
     --export /home/angus/worktrees/spacepdhcg-gtoc12/results/gtoc12/viewer-exports/fleet_master_v1 \
     --catalogue /home/angus/worktrees/spacepdhcg-gtoc12/benchmarks/gtoc12/data/GTOC12_Asteroids_Data.txt \
     --solution /home/angus/worktrees/spacepdhcg-gtoc12/results/gtoc12/runs/fleet_master_v1/fleet/Result.txt \
     --fleet /home/angus/worktrees/spacepdhcg-gtoc12/results/gtoc12/runs/fleet_master_v1/fleet/fleet.json
   ```

   `--solution` and `--fleet` are optional cross-checks. The importer refuses to write unless the export's `trajectories.json` matches its manifest hash, the catalogue matches the pinned SHA-256 (`99a42cc3…c46675`, 6,840,111 bytes), `Result.txt` matches the export's source hash, every ship's replay/transcription series is consistent (counts, monotone epochs, exact event epochs present as archived samples), per-ship collected mass from the archived events equals `fleet.json`, and the viewer's own Kepler propagation reproduces the exporter's context orbits (fleet_master_v1: 41,179 samples, max 3.5e-6 km). It writes `data/gtoc12/fleet.json` (exact replay samples, selected indices and original-sample hashes copied verbatim; events classified as launch / deploy / collect / Earth return; pinned Keplerian elements for every visited asteroid and Earth) plus `data/gtoc12/manifest.json`.

3. `npm run check` now also validates the fleet dataset; `npm run serve` and open `http://127.0.0.1:4173/?dataset=gtoc12`, or pick **GTOC12 fleet** in the selector. Deep-link parameters: `dataset=gtoc12`, `ship=N` (1-based), `epoch=MJD`, `focus=1`, `follow=1`, `preset=top|oblique|edge`, `exaggeration=1..20`.

To visualise a different verified fleet (for example a later `results/gtoc12/runs/<run>/fleets/<fleet>/Result.txt`), run steps 1–2 with that solution and its `fleet.json`. The currently imported dataset is `fleet_master_v4_fleet` (19 ships, 158 asteroids, 10 700.48 kg summed collects; official verifier 10 700.5 kg).

### 3D scene

The fleet view is a perspective WebGL2 scene, no dependencies or textures:

- **Camera** — orbit (drag, with release inertia), pan (right-drag / Shift+drag), wheel dolly towards the cursor (the pointed-at world point stays put), presets **Top-down**, **30° oblique** (default), **Edge-on** (inclinations) and **Follow ship**, all with eased transitions (`camera.js`, pure functions). The **Vertical exaggeration** slider (1×–20×, opens at 6×) scales Z for display only and is labelled *not physical* in the panel and the scene overlay.
- **Geometry** — one instanced draw call of lit spheres for the Sun (emissive, with a procedural two-layer glow sprite), Earth, the 158 visited asteroids (pending grey → deployed tint → mined ship colour, brightening) and the ship markers; each ship arc is a 6-sided lit tube mesh whose ring vertices sit exactly on the archived samples (`tubeArrays`), drawn segment-by-segment up to the current epoch so it never extends past an archived node; Earth and asteroid orbits are depth-faded ribbons; the ecliptic is a grid disc with 1 AU rings.
- **Lighting** — the Sun at the origin is the point light (Lambert + Blinn-Phong), a sky ambient term, a procedural gradient/vignette background with a deterministic star field, and exponential distance fog scaled to the camera distance so depth reads.
- **Motion** — the 2035–2050 clock plays at 0.25–4 yr/s in a single `requestAnimationFrame` loop; ships slide along their archived samples, deploy/collect events flash rings and glows for 60 days, per-ship and fleet mass counters tick up, hovering a ship or asteroid brightens it. Tube and sphere radii scale with camera distance so the scene stays legible at every zoom (they are not to scale).

### Static fallback figure

`python scripts/plot_gtoc12_fleet.py` (numpy + matplotlib) renders `test-artifacts/gtoc12-fleet-ecliptic.png`: ecliptic XY projection with per-ship colours matching the viewer, Earth orbit and epoch position, faint Keplerian asteroid orbits, asteroid positions at the archived visit epochs (hollow = deploy, filled = collect), title/axes/legend and a source caption with run, commit, hashes and verifier status. `--epoch MJD` truncates the arcs, `--ship N` highlights one ship.

## Controls

- Left-drag or use arrow keys to orbit.
- Right-drag, Shift+drag or Shift+arrow keys to pan.
- Wheel or `+`/`-` to zoom (in the fleet view the wheel dollies towards the cursor).
- Space or the play button starts and pauses playback (archive replay, or the GTOC12 mission clock).
- **Reset view** restores the camera without changing the selection. Keys `1`–`4` select the fleet camera presets.

Archive view:

- The timeline selects an exact archived sample.
- **Dense replay** is the default. **Transcription nodes** shows source solver/reference nodes.

GTOC12 fleet view:

- The **Epoch** slider (MJD 64328–69807, 2035-01-01 to 2050-01-01) sits under the scene as a mission clock with a time cursor and 1 January year ticks; **Play mission** runs the 15-year window at the selected speed (default 1 yr/s). Every frame shows each launched ship at its last archived sample at or before the epoch — nothing between samples is interpolated — while Earth and the asteroids move on their Keplerian orbits.
- The **Ships** list (colour, collected kg so far over the ship's total with a bar that fills as the clock runs, asteroid count, launch date) selects one ship: its tube and its asteroids' orbits are highlighted, the others dimmed, its archived events are labelled in the scene and tabulated in **Selected ship**. **Whole fleet** clears the selection; **Frame ship arc** fits the selected arc; **Follow ship** keeps the ship centred while the clock runs.
- Hover identifies ships, event markers (asteroid, epoch, mass before/after), asteroids (elements, visits, state) and Earth; click selects the ship or pins an asteroid.
- The compact legend maps ship colours and markers: hollow ring = deploy miner, filled disc = collect mined mass, green ring = launch, amber ring = Earth return; grey sphere = asteroid not yet reached, ship-coloured sphere = mined; blue = Earth orbit; faint grey = asteroid orbits; tube + bright 450-day trail = ship arc; spheres are not to scale; rings every 1 AU.

The canvas is focusable and all controls have keyboard focus indicators. Layouts adapt to narrow screens (on phones the scene, timeline and verification strip come first, then the ship list and camera controls, then the tables). Reduced-motion preferences suppress decorative motion; nothing animates on its own and playback only starts on explicit input.

### Visual design

The chrome is styled as a matte instrument panel around a black porthole: graphite surfaces, warm-white legends, hairline rules, one system sans-serif family with tabular numerals (monospace only for hashes and commands), no web fonts, no CDN. Colour is reserved for meaning — ship colours mirror the scene, green/amber/red mark verified/caution/error — and structure encodes information: the timeline is a strip chart with a thin time cursor, ship rows carry a bar of collected mass, and the qualification/verifier state is repeated as a labelled dot in the scene toolbar, the list rows and the verification strip. The layout follows Anthropic's `frontend-design` skill guidance (dependency-free, visible focus, reduced motion respected, no decorative gradients or motion). Before/after captures live in `test-artifacts/redesign-*.png`.

## Evidence and smoothness

The archive display defaults to the compact dataset's real dense replay arrays: 201 points for P1-C, 251 for P1-B, and 512 deterministically selected exact archived replay points for P1-D, P1-E and P2. Smooth paths come from those dense physical samples plus a GPU antialiased ribbon and line. No visual interpolation is performed. Switching to transcription mode shows the exact archived nodes, including the two P2 Lambert endpoints. Display geometry never changes validation metrics.

The GTOC12 arcs are lit tube meshes whose straight segments connect the exporter's ≤ 512 exact propagated samples per ship (fleet_master_v4: 9,643 samples for 19 ships, every event epoch preserved). They are connections between archived nodes, not interpolation; the legend and scene overlay say so, and the tube is only ever drawn up to an archived sample. Earth and asteroid orbits, and their positions at the current epoch, are two-body Keplerian curves from the pinned GTOC12 elements (Appendix 6.1 model, `kepler.js`), cross-checked against the exporter's ephemeris at import time. Event markers sit on the archived event states (the transcription nodes). Sun, Earth, asteroid and ship spheres are display markers, not to scale; vertical exaggeration is a display transform and is labelled as such.

The archive importer verifies authoritative source SHA-256 `83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2` before conversion. It preserves the compact source fields, selected indices, original point hashes, replay/transcription arrays, source metadata, qualification and validation records. Stable key ordering produces deterministic `data/trajectories.json` and a SHA-256 manifest.

Qualification is record-specific:

- P1-B is qualified CPU solver and replay evidence in Hill/LVLH relative coordinates. It has an LVLH plane and no globe.
- P1-C and P1-D are unqualified diagnostic powered-descent records. Their generic local surface is physical `Z = 0`; no planetary identity or radius is inferred.
- P1-E is unqualified because the archived host optimizer dual is absent. Its unnamed 6,500 km sphere is the minimum-radius constraint, not a claimed planetary surface.
- P2 has GPU Lambert component parity evidence, while the plotted dense history is explicitly a CPU replay. Its Earth ECI sphere uses radius 6,378,136.3 m.
- P1-A is non-trajectory evidence. Other absent, unsupported or synthetic histories remain excluded as documented in the embedded validation report.
- The GTOC12 fleet badge reads **Verified fleet** only when the export's validation report records an independent-verifier pass; the provenance panel lists the official verifier result, `Result.txt` / export / catalogue hashes and the Kepler cross-check.

## Browser verification

`scripts/browser-check.cjs` drives headless Chromium (SwiftShader WebGL2) through both datasets and writes `test-artifacts/browser-report.json` plus screenshots. It needs a local Playwright module: `PLAYWRIGHT_PATH=<path to node_modules/playwright> node scripts/browser-check.cjs` with `npm run serve` running on port 4173. GTOC12 steps run only when `data/gtoc12/` is installed; the absent-dataset degrade path is always exercised. The fleet steps assert the 3D contract from `window.viewerDebug.glInfo` (antialiased WebGL2 context, depth test on, one sphere instance per body, 6-sided tubes), the default 30° oblique / 6× opening, the ≥ 70 %-height canvas, wheel dolly towards the cursor, preset transitions, follow-ship centring during playback, hover picking and the running mass counters. Screenshots at 1440×900: `gtoc12-3d-oblique-fleet.png` (opening view), `gtoc12-3d-desktop-window.png` (full window), `gtoc12-3d-desktop-fullpage.png`, `gtoc12-3d-edge-on.png` (10×, inclinations), `gtoc12-3d-timeline-mid-mission.png`, `gtoc12-3d-follow-ship.png` (tube + trail close-up), `gtoc12-3d-ship-arc-framed.png` (hover tooltip), the ten-frame sequence `gtoc12-3d-frame-01..10.png` (2036 → 2050), and `gtoc12-3d-mobile.png` (390 px). `python scripts/build_gif.py` (Pillow) assembles the frames into `gtoc12-3d-preview.gif`. Archive: `desktop-p2-earth.png`, `mobile-p1c-local-surface.png`. Static fallback (`scripts/plot_gtoc12_fleet.py`): `gtoc12-fleet-ecliptic.png`, `gtoc12-fleet-ecliptic-ship15.png` (ship 15, the richest at 603.7 kg).

## Package structure

- `index.html` — semantic application shell (top bar with dataset selector, inventory/camera rail, porthole, timeline strip, verification strip, detail tables).
- `app.js` — UI state, dataset switching, interaction and the archive WebGL2 renderer.
- `gtoc12.js` — GTOC12 scene builder, fleet renderer, picking, panels and event labels.
- `kepler.js` — pure Keplerian ephemeris helpers (GTOC12 Appendix 6.1) and MJD calendar conversion.
- `webgl.js` — shared WebGL2 primitives (ribbons, tube meshes, instanced lit spheres, star field, procedural sky), shaders with fog and resource management.
- `camera.js` — pure orbit-camera helpers: presets, eased transitions, inertia, bounds fitting, exaggeration, cursor dolly.
- `math.js` — pure camera and matrix helpers; `dom.js` — text/DOM helpers.
- `styles.css` — responsive local styling; no fonts or external assets.
- `data/trajectories.json`, `data/manifest.json` — deterministic transformed archive evidence.
- `data/gtoc12/` — ignored, generated GTOC12 fleet dataset (`npm run import-gtoc12`).
- `scripts/import-data.mjs` — verified lossless archive transform.
- `scripts/import-gtoc12.mjs` — verified GTOC12 fleet transform.
- `scripts/plot_gtoc12_fleet.py` — static matplotlib fallback figure.
- `scripts/check.mjs` — schema, data, physical-rule, DOM and asset checks (both datasets).
- `scripts/serve.mjs` — localhost static server.
- `scripts/browser-check.cjs` — headless browser verification; `scripts/build_gif.py` — frame sequence → GIF.
- `tests/` — Node standard-runner tests for data, import determinism, Kepler helpers, camera and geometry (tubes, ribbons, star field), math and server behaviour.
- `test-artifacts/` — browser-verification screenshots and report.

WebGL2 context loss pauses playback and reports status; restoration recreates shaders and buffers. Page shutdown deletes all allocated buffers and programs.
