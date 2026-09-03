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

3. `npm run check` now also validates the fleet dataset; `npm run serve` and open `http://127.0.0.1:4173/?dataset=gtoc12`, or pick **GTOC12 fleet** in the selector. Deep-link parameters: `dataset=gtoc12`, `ship=N` (1-based), `epoch=MJD`, `focus=1`.

To visualise a different verified fleet (for example a later `results/gtoc12/runs/<run>/fleets/<fleet>/Result.txt`), run steps 1–2 with that solution and its `fleet.json`.

### Static fallback figure

`python scripts/plot_gtoc12_fleet.py` (numpy + matplotlib) renders `test-artifacts/gtoc12-fleet-ecliptic.png`: ecliptic XY projection with per-ship colours matching the viewer, Earth orbit and epoch position, faint Keplerian asteroid orbits, asteroid positions at the archived visit epochs (hollow = deploy, filled = collect), title/axes/legend and a source caption with run, commit, hashes and verifier status. `--epoch MJD` truncates the arcs, `--ship N` highlights one ship.

## Controls

- Left-drag or use arrow keys to orbit.
- Right-drag, Shift+drag or Shift+arrow keys to pan.
- Wheel or `+`/`-` to zoom.
- Space or the play button starts and pauses playback (archive replay, or the GTOC12 mission clock).
- **Reset view** restores the camera without changing the selection.

Archive view:

- The timeline selects an exact archived sample.
- **Dense replay** is the default. **Transcription nodes** shows source solver/reference nodes.

GTOC12 fleet view:

- The **Epoch** slider (MJD 64328–69807, 2035-01-01 to 2050-01-01) scrubs the mission; **Play mission** runs the 15-year window in about 24 s. Every frame shows each launched ship at its last archived sample at or before the epoch — nothing between samples is interpolated — while Earth and the asteroids move on their Keplerian orbits.
- The **Ships** list (colour, collected kg, asteroid count, launch date) selects one ship: its arc and its asteroids' orbits are highlighted, the others dimmed, its archived events are labelled in the scene and tabulated in **Selected ship**. **Whole fleet** clears the selection; **Focus ship** frames the selected arc top-down.
- Hover identifies ships, event markers (asteroid, epoch, mass before/after), asteroids (elements and visits) and Earth; click selects the ship or pins an asteroid.
- The legend maps ship colours and markers: hollow ring = deploy miner, filled disc = collect mined mass, green ring = launch, amber ring = Earth return; blue = Earth orbit; faint grey = asteroid orbits; the Sun marker is not to scale; rings every 1 AU.

The canvas is focusable and all controls have keyboard focus indicators. Layouts adapt to narrow screens. Reduced-motion preferences suppress decorative motion; playback only starts on explicit input.

## Evidence and smoothness

The archive display defaults to the compact dataset's real dense replay arrays: 201 points for P1-C, 251 for P1-B, and 512 deterministically selected exact archived replay points for P1-D, P1-E and P2. Smooth paths come from those dense physical samples plus a GPU antialiased ribbon and line. No visual interpolation is performed. Switching to transcription mode shows the exact archived nodes, including the two P2 Lambert endpoints. Display geometry never changes validation metrics.

The GTOC12 arcs are straight GPU segments (ribbon + line) connecting the exporter's ≤ 512 exact propagated samples per ship (fleet_master_v1: 7,622 samples for 15 ships, every event epoch preserved). They are connections between archived nodes, not interpolation; the legend and scene overlay say so. Earth and asteroid orbits, and their positions at the current epoch, are two-body Keplerian curves from the pinned GTOC12 elements (Appendix 6.1 model, `kepler.js`), cross-checked against the exporter's ephemeris at import time. Event markers sit on the archived event states (the transcription nodes). The Sun is drawn as a small marker, not to scale.

The archive importer verifies authoritative source SHA-256 `83fc5031ecafccbdc7ae624df4a61679fd2af342ce315e528adda9e6325ae6d2` before conversion. It preserves the compact source fields, selected indices, original point hashes, replay/transcription arrays, source metadata, qualification and validation records. Stable key ordering produces deterministic `data/trajectories.json` and a SHA-256 manifest.

Qualification is record-specific:

- P1-B is qualified CPU solver and replay evidence in Hill/LVLH relative coordinates. It has an LVLH plane and no globe.
- P1-C and P1-D are unqualified diagnostic powered-descent records. Their generic local surface is physical `Z = 0`; no planetary identity or radius is inferred.
- P1-E is unqualified because the archived host optimizer dual is absent. Its unnamed 6,500 km sphere is the minimum-radius constraint, not a claimed planetary surface.
- P2 has GPU Lambert component parity evidence, while the plotted dense history is explicitly a CPU replay. Its Earth ECI sphere uses radius 6,378,136.3 m.
- P1-A is non-trajectory evidence. Other absent, unsupported or synthetic histories remain excluded as documented in the embedded validation report.
- The GTOC12 fleet badge reads **Verified fleet** only when the export's validation report records an independent-verifier pass; the provenance panel lists the official verifier result, `Result.txt` / export / catalogue hashes and the Kepler cross-check.

## Browser verification

`scripts/browser-check.cjs` drives headless Chromium (SwiftShader WebGL2) through both datasets and writes `test-artifacts/browser-report.json` plus screenshots. It needs a local Playwright module: `PLAYWRIGHT_PATH=<path to node_modules/playwright> node scripts/browser-check.cjs` with `npm run serve` running on port 4173. GTOC12 steps run only when `data/gtoc12/` is installed; the absent-dataset degrade path is always exercised. Screenshots: `desktop-p2-earth.png`, `mobile-p1c-local-surface.png`, `gtoc12-fleet-heliocentric.png`, `gtoc12-timeline-mid-mission.png`, `gtoc12-ship4-hop-sequence.png`, `gtoc12-desktop-fullpage.png`, `gtoc12-mobile.png`; static fallback `gtoc12-fleet-ecliptic.png` and `gtoc12-fleet-ecliptic-ship4.png`.

## Package structure

- `index.html` — semantic application shell (dataset selector, archive and fleet panels).
- `app.js` — UI state, dataset switching, interaction and the archive WebGL2 renderer.
- `gtoc12.js` — GTOC12 scene builder, fleet renderer, picking, panels and event labels.
- `kepler.js` — pure Keplerian ephemeris helpers (GTOC12 Appendix 6.1) and MJD calendar conversion.
- `webgl.js` — shared WebGL2 primitives, shaders and resource management.
- `math.js` — pure camera and matrix helpers; `dom.js` — text/DOM helpers.
- `styles.css` — responsive local styling; no fonts or external assets.
- `data/trajectories.json`, `data/manifest.json` — deterministic transformed archive evidence.
- `data/gtoc12/` — ignored, generated GTOC12 fleet dataset (`npm run import-gtoc12`).
- `scripts/import-data.mjs` — verified lossless archive transform.
- `scripts/import-gtoc12.mjs` — verified GTOC12 fleet transform.
- `scripts/plot_gtoc12_fleet.py` — static matplotlib fallback figure.
- `scripts/check.mjs` — schema, data, physical-rule, DOM and asset checks (both datasets).
- `scripts/serve.mjs` — localhost static server.
- `scripts/browser-check.cjs` — headless browser verification.
- `tests/` — Node standard-runner tests for data, import determinism, Kepler helpers, math and server behaviour.
- `test-artifacts/` — browser-verification screenshots and report.

WebGL2 context loss pauses playback and reports status; restoration recreates shaders and buffers. Page shutdown deletes all allocated buffers and programs.
