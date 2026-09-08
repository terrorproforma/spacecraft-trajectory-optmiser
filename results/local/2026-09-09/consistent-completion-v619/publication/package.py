"""Package immutable v619 evidence with deduplicated, verified reconstruction archives."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "build/performance/incumbent-admission-v619"
RETURN = ROOT / "build/performance/return-mass-bias-v619"
DEST = ROOT / "results/local/2026-09-09/consistent-completion-v619"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert sha(source) == sha(destination)


def archive(name, sources):
    members = {}
    with (DEST / name).open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as stream:
            for relative, source in sorted(sources.items()):
                payload = source.read_bytes()
                entry = tarfile.TarInfo(relative)
                entry.size, entry.mtime, entry.mode = len(payload), 0, 0o644
                stream.addfile(entry, io.BytesIO(payload))
                members[relative] = hashlib.sha256(payload).hexdigest()
    with tarfile.open(DEST / name, "r:gz") as stream:
        assert len(stream.getmembers()) == len(members)
        for member in stream.getmembers():
            assert member.isfile() and ".." not in Path(member.name).parts
            assert hashlib.sha256(stream.extractfile(member).read()).hexdigest() == members[member.name]
    return members


def main():
    assert not DEST.exists(), "New publication destination required"
    evidence = read(KIT / "evidence-manifest.json")
    assert sha(KIT / "evidence-manifest.json") == "1915531ba8c539316ff2f4ed1118df8b0b304bd0c7fc7ecd90f1d066139838ab"
    for name, expected in evidence["files"].items():
        assert sha(KIT / name) == expected["sha256"] and (KIT / name).stat().st_size == expected["bytes"]
    assert len(evidence["files"]) == 612
    roots = {str(p.relative_to(RETURN)): sha(p) for p in sorted(RETURN.rglob("*")) if p.is_file()}
    assert len(roots) == 26
    assert roots["report.json"] == "54aad2bb302738883ba9aedfcd1def85b4277eba40d48daf14d07056b7e08048"
    DEST.mkdir(parents=True)
    old = read(KIT / "source-sha256.json")
    final = read(KIT / "final-source-sha256.json")
    archives = {}
    archives["source-base.tar.gz"] = archive("source-base.tar.gz", {name: KIT / "source/baseline" / name for name in old["baseline"]})
    search = "src/spacepdhcg/gtoc12/search.py"
    archives["source-overlays.tar.gz"] = archive("source-overlays.tar.gz", {f"{arm}/{search}": KIT / "source" / arm / search for arm in ("candidate", "candidate-final")})
    input_sources = {f"routes/{p.name}": p for p in sorted((RETURN / "inputs").glob("ship-*.json"))}
    assert len(input_sources) == 23
    input_sources["shared/incumbent-inventory.json"] = RETURN / "inputs/incumbent-inventory.json"
    for p in sorted((KIT / "inputs").iterdir()):
        if p.name.startswith("ship-") or p.name in ("incumbent-inventory.json", "root-return-components.json"):
            continue
        input_sources[f"kit/{p.name}"] = p
    archives["inputs.tar.gz"] = archive("inputs.tar.gz", input_sources)
    reconstruction = {}
    for name, expected in evidence["files"].items():
        if name.startswith("source/"):
            _, arm, relative = name.split("/", 2)
            index = final if arm == "candidate-final" else old[arm]
            assert index[relative] == expected["sha256"]
            if arm != "baseline" and relative == search:
                location = {"archive": ["source-overlays.tar.gz", f"{arm}/{relative}"]}
            else:
                location = {"archive": ["source-base.tar.gz", relative]}
        elif name.startswith("inputs/"):
            filename = name.split("/", 1)[1]
            if filename.startswith("ship-"):
                location = {"archive": ["inputs.tar.gz", "routes/" + filename]}
            elif filename == "incumbent-inventory.json":
                location = {"archive": ["inputs.tar.gz", "shared/" + filename]}
            elif filename == "root-return-components.json":
                location = {"file": "root-audit/report.json"}
            else:
                location = {"archive": ["inputs.tar.gz", "kit/" + filename]}
        else:
            copy(KIT / name, DEST / "kit" / name)
            location = {"file": "kit/" + name}
        actual = (archives[location["archive"][0]][location["archive"][1]] if "archive" in location
                  else sha(KIT / name))
        assert actual == expected["sha256"], name
        reconstruction[name] = location
    root_map = {}
    for name, digest in roots.items():
        if name.startswith("inputs/ship-"):
            root_map[name] = {"archive": ["inputs.tar.gz", "routes/" + Path(name).name]}
        elif name == "inputs/incumbent-inventory.json":
            root_map[name] = {"archive": ["inputs.tar.gz", "shared/incumbent-inventory.json"]}
        else:
            copy(RETURN / name, DEST / "root-audit" / name)
            root_map[name] = {"file": "root-audit/" + name}
    report = read(RETURN / "report.json")
    source_notes = {}
    for name, digest in report["source_hashes"].items():
        live = ROOT / "src/spacepdhcg/gtoc12" / name
        assert sha(live) == digest
        copy(live, DEST / "root-audit/source" / name)
        source_notes[name] = {"sha256": digest, "normalized_matches_frozen_baseline": live.read_bytes().replace(b"\r\n", b"\n") == (KIT / "source/baseline/src/spacepdhcg/gtoc12" / name).read_bytes().replace(b"\r\n", b"\n")}
        assert source_notes[name]["normalized_matches_frozen_baseline"]
    copy(KIT / "evidence-manifest.json", DEST / "kit/evidence-manifest.json")
    copy(ROOT / "build/performance/reproduce_consistent_completion_v619.py", DEST / "reproduce.py")
    copy(Path(__file__), DEST / "publication/package.py")
    write(DEST / "reconstruction.json", {"kit": reconstruction, "root_audit": root_map,
          "root_audit_sha256": roots, "archives": archives, "root_source_notes": source_notes})
    (DEST / ".gitattributes").write_text("# Preserve exact evidence bytes and archive hashes.\n* -text\n")
    (DEST / "README.md").write_text(README)
    write(DEST / "archive-audit.json", {
        "prepared_files_reconstructed": len(reconstruction), "prepared_bytes": evidence["total_bytes"],
        "all_prepared_hashes_verified": True, "original_root_files_reconstructed": len(root_map),
        "all_root_hashes_verified": True, "archived_routes": 23,
        "unique_source_files_archived": 192, "source_entries_reconstructed": 570,
        "original_kit_manifest_sha256": sha(KIT / "evidence-manifest.json"),
        "scope": "Archive bytes and reconstruction mappings verified; no scalar, solver or physics run repeated during packaging.",
    })
    assert {name: sha(KIT / name) for name in evidence["files"]} == {name: value["sha256"] for name, value in evidence["files"].items()}
    assert {name: sha(RETURN / name) for name in roots} == roots
    files = {str(p.relative_to(DEST)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(DEST.rglob("*")) if p.is_file()}
    write(DEST / "sha256.json", {"files": files, "file_count": len(files), "total_bytes": sum(x["bytes"] for x in files.values())})
    print(json.dumps({"path": str(DEST), "files": len(files), "bytes": sum(x["bytes"] for x in files.values()),
                      "index_sha256": sha(DEST / "sha256.json")}, indent=2))


README = """# Consistent completion costs, v619

The correction preserves the configured DP hop-cost model, prices every collection/return flight once at its actual sequential mass, retains certified return-cell overrides, and records the inflation actually spent. Authority, mining-stay, cargo and final mass gates are unchanged. This is a CPU finishing-bridge correction, with no measured GPU speedup or fleet score gain.

These are historical positive controls pinned to the v595/v616 incumbent, not the newer fleet published in commit 2ccb93c1. The comparison does not describe the current fleet's score. The matched comparison has 40 CPU bridge calls on preselected ships 23, 1, 4, 7 and 10: old/new source × flat/existing-fit model × flat-proxy/measured deployment prefix. Baseline admits 0/20 cases and corrected source admits 2/20, both ship 4 with an optimistic flat deployment prefix. Every measured-prefix case still fails the final proxy mass gate. For ship 23, measured-prefix margin improves from −35.563593 to −9.587670 kg without the fit, and to −6.385482 kg with the unchanged existing fit. A negative proxy margin is not physical infeasibility: the historical ship 23 trajectory has an archived positive dry-mass margin of 2.861384 kg.

The independent all-23 return-component check confirms median guessed-mass excess 707.493615 kg and median return-fuel overprediction 22.811826 kg from that mass argument alone. Even at actual mass the generic return model overpredicts 20/23 returns, median 18.364126 kg. These component diagnostics are not new trajectory certificates. The 45 CPU regression tests and Ruff pass; the first 43-pass/two-harness-failure run is preserved and explained in the detailed kit README. No tolerances, epochs, cargo, fitting coefficients or incumbent fleet changed.

## Contents and integrity

`kit/README.md` gives the full interpretation in the historical experiment's context. `kit/output/` retains every baseline/final control and failure trace plus the independent audit. `root-audit/` contains the original 23-control diagnostic, its executed script and the exact source bytes it read. All 23 route inputs are included in `inputs.tar.gz`. Archived certification flags and the pre-existing inventory-to-Result binding are retained; this package does not rerun the official/internal physics verifiers, and it does not contain trajectory files sufficient to claim a fresh full-fleet certificate.

The original 612 indexed kit files reconstruct byte-for-byte: the 190-file f8b2ac7a source is stored once in `source-base.tar.gz`, and the initial/final `search.py` overlays are in `source-overlays.tar.gz`. The initial overlay was never executed. `reconstruction.json` maps the 612 entries and all 26 original root-diagnostic files to these archives/raw files. The final production source hash is 9c5d64e37ceea7e5fd8c680556cd41bfbbb1e5a282aacba5d3ede0cbe0797d18. The complete raw kit manifest and all superseded failure evidence are retained. No binaries, credentials, caches or new numerical run outputs were added.

## Package-only use

Python 3.12 is the recorded runtime. `verify`, `extract` and `audit` need only Python's standard library and this package. `bridges` requires NumPy and the existing pinned catalogue/bonus data; `tests` additionally needs pytest/scipy and their dependencies. No command downloads data or loads a GPU/solver library. The test harness allows only libc.so.6 for heap accounting. Reproduction adapters relocate input/data paths in memory; archived files and source hashes stay unchanged.

```sh
python -B reproduce.py verify
python -B reproduce.py extract --work /tmp/consistent-completion-v619-extracted
python -B reproduce.py audit --work /tmp/consistent-completion-v619-audit
python -B reproduce.py bridges --work /tmp/consistent-completion-v619-bridges --data /path/to/pinned/gtoc12/data
python -B reproduce.py tests --work /tmp/consistent-completion-v619-tests --data /path/to/pinned/gtoc12/data
```

Every work directory must be new. `bridges` deliberately repeats the 40 CPU bridge evaluations, then audits them; `audit` recomputes only the saved-output/component checks; `tests` repeats the 45 CPU tests. Packaging validated archive reconstruction only and did not execute these numerical replays again. Runtime dependencies/data are external; the geometry-free saved-output audit is fully self-contained.

## Next bounded proposal, not launched

The next milestone is same-pool native completion/parity and systematic residual-cost attribution, before wider generation or another timing search. Use exactly these five preselected schedules, two prefix masses and two frozen model configurations: 20 native forward evaluations with saved DVs, compared against the 20 corrected CPU outcomes at every leg. There should be zero generation, Lambert search or cargo changes. The current native joint API uses retimer flat/ratio/return models and per-pair calibration; it does not implement the DP five-feature fit used by the −6.39 kg CPU case. A small native finishing adapter must therefore preserve that fit and certified-cell provenance before claiming parity. Silently replacing it with the retimer model would invalidate this comparison.

In parallel, decompose residual error at each archived measured burn mass across all 23 historical controls, separating deployment, collection and return costs without refitting. The saved final 27463→30805 collect hop on ship 23 has 84.296016 kg sequential proxy fuel versus 76.686179 kg archived measured fuel, at masses 1370.877223 versus 1369.983590 kg. That identifies a useful residual to investigate; it is not a guaranteed saving or proof that this one leg causes the entire route deficit. If new solver truth is needed, permit at most two fixed-epoch native leg refinements for that hop: an archived-mass positive control and the exact sequential-proxy mass case, with a hard 120-second budget and unchanged certification gates. A failed positive control stops interpretation of the probe.

Do not run wider generation until measured-prefix incumbent admission is resolved under an explicit validated cost model. Any resulting route must still undergo the complete low-thrust refinement and both final fleet checkers before promotion. This is a proposal only; the 20 native parity evaluations, residual analysis and two leg refinements have not been executed or prepared as a runnable experiment here.
"""


if __name__ == "__main__":
    main()
