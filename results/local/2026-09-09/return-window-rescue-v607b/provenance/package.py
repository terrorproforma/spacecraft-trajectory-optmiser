"""Publish exact v607b evidence locally; no solver, network, or Git operations."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "build/performance/return-window-rescue-v607b"
AUDIT = REPO / "build/performance/return-rescue-audit-v607b"
DEST = REPO / "results/local/2026-09-09/return-window-rescue-v607b"
READY_SHA = "9414730e31bc2d896285761dc6436277cefb7d92af8a297291d9e3116a2c28ec"
AUDIT_SHA = "0005b09c653bf47f3e07f3d161387c1fdbb30f6624641213bff42087dfb8babb"
FORBIDDEN_SUFFIXES = {".so", ".dll", ".exe", ".a", ".lib", ".o", ".obj", ".pyd", ".pyc"}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inventory(folder):
    return {
        p.relative_to(folder).as_posix(): sha(p)
        for p in sorted(folder.rglob("*"))
        if p.is_file() and not set(p.parts) & {"__pycache__", ".pytest_cache", ".ruff_cache"}
    }


def checked_copy(source, destination):
    if source.suffix in FORBIDDEN_SUFFIXES:
        raise ValueError(f"Compiled artifact excluded: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != sha(destination):
        raise AssertionError(f"Copy mismatch: {source}")


def archive(folder, label, expected):
    path = DEST / (label + ".tar.gz")
    with (
        path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, checksum in sorted(expected.items()):
                source = folder / name
                if source.suffix in FORBIDDEN_SUFFIXES or sha(source) != checksum:
                    raise ValueError(f"Disallowed or changed archive member: {source}")
                data = source.read_bytes()
                info = tarfile.TarInfo(label + "/" + name)
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    restored = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise AssertionError("Archive contains unsafe/nonregular member")
            name = Path(member.name).relative_to(label).as_posix()
            if name in restored:
                raise AssertionError("Duplicate archive member")
            blob = tar.extractfile(member).read()
            restored[name] = hashlib.sha256(blob).hexdigest()
            if blob != (folder / name).read_bytes():
                raise AssertionError("Archive full roundtrip byte mismatch")
    if restored != expected:
        raise AssertionError("Archive inventory differs from exact frozen manifest")
    return {
        "path": path.name,
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "file_count": len(restored),
        "all_members_roundtrip_byte_identical": True,
        "members_sha256": restored,
    }


def main():
    if DEST.exists():
        raise FileExistsError("Publication directory must be new")
    original_kit, original_audit = inventory(KIT), inventory(AUDIT)
    if (
        sha(KIT / "ready-manifest.json") != READY_SHA
        or sha(AUDIT / "final-report.json") != AUDIT_SHA
    ):
        raise AssertionError("Reviewed run/audit identity changed")
    ready, report, launch = (
        read(KIT / "ready-manifest.json"),
        read(KIT / "output/report.json"),
        read(KIT / "launch.json"),
    )
    for name, checksum in ready["files"].items():
        if sha(KIT / name) != checksum:
            raise AssertionError(f"Frozen kit changed: {name}")
    if not report["complete"] or report["promotions"] or report["native_returns_completed"] != 5:
        raise AssertionError("Unexpected result scope")
    if launch["returncode"] != 0 or len(report["cases"]) != 5 or not report["control_passed"]:
        raise AssertionError("Incomplete execution/control")
    raw_files = inventory(KIT / "output")
    if (
        len(raw_files) != 33
        or {"output/" + k: v for k, v in raw_files.items()}
        != read(AUDIT / "final-report.json")["output_files"]
    ):
        raise AssertionError("Raw output/audit disagreement")
    source = read(KIT / "source-sha256.json")
    inputs = {name: entry["sha256"] for name, entry in read(KIT / "inputs-sha256.json").items()}
    if len(source) != 190 or len(inputs) != 78:
        raise AssertionError("Frozen source/input cardinality changed")
    DEST.mkdir(parents=True)
    copied = {}
    for name in [name for name in ready["files"] if not name.startswith(("source/", "inputs/"))] + [
        "ready-manifest.json",
        "launch.json",
        "run.log",
    ]:
        target = "execution/" + name
        checked_copy(KIT / name, DEST / target)
        copied[target] = {"origin": "kit/" + name, "sha256": sha(KIT / name)}
    for name in raw_files:
        target = "output/" + name
        checked_copy(KIT / "output" / name, DEST / target)
        copied[target] = {"origin": "kit/output/" + name, "sha256": raw_files[name]}
    for name, checksum in original_audit.items():
        target = "audit/" + name
        checked_copy(AUDIT / name, DEST / target)
        copied[target] = {"origin": "independent-audit/" + name, "sha256": checksum}
    checked_copy(Path(__file__), DEST / "provenance/package.py")
    source_archive, input_archive = (
        archive(KIT / "source", "source", source),
        archive(KIT / "inputs", "inputs", inputs),
    )
    restored_ready = {}
    for name, checksum in ready["files"].items():
        if name.startswith("source/"):
            actual = source_archive["members_sha256"][name.removeprefix("source/")]
        elif name.startswith("inputs/"):
            actual = input_archive["members_sha256"][name.removeprefix("inputs/")]
        else:
            actual = sha(DEST / "execution" / name)
        if actual != checksum:
            raise AssertionError("Ready reconstruction mismatch")
        restored_ready[name] = actual
    if inventory(KIT) != original_kit or inventory(AUDIT) != original_audit:
        raise AssertionError("Original run/audit changed during publication")
    write(
        DEST / "provenance/archive-audit.json",
        {
            "roundtrip_ok": True,
            "archives": [source_archive, input_archive],
            "ready_manifest_sha256": READY_SHA,
            "ready_files_reconstructed": len(restored_ready),
            "ready_files_reconstructed_exactly": restored_ready == ready["files"],
            "all_raw_outputs_preserved": len(raw_files),
            "copied_files": copied,
            "original_kit_and_audit_unchanged": True,
            "compiled_binaries_included": 0,
            "GPU_calls": 0,
            "remote_operations": 0,
        },
    )
    native_source = (
        REPO / "results/local/2026-09-09/joint-candidates-v596/original-v590/native-source.tar.gz"
    )
    write(
        DEST / "publication.json",
        {
            "experiment": "return-window-rescue-v607b",
            "execution_environment": "local RTX 5090",
            "actual_native_return_solves": 5,
            "actual_control_returns": 1,
            "actual_candidate_returns": 4,
            "whole_route_reruns": 0,
            "H100_executions": 0,
            "status": report["status"],
            "promotions": 0,
            "raw_mass_kg": report["best"]["total_mass_kg"],
            "fixed_bonus_weighted_score_kg": report["best"]["score_kg"],
            "ready_sha256": READY_SHA,
            "output_report_sha256": sha(KIT / "output/report.json"),
            "independent_audit_sha256": AUDIT_SHA,
            "source_commit": report["source_commit"],
            "source_manifest_sha256": sha(KIT / "source-sha256.json"),
            "native_libraries": report["native_libraries"],
            "native_source": {
                "relative_path": "../joint-candidates-v596/original-v590/native-source.tar.gz",
                "sha256": sha(native_source),
            },
            "inherited_metadata": {
                "execution/preparation.json": "v606 provenance and demonstrated full-route settings; not v607b actual work counts",
                "execution/profiles.json": "v606 profiles retained exactly; v607b used only local profile",
                "execution/README.md": "Historical frozen preparation; GPU execution was pending when written",
            },
            "reproduction": "Restore source/ and inputs/ archives alongside execution files in a fresh kit directory; requires pinned external data and native libraries. Historical launch marker prevents automatic replay.",
        },
    )
    (DEST / ".gitattributes").write_text("* -text whitespace=cr-at-eol\n*.log -whitespace\n")
    (DEST / "README.md").write_text(README)
    index = {
        name: {"bytes": (DEST / name).stat().st_size, "sha256": checksum}
        for name, checksum in inventory(DEST).items()
    }
    write(DEST / "sha256.json", index)
    for name, entry in read(DEST / "sha256.json").items():
        if sha(DEST / name) != entry["sha256"] or (DEST / name).stat().st_size != entry["bytes"]:
            raise AssertionError("Final publication hash mismatch")
    if set(inventory(DEST)) != set(index) | {"sha256.json"}:
        raise AssertionError("Unindexed publication file")
    print(
        json.dumps(
            {
                "destination": str(DEST),
                "indexed_files": len(index),
                "indexed_bytes": sum(v["bytes"] for v in index.values()),
                "sha256_index": sha(DEST / "sha256.json"),
                "archive_audit_sha256": sha(DEST / "provenance/archive-audit.json"),
                "source_archive_sha256": source_archive["sha256"],
                "inputs_archive_sha256": input_archive["sha256"],
                "ready_files_reconstructed": len(restored_ready),
                "raw_outputs": len(raw_files),
            },
            indent=2,
        )
    )


README = """# Fixed-cargo return-window rescue v607b

The bounded local RTX 5090 experiment **did not improve the fleet**. The retained
23-ship solution remains **14,051.854893908598 raw kg / 12,810.135953048577 fixed-bonus
weighted kg**. No candidate was promoted and no new visualizer result was added.

The test preserved the 16 certified prefix legs of the v606 ship-7 replacement,
including its cargo, deployment/collection events and emitted thrust samples.
It changed only waiting after final collection and the Earth-return window.
Actual work was **one original-prefix control return plus four candidate returns**,
with **zero whole-route reruns**. The successful control was freshly solved and
passed both fleet checkers. All four replacement returns remained uncertified.
The actual execution was local; **no H100 job or transfer occurred for v607b**.

| Recorded work | Count / time |
| --- | ---: |
| Legal screened windows | 7,022 coarse + 2,416 fine = 9,438 |
| Lambert direction requests | 18,876 |
| Fresh native return solves | 5 |
| SCvx iterations / accepted iterations | 142 / 53 |
| QOCO report rows / reported inner iterations | 142 / 12,117 |
| Total wall time | 86.221170 s |
| Native-wrapper wall time, including host overhead | 41.463645 s |
| Baseline and control full-fleet verification | 41.723452 s |

These work units describe stages of five return solves. They are not independent
whole-route solutions, and the wall times are not pure GPU kernel timings.

The replacement prefix left **91.543915 kg** of return propellant. The cheapest
screened return estimate was **213.617946 kg**, a **122.074031 kg** deficit.
All four selected windows received native refinement despite negative proxy
reserves. They reached the fixed optimizer fuel allowance with nonzero defects
of approximately 0.00218944, 0.00218944, 0.00744544 and 0.01000789. Every failure's
arrays, diagnostics and prescribed cargo were retained; no partial trajectory
was scored as a fleet.

The replacement prefix had already spent approximately **95.392567 kg more fuel**
than its original counterpart. The tested return-date changes did not recover
that loss. A distinct next hypothesis should reduce upstream prefix fuel use or
change the replacement geometry. Proxy estimates and failed SCvx solves do not
prove physical infeasibility, and this sampled epoch grid is not exhaustive.

The [independent artifact audit](audit/final-report.json) verified all five native
calls, the preserved control prefix and other 22 ships, both control fleet gates,
all raw screening arithmetic and deterministic diverse selection, and absence
of false promotion. Its [readable interpretation](audit/README.md) includes all
four failures. The frozen kit passed **44 CPU behavioral tests** and Ruff checks;
see [validation evidence](execution/validation/report.json). Those tests include
byte-identical original-control reconstruction, waiting-coast/perihelion checks,
immutable events/cargo, failed-call retention and exclusive launch guards.

`execution/` preserves the exact ready manifest, driver, foreground supervisor,
completed launch record, inherited profiles/settings and validation logs.
`source.tar.gz` contains all **190** frozen Python/benchmark source files, and
`inputs.tar.gz` contains all **78** frozen route/source-lineage inputs. Members
retain the original bytes. Restoring both archives alongside the execution
files reconstructs every one of the **286** ready-indexed files exactly.
`output/` contains **all 33 raw outputs**, including failed NPZs; `audit/` preserves
the unchanged independent audit. [Archive/copy audit](provenance/archive-audit.json)
checks full member roundtrips, all copied hashes and unchanged original files.
`sha256.json` indexes every publication file except itself; raw Git attributes
prevent line-ending conversion from changing evidence hashes. No compiled
binaries are included.

Python is frozen at `b08b1f5a`; native libraries are the demonstrated local v590
core and QOCO540, with exact hashes in [publication provenance](publication.json).
The [previously published native source](../joint-candidates-v596/original-v590/native-source.tar.gz)
has SHA256 `6f10438bd50bb52401f25d1fe3d44524fdf1028080c0e9e5644c0037e4dbdf33`.
Re-execution also requires the pinned catalogue, bonus data and native builds;
this publication neither includes binaries nor automatically launches a run.

`execution/preparation.json` and `execution/profiles.json` are inherited v606
lineage, including an unused H100 profile. They do not describe extra v607b work.
`execution/README.md` is the unchanged pre-run preparation record; its pending
execution statement is historical. The authoritative actual run is
[output/report.json](output/report.json) with the completed local
[launch record](execution/launch.json). Historical absolute paths identify the
original workspace and are preserved without rewriting raw evidence.
"""


if __name__ == "__main__":
    main()
