"""Package the completed v611 evidence without changing its frozen kit or raw output."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "build/performance/coupled-fuel-search-v611"
AUDIT = REPO / "build/performance/coupled-fuel-audit-v611"
DEST = REPO / "results/local/2026-09-09/coupled-fuel-search-v611"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def copy(origin, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    assert sha(origin) == sha(target)


def archive(name, members):
    target = DEST / (name + ".tar.gz")
    with target.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as stream:
            for relative in sorted(members):
                payload = (KIT / name / relative).read_bytes()
                member = tarfile.TarInfo(relative)
                member.size, member.mode, member.mtime = len(payload), 0o644, 0
                stream.addfile(member, io.BytesIO(payload))
    roundtrip = {}
    with tarfile.open(target, "r:gz") as stream:
        for member in stream.getmembers():
            assert member.isfile() and not Path(member.name).is_absolute()
            assert ".." not in Path(member.name).parts
            digest = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            assert digest == members[member.name]
            roundtrip[member.name] = digest
    assert roundtrip == members
    return {"archive_sha256": sha(target), "files": len(members), "members": roundtrip}


def main():
    if DEST.exists():
        raise FileExistsError("New publication directory required")
    ready = read(KIT / "ready-manifest.json")
    initial = {name: sha(KIT / name) for name in ready["files"]}
    assert initial == ready["files"]
    report = read(AUDIT / "report.json")
    raw_before = {name: sha(KIT / "output" / name) for name in report["raw_output_files"]}
    assert raw_before == report["raw_output_files"]
    DEST.mkdir(parents=True)
    archives = {}
    for name in ("source", "inputs"):
        members = {key[len(name) + 1:]: value for key, value in initial.items() if key.startswith(name + "/")}
        archives[name] = archive(name, members)
    assert archives["source"]["files"] == 190 and archives["inputs"]["files"] == 33
    copied = {}
    for name, digest in initial.items():
        if name.startswith(("source/", "inputs/")):
            continue
        copy(KIT / name, DEST / "kit" / name)
        copied[name] = digest
    copy(KIT / "ready-manifest.json", DEST / "kit/ready-manifest.json")
    for name in ("launch.json", "run.log"):
        copy(KIT / name, DEST / "execution" / name)
    for name in raw_before:
        copy(KIT / "output" / name, DEST / "output" / name)
    for name in ("audit.py", "report.json"):
        copy(AUDIT / name, DEST / "audit" / name)
    copy(Path(__file__), DEST / "audit/package.py")
    (DEST / ".gitattributes").write_text("# Preserve exact evidence bytes and indexed hashes.\n* -text\n")
    (DEST / "README.md").write_text("""# Coupled itinerary fuel search v611 — no verified gain

The single local RTX 5090 run completed normally in **34.844880 seconds**. It evaluated **39,852 complete itinerary proxy candidates** across 12 device searches and retained 202 accepted timing moves. Actual screening work was 793,152 Lambert direction requests. The frozen weighted score remains **12,810.135953048577 kg**, with **14,051.854893908598 raw kg** across 23 ships. The retained baseline passed both full-fleet checkers during this run.

Four independent eight-miner routes (ships20,7,10,11) were searched with the same mesh/caps at margin prices0.05,0.25,1.0. All four price0.05 outcomes increased predicted weighted cargo. The frozen one-candidate-per-price policy selected only the largest gain: ship20, +6.033277350309 predicted weighted kg (+6.351813826147 raw kg). The other price settings increased estimated spare fuel at the expense of weighted cargo and failed the improvement gate.

The selected route used **17 native leg solves**. Its first16 legs obtained independent certificates; its Earth-return leg34525→Earth, MJD69298→69698, remained uncertified after25 SCvx iterations/four accepted steps. Dynamics defect was0.0136137792152. Its fixed598.220396988364kg cargo left292.503658693290kg return propellant, which the uncertified optimizer exhausted. Cargo and epochs stayed unchanged. No candidate full-fleet Result or promotion was emitted. This is a failed refinement, not proof of physical infeasibility.

The three unrefined price0.05 gains remain in the raw searches: ship7 +0.372079193363, ship10 +0.245577079815, ship11 +0.123515282719 weighted kg. They were outside the frozen one-per-price shortlist, not proven infeasible. The unused refinement allowance was not spent, and no retry or extra GPU job was launched.

Measured scopes: search wrappers0.450427451s (about88,476 proxy evaluations/s, including first-call overhead); native leg wrappers12.091912440s; fresh baseline independent+official checks20.329613125s. These are distinct stages. Proxy evaluations/s is not certified solutions/s, and this experiment is not an A/B speedup claim.

The one-shot supervisor ran PID290/child378 with a600s wall budget,630s outer timeout and10s termination grace. It exited0 before timeout. `execution/launch.json` preserves process, device, environment and deadline evidence; the CPU validation includes a real harmless child timeout test. The single-launch marker and all failures remain intact in the original kit.

## Reproducibility and evidence

- `output/` preserves all50 raw files, including17 leg NPZs and all12 search outcomes.
- `kit/ready-manifest.json` retains the exact247-file preparation index, SHA256 `1d9428be654cf30f49367d5ba98f4a2674d810fc4e2ee728906b4518a1cf8986`.
- `source.tar.gz` contains all190 frozen source files from `f8b2ac7aeba9e29ea90f5814060bb8cfe5db22d7`; `inputs.tar.gz` contains all33 exact inputs, including the incumbent and archived equivalent controls. Extract them into `kit/source/` and `kit/inputs/` respectively to reconstruct the prepared kit.
- `kit/validation/` retains31 passing CPU tests and Ruff check/format logs. No CUDA was loaded during that validation or the post-run audit.
- `kit/profile.json` pins the actual local v702 core (`65f1335e…`) and v683 QOCO (`5b1b1a04…`). Binaries are omitted. The original v707 implementation/runtime evidence is under `results/lambda/2026-09-09/gpu-device-search-v707`; this v611 run was local. Prior local active-loop memcheck CUDA999 remains unresolved and is not represented as a pass.
- `audit/report.json` independently checks indexed bytes, weighted totals, raw fleet eligibility, shortlist exclusions, actual work, all leg mass/event accounting, retained certificates and the absence of false promotion. It audits recorded physical certificates; it does not repeat numerical propagation.
- `archive-audit.json` verifies all archive members and reconstructs every prepared hash. `sha256.json` indexes every publication file except itself. `.gitattributes` preserves raw evidence bytes.

This fixed-order search does not establish that nine- or ten-miner route families are unavailable. The broader frozen search defaults permit ten deployments; several retained fleet ships already have nine miners. Further route-family work needs separate source/pool diagnosis. No new experiment is authorized by this evidence bundle.
""")
    reconstructed = dict(copied)
    for name, entry in archives.items():
        reconstructed.update({name + "/" + member: digest for member, digest in entry["members"].items()})
    assert reconstructed == ready["files"]
    write(DEST / "archive-audit.json", {
        "archives": archives, "reconstructed_ready_files": len(reconstructed),
        "all_ready_hashes_match": True, "all_raw_output_hashes_match": True,
        "raw_output_files": len(raw_before), "ready_sha256": sha(KIT / "ready-manifest.json"),
    })
    assert {name: sha(KIT / name) for name in initial} == initial
    assert {name: sha(KIT / "output" / name) for name in raw_before} == raw_before
    files = {str(p.relative_to(DEST)): {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted(DEST.rglob("*")) if p.is_file()}
    write(DEST / "sha256.json", {"files": files, "file_count": len(files),
          "total_bytes": sum(item["bytes"] for item in files.values())})
    for name, entry in files.items():
        assert sha(DEST / name) == entry["sha256"]
    print(json.dumps({"destination": str(DEST), "indexed_files": len(files),
          "indexed_bytes": sum(item["bytes"] for item in files.values()),
          "index_sha256": sha(DEST / "sha256.json"),
          "archive_audit_sha256": sha(DEST / "archive-audit.json"),
          "audit_report_sha256": sha(DEST / "audit/report.json")}, indent=2))


if __name__ == "__main__":
    main()
