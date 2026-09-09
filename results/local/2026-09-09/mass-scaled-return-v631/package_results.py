"""Package immutable saved evidence only; no numerical rerun or remote operation."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

KIT = Path(__file__).resolve().parent
ROOT = KIT.parents[2]
DEST = ROOT / "results/local/2026-09-09/mass-scaled-return-v631"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def archive(target, items):
    manifest = {}
    with tarfile.open(target, "w:gz", compresslevel=6) as output:
        for name, source in sorted(items.items()):
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
            payload = source.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = 0
            output.addfile(info, io.BytesIO(payload))
            manifest[name] = {"sha256": sha(payload), "bytes": len(payload)}
    write(target.with_name(target.name.removesuffix(".tar.gz") + "-files.json"), manifest)
    return manifest


assert __debug__
ready = json.loads((KIT / "ready.json").read_bytes())
audit = json.loads((KIT / "saved-audit.json").read_bytes())
assert audit["passed"] and audit["score_unchanged"]
assert sha((KIT / "ready.json").read_bytes()) == "f053a57f9996244c815924d6b41e9cefa8f1fc7c47a652d50a4cd1d3d643fefe"
DEST.mkdir()
prepared = {name: KIT / name for name in ready["files"]} | {"ready.json": KIT / "ready.json"}
for name, digest in ready["files"].items():
    assert sha(prepared[name].read_bytes()) == digest
external = {}
external_map = {}
for name, digest in ready["external_files"].items():
    source = (KIT / name).resolve()
    member = str(source.relative_to(ROOT)).replace("\\", "/")
    assert source.suffix.lower() not in {".so", ".dll", ".exe", ".key", ".pem"}
    assert sha(source.read_bytes()) == digest
    external[member] = source
    external_map[name] = member
raw = {name: KIT / "output" / name for name in audit["raw_output_files"]}
for name, expected in audit["raw_output_files"].items():
    assert sha(raw[name].read_bytes()) == expected["sha256"]
archive(DEST / "frozen-kit.tar.gz", prepared)
archive(DEST / "external-evidence.tar.gz", external)
archive(DEST / "raw-output.tar.gz", raw)
gpu = ROOT / "build/performance/gpu-scaled-zoh-v631/gpu-tests-a"
archive(DEST / "native-validation.tar.gz", {str(p.relative_to(gpu)): p for p in gpu.rglob("*") if p.is_file()})
write(DEST / "external-map.json", external_map)
for name, source in {
    "ready.json": KIT / "ready.json", "protocol.json": KIT / "protocol.json",
    "profile.json": KIT / "profile.json", "report.json": KIT / "output/report.json",
    "launch-report.json": KIT / "output/launch-report.json", "saved-audit.json": KIT / "saved-audit.json",
    "audit_saved.py": KIT / "audit_saved.py", "package_results.py": Path(__file__),
    "native-manifest.json": ROOT / "build/performance/gpu-scaled-zoh-v631/manifest.json",
    "verify_package.py": KIT / "verify_package.py", "README.md": KIT / "PACKAGE_README.md",
}.items():
    (DEST / name).write_bytes(source.read_bytes())
(DEST / ".gitattributes").write_text("* -text -whitespace\n")
write(DEST / "sha256.json", {str(p.relative_to(DEST)): sha(p.read_bytes()) for p in sorted(DEST.rglob("*")) if p.is_file()})
print(json.dumps({"package": str(DEST), "indexed_files": len(json.loads((DEST / "sha256.json").read_bytes())),
                  "index_sha256": sha((DEST / "sha256.json").read_bytes())}, indent=2))
