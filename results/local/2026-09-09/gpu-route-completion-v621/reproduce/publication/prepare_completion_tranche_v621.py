"""Audit the exact completion change and evidence before any Git write or upload."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "results/local/2026-09-09/gpu-route-completion-v621"
OUT = ROOT / "build/performance/completion-tranche-v621"
SHA = lambda data: hashlib.sha256(data).hexdigest()
native = {
    "cpp/cuda/include/spacepdhcg/cuda/gtoc12_completion_c_api.h": "e176bb0d5848981a140e264a71d10ac744ce1babc946d5edf5caf01dc3f9dc76",
    "cpp/cuda/src/gtoc12_completion.cu": "b81d72702238a1096a5cec0e49b9b07c3738c4d3ac657cafb116993651a99c2b",
    "cpp/cuda/tests/gtoc12_completion_test.cu": "52c127e2f66b2c28eb046016a72d8a2c919ae044720ff139cea9679928329faf",
    "cpp/cuda/CMakeLists.txt": "4fa338c5ae6f153e9ebe89def13da183b0b6f04f1c531e0c0680e6cfdb9b9aac",
}
host = json.loads((ROOT / "build/performance/completion-adapter-cpu-v621b/report.json").read_text())
python_paths = [
    "src/spacepdhcg/gtoc12/gpu_completion.py", "src/spacepdhcg/gtoc12/gpu_lambert.py",
    "src/spacepdhcg/gtoc12/search.py", "tests/test_gtoc12_gpu_completion.py",
]
for name, digest in {**native, **{p: host["source_sha256"][p] for p in python_paths}}.items():
    assert SHA((ROOT / name).read_bytes()) == digest, name
owned = [*native, *python_paths, "docs/GPU_ROUTE_COMPLETION.md", "docs/SOTA_EXECUTION_PLAN_2026-09-09.md"]
index_bytes = (PACKAGE / "sha256.json").read_bytes()
records = json.loads(index_bytes)
records = records.get("files", records)
paths = {p.relative_to(PACKAGE).as_posix() for p in PACKAGE.rglob("*") if p.is_file()}
assert paths == set(records) | {"sha256.json"}, paths ^ (set(records) | {"sha256.json"})
for name, record in records.items():
    path = (PACKAGE / name).resolve()
    assert path.is_relative_to(PACKAGE.resolve()) and not path.is_symlink(), name
    data = path.read_bytes()
    assert len(data) == record["bytes"] and SHA(data) == record["sha256"], name
owned += [(PACKAGE / name).relative_to(ROOT).as_posix() for name in sorted(paths)]

secret = re.compile(rb"(?m)^-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----\r?$")
denied = {".pem", ".key", ".so", ".dll", ".exe", ".pyd", ".pyc", ".a", ".o", ".obj"}
member_count = 0

def check(name, data):
    global member_count
    path = PurePosixPath(name.replace("\\", "/"))
    assert not path.is_absolute() and ".." not in path.parts, name
    assert path.suffix.lower() not in denied and "__pycache__" not in path.parts, name
    assert not secret.search(data), "private key material: " + name
    assert not data.startswith((b"\x7fELF", b"MZ")), "executable file: " + name
    if name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            seen = set()
            for member in archive:
                assert member.name not in seen and (member.isfile() or member.isdir()), member.name
                seen.add(member.name)
                p = PurePosixPath(member.name)
                assert not p.is_absolute() and ".." not in p.parts, member.name
                if member.isfile():
                    check(member.name, archive.extractfile(member).read())
                    member_count += 1
    elif name.endswith((".npz", ".zip")):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            assert len(archive.namelist()) == len(set(archive.namelist()))
            for member in archive.infolist():
                if not member.is_dir():
                    check(member.filename, archive.read(member))
                    member_count += 1

files = {}
for name in sorted(owned):
    path = ROOT / name
    assert not path.is_symlink()
    data = path.read_bytes()
    check(name, data)
    assert len(data) < 100_000_000, name
    files[name] = {"bytes": len(data), "sha256": SHA(data)}
assert len(files) == len(owned)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
report = {
    "scope": "Opt-in retained CUDA route completion, search batching, evidence and roadmap",
    "head_at_audit": head, "paths": files, "file_count": len(files),
    "bytes": sum(x["bytes"] for x in files.values()), "archive_members_inspected": member_count,
    "package_index_sha256": SHA(index_bytes), "package_files": len(paths),
    "destination_repository": "https://github.com/terrorproforma/spacecraft-trajectory-optmiser.git",
    "frozen_owned_sources_match": True, "private_keys_binaries_caches": "none found",
    "network_actions": 0,
}
OUT.mkdir(exist_ok=False)
(OUT / "payload.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
(OUT / "paths.nul").write_bytes(b"\0".join(p.encode() for p in files) + b"\0")
print(json.dumps({k: v for k, v in report.items() if k != "paths"}, indent=2))
