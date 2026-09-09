"""Portable, stdlib-only saved evidence verification. Executes no archived code."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
import zipfile

BUILD = "build/performance/gpu-scaled-zoh-v631"
MATH = "build/performance/mass-scaled-seed-review-v631"
SEED = "build/performance/native-seed-audit-v631"
OLD = "build/performance/gpu-route-ephemeris-v630"
GUARD = "c7039b63b8ba8e1745d77fcb759c83460f6eca309879d304bb60ae3e671f4422"
MANIFEST = "b92df5cc06f0119e0958b85935334b82910967f580f0d02bab2bde8d39702e11"
MATH_INDEX = "fca257571227ad11d813a22eb2ae4b38b702875b73ecf9bf42caf223a7cacaf4"
SEED_INDEX = "157f0a8a0891709554f5aa12e0456b50f17c5ae26cc32ac7e4b2ea5f7e790bc0"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def namesafe(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and "\\" not in name and ":" not in name
        and all(v not in ("", ".", "..") for v in name.split("/")), "Unsafe name: " + name)
    require(not any(v in (".git", "__pycache__", ".pytest_cache", ".ruff_cache") for v in p.parts), "Cache/Git member")
    require(not re.search(r"\.(?:exe|dll|dylib|pyd|pyc|o|obj|a|lib|cubin|fatbin|so(?:\.\d+)*)$", name, re.I), "Compiled member")


def safe_content(name, raw, counts, depth=0):
    namesafe(name)
    require(depth < 8, "Archive nesting limit")
    require(not raw.startswith((b"\x7fELF", b"MZ")), "Compiled executable content")
    require(not re.search(rb"^-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\r\n]", raw, re.M), "Private key content")
    if name.endswith((".tar.gz", ".tgz")):
        counts["nested_archives"] += 1
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as t:
            members = t.getmembers()
            require(len(members) == len({m.name for m in members}), "Duplicate tar member")
            for m in members:
                require(m.isfile() and m.size <= 64 * 1024**2, "Nonregular/oversized tar member")
                counts["nested_members"] += 1
                safe_content(m.name, t.extractfile(m).read(), counts, depth + 1)
    elif raw.startswith(b"PK\x03\x04"):
        counts["nested_archives"] += 1
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = z.infolist()
            require(len(members) == len({m.filename for m in members}), "Duplicate nested zip member")
            for m in members:
                require(not m.is_dir() and not stat.S_ISLNK(m.external_attr >> 16)
                    and m.file_size <= 64 * 1024**2, "Nonregular/oversized nested member")
                counts["nested_members"] += 1
                safe_content(m.filename, z.read(m), counts, depth + 1)


def verify(package, expected_index=None, roundtrip=False):
    package = Path(package).resolve()
    raw_index = (package / "index.json").read_bytes()
    if expected_index:
        require(sha(raw_index) == expected_index, "Index pin mismatch")
    index = json.loads(raw_index)
    require(index["schema"] == "gpu-scaled-zoh-v631-v1", "Wrong schema")
    for name, pin in index["top_files"].items():
        namesafe(name)
        raw = (package / name).read_bytes()
        require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "Top file pin: " + name)
    raw_archive = (package / "evidence.zip").read_bytes()
    require(len(raw_archive) == index["archive_bytes"] and sha(raw_archive) == index["archive_sha256"], "Archive pin")
    counts = {"nested_archives": 0, "nested_members": 0}
    files = {}
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as z:
        members = z.infolist()
        require(len(members) == len({m.filename for m in members}) == index["file_count"], "Duplicate/count mismatch")
        require(set(z.namelist()) == set(index["files"]), "Member map mismatch")
        for m in members:
            require(not m.is_dir() and not stat.S_ISLNK(m.external_attr >> 16), "Link/directory in ZIP")
            require(m.file_size <= 64 * 1024**2, "Member exceeds size bound")
            raw = z.read(m)
            pin = index["files"][m.filename]
            require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "Member pin: " + m.filename)
            safe_content(m.filename, raw, counts)
            files[m.filename] = raw
    require(sum(map(len, files.values())) == index["total_bytes"], "Total bytes")

    def read(name):
        require(name in files, "Missing dependency: " + name)
        return files[name]

    def js(name):
        return json.loads(read(name))

    def source_tree(prefix):
        manifest = js(prefix + "/manifest.json")
        archive = read(prefix + "/source.tar.gz")
        require(sha(archive) == manifest["source_archive_sha256"], "Source archive mismatch")
        values = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as t:
            members = t.getmembers()
            require(len(members) == len({m.name for m in members}), "Source duplicates")
            for m in members:
                require(m.isfile(), "Source link/directory")
                values[m.name] = t.extractfile(m).read()
        require(set(values) == set(manifest["sources"]), "Source map mismatch")
        require(all(sha(values[k]) == v for k, v in manifest["sources"].items()), "Source member hash")
        tree = sha("".join(k + ":" + v + "\n" for k, v in manifest["sources"].items()).encode())
        require(tree == manifest["source_tree_sha256"], "Source tree digest")
        return manifest, values

    require(sha(read(BUILD + "/manifest.json")) == MANIFEST, "Final build manifest pin")
    manifest, sources = source_tree(BUILD)
    require(manifest["complete"] and manifest["passed"] and manifest["GPU_calls"] == 0, "Build status")
    require(manifest["compiled_object_reuse"] is False, "Build scope")
    for stage in manifest["stages"]:
        require(stage["exit_code"] == stage["expected_exit"] == 0, "Failed build stage")
        require(sha(read(BUILD + "/" + stage["name"] + ".log")) == stage["log_sha256"], "Build log pin")
    require(sha(read(BUILD + "/compile-commands.json")) == manifest["compile_commands_sha256"], "Compile flags pin")
    require(b"92 passed, 27 skipped" in read(BUILD + "/cpu-tests.log"), "CPU test scope")
    for prefix, pin in ((MATH, MATH_INDEX), (SEED, SEED_INDEX)):
        require(sha(read(prefix + "/index.json")) == pin, "Review index pin")
        review_index = js(prefix + "/index.json")
        require(review_index["file_count"] == len(review_index["files"]), "Review count")
        for name, entry in review_index["files"].items():
            raw = read(prefix + "/" + name)
            require(len(raw) == entry["bytes"] and sha(raw) == entry["sha256"], "Review file pin")
    review = js(MATH + "/findings.json")
    require(review["source_review_passed"] and review["original_gates_unchanged"], "Review status")
    require(len(review["source_sha256"]) == 6 and review["source_sha256"] == manifest["owned_sha256"], "Six reviewed sources")
    require(all(sha(sources[k]) == v for k, v in review["source_sha256"].items()), "Six source bytes")
    seed = js(SEED + "/findings.json")
    for name, pin in seed["input_hashes"].items():
        raw = read(name)
        require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "Historical diagnosis input pin")
    old_manifest, old_sources = source_tree(OLD)
    for name, pin in seed["source_hashes"].items():
        require(sha(old_sources[name]) == pin["compiled_sha256"], "Historical compiled source")
        require(read(SEED + "/frozen-source/" + name) == old_sources[name], "Recovered historical source")

    runs = BUILD + "/gpu-tests-a"
    result = js(runs + "/report.json")
    require(result["complete"] and result["passed"] and result["optimizer_calls"] == 0, "Native test status")
    require(result["manifest_sha256"] == MANIFEST and result["source_tree_sha256"] == manifest["source_tree_sha256"], "Native/build binding")
    require(sha(read(runs + "/run.py")) == result["runner_sha256"], "Supervisor binding")
    require(sha(read(runs + "/process_guard.py")) == result["process_guard_sha256"] == GUARD, "Original guard binding")
    require(result["process_budget"] == 4 and [s["name"] for s in result["stages"]] ==
        ["original-zoh", "scaled-zoh", "controller", "scaled-memcheck"], "Finite native run scope")
    for stage in result["stages"]:
        require(stage["exit_code"] == 0 and stage["cleanup"]["verified_empty"]
            and not stage["cleanup"]["survivors"] and not stage["compute_after"], "Native cleanup/status")
        require(sha(read(runs + "/" + stage["name"] + ".log")) == stage["log_sha256"], "Native raw log")
    require(result["final_cleanup"]["verified_empty"] and not result["final_compute_inventory"], "Final cleanup")
    require(b"reference_calls=1 finite_scaled_calls=3 invalid_calls=8 optimizer_calls=0" in read(runs + "/scaled-zoh.log"), "Scaled cases")
    require(b"ERROR SUMMARY: 0 errors" in read(runs + "/scaled-memcheck.log"), "Memcheck result")
    require(b"PASS: exact ZOH schedule" in read(runs + "/original-zoh.log"), "Legacy regression")
    require(b"PASS: boundary-corrective acceptance" in read(runs + "/controller.log"), "Controller regression")
    require(index["external_runtime_binaries"] == manifest["outputs"], "Runtime dependency declaration")
    if roundtrip:
        with tempfile.TemporaryDirectory(prefix="scaled-zoh-saved-") as temp:
            dest = Path(temp)
            for name, raw in files.items():
                p = dest.joinpath(*PurePosixPath(name).parts)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(raw)
            for name, pin in index["files"].items():
                p = dest.joinpath(*PurePosixPath(name).parts)
                require(p.stat().st_size == pin["bytes"] and sha(p.read_bytes()) == pin["sha256"], "Roundtrip pin")
    return {"passed": True, "index_sha256": sha(raw_index), "archive_sha256": sha(raw_archive),
        "file_count": len(files), "total_bytes": index["total_bytes"], "nested": counts,
        "new_source_members": len(sources), "historical_source_members": len(old_sources),
        "reviewed_source_files": 6, "native_processes": 4, "unit_optimizer_calls": 0,
        "package_GPU_calls": 0, "package_numerical_calls": 0, "roundtrip": roundtrip,
        "scope": "Static saved bytes/source/status only; absent external runtime binaries are declared, not reverified."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--expected-index")
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package, args.expected_index, args.roundtrip)
    raw = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(raw)
    print(raw, end="")


if __name__ == "__main__":
    main()
