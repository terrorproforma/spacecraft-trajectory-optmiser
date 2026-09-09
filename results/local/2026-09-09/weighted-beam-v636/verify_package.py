"""Portable saved-evidence verification; optional dependency hashes and stdlib replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(test, message):
    if not test:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--roundtrip", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package.parents[3]
    index_raw = (package / "index.json").read_bytes()
    require(sha(index_raw) == args.index_sha256, "Index hash")
    index = json.loads(index_raw)
    for name, pin in index["files"].items():
        path = (package / name).resolve()
        require(path.is_relative_to(package), "Index path boundary")
        data = path.read_bytes()
        require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], name)
    mapping = json.loads((package / "package-map.json").read_bytes())["files"]
    with zipfile.ZipFile(package / "evidence.zip") as archive:
        names = archive.namelist()
        pins = json.loads(archive.read("FILES.json"))
        require(len(names) == len(set(names)) and set(names) == set(pins) | {"FILES.json"}, "Archive inventory")
        for name, pin in pins.items():
            data = archive.read(name)
            require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], name)
            require(mapping[name]["kind"] == "included" and mapping[name]["sha256"] == pin["sha256"], "Map binding")
        for kit, key in (("weighted-beam-v636", "original_ready_sha256"),
                          ("weighted-beam-v636b", "continuation_ready_sha256")):
            prefix = "build/performance/" + kit
            ready_raw = archive.read(prefix + "/ready.json")
            require(sha(ready_raw) == index[key], "Ready hash")
            ready = json.loads(ready_raw)
            for name, digest in ready["files"].items():
                require(mapping[prefix + "/" + name]["sha256"] == digest, "Prepared input binding")
            for name, digest in ready["external_files"].items():
                require(mapping[name]["sha256"] == digest, "External input binding")
            launch = json.loads(archive.read(prefix + "/output/launch-report.json"))
            require(launch["complete"] and launch["passed"] and launch["exit_code"] == 0, "Launch outcome")
            require(launch["cleanup"]["verified_empty"] and launch["compute_after"] in ("", "No running processes found"), "Owned cleanup")
        audit = json.loads((package / "results-analysis.json").read_bytes())
        require(audit["combined_search_calls"] == 4 and audit["GPU_calls_in_this_audit"] == 0, "Actual call count")
        if args.dependencies:
            groups = {}
            for name, pin in mapping.items():
                if pin["kind"] == "existing_file":
                    data = (repo / pin["path"]).read_bytes()
                    require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], name)
                elif pin["kind"] == "existing_tar_member":
                    groups.setdefault(pin["archive"], {})[pin["member"]] = (name, pin)
            for path, members in groups.items():
                expected_archive = next(iter(members.values()))[1]["archive_sha256"]
                require(sha((repo / path).read_bytes()) == expected_archive, "Dependency archive")
                seen = set()
                with tarfile.open(repo / path, "r|gz") as source:
                    for member in source:
                        if member.name in members:
                            name, pin = members[member.name]
                            data = source.extractfile(member).read()
                            require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], name)
                            seen.add(member.name)
                require(seen == set(members), "Dependency member inventory")
        if args.roundtrip:
            with tempfile.TemporaryDirectory(prefix="v636-saved-audit-") as temporary:
                root = Path(temporary).resolve()
                for name in pins:
                    path = (root / name).resolve()
                    require(path.is_relative_to(root), "Extraction boundary")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.read(name))
                here = root / "build/performance/weighted-beam-analysis-v636"
                expected_report = json.loads((here / "results-analysis.json").read_bytes())
                expected_metrics = json.loads((here / "candidate-metrics.json").read_bytes())
                (here / "results-analysis.json").unlink()
                (here / "candidate-metrics.json").unlink()
                result = subprocess.run([sys.executable, "-B", str(here / "analyse_results.py")],
                                        capture_output=True, text=True, timeout=60)
                require(result.returncode == 0, result.stderr)
                require(json.loads((here / "results-analysis.json").read_bytes()) == expected_report, "Saved arithmetic replay")
                require(json.loads((here / "candidate-metrics.json").read_bytes()) == expected_metrics, "Saved metrics replay")
    print(json.dumps({"passed": True, "archive_files": len(pins),
                      "dependency_bytes_checked": args.dependencies, "saved_arithmetic_roundtrip": args.roundtrip,
                      "new_GPU_search_refinement_or_certification_calls": 0}))


if __name__ == "__main__":
    main()
