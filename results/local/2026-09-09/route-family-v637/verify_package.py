"""Portable byte/retained-arithmetic audit; no project numerical imports or GPU."""
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


def require(condition, message):
    if not condition:
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
    require(sha(index_raw) == args.index_sha256, "Index trust anchor")
    index = json.loads(index_raw)
    for name, pin in index["files"].items():
        path = (package / name).resolve()
        require(path.is_relative_to(package), "Package path boundary")
        data = path.read_bytes()
        require(len(data) == pin["bytes"] and sha(data) == pin["sha256"], name)
    mapping = json.loads((package / "package-map.json").read_bytes())["files"]
    with zipfile.ZipFile(package / "evidence.zip") as z:
        names = z.namelist()
        pins = json.loads(z.read("FILES.json"))
        require(len(names) == len(set(names)) and set(names) == set(pins) | {"FILES.json"}, "Archive inventory")
        for name, pin in pins.items():
            raw = z.read(name)
            require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], name)
            require(mapping[name]["kind"] == "included" and mapping[name]["sha256"] == pin["sha256"], "Included source mapping")
        prefix = "build/performance/route-family-v637/"
        ready_raw = z.read(prefix + "ready.json")
        require(sha(ready_raw) == index["ready_sha256"], "Ready identity")
        ready = json.loads(ready_raw)
        require(len(ready["files"]) == ready["file_count"] == 221 and
            sum(mapping[prefix + n]["bytes"] for n in ready["files"]) == ready["bytes"], "Prepared count/bytes")
        for name, digest in ready["files"].items():
            require(mapping[prefix + name]["sha256"] == digest, "Prepared member")
        for name, digest in ready["external_files"].items():
            require(mapping[name]["sha256"] == digest, "External input")
        launch = json.loads(z.read(prefix + "output/launch-report.json"))
        require(launch["complete"] and launch["passed"] and launch["exit_code"] == 0 and
            launch["cleanup"]["verified_empty"] and launch["compute_after"] == "", "Terminal cleanup")
        report = json.loads(z.read(prefix + "output/report.json"))
        audit = json.loads((package / "saved-audit.json").read_bytes())
        require(report["complete"] and report["route_searches_started"] == report["route_searches_completed"] == 4 and
            report["screen_calls_started"] == report["screen_calls_completed"] == 8, "Actual bounded counts")
        require(audit["totals"]["plans"] == 923 and audit["selection"]["positive_raw_eligible_queries"] == 0 and
            audit["GPU_search_refinement_propagation_or_checker_calls_in_audit"] == 0, "Saved outcome")
        if args.dependencies:
            groups = {}
            for name, pin in mapping.items():
                if pin["kind"] == "existing_file":
                    raw = (repo / pin["path"]).read_bytes()
                    require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], name)
                elif pin["kind"] == "existing_tar_member":
                    groups.setdefault(pin["archive"], {})[pin["member"]] = (name, pin)
            for archive_path, members in groups.items():
                expected_archive = next(iter(members.values()))[1]["archive_sha256"]
                require(sha((repo / archive_path).read_bytes()) == expected_archive, "Dependency archive")
                seen = set()
                with tarfile.open(repo / archive_path, "r|gz") as source:
                    for member in source:
                        if member.name in members:
                            name, pin = members[member.name]
                            raw = source.extractfile(member).read()
                            require(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], name)
                            seen.add(member.name)
                require(seen == set(members), "Source reconstruction inventory")
        if args.roundtrip:
            context = json.loads(z.read(prefix + "fleet-context.json"))
            fleet = repo / context["current_Result_path"]
            require(sha(fleet.read_bytes()) == context["current_Result_sha256"], "Roundtrip fleet dependency")
            with tempfile.TemporaryDirectory(prefix="v637-saved-audit-") as tmp:
                root = Path(tmp).resolve()
                for name in pins:
                    path = (root / name).resolve()
                    require(path.is_relative_to(root), "Safe roundtrip path")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(z.read(name))
                script = root / "build/performance/route-family-audit-v637/analyse_saved.py"
                destination = root / "replayed.json"
                completed = subprocess.run([sys.executable, "-B", str(script), "--fleet", str(fleet),
                    "--output", str(destination)], capture_output=True, text=True, timeout=60)
                require(completed.returncode == 0, completed.stderr)
                require(json.loads(destination.read_bytes()) == audit, "Exact saved scalar/NPZ replay")
    print(json.dumps({"passed": True, "archive_files": len(pins), "prepared_files": 221,
        "dependency_bytes_checked": args.dependencies, "exact_saved_arithmetic_roundtrip": args.roundtrip,
        "GPU_search_refinement_propagation_or_checker_calls": 0}))


if __name__ == "__main__":
    main()
