"""Independent stdlib-only read-only validation of the frozen finite benchmark kit.

Does not import project code, load native code, or execute the benchmark.
"""
from __future__ import annotations

import ast
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
KIT = ROOT / "build/performance/completion-pack-benchmark-v622"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    ready = read(KIT / "ready.json")
    files = ready["files"]
    for name, digest in files.items():
        target = KIT / name
        assert target.is_file() and sha(target) == digest, name
    source = read(KIT / "source-sha256.json")
    inputs = read(KIT / "input-sha256.json")
    for folder, index in (("source", source), ("inputs", inputs)):
        for name, digest in index.items():
            assert sha(KIT / folder / name) == digest, (folder, name)
    with tarfile.open(KIT / "source-full.tar.gz") as archive:
        members = [m for m in archive if m.isfile()]
        assert len(members) == len(source) == 341
        for member in members:
            path = member.name.removeprefix("./").removeprefix("source/")
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == source[path]
    original = read(KIT / "inputs/compact-g-report.json")
    assert len(original["owned_sources"]) == 12
    with tarfile.open(KIT / "inputs/compact-g-source.tar.gz") as archive:
        members = [m for m in archive if m.isfile()]
        assert len(members) == 12
        for member in members:
            path = member.name.removeprefix("./").removeprefix("source/")
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert digest == original["owned_sources"][path] == source[path]
    profile, plan = read(KIT / "profile.json"), read(KIT / "plan.json")
    prerequisite = ROOT / "build/performance/completion-model-gpu-v622d/report.json"
    assert sha(prerequisite) == "643caf4e5550373ac3b06761197ae4ee1f7801569e614498786ba089b34a54ff"
    previous = read(prerequisite)
    assert previous["complete"] and previous["exit_code"] == 0
    assert previous["junit"] == {"tests": 7, "failures": 0, "errors": 0, "skipped": 0}
    assert previous["expected_readbacks_present"] is True
    assert sha(KIT / "inputs/compact-g-report.json") == profile["source_report_sha256"]
    assert previous["library_sha256"] == original["library"]["sha256"] == profile["library"]["sha256"]
    assert profile["library"]["sha256"] == "9c3e6a250002893363d2f4dcf68babe3b7158508b5b97149f83c8261437f1fa7"
    fixtures = read(KIT / "inputs/fixtures.json")["historical"]
    totals = dict(ordinary=0, compact=0)
    unique, groups, calls = set(), [], 0
    assert len(plan["groups"]) == 8
    for number, group in enumerate(plan["groups"]):
        indices = group["fixture_indices"]
        assert len(indices) == group["size"]
        assert group["size"] in (24, 48, 256, 1024)
        assert len(set(indices)) == 10
        unique.update(indices)
        accepted = sum(fixtures[i]["expected"]["failure_name"] == "ok" for i in indices)
        assert accepted == group["expected_accepted"]
        sequence = ["ordinary", "compact"] if number % 2 == 0 else ["compact", "ordinary"]
        sequence += plan["warm_order"]
        assert sequence[2:] == ["ordinary", "compact", "compact", "ordinary", "compact", "ordinary", "ordinary", "compact"]
        assert sequence.count("ordinary") == sequence.count("compact") == 5
        for backend in sequence:
            calls += 1
            totals[backend] += len(indices)
        groups.append({"id": group["id"], "size": group["size"], "accepted": accepted, "order": sequence})
    assert len(unique) == 20 and calls == 80
    assert totals == {"ordinary": 13520, "compact": 13520}
    launch = (KIT / "launch.py").read_text()
    run = (KIT / "run.py").read_text()
    for name in ("launch.py", "run.py", "common.py", "test_preparation.py"):
        ast.parse((KIT / name).read_text(), filename=name)
    assert "pass_fds=(lock.fileno(),)" in launch
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in launch
    assert "child.wait(timeout=180)" in launch
    assert "start_new_session=True" in launch
    assert 'assert not processes or processes == "No running processes found"' in launch
    assert "os.killpg(child.pid, signal.SIGTERM)" in launch
    assert "os.killpg(child.pid, signal.SIGKILL)" in launch
    assert 'rows = owner.run(search, inputs)' in run
    assert 'assert not search.collect_table.return_sweeps' in run
    assert 'assert search.collect_table.lambert_evaluations == 0' in run
    assert 'rows, elapsed, post = measured_call(owner, search, inputs, folder, label)' in run
    measure = next(n for n in ast.parse(run).body if isinstance(n, ast.FunctionDef) and n.name == "measured_call")
    timed = next(n for n in measure.body if isinstance(n, ast.Try))
    assert timed.finalbody and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "savez_compressed" for block in timed.finalbody for n in ast.walk(block))
    assert '"unknown_until_return"' in run and 'not yet validated' in run
    assert 'may be stale if run did not return' in run
    assert run.index('write(folder / f"{label}-comparison.json", comparison)') < run.index('assert comparison["passed"]')
    tests = ET.parse(KIT / "validation/attempt-04/pytest.xml").getroot()
    suites = list(tests.iter("testsuite"))
    assert sum(int(s.attrib["tests"]) for s in suites) == 18
    assert all(int(s.attrib.get(k, 0)) == 0 for s in suites for k in ("errors", "failures", "skipped"))
    assert all(s["exit_code"] == 0 for s in read(KIT / "validation/attempt-04/report.json"))
    findings = {
        "scope": "CPU-only frozen source/manifest/finite-budget review; no benchmark or native execution",
        "ready_sha256": sha(KIT / "ready.json"),
        "indexed_files": len(files),
        "indexed_bytes": sum((KIT / name).stat().st_size for name in files),
        "source_files": len(source),
        "owned_overlay_files": 12,
        "source_and_input_indices_verified": True,
        "archives_roundtrip_verified": True,
        "source_report_sha256": profile["source_report_sha256"],
        "core_sha256": profile["library"]["sha256"],
        "prerequisite_sha256": sha(prerequisite),
        "source_files_sha256": {name: sha(KIT / name) for name in ("launch.py", "run.py", "common.py", "test_preparation.py", "profile.json", "plan.json")},
        "expected_counts": {"evaluate_calls": calls, "kernels_from_source": 160, "candidates_per_arm": totals, "unique_historical_controls": 20},
        "groups": groups,
        "preparation_tests": 18,
        "failed_readback_retention_fixed": True,
        "launch_review": "GO for the single finite supervised run specified by this exact ready hash; reviewer has not executed it",
        "timing_review": "Both arms time production GpuCompletion.run including packing, metadata/hash validation, synchronous native work/transfers and plan reconstruction. Catalogue/request construction and ordinary workspace create/close are separately recorded. Compact model configuration is timed. First calls share one process; four warm samples/arm use ABBA/BAAB.",
        "limitations": ["Repeated proxy completions, not fresh routes or physics-qualified solutions", "Historical 20-control corpus; not latest fleet", "No default promotion or full-mission/SOTA claim", "Kernel count is source-derived, not a profiler measurement"],
    }
    (OUT / "preparation-findings.json").write_text(json.dumps(findings, indent=2) + "\n")
    print(json.dumps({"passed": True, "ready_sha256": findings["ready_sha256"], "files": len(files), "calls": calls, "candidate_evaluations": sum(totals.values())}))


if __name__ == "__main__":
    main()
