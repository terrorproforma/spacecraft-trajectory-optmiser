"""Read saved L1 build evidence only. Does not import a solver or launch CUDA.

This verifies identities, CPU-emitted maps and supplied point bits. The source
review findings recorded below are a human code review, not a formal proof.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import tarfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def records(path):
    return [tuple((line.split(" ", 1)[0], json.loads(line.split(" ", 1)[1])))
            for line in path.read_text().splitlines()]


def bits(values):
    return b"".join(struct.pack("!d", float(x)) for x in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build = args.build
    manifest_bytes = (build / "manifest.json").read_bytes()
    assert digest(manifest_bytes) == args.manifest_sha256
    manifest = json.loads(manifest_bytes)
    assert manifest["complete"]
    source = {}
    with tarfile.open(build / "source.tar.gz") as archive:
        for name in manifest["owned_paths"]:
            data = archive.extractfile(name).read()
            assert digest(data) == manifest["source_sha256"][name], name
            source[name] = data.decode()
    mathematical = json.loads(Path(__file__).with_name("findings.json").read_bytes())
    inputs = Path("build/performance/known-point-replay-v606/inputs")
    evidence = {}
    for case in ("conditioning", "difficult"):
        expected = mathematical["captures"][case]
        count = expected["original_dimensions"]["p"]
        expected_map = [[p["t"], p["v"], p["positive_row"] + count,
                         p["negative_row"] + count, p["lambda_value"]]
                        for p in expected["detection_map"]]
        parsed = dict(records(build / f"validate-{case}-l1-seeded.log"))
        meta = parsed["PERSISTENT_REPLAY_META"]
        point = parsed["PERSISTENT_REPLAY_INITIAL_POINT"]
        assert meta["source_commit"] == manifest["frozen_commit"]
        assert meta["source_sha256"] == manifest["compiled_snapshot_source_sha256"]
        assert meta["input_sha256"] == expected["snapshot_sha256"]
        assert meta["l1_map"] == expected_map
        assert meta["l1_active_variables"] == expected["logically_reduced_dimensions"]["n"]
        assert meta["l1_active_rows"] == (expected["logically_reduced_dimensions"]["p"]
                                          + expected["logically_reduced_dimensions"]["m"])
        assert meta["audit_tolerance"] == 1e-9 and meta["cone_tolerance"] == 1e-8
        assert meta["halpern_mode"] == "off" and not meta["shifted"]
        assert meta["common_kkt_initial_check"] and meta["common_kkt_recovery_disabled"]
        encoded_bytes = (inputs / f"{case}-initial.txt").read_bytes()
        assert digest(encoded_bytes) == point["point_sha256"] == expected["point_sha256"]
        encoded = {}
        for line in encoded_bytes.decode().splitlines()[3:]:
            label, length, *values = line.split()
            assert int(length) == len(values)
            encoded[label] = list(map(float, values))
        assert all(bits(point[label]) == bits(encoded[label]) for label in ("x", "y", "z", "s"))
        assert bits(point["x_solver"]) == bits(encoded["x"])
        assert point["supplied_qualified"] and point["mapped_reference_qualified"]
        for mode in ("default", "l1-cold", "l1-seeded"):
            records(build / f"validate-{case}-{mode}.log")
        evidence[case] = {"snapshot_sha256": expected["snapshot_sha256"],
                          "point_sha256": expected["point_sha256"],
                          "map_count": len(expected_map), "all_map_entries_match": True,
                          "all_original_point_bits_preserved_before_cuda": True}
    resource_text = (build / "resource-usage.log").read_text()
    resources = {}
    for match in re.finditer(r"Function ([^\n]+):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", resource_text):
        for marker in ("cooperative_solve_kernelILb0", "cooperative_solve_kernelILb1",
                       "12solve_kernelILb0", "12solve_kernelILb1", "cooperative_l1_kernel",
                       "cooperative_l1_initialise_kernel", "cooperative_halpern_kernel"):
            if marker in match[1]:
                resources[marker] = dict(zip(("registers", "stack", "shared", "local"), map(int, match.groups()[1:])))
    for marker, registers in (("cooperative_solve_kernelILb0", 80), ("cooperative_solve_kernelILb1", 96),
                              ("12solve_kernelILb0", 148), ("12solve_kernelILb1", 204)):
        assert resources[marker]["registers"] == registers
    kernel = source["cpp/cuda/src/persistent_l1.cuh"]
    initializer = kernel.split("__global__ void cooperative_l1_initialise_kernel", 1)[1].split("__device__", 1)[0]
    assert "problem->q[" not in initializer and "const double q_norm = 0.0" in initializer
    result = {"scope": "CPU saved-evidence verification and independent source review; no GPU calls",
              "manifest_sha256": args.manifest_sha256, "source_commit": manifest["frozen_commit"],
              "source_archive_sha256": digest((build / "source.tar.gz").read_bytes()),
              "core_sha256": manifest["library_sha256"],
              "replay_sha256": manifest["persistent_snapshot_replay_sha256"],
              "test_sha256": manifest["persistent_l1_test_sha256"],
              "script_sha256": digest(Path(__file__).read_bytes()),
              "owned_source_hashes_verified": {name: manifest["source_sha256"][name] for name in source},
              "captures": evidence, "compiled_resources": resources,
              "source_review": {
                  "blocking_findings_remaining": [],
                  "fixed_before_gpu": "Deleted exactly-zero Q traversal: atomic writes at inactive t could race ordinary dummy stores even though the values were zero.",
                  "mapping": "Disjoint native row/variable descriptor bounds on host; exact Q=0, isolated positive-cost t and +/-v-t zero-upper rows proved from original device CSC/CSR.",
                  "working_state": "Separate masked A/smooth c view shares iterate buffers; inactive t and pair rows excluded from Ruiz, norms, power products and updates. Original coefficients/common view retained.",
                  "prox_and_recovery": "Dual-first retained Moreau step, diagonal-metric soft(v-d*g,d*lambda), then t=abs(v). Nonzero v selects strict endpoint; exactly zero uses clipped negative retained gradient.",
                  "accuracy": "Initial original point audited before completion. Every acceptance uses original objective, grouping, cones, gap and block complementarity with unchanged thresholds.",
                  "lifecycle": "Disable resets original history and forces original scaling; enable forces reduced refresh; checkpoint/numeric/common-policy changes rejected while enabled. Cancellation takes precedence.",
                  "occupancy": "Setter uses min(original common capacity, L1 solve occupancy, L1 initializer occupancy); subsequent grid changes recheck L1 capacity.",
                  "limits": "20-step norm estimate remains heuristic. Full storage retained. Build/analytic evidence cannot establish cold convergence, equal-accuracy speedup, mission score or SOTA."}}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output.read_bytes()),
                      "owned_sources": len(source), "maps_verified": sum(v["map_count"] for v in evidence.values())}))


if __name__ == "__main__":
    main()
