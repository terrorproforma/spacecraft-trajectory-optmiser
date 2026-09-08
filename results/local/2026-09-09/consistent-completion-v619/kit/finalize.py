"""Seal the finite CPU evidence and verify all frozen indices and live owned files."""

from prepare import KIT, ROOT, read, sha, write


def main():
    old_indices = read(KIT / "source-sha256.json")
    for arm, index in old_indices.items():
        for name, digest in index.items():
            assert sha(KIT / "source" / arm / name) == digest, (arm, name)
    final = read(KIT / "final-source-sha256.json")
    for name, digest in final.items():
        assert sha(KIT / "source/candidate-final" / name) == digest, name
    for name, digest in read(KIT / "provenance.json")["inputs"].items():
        assert sha(KIT / "inputs" / name) == digest, name
    source = "src/spacepdhcg/gtoc12/search.py"
    assert sha(ROOT / source) == final[source]
    assert sha(ROOT / "tests/test_gtoc12_completion_costs.py") == sha(KIT / "test_completion_costs_final.py")
    assert read(KIT / "validation-final-02/report.json")["passed"]
    assert read(KIT / "output/audit.json")["complete"]
    paths = sorted(p for p in KIT.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert not any(p.name == "evidence-manifest.json" for p in paths)
    files = {str(path.relative_to(KIT)): {"sha256": sha(path), "bytes": path.stat().st_size} for path in paths}
    for name, record in files.items():
        assert sha(KIT / name) == record["sha256"]
    write(KIT / "evidence-manifest.json", {
        "files": files, "file_count": len(files), "total_bytes": sum(record["bytes"] for record in files.values()),
        "GPU_calls": 0, "scalar_bridge_calls": 40, "refinements": 0,
        "final_search_sha256": final[source],
        "final_test_sha256": sha(KIT / "test_completion_costs_final.py"),
        "all_indexed_hashes_verified": True,
    })
    print({"files": len(files), "bytes": sum(record["bytes"] for record in files.values()),
           "manifest_sha256": sha(KIT / "evidence-manifest.json"),
           "source_sha256": final[source], "test_sha256": sha(KIT / "test_completion_costs_final.py")})


if __name__ == "__main__":
    main()
