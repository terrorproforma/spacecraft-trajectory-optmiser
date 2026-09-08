"""Retain the initial overlay; freeze the reviewed once-per-flight final version."""

from prepare import KIT, ROOT, read, sha, write
import shutil


def main():
    baseline = read(KIT / "source-sha256.json")["baseline"]
    dest_root = KIT / "source/candidate-final"
    assert not dest_root.exists()
    for name in baseline:
        source = KIT / "source/baseline" / name
        dest = dest_root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    relative = "src/spacepdhcg/gtoc12/search.py"
    shutil.copyfile(ROOT / relative, dest_root / relative)
    final = {name: sha(dest_root / name) for name in baseline}
    assert [name for name in baseline if baseline[name] != final[name]] == [relative]
    write(KIT / "final-source-sha256.json", final)
    shutil.copyfile(
        ROOT / "tests/test_gtoc12_completion_costs.py", KIT / "test_completion_costs_final.py"
    )
    write(
        KIT / "final-source-provenance.json",
        {
            "reason": "Root review requested removing redundant temporary guessed-mass model evaluation before the actual forward pricing pass.",
            "initial_candidate_preserved": "source/candidate",
            "initial_candidate_executed": False,
            "final_source_sha256": sha(KIT / "final-source-sha256.json"),
            "final_search_sha256": final[relative],
            "test_sha256": sha(KIT / "test_completion_costs_final.py"),
            "baseline_executed_replay_script_sha256": sha(KIT / "replay-initial.py"),
            "candidate_replay_script_sha256": sha(KIT / "replay.py"),
            "only_model_call_redundancy_removed_after_initial_freeze": True,
        },
    )
    print(final[relative])


if __name__ == "__main__":
    main()
