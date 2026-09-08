"""Derive benchmark results from saved measurements only; no project/native imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

KIT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    source = KIT / "output/benchmark-report.json"
    report, launch, plan = read(source), read(KIT / "output/launch-report.json"), read(KIT / "plan.json")
    assert report["complete"] and report["passed"] and not report["failures"]
    assert launch["passed"] and launch["exit_code"] == 0 and not launch["failures"]
    assert report["GPU_evaluate_calls"] == plan["gpu_evaluate_calls"] == 60
    assert report["CPU_candidate_evaluations"] == report["GPU_candidate_evaluations"] == 14200
    assert len(report["groups"]) == len(plan["groups"]) == 12
    rows = []
    total = {"cpu_calls": 0, "gpu_calls": 0, "cpu_candidates": 0, "gpu_candidates": 0,
             "cpu_method_seconds": 0.0, "gpu_method_seconds": 0.0,
             "post_method_check_capture_seconds": 0.0, "construction_seconds": 0.0,
             "backend_setup_seconds": 0.0, "backend_close_seconds": 0.0,
             "accepted_evaluations_per_backend": 0, "rejected_evaluations_per_backend": 0}
    for group_number, (saved, prescribed) in enumerate(zip(report["groups"], plan["groups"], strict=True)):
        assert saved["group"] == prescribed
        measurements = saved["measurements"]
        assert len(measurements) == 10
        n = prescribed["size"]
        cpu = [x for x in measurements if x["backend"] == "cpu"]
        gpu = [x for x in measurements if x["backend"] == "gpu"]
        assert len(cpu) == len(gpu) == 5
        warm_cpu = [x for x in cpu if not x["first_call"]]
        warm_gpu = [x for x in gpu if not x["first_call"]]
        first_cpu = [x for x in cpu if x["first_call"]]
        first_gpu = [x for x in gpu if x["first_call"]]
        assert (len(warm_cpu), len(warm_gpu), len(first_cpu), len(first_gpu)) == (4, 4, 1, 1)
        assert [x["backend"] for x in measurements[2:]] == plan["warm_order"]
        for x in measurements:
            assert x["plan_validation"] == {"passed": True, "accepted": prescribed["expected_accepted"], "rejected": n-prescribed["expected_accepted"]}
            assert x["method_seconds"] > 0.0
            assert x["proxy_evaluations_per_second"] == n / x["method_seconds"]
            assert len(x["raw_outcome_sha256"]) == 64
            if not x["first_call"]:
                assert x["raw_outcome_identical_to_first"]
            total[x["backend"] + "_calls"] += 1
            total[x["backend"] + "_candidates"] += n
            total[x["backend"] + "_method_seconds"] += x["method_seconds"]
            total["post_method_check_capture_seconds"] += x["all_post_method_check_capture_seconds"]
        for x in gpu:
            assert x["telemetry_delta"]["completion_batches"] == 1
            assert x["telemetry_delta"]["completion_candidates"] == n
            parity_path = KIT / "output" / prescribed["id"] / f"{x['trial']:02}-gpu-parity.json"
            parity = read(parity_path)
            assert parity["passed"] and not parity["failures"]
        assert saved["telemetry"]["completion_batches"] == 5
        assert saved["telemetry"]["completion_candidates"] == 5*n
        assert saved["telemetry"]["completed_branch_requests"] == 0
        for key in ("construction_seconds", "backend_setup_seconds", "backend_close_seconds"):
            total[key] += saved[key]
        total["accepted_evaluations_per_backend"] += 5*prescribed["expected_accepted"]
        total["rejected_evaluations_per_backend"] += 5*(n-prescribed["expected_accepted"])
        cpu_seconds = statistics.median(x["method_seconds"] for x in warm_cpu)
        gpu_seconds = statistics.median(x["method_seconds"] for x in warm_gpu)
        ratio = cpu_seconds/gpu_seconds
        assert cpu_seconds == saved["cpu_warm"]["median_seconds"]
        assert gpu_seconds == saved["gpu_warm"]["median_seconds"]
        assert ratio == saved["warm_host_inclusive_speed_ratio_cpu_over_gpu"]

        def phase(name):
            return statistics.median(x["telemetry_delta"][name] for x in warm_gpu)

        def share(name):
            return statistics.median(x["telemetry_delta"][name]/x["method_seconds"] for x in warm_gpu)

        rows.append({"id": prescribed["id"], "model": prescribed["model"], "size": n,
                     "size_scope": prescribed["role"], "unique_controls_in_group": prescribed["unique_control_count"],
                     "accepted_per_call": prescribed["expected_accepted"], "rejected_per_call": n-prescribed["expected_accepted"],
                     "samples_per_backend": {"first": 1, "warm": 4},
                     "warm_cpu_seconds": cpu_seconds, "warm_gpu_seconds": gpu_seconds,
                     "warm_cpu_over_gpu_speed_ratio": ratio,
                     "warm_cpu_proxy_evaluations_per_second": n/cpu_seconds,
                     "warm_gpu_proxy_evaluations_per_second": n/gpu_seconds,
                     "warm_gpu_pack_seconds_median": phase("completion_pack_seconds"),
                     "warm_gpu_native_call_seconds_median": phase("completion_native_call_seconds"),
                     "warm_gpu_kernel_seconds_median": phase("completion_kernel_seconds"),
                     "warm_gpu_matching_sample_pack_share_median": share("completion_pack_seconds"),
                     "warm_gpu_matching_sample_native_call_share_median": share("completion_native_call_seconds"),
                     "warm_gpu_matching_sample_readout_share_median": statistics.median(
                         (x["telemetry_delta"]["completion_total_seconds"]-x["telemetry_delta"]["completion_pack_seconds"]-x["telemetry_delta"]["completion_native_call_seconds"])/x["method_seconds"] for x in warm_gpu),
                     "first_cpu_seconds": first_cpu[0]["method_seconds"],
                     "first_gpu_seconds": first_gpu[0]["method_seconds"],
                     "first_gpu_telemetry": first_gpu[0]["telemetry_delta"],
                     "backend_setup_seconds": saved["backend_setup_seconds"],
                     "backend_close_seconds": saved["backend_close_seconds"],
                     "first_process_GPU_completion": group_number == 0,
                     "input_construction_seconds": saved["construction_seconds"],
                     "workspace_capacities": saved["workspace_capacities"],
                     "warm_cpu_samples_seconds": [x["method_seconds"] for x in warm_cpu],
                     "warm_gpu_samples_seconds": [x["method_seconds"] for x in warm_gpu]})
    assert total["cpu_calls"] == total["gpu_calls"] == 60
    assert total["cpu_candidates"] == total["gpu_candidates"] == 14200
    files = {p.relative_to(KIT).as_posix(): sha(p) for p in sorted((KIT / "output").rglob("*")) if p.is_file()}
    raw_index = {"files": files, "file_count": len(files), "bytes": sum((KIT/name).stat().st_size for name in files),
                 "tree_sha256": hashlib.sha256("".join(name+":"+digest+"\n" for name,digest in files.items()).encode()).hexdigest()}
    write(KIT / "raw-index.json", raw_index)
    summary = {"complete": True, "saved_output_derivation_only": True,
               "benchmark_report_sha256": sha(source), "launch_report_sha256": sha(KIT / "output/launch-report.json"),
               "ready_sha256": sha(KIT / "ready.json"), "raw_index_sha256": sha(KIT / "raw-index.json"),
               "plan_sha256": sha(KIT / "plan.json"), "unique_historical_controls": 20,
               "total_worker_seconds": report["total_worker_seconds"], "totals": total, "groups": rows,
               "packing_share_definition": "Median of pack_seconds/method_seconds for each matching warm GPU sample; not a ratio of unmatched medians",
               "kernel_time_definition": "Median of the four recorded warm CUDA-event kernel timings, seconds",
               "scope": "Repeated proxy completion evaluations with cached historical geometry; not new solutions, physics certifications or whole-mission speedup",
               "default_changed": False, "new_GPU_calls_in_derivation": 0, "new_CPU_completion_calls_in_derivation": 0}
    write(KIT / "summary.json", summary)
    lines = ["# Completion benchmark results (v621)", "",
             "The production GPU completion adapter helps the fitted cost model at practical shortlist sizes. The small flat-model batch is slower on the GPU. The production default remains unchanged.", "",
             "This was one local RTX 5090 run of the frozen final-b Python adapter and normal full CUDA core. It evaluated **20 unique historical controls**, repeated in fixed sequences: **60 GPU calls and 14,200 proxy candidate evaluations per backend**. All prescribed decisions, cargo, metadata and numerical comparisons passed; retained workspace buffers were reused. There were no new trajectories, physics certificates or score changes.", "",
             "Each row has one first call and four warm calls per backend. Warm timings are medians; the speed ratio is CPU median divided by GPU median. Packing share is the median share from matching individual GPU samples. Method timings include normal host packing and plan readout.", "",
             "| Model | Batch | Warm CPU ms | Warm GPU ms | CPU/GPU ratio | GPU proxy eval/s | Host packing share | Native kernel µs |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        label = "Flat" if r["model"] == "v616_no_fit" else "Existing fit"
        lines.append(f"| {label} | {r['size']} | {1000*r['warm_cpu_seconds']:.3f} | {1000*r['warm_gpu_seconds']:.3f} | {r['warm_cpu_over_gpu_speed_ratio']:.3f}× | {r['warm_gpu_proxy_evaluations_per_second']:,.0f} | {100*r['warm_gpu_matching_sample_pack_share_median']:.1f}% | {1e6*r['warm_gpu_kernel_seconds_median']:.2f} |")
    lines += ["", "The observed v616 shortlist setting was 24; the source default is 48. The four-request case uses historical DP metadata at a schedule-sized batch, not a measured heuristic scheduler invocation. Sizes 64/256/1024 are prospective scaling probes. Repetition provides a controlled component comparison; it does not show that a live search can assemble those batches at no cost.", "",
              "The fitted model gains about 3.000× at 24 and 3.394× at 48. The flat model is approximately tied at 24 (0.963×), and its four-request batch loses substantially (0.443×). At 1024 fitted requests the gain is 4.856×, with host metadata packing dominating the remaining GPU method time. This points toward retained metadata/device inputs as a next performance hypothesis; it does not justify making every small completion batch use the GPU.", "",
              "## First call and setup", "",
              "A new completion workspace was created for each group. Only the first group is the first GPU completion in this process; these are not repeated cold-device measurements. Backend setup is separate from the first method call. First-call latency must not be averaged into warm throughput.", "",
              "| Model | Batch | First CPU ms | First GPU ms | Backend setup ms | Backend close ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        label = "Flat" if r["model"] == "v616_no_fit" else "Existing fit"
        lines.append(f"| {label} | {r['size']} | {1000*r['first_cpu_seconds']:.3f} | {1000*r['first_gpu_seconds']:.3f} | {1000*r['backend_setup_seconds']:.3f} | {1000*r['backend_close_seconds']:.3f} |")
    lines += ["", f"The measured worker took {report['total_worker_seconds']:.6f} s. Timed CPU method calls summed to {total['cpu_method_seconds']:.6f} s and GPU method calls to {total['gpu_method_seconds']:.6f} s. Input construction summed to {total['construction_seconds']:.6f} s; post-method checking/capture summed to {total['post_method_check_capture_seconds']:.6f} s. Setup, close, signatures, journal writes and other runner work account for further time. These diagnostic checks are not full trajectory/fleet verification and are excluded from the method throughput numbers.", "",
              "Raw vectors, parity results, compact first outcomes, every timing sample and outcome hash are retained under `output/`; `raw-index.json` freezes their hashes. `summary.json` contains exact values and sample definitions. Four warm samples per row support this bounded comparison, not a broad statistical performance claim.", "",
              "No production source, defaults or objective changed as a result of this summary. GPU-native search work remains incomplete outside the tested completion component; the retained mission score is unchanged."]
    with (KIT / "RESULTS.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"summary_sha256": sha(KIT / "summary.json"), "results_sha256": sha(KIT / "RESULTS.md"),
                      "raw_index_sha256": sha(KIT / "raw-index.json"), "raw_files": len(files),
                      "raw_bytes": raw_index["bytes"], "groups": len(rows)}))


if __name__ == "__main__":
    main()
