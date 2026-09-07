#!/usr/bin/env python3
"""Expose prepared IPM node emission for a GPU-controlled outer graph."""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_qoco_device_control import once


def prepare(root: Path) -> dict[str, str]:
    path = root / "algebra/cuda"
    graph = (path / "qoco_ipm_graph.cuh").read_text()
    start = graph.index("    cudaGraphConditionalHandle valid_update;")
    end = graph.index("    qoco_ipm_current = nullptr;", start)
    end += len("    qoco_ipm_current = nullptr;")
    body = graph[start:end]
    # Default conditional values reset only at top-level graph launch. A nested
    # solve must restart its loop and counters on every enclosing iteration.
    marker = "    cudaGraphNodeParams params{};\n    params.type = cudaGraphNodeTypeConditional;"
    body = once(
        body,
        marker,
        "    qoco_ipm_resume();\n"
        "    qoco_ipm_reset_nested<<<1,1>>>(iterations,loop);\n"
        "    qoco_ipm_pause();\n" + marker,
    )
    helper = """
__global__ void qoco_ipm_reset_nested(int* iterations,cudaGraphConditionalHandle loop) {
    *iterations=0;
    cudaGraphSetConditional(loop,1);
}
static cudaGraphNode_t qoco_ipm_emit_body(QOCOSolver* solver,QocoIpmCapture& capture,
    bool initialization,bool terminal) {
    auto* s=solver->linsys_data;
    auto* work=solver->work;
    auto* data=work->data;
    auto& cache=s->ir->ipm;
    const auto root=capture.graph;
""" + body + "\n    return guard_node;\n}\n"
    graph = (graph[:start] + "    qoco_ipm_emit_body(solver,capture,initialization,terminal);"
             + graph[end:])
    graph = once(graph, 'extern "C" int qoco_gpu_ipm_loop(QOCOSolver* solver) {',
                 helper + '\nextern "C" int qoco_gpu_ipm_loop(QOCOSolver* solver) {')
    graph += '\n#include "qoco_ipm_outer_graph.cuh"\n'
    header = (path / "qoco_gpu_replay.h").read_text()
    declaration = """// Emit a prepared solve into an existing graph, including a conditional body.
// No graph is executed. All dependencies must belong to graph. Outputs and
// solver-owned buffers must remain alive until every executable using the nodes
// is destroyed. No other solver API may run during that ownership period;
// launches using this workspace must be serialized. Destroy external graphs
// and complete their streams before settings changes, replay or cleanup.
// Capture/execute on the solver's original host thread and CUDA device. Call
// after synchronous priming and finish_device, with no active stream capture.
// A borrowed numeric_update may supply changing scaling/invalid values on GPU.
// Null uses host settings captured at emission. Returned node gates consumers.
// Each enclosing loop iteration resets IPM control/counters on GPU. No report
// downloads or host callbacks are emitted. Host solution metadata is unchanged.
// Returns 0 emitted, 1 invalid, 2 unprepared/pending/stale, 4 CUDA error.
int qoco_gpu_ipm_emit_graph(QOCOSolver*,cudaGraph_t,const cudaGraphNode_t*,size_t,
    const double* numeric_update,QocoGpuOutput*,cudaGraphNode_t* completion_node);
// Capture the prepared numeric-update kernels on an actively capturing stream.
// Same exclusive external-graph ownership rules as emit_graph. Prime numeric
// scaling synchronously first. Output is the borrowed nine-double device packet.
int qoco_gpu_capture_numeric_update(void*,const double*,cudaStream_t,const double**);
"""
    header = once(header, "#ifdef __cplusplus\n}\n#endif",
                  declaration + "#ifdef __cplusplus\n}\n#endif")
    extension_path = (Path(__file__).resolve().parents[2]
                      / "cpp/cuda/patches/qoco_ipm_outer_graph.cuh")
    extension = extension_path.read_text()
    update = (path / "qoco_device_update.cuh").read_text()
    start = update.index('extern "C" int qoco_gpu_update_numeric_device(')
    end = update.index('extern "C" int qoco_gpu_finish_numeric_update(', start)
    captured = update[start:end]
    captured = once(captured, "qoco_gpu_update_numeric_device(", "qoco_gpu_capture_numeric_update(")
    captured = once(captured, "if (w->pending && producer != w->pending_stream) return 2;",
                    "if (w->pending) return 2;")
    captured = once(captured, "if (capture != cudaStreamCaptureStatusNone) return 2;",
                    "if (capture != cudaStreamCaptureStatusActive) return 2;")
    captured = once(
        captured, "    w->pending = true; w->host_stale = true; w->pending_stream = producer;",
        "    w->host_stale = true; w->pending_stream = producer;")
    update += "\n" + captured
    files = {"qoco_ipm_graph.cuh": graph, "qoco_gpu_replay.h": header,
             "qoco_device_update.cuh": update,
             "qoco_ipm_outer_graph.cuh": extension}
    for name, contents in files.items():
        (path / name).write_text(contents)
    report = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in files}
    (root / "spacepdhcg-outer-graph.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().destination), indent=2))
