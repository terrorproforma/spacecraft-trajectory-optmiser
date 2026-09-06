// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qoco.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct QocoGpuCompletion {
    int abi_version, status, iterations, ir_iterations, step_ir_iterations, restored;
    double primal_residual, dual_residual, gap, objective, dynamic_reg;
} QocoGpuCompletion;
typedef struct QocoGpuOutput {
    const QocoGpuCompletion* completion;
    const double *x, *y, *s, *z;
    int n, p, m;
} QocoGpuOutput;
// Prime with a synchronous graph-enabled qoco_solve first. Replay requires
// captured initialisation AND terminal handling with unchanged static settings.
// Numeric values/scalings may change in their existing buffers before replay.
// Queues on the supplied CUDA stream without downloading or waiting for results.
// GPU consumers must use that same stream. Before changing streams, updating,
// or using any other solver API, finish_device must complete all queued work.
// Same-stream replays can be chained, with consumers between them; outputs are
// overwritten by the next replay. Host sol metadata is invalidated, not updated.
// Current host settings supply each replay's tolerances/budgets/dynamic reg/warm
// flag. No implicit host update from the preceding device solve is performed.
// Returns 0 queued, 1 invalid arguments/settings, 2 unprepared/stale/incompatible,
// 3 pending on another stream, 4 CUDA error. Failure clears the output descriptor.
int qoco_gpu_ipm_replay_device(QOCOSolver*, void* stream, QocoGpuOutput*);
// Waits for the replay stream INCLUDING subsequently queued GPU consumers.
// Does not download or materialise legacy host solution metadata.
int qoco_gpu_ipm_finish_device(QOCOSolver*);
#ifdef __cplusplus
}
#endif
