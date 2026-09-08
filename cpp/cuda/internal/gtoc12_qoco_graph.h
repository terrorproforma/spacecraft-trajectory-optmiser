#pragma once
#include "spacepdhcg/cuda/gtoc12_qoco_c_api.h"
#include "native_qoco_gpu.h"
#include <cuda_runtime.h>

int spacepdhcg_gtoc12_qoco_set_conditioning_retry(spacepdhcg_gtoc12_qoco*,const int*);

// Internal graph lease over assembly, conversion, IPM and original-coordinate
// qualification. Prime the ordinary cold device path first. Graph resources
// and borrowed inputs must outlive every launch. Destroy all emitted graphs
// and executables before end_graph; launches use the lease stream/thread/device.
// Per-attempt device reports carry qualification/iterations/objectives; their
// legacy host phase timings are unset. The owner measures complete graph time.
int spacepdhcg_gtoc12_qoco_begin_graph(spacepdhcg_gtoc12_qoco*,cudaStream_t,
    const QocoGraphProgress**);
int spacepdhcg_gtoc12_qoco_emit_graph(spacepdhcg_gtoc12_qoco*,cudaGraph_t,
    const cudaGraphNode_t*,size_t,cudaStream_t,const double*,const double*,
    const spacepdhcg_gtoc12_conic_parameters*,const int*,
    spacepdhcg_gtoc12_qoco_consumer,void*,cudaGraphNode_t*);
int spacepdhcg_gtoc12_qoco_end_graph(spacepdhcg_gtoc12_qoco*,cudaStream_t,
    spacepdhcg_gtoc12_qoco_report* totals=nullptr);
