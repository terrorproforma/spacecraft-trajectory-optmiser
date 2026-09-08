#pragma once
#include "spacepdhcg/cuda/gtoc12_joint_c_api.h"
#include <cuda_runtime_api.h>

// Device buffers belong to the joint workspace and remain on its stream.
cudaError_t spacepdhcg_joint_geometry_launch(
    int count, int n, const spacepdhcg_orbitweaver_hop_elements* elements,
    const double* arrivals, const double* departures,
    const spacepdhcg_gtoc12_joint_result* preflight,
    const spacepdhcg_gtoc12_joint_cached_cost* records, int record_count,
    spacepdhcg_gtoc12_joint_cost* costs,
    spacepdhcg_orbitweaver_hop_request* requests,
    spacepdhcg_orbitweaver_hop_result* results,
    spacepdhcg_gtoc12_joint_geometry_stats* stats, cudaStream_t stream);
