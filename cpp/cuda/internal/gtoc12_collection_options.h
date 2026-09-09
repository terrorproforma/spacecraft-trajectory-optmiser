#pragma once
#include "spacepdhcg/cuda/gtoc12_collection_c_api.h"
#include <cuda_runtime_api.h>
// Copies immutable producer rows on its stream and drains that stream before
// returning. The table owns its allocation independently of producer scratch.
spacepdhcg_cuda_status gtoc12_collection_options_copy_device(
    const spacepdhcg_gtoc12_collection_option*, int, cudaStream_t,
    spacepdhcg_gtoc12_collection_options**);
// Borrow immutable rows on the owning thread/device. Retain the table until
// the consuming stream completes; no host row download.
spacepdhcg_cuda_status gtoc12_collection_options_view(
    spacepdhcg_gtoc12_collection_options*,
    const spacepdhcg_gtoc12_collection_option**, int*);
