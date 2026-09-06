// Test-only host launch observer. It does not synchronize or modify CUDA work.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

extern "C" cudaError_t cudaLaunchKernel(
    const void* function, dim3 grid, dim3 block, void** arguments,
    size_t shared_bytes, cudaStream_t stream
) {
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
    static const auto actual = reinterpret_cast<Launch>(
        dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    static const auto gather = dlsym(RTLD_DEFAULT,
        "_ZN10spacepdhcg4cuda6detail28gather_scvx_candidate_kernelEPKdPKiS5_PdS6_mm");
    if (!actual || !gather) std::abort();
    const auto status = actual(function, grid, block, arguments, shared_bytes, stream);
    if (function == gather)
        std::fprintf(stderr, "{\"case\":\"scvx_gather_launch\",\"grid_x\":%u,"
            "\"grid_y\":%u,\"grid_z\":%u,\"block_x\":%u,\"status\":%d}\n",
            grid.x, grid.y, grid.z, block.x, int(status));
    return status;
}
