#pragma once
#include <cuda_runtime.h>
#include <vector>

// Append a stream-ordered segment to an existing graph/conditional body. The
// caller owns that graph, including any partial nodes after a failed capture.
template<class Emit>
cudaError_t spacepdhcg_graph_append(cudaStream_t stream,cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,size_t count,Emit emit,cudaGraphNode_t* last) {
    if (!last || !graph || (count && !dependencies)) return cudaErrorInvalidValue;
    *last=nullptr;
    auto error=cudaStreamBeginCaptureToGraph(stream,graph,dependencies,nullptr,count,
        cudaStreamCaptureModeThreadLocal);
    if (error!=cudaSuccess) return error;
    struct Capture {
        cudaStream_t stream;
        bool active=true;
        ~Capture() { if(active) { cudaGraph_t unused{};cudaStreamEndCapture(stream,&unused); } }
    } capture{stream};
    error=emit();
    if (error!=cudaSuccess) return error;
    cudaStreamCaptureStatus status{};cudaGraph_t found{};
    const cudaGraphNode_t* nodes{};size_t size{};
    error=cudaStreamGetCaptureInfo(stream,&status,nullptr,&found,&nodes,&size);
    if (error!=cudaSuccess) return error;
    if (status!=cudaStreamCaptureStatusActive || found!=graph || !size) return cudaErrorInvalidValue;
    std::vector<cudaGraphNode_t> tail(nodes,nodes+size);
    error=cudaStreamEndCapture(stream,&found);capture.active=false;
    if (error!=cudaSuccess) return error;
    if (found!=graph) return cudaErrorInvalidValue;
    if (tail.size()==1) { *last=tail[0];return cudaSuccess; }
    return cudaGraphAddEmptyNode(last,graph,tail.data(),tail.size());
}
