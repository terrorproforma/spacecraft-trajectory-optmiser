// Experimental factorization graph with retained vendor allocations.
// Only same-stream free blocks are reusable; no new allocation during capture.
struct QocoFactorMemory {
    struct Block { void* pointer; size_t size; cudaStream_t stream; bool used; };
    std::vector<Block> blocks;
    cudaGraph_t factor_graph{};
    cudaGraphExec_t factor_exec{};
    bool capturing=false;
    size_t bytes=0, misses=0;
    static int allocate(void* context,void** output,size_t size,cudaStream_t stream) {
        if(!size){*output=nullptr;return 0;}
        auto* self=static_cast<QocoFactorMemory*>(context);
        for(auto& block:self->blocks) if(!block.used && block.size>=size && block.stream==stream) {
            block.used=true;*output=block.pointer;return 0;
        }
        if(self->capturing) {
            ++self->misses;
            fprintf(stderr,"FACTOR_GRAPH allocation miss bytes=%zu\n",size);
            return 1;
        }
        void* pointer=nullptr;
        const auto status=cudaMalloc(&pointer,std::max(size,size_t(1)));
        if(status!=cudaSuccess)return int(status);
        self->blocks.push_back({pointer,size,stream,true});self->bytes+=size;*output=pointer;return 0;
    }
    static int release(void* context,void* pointer,size_t,cudaStream_t stream) {
        if(!pointer)return 0;
        auto* self=static_cast<QocoFactorMemory*>(context);
        for(auto& block:self->blocks) if(block.pointer==pointer) {
            if(!block.used){fprintf(stderr,"FACTOR_GRAPH duplicate free\n");return 1;}
            block.used=false;block.stream=stream;return 0;
        }
        fprintf(stderr,"FACTOR_GRAPH unowned free\n");return 1;
    }
};
static QocoFactorMemory* qoco_factor_memory_create(cudssHandle_t handle) {
    auto* memory=new QocoFactorMemory;
    cudssDeviceMemHandler_t handler{};
    handler.ctx=memory;handler.device_alloc=QocoFactorMemory::allocate;handler.device_free=QocoFactorMemory::release;
    std::memcpy(handler.name,"SpacePDHCG retained graph probe",31);
    auto set=reinterpret_cast<decltype(&::cudssSetDeviceMemHandler)>(dlsym(g_cudss_handle,"cudssSetDeviceMemHandler"));
    if(!set)exit(111);
    CUDSS_CHECK(set(handle,&handler));
    return memory;
}
static void qoco_factor_memory_destroy(QocoFactorMemory* memory) {
    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr,"FACTOR_GRAPH allocator bytes=%zu blocks=%zu misses=%zu\n",memory->bytes,memory->blocks.size(),memory->misses);
    if(memory->factor_exec)CUDA_CHECK(cudaGraphExecDestroy(memory->factor_exec));
    if(memory->factor_graph)CUDA_CHECK(cudaGraphDestroy(memory->factor_graph));
    for(auto block:memory->blocks)CUDA_CHECK(cudaFree(block.pointer));
    delete memory;
}

static cudssStatus_t qoco_factor_execute(QocoFactorMemory* memory,cudaStream_t stream,
    cudssHandle_t handle,int phase,const cudssConfig_t config,cudssData_t data,
    const cudssMatrix_t matrix,cudssMatrix_t solution,const cudssMatrix_t rhs) {
    if(phase!=CUDSS_PHASE_FACTORIZATION || getenv("SPACEPDHCG_TEST_QOCO_FACTOR_GRAPH_DISABLE"))
        return g_cuda_funcs.cudssExecute(handle,phase,config,data,matrix,solution,rhs);
    if(memory->factor_exec) {
        CUDA_CHECK(cudaGraphLaunch(memory->factor_exec,stream));
        return CUDSS_STATUS_SUCCESS;
    }
    // Warm the vendor factorization once. Its normal allocator callbacks populate
    // the retained pool before capture. Every later factorization replays this graph.
    auto status=g_cuda_funcs.cudssExecute(handle,phase,config,data,matrix,solution,rhs);
    if(status!=CUDSS_STATUS_SUCCESS)return status;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    memory->capturing=true;
    CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    status=g_cuda_funcs.cudssExecute(handle,phase,config,data,matrix,solution,rhs);
    const auto captured=cudaStreamEndCapture(stream,&memory->factor_graph);
    memory->capturing=false;
    CUDA_CHECK(captured);
    if(status!=CUDSS_STATUS_SUCCESS)return status;
    size_t count=0;CUDA_CHECK(cudaGraphGetNodes(memory->factor_graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA_CHECK(cudaGraphGetNodes(memory->factor_graph,nodes.data(),&count));
    for(auto node:nodes) {
        cudaGraphNodeType type;CUDA_CHECK(cudaGraphNodeGetType(node,&type));
        if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms copy{};CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node,&copy));
            if(copy.kind!=cudaMemcpyDeviceToDevice) {
                fprintf(stderr,"Factor graph requires a host copy; refusing stale host inputs\n");exit(115);
            }
        } else if(type!=cudaGraphNodeTypeKernel && type!=cudaGraphNodeTypeMemset && type!=cudaGraphNodeTypeEmpty) {
            fprintf(stderr,"Unsupported factor graph node type %d\n",int(type));exit(116);
        }
    }
    CUDA_CHECK(cudaGraphInstantiate(&memory->factor_exec,memory->factor_graph,0));
    return CUDSS_STATUS_SUCCESS;
}
