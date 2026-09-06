// Diagnostic allocator: retain vendor allocations to test graph address lifetime.
// Only same-stream free blocks are reusable; no new allocation during capture.
struct QocoFactorMemory {
    struct Block { void* pointer; size_t size; cudaStream_t stream; bool used; };
    std::vector<Block> blocks;
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
        fprintf(stderr,"FACTOR_GRAPH malloc bytes=%zu status=%d\n",size,int(status));
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
    for(auto block:memory->blocks)CUDA_CHECK(cudaFree(block.pointer));
    delete memory;
}
