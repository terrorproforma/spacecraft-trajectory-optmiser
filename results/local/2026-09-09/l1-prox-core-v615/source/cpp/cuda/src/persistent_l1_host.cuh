// SPDX-License-Identifier: Apache-2.0
// Included only after workspace ownership and common-KKT setup helpers.
extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_set_l1_options(
    spacepdhcg_cuda_workspace* workspace,const spacepdhcg_cuda_l1_options* options,
    const spacepdhcg_cuda_l1_pair* host_pairs
) try {
    if(!workspace||!options||options->abi_version!=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        ||(options->enabled!=0 && options->enabled!=1)||options->reserved
        ||options->pair_count<0||options->pair_count>workspace->structure.variables/2
        ||(options->enabled && (!options->pair_count||!host_pairs))
        ||(!options->enabled && (options->pair_count||host_pairs)))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::lock_guard lock(workspace->mutex);
    auto status=finalize_if_complete(workspace);if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    const auto stream=native_stream(workspace->consumer_stream);
    if(!options->enabled) {
        if(workspace->l1_enabled) {
            halpern_restore_default_history<<<64,kThreads,0,stream>>>(workspace->device_problem);
            halpern_force_refresh<<<1,1,0,stream>>>(workspace->control);
            auto error=cudaGetLastError();if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
            if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 disable restores history and original scaling refresh");
        }
        workspace->l1_enabled=workspace->l1_valid=workspace->common_valid=false;
        workspace->termination=SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;return SPACEPDHCG_CUDA_SUCCESS;
    }
    if(workspace->l1_enabled||workspace->halpern_mode||!workspace->common_enabled||workspace->cooperative_blocks<=0) {
        set_error(workspace,"L1 requires common KKT and positive cooperative grid; disable any existing L1/Halpern mode first");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    const int n=workspace->structure.variables,rows=workspace->structure.scalar_rows;
    std::vector<int> used(n,0),inactive(n,0),removed(rows,0);
    for(int i=0;i<options->pair_count;++i) {
        const auto p=host_pairs[i];
        if(p.epigraph_variable<0||p.epigraph_variable>=n||p.absolute_variable<0||p.absolute_variable>=n
            ||p.epigraph_variable==p.absolute_variable||p.positive_scalar_row<0||p.positive_scalar_row>=rows
            ||p.negative_scalar_row<0||p.negative_scalar_row>=rows||p.positive_scalar_row==p.negative_scalar_row
            ||used[p.epigraph_variable]||used[p.absolute_variable]||removed[p.positive_scalar_row]||removed[p.negative_scalar_row])
            return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        used[p.epigraph_variable]=used[p.absolute_variable]=1;inactive[p.epigraph_variable]=1;
        removed[p.positive_scalar_row]=removed[p.negative_scalar_row]=1;
    }
    cudaDeviceProp properties{};auto error=cudaGetDeviceProperties(&properties,workspace->consumer_stream.device.id);
    int active=0,init_active=0;
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,cooperative_l1_kernel,kThreads,0);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&init_active,cooperative_l1_initialise_kernel,kThreads,0);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 cooperative occupancy");
    const int capacity=std::min(workspace->common_cooperative_capacity,std::min(active,init_active)*properties.multiProcessorCount);
    if(workspace->cooperative_blocks>capacity) {
        set_error(workspace,"requested grid exceeds L1 solve/scaling occupancy; select fewer positive blocks first");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    if(!workspace->l1) {
        L1State setup{};L1State* device_state=nullptr;spacepdhcg_cuda_l1_diagnostics* host_report=nullptr;
#define L1_ALLOC(pointer,count,type) \
        status=allocate_device(workspace,reinterpret_cast<void**>(&(pointer)),std::max<std::size_t>(1,(count))*sizeof(type),AllocationCategory::iterate); \
        if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
        L1_ALLOC(setup.pairs,n/2,spacepdhcg_cuda_l1_pair)
        L1_ALLOC(setup.inactive_variable,n,int)
        L1_ALLOC(setup.removed_scalar,rows,int)
        L1_ALLOC(setup.masked_a,workspace->structure.scalar_nonzeros,double)
        L1_ALLOC(setup.smooth_c,n,double)
        L1_ALLOC(setup.lambda,n,double)
        L1_ALLOC(setup.working,1,DeviceProblem)
        L1_ALLOC(device_state,1,L1State)
#undef L1_ALLOC
        error=workspace->ledger.allocate_pinned(reinterpret_cast<void**>(&host_report),sizeof(*host_report),
            AllocationCategory::diagnostics,workspace->update_epoch+1U);
        if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 pinned diagnostics");
        *host_report={};workspace->l1=device_state;workspace->l1_setup=setup;workspace->host_l1=host_report;
    }
    auto& setup=workspace->l1_setup;setup.options=*options;setup.result={};
#define L1_COPY(destination,source,bytes) \
    status=copy_async(workspace,(destination),(source),(bytes),cudaMemcpyHostToDevice,stream,false); \
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    L1_COPY(setup.pairs,host_pairs,options->pair_count*sizeof(*host_pairs))
    L1_COPY(setup.inactive_variable,inactive.data(),n*sizeof(int))
    L1_COPY(setup.removed_scalar,removed.data(),rows*sizeof(int))
    L1_COPY(workspace->l1,&setup,sizeof(setup))
#undef L1_COPY
    int* invalid=nullptr;status=allocate_device(workspace,reinterpret_cast<void**>(&invalid),sizeof(int),AllocationCategory::diagnostics);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    error=cudaMemsetAsync(invalid,0,sizeof(int),stream);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 proof initialization");
    l1_prepare<<<64,kThreads,0,stream>>>(workspace->device_problem,workspace->l1);
    l1_validate_pairs<<<64,kThreads,0,stream>>>(workspace->device_problem,workspace->l1,invalid);
    error=cudaGetLastError();if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 exact map proof");
    int host_invalid=0;status=copy_async(workspace,&host_invalid,invalid,sizeof(int),cudaMemcpyDeviceToHost,stream,false);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    error=cudaStreamSynchronize(stream);if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 exact map proof wait");
    error=workspace->ledger.release(invalid,workspace->update_epoch+workspace->solve_epoch+2U);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 proof scratch cleanup");
    if(host_invalid) {
        set_error(workspace,"L1 map fails exact Q=0, positive-cost isolated epigraph or +/-v-t upper-row proof");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    halpern_force_refresh<<<1,1,0,stream>>>(workspace->control);
    error=cudaGetLastError();if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"L1 reduced scaling refresh request");
    workspace->l1_enabled=true;workspace->l1_valid=workspace->common_valid=false;
    workspace->l1_cooperative_capacity=capacity;workspace->termination=SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;
    return SPACEPDHCG_CUDA_SUCCESS;
} catch(const std::bad_alloc&) {return SPACEPDHCG_CUDA_OUT_OF_MEMORY;}
  catch(...) {return SPACEPDHCG_CUDA_INTERNAL_ERROR;}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_l1_diagnostics(
    spacepdhcg_cuda_workspace* workspace,spacepdhcg_cuda_l1_diagnostics* diagnostics) {
    if(!workspace||!diagnostics)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::lock_guard lock(workspace->mutex);const auto status=finalize_if_complete(workspace);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    *diagnostics={};diagnostics->abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;diagnostics->enabled=workspace->l1_enabled;
    if(workspace->l1_valid)*diagnostics=*workspace->host_l1;return SPACEPDHCG_CUDA_SUCCESS;
}
