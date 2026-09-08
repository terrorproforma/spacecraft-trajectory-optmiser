// SPDX-License-Identifier: Apache-2.0
// Included after the workspace owner and retained common/L1 helpers.
extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_set_mass_options(
    spacepdhcg_cuda_workspace* workspace,const spacepdhcg_cuda_mass_options* options,
    const spacepdhcg_cuda_mass_node* host_nodes
) try {
    if(!workspace||!options||options->abi_version!=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION
        ||(options->enabled!=0&&options->enabled!=1)||options->reserved
        ||(options->enabled && (options->node_count<2||options->node_count>4096||!host_nodes))
        ||(!options->enabled && (options->node_count||host_nodes)))return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::lock_guard lock(workspace->mutex);
    auto status=finalize_if_complete(workspace);if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    const auto stream=native_stream(workspace->consumer_stream);
    if(!options->enabled) {
        if(workspace->mass_enabled) {
            halpern_restore_default_history<<<64,kThreads,0,stream>>>(workspace->device_problem);
            halpern_force_refresh<<<1,1,0,stream>>>(workspace->control);
            auto error=cudaGetLastError();if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
            if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass disable restores history and L1 scaling refresh");
        }
        workspace->mass_enabled=workspace->mass_valid=workspace->l1_valid=workspace->common_valid=false;
        workspace->termination=SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;return SPACEPDHCG_CUDA_SUCCESS;
    }
    if(workspace->mass_enabled||!workspace->l1_enabled||!workspace->common_enabled
        ||workspace->halpern_mode||workspace->cooperative_blocks<=0||workspace->l1_setup.omega!=1.0) {
        set_error(workspace,"mass elimination requires common KKT, unit L1, positive cooperative grid and disabled prior mass mode");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    const int n=workspace->structure.variables,rows=workspace->structure.scalar_rows;
    if(options->node_count>n || 3LL*options->node_count-2>n)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::vector<int> variable(n,-1),equality(rows,-1),gamma(n,-1),virtuals(n,-1),used(n,0);
    for(int i=0;i<options->node_count;++i) {
        const auto node=host_nodes[i];
        if(node.mass_variable<0||node.mass_variable>=n||node.equality_row<0||node.equality_row>=rows
            ||used[node.mass_variable]||equality[node.equality_row]>=0)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        used[node.mass_variable]=1;variable[node.mass_variable]=i;equality[node.equality_row]=i;
        if(!i) {if(node.gamma_variable!=-1||node.virtual_variable!=-1)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;continue;}
        if(node.gamma_variable<0||node.gamma_variable>=n||node.virtual_variable<0||node.virtual_variable>=n
            ||node.gamma_variable==node.virtual_variable||used[node.gamma_variable]||used[node.virtual_variable])return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
        used[node.gamma_variable]=used[node.virtual_variable]=1;gamma[node.gamma_variable]=virtuals[node.virtual_variable]=i;
    }
    cudaDeviceProp properties{};auto error=cudaGetDeviceProperties(&properties,workspace->consumer_stream.device.id);
    int active=0,initial=0;
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,cooperative_mass_kernel,kThreads,0);
    if(error==cudaSuccess)error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&initial,cooperative_mass_initialise_kernel,kThreads,0);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass cooperative occupancy");
    const int capacity=std::min(workspace->common_cooperative_capacity,std::min(active,initial)*properties.multiProcessorCount);
    if(workspace->cooperative_blocks>capacity) {
        set_error(workspace,"requested cooperative grid exceeds mass solve/metric occupancy; select fewer positive blocks first");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    if(!workspace->mass) {
        MassState setup{};MassState* device=nullptr;spacepdhcg_cuda_mass_diagnostics* host=nullptr;
#define MASS_ALLOC(pointer,count,type) \
        status=allocate_device(workspace,reinterpret_cast<void**>(&(pointer)),std::max<std::size_t>(1,(count))*sizeof(type),AllocationCategory::iterate); \
        if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
        MASS_ALLOC(setup.nodes,std::min(n,4096),spacepdhcg_cuda_mass_node)
        MASS_ALLOC(setup.variable_node,n,int)
        MASS_ALLOC(setup.equality_node,rows,int)
        MASS_ALLOC(setup.gamma_node,n,int)
        MASS_ALLOC(setup.virtual_node,n,int)
        MASS_ALLOC(setup.row_node,rows,int)
        MASS_ALLOC(setup.row_sign,rows,int)
        MASS_ALLOC(setup.g,std::min(n,4096),double)
        MASS_ALLOC(setup.affine,std::min(n,4096),double)
        MASS_ALLOC(setup.constant,std::min(n,4096),double)
        MASS_ALLOC(setup.scan0,std::min(n,4096),double)
        MASS_ALLOC(setup.scan1,std::min(n,4096),double)
        MASS_ALLOC(setup.suffix,std::min(n,4096),double)
        MASS_ALLOC(setup.row_sum,rows+workspace->structure.affine_rows,double)
        MASS_ALLOC(setup.column_sum,n,double)
        MASS_ALLOC(setup.upper,rows,double)
        MASS_ALLOC(device,1,MassState)
#undef MASS_ALLOC
        error=workspace->ledger.allocate_pinned(reinterpret_cast<void**>(&host),sizeof(*host),AllocationCategory::diagnostics,workspace->update_epoch+1U);
        if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass pinned diagnostics");
        *host={};workspace->mass=device;workspace->mass_setup=setup;workspace->host_mass=host;
    }
    auto& setup=workspace->mass_setup;setup.options=*options;setup.result={};
#define MASS_COPY(destination,source,bytes) \
    status=copy_async(workspace,(destination),(source),(bytes),cudaMemcpyHostToDevice,stream,false); \
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    MASS_COPY(setup.nodes,host_nodes,options->node_count*sizeof(*host_nodes))
    MASS_COPY(setup.variable_node,variable.data(),n*sizeof(int))
    MASS_COPY(setup.equality_node,equality.data(),rows*sizeof(int))
    MASS_COPY(setup.gamma_node,gamma.data(),n*sizeof(int))
    MASS_COPY(setup.virtual_node,virtuals.data(),n*sizeof(int))
    MASS_COPY(workspace->mass,&setup,sizeof(setup))
#undef MASS_COPY
    int* invalid=nullptr;status=allocate_device(workspace,reinterpret_cast<void**>(&invalid),sizeof(int),AllocationCategory::diagnostics);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    error=cudaMemsetAsync(invalid,0,sizeof(int),stream);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass proof initialization");
    mass_prepare<<<64,kThreads,0,stream>>>(workspace->device_problem,workspace->mass);
    mass_validate<<<64,kThreads,0,stream>>>(workspace->device_problem,workspace->l1,workspace->mass,invalid);
    error=cudaGetLastError();if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass exact map proof");
    int host_invalid=0;status=copy_async(workspace,&host_invalid,invalid,sizeof(int),cudaMemcpyDeviceToHost,stream,false);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    error=cudaStreamSynchronize(stream);if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass exact map proof wait");
    error=workspace->ledger.release(invalid,workspace->update_epoch+workspace->solve_epoch+2U);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass proof scratch cleanup");
    if(host_invalid) {
        set_error(workspace,"mass map fails exact causal Q=0/zero-cost/three-singleton-row proof");
        return SPACEPDHCG_CUDA_UNSUPPORTED;
    }
    halpern_restore_default_history<<<64,kThreads,0,stream>>>(workspace->device_problem);
    halpern_force_refresh<<<1,1,0,stream>>>(workspace->control);
    error=cudaGetLastError();if(error==cudaSuccess)error=cudaStreamSynchronize(stream);
    if(error!=cudaSuccess)return cuda_failure(workspace,error,"mass history and metric refresh");
    workspace->mass_enabled=true;workspace->mass_valid=workspace->l1_valid=workspace->common_valid=false;
    workspace->mass_cooperative_capacity=capacity;workspace->termination=SPACEPDHCG_CUDA_TERMINATION_UNSPECIFIED;
    return SPACEPDHCG_CUDA_SUCCESS;
} catch(const std::bad_alloc&) {return SPACEPDHCG_CUDA_OUT_OF_MEMORY;}
  catch(...) {return SPACEPDHCG_CUDA_INTERNAL_ERROR;}

extern "C" spacepdhcg_cuda_status spacepdhcg_cuda_workspace_mass_diagnostics(
    spacepdhcg_cuda_workspace* workspace,spacepdhcg_cuda_mass_diagnostics* diagnostics) {
    if(!workspace||!diagnostics)return SPACEPDHCG_CUDA_INVALID_ARGUMENT;
    std::lock_guard lock(workspace->mutex);const auto status=finalize_if_complete(workspace);
    if(status!=SPACEPDHCG_CUDA_SUCCESS)return status;
    *diagnostics={};diagnostics->abi_version=SPACEPDHCG_CUDA_WORKSPACE_ABI_VERSION;diagnostics->enabled=workspace->mass_enabled;
    if(workspace->mass_valid)*diagnostics=*workspace->host_mass;return SPACEPDHCG_CUDA_SUCCESS;
}
