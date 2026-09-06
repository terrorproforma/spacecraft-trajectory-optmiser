// Diagnostic only: repeat actual KKT numerical factorization and solve in a
// retained graph while matrices/RHS change through normal trajectory IPM steps.
namespace qoco_factor_graph_probe {
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    std::vector<void*> snapshots;
    int calls=0;
};
static std::unordered_map<LinSysData*,Graph> graphs;
static void snapshot_inputs(Graph& g) {
    size_t count=0;CUDA_CHECK(cudaGraphGetNodes(g.graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);CUDA_CHECK(cudaGraphGetNodes(g.graph,nodes.data(),&count));
    int types[20]={};
    for(auto node:nodes) {
        cudaGraphNodeType type;CUDA_CHECK(cudaGraphNodeGetType(node,&type));
        if(int(type)<20)++types[int(type)];
        if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms copy{};CUDA_CHECK(cudaGraphMemcpyNodeGetParams(node,&copy));
            if(copy.kind!=cudaMemcpyHostToDevice)continue;
            if(copy.srcArray || copy.dstArray || copy.extent.height!=1 || copy.extent.depth!=1
                || copy.srcPos.x || copy.srcPos.y || copy.srcPos.z)exit(112);
            fprintf(stderr,"FACTOR_GRAPH snapshot bytes=%zu\n",copy.extent.width);
            void* owned;CUDA_CHECK(cudaMalloc(&owned,copy.extent.width));
            CUDA_CHECK(cudaMemcpy(owned,copy.srcPtr.ptr,copy.extent.width,cudaMemcpyHostToDevice));
            copy.srcPtr.ptr=owned;copy.kind=cudaMemcpyDeviceToDevice;
            CUDA_CHECK(cudaGraphMemcpyNodeSetParams(node,&copy));g.snapshots.push_back(owned);
        } else if(type!=cudaGraphNodeTypeKernel && type!=cudaGraphNodeTypeMemset && type!=cudaGraphNodeTypeEmpty) {
            fprintf(stderr,"FACTOR_GRAPH unsupported node=%d\n",int(type));exit(113);
        }
    }
    fprintf(stderr,"FACTOR_GRAPH nodes=%zu types=",count);
    for(int i=0;i<20;++i)if(types[i])fprintf(stderr,"%d:%d,",i,types[i]);
    fprintf(stderr,"\n");
}
static void run(LinSysData* s,QOCOWorkspace* work,const double* b,const double* expected_x) {
    const char* limit=getenv("SPACEPDHCG_TEST_QOCO_FACTOR_GRAPH_PROBE");
    if(!limit)return;
    auto& g=graphs[s];
    if(g.calls>=atoi(limit))return;
    ++g.calls;
    CUDA_CHECK(cudaDeviceSynchronize());
    const int n=s->Kn;const size_t bytes=n*sizeof(double);auto stream=s->ir->stream;
    std::vector<double> expected(n),actual(n);
    CUDA_CHECK(cudaMemcpy(expected.data(),expected_x,bytes,cudaMemcpyDeviceToHost));
    const double expected_norm=compute_linsys_residual(s,work,b,expected_x,s->d_xyz_matrix_data);
    if(!g.executable) {
        // Factor and solve have already run once using this allocator/stream.
        s->factor_memory->capturing=true;
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        const auto factor=g_cuda_funcs.cudssExecute(s->handle,CUDSS_PHASE_FACTORIZATION,s->config,s->data,
            s->K_csr,s->d_xyz_matrix,s->d_rhs_matrix);
        CUDA_CHECK(cudaMemsetAsync(s->d_xyz_matrix_data,0,bytes,stream));
        const auto solve=g_cuda_funcs.cudssExecute(s->handle,CUDSS_PHASE_SOLVE,s->config,s->data,
            s->K_csr,s->d_xyz_matrix,s->d_rhs_matrix);
        const auto capture=cudaStreamEndCapture(stream,&g.graph);
        s->factor_memory->capturing=false;
        fprintf(stderr,"FACTOR_GRAPH capture factor=%d solve=%d cuda=%d Kn=%d\n",int(factor),int(solve),int(capture),n);
        if(factor || solve || capture)exit(114);
        snapshot_inputs(g);
        CUDA_CHECK(cudaGraphInstantiate(&g.executable,g.graph,0));
    }
    for(int repeat=0;repeat<2;++repeat) {
        CUDA_CHECK(cudaMemcpyAsync(s->d_rhs_matrix_data,b,bytes,cudaMemcpyDeviceToDevice,stream));
        CUDA_CHECK(cudaGraphLaunch(g.executable,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(actual.data(),s->d_xyz_matrix_data,bytes,cudaMemcpyDeviceToHost));
        double error=0,scale=1;bool finite=true;
        for(int i=0;i<n;++i) {
            finite=finite && std::isfinite(actual[i]) && std::isfinite(expected[i]);
            error=std::max(error,std::abs(actual[i]-expected[i]));scale=std::max(scale,std::abs(expected[i]));
        }
        const double norm=compute_linsys_residual(s,work,b,s->d_xyz_matrix_data,s->d_xyz_matrix_data);
        fprintf(stderr,"FACTOR_GRAPH result call=%d repeat=%d Kn=%d finite=%d scaled_error=%.17g parity=%d direct_norm=%.17g graph_norm=%.17g\n",
            g.calls,repeat,n,finite,error/scale,finite && error<=1e-12*scale,expected_norm,norm);
    }
    // expected_x is untouched; production residual assembly refreshes scratch.
}
static void cleanup(LinSysData* s) {
    auto found=graphs.find(s);if(found==graphs.end())return;auto& g=found->second;
    CUDA_CHECK(cudaDeviceSynchronize());
    if(g.executable)CUDA_CHECK(cudaGraphExecDestroy(g.executable));
    if(g.graph)CUDA_CHECK(cudaGraphDestroy(g.graph));
    for(auto pointer:g.snapshots)CUDA_CHECK(cudaFree(pointer));graphs.erase(found);
}
}
