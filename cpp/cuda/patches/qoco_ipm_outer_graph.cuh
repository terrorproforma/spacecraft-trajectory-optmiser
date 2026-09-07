// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
extern "C" int qoco_gpu_begin_reduction_scope();
extern "C" void qoco_gpu_end_reduction_scope();

extern "C" int qoco_gpu_ipm_emit_graph(QOCOSolver* solver,cudaGraph_t graph,
    const cudaGraphNode_t* dependencies,size_t count,const double* numeric_update,
    QocoGpuOutput* output,cudaGraphNode_t* completion_node) {
    if (output) *output={};
    if (completion_node) *completion_node=nullptr;
    if (!output || !completion_node || !graph || (count && !dependencies) || !solver ||
        !solver->work || !solver->settings || !solver->sol || !solver->linsys_data ||
        !solver->linsys_data->ir || qoco_validate_settings(solver->settings)) return 1;
    auto* s=solver->linsys_data;
    auto* work=solver->work;
    auto* settings=solver->settings;
    auto& cache=s->ir->ipm;
    if (!cache.executable || !cache.linear || !cache.initialization || !cache.terminal ||
        cache.replay_pending || qoco_ipm_current || !qoco_gpu_ipm_initialization_enabled() ||
        !qoco_ir_counts_enabled() || settings->verbose ||
        getenv("SPACEPDHCG_TEST_QOCO_IPM_TERMINAL_DISABLE") ||
        getenv("SPACEPDHCG_TEST_QOCO_IPM_CACHE_DISABLE") ||
        getenv("SPACEPDHCG_TEST_QOCO_HOST_INITIAL_CONE") ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_STEPS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_COMBINED_RHS_COMPARE") ||
        getenv("SPACEPDHCG_TEST_QOCO_DEVICE_CONTROL_COMPARE") ||
        !qoco_gpu_ipm_resources_compatible(cache.resources) ||
        cache.work!=work || cache.control!=work->gpu_control ||
        cache.static_p!=settings->kkt_static_reg_P || cache.static_a!=settings->kkt_static_reg_A ||
        cache.static_g!=settings->kkt_static_reg_G || cache.linsys_static_p!=s->kkt_static_reg_P) return 2;
    cudaStreamCaptureStatus status{};
    if (cudaStreamIsCapturing(cudaStreamPerThread,&status)!=cudaSuccess) return 4;
    if (status!=cudaStreamCaptureStatusNone) return 2;
    size_t existing{};
    if (cudaGraphGetNodes(graph,nullptr,&existing)!=cudaSuccess) return 1;
    std::vector<cudaGraphNode_t> nodes(existing);
    if (existing && cudaGraphGetNodes(graph,nodes.data(),&existing)!=cudaSuccess) return 1;
    for(size_t i=0;i<count;++i)
        if(std::find(nodes.begin(),nodes.end(),dependencies[i])==nodes.end()) return 1;
    if (qoco_gpu_begin_reduction_scope()) return 4;
    cache.resources=qoco_gpu_ipm_resources_enter(cache.resources);
    QocoIpmCapture capture;
    capture.graph=graph;
    capture.linear=cache.linear;
    capture.parameters=cache.parameters;
    if (count) capture.dependencies.assign(dependencies,dependencies+count);
    const QocoIpmParameters parameters{work->scaling->k,work->scaling->kinv,
        settings->abstol,settings->reltol,settings->abstol_inacc,settings->reltol_inacc,
        settings->ir_tol,settings->max_iters,settings->max_ir_iters,
        settings->kkt_dynamic_reg,int(work->use_x0)};
    qoco_ipm_current=&capture;
    qoco_ipm_resume();
    if (numeric_update)
        qoco_ipm_parameters_from_update<<<1,1>>>(cache.parameters,parameters,numeric_update);
    else qoco_ipm_parameters_set<<<1,1>>>(cache.parameters,parameters);
    qoco_ipm_pause();
    *completion_node=qoco_ipm_emit_body(solver,capture,true,true);
    qoco_gpu_ipm_resources_leave(cache.resources);
    qoco_gpu_end_reduction_scope();
    *output={cache.completion,work->x->d_data,work->y->d_data,work->s->d_data,
        work->z->d_data,work->data->n,work->data->p,work->data->m};
    return 0;
}
