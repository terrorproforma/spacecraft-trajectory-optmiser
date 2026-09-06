// Standalone vendor diagnostic: no QOCO, graphs, custom allocator or refinement.
// Strictly diagonally dominant symmetric systems have the known solution x=1.
#include <cuda_runtime.h>
#include <cudss.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { auto s=(call); if(s!=cudaSuccess) { std::fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(s)); return 2; } } while(0)
#define CUDSS(call) do { auto s=(call); if(s!=CUDSS_STATUS_SUCCESS) { std::fprintf(stderr,"cuDSS line %d: %d\n",__LINE__,int(s)); return 3; } } while(0)

int main(int argc,char** argv) {
    if(argc!=5 && argc!=6) return 1;
    const int n=std::atoi(argv[1]),deterministic=std::atoi(argv[2]),superpanels=std::atoi(argv[3]),indefinite=std::atoi(argv[4]);
    if(n<2 || n>100000) return 1;
    std::vector<int> offsets{0},columns;
    std::vector<double> values,rhs(n),x(n);
    for(int row=0;row<n;++row) {
        const double diagonal=indefinite && row%2 ? -4 : 4;
        if(row) { columns.push_back(row-1); values.push_back(.25); }
        columns.push_back(row); values.push_back(diagonal); offsets.push_back(columns.size());
        if(indefinite==2 && row+1<n) {
            columns.push_back(row+1); values.push_back(.25); offsets.back()=columns.size();
        }
        rhs[row]=diagonal+(row ? .25 : 0)+(row+1<n ? .25 : 0);
    }
    int *dr{},*dc{}; double *dv{},*db{},*dx{};
    CUDA(cudaMalloc(&dr,offsets.size()*sizeof(int))); CUDA(cudaMalloc(&dc,columns.size()*sizeof(int)));
    CUDA(cudaMalloc(&dv,values.size()*sizeof(double))); CUDA(cudaMalloc(&db,n*sizeof(double))); CUDA(cudaMalloc(&dx,n*sizeof(double)));
    CUDA(cudaMemcpy(dr,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dc,columns.data(),columns.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dv,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db,rhs.data(),n*sizeof(double),cudaMemcpyHostToDevice)); CUDA(cudaMemset(dx,0,n*sizeof(double)));
    cudssHandle_t handle{}; cudssConfig_t config{}; cudssData_t data{}; cudssMatrix_t matrix{},solution{},right{};
    CUDSS(cudssCreate(&handle)); CUDSS(cudssConfigCreate(&config)); CUDSS(cudssDataCreate(handle,&data));
    CUDSS(cudssConfigSet(config,CUDSS_CONFIG_DETERMINISTIC_MODE,&deterministic,sizeof(int)));
    CUDSS(cudssConfigSet(config,CUDSS_CONFIG_USE_SUPERPANELS,&superpanels,sizeof(int)));
    if(argc==6 && std::atoi(argv[5])) {
        const cudssPivotType_t pivot=CUDSS_PIVOT_NONE;
        CUDSS(cudssConfigSet(config,CUDSS_CONFIG_PIVOT_TYPE,&pivot,sizeof(pivot)));
    }
    int vendor_ir=-1;size_t written=0;
    CUDSS(cudssConfigGet(config,CUDSS_CONFIG_IR_N_STEPS,&vendor_ir,sizeof(int),&written));
    if(written!=sizeof(int) || vendor_ir!=0) return 4;
    CUDSS(cudssMatrixCreateCsr(&matrix,n,n,values.size(),dr,nullptr,dc,dv,CUDSS_R_32I,CUDSS_R_32I,CUDSS_R_64F,
        indefinite==2 ? CUDSS_MTYPE_GENERAL : indefinite ? CUDSS_MTYPE_SYMMETRIC : CUDSS_MTYPE_SPD,
        indefinite==2 ? CUDSS_MVIEW_FULL : CUDSS_MVIEW_LOWER,CUDSS_BASE_ZERO));
    CUDSS(cudssMatrixCreateDn(&right,n,1,n,db,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    CUDSS(cudssMatrixCreateDn(&solution,n,1,n,dx,CUDSS_R_64F,CUDSS_LAYOUT_COL_MAJOR));
    for(int phase:{int(CUDSS_PHASE_ANALYSIS),int(CUDSS_PHASE_FACTORIZATION),int(CUDSS_PHASE_SOLVE)}) {
        std::printf("phase=%d n=%d deterministic=%d superpanels=%d indefinite=%d\n",phase,n,deterministic,superpanels,indefinite); std::fflush(stdout);
        CUDSS(cudssExecute(handle,phase,config,data,matrix,solution,right)); CUDA(cudaDeviceSynchronize());
    }
    CUDA(cudaMemcpy(x.data(),dx,n*sizeof(double),cudaMemcpyDeviceToHost));
    double error=0,residual=0;
    for(int i=0;i<n;++i) {
        if(!std::isfinite(x[i])) return 5;
        error=std::max(error,std::abs(x[i]-1));
        const double diagonal=indefinite && i%2 ? -4 : 4;
        residual=std::max(residual,std::abs(diagonal*x[i]+(i ? .25*x[i-1] : 0)+(i+1<n ? .25*x[i+1] : 0)-rhs[i]));
    }
    std::printf("max_coordinate_error=%.17g max_residual=%.17g\n",error,residual);
    CUDSS(cudssMatrixDestroy(matrix)); CUDSS(cudssMatrixDestroy(solution)); CUDSS(cudssMatrixDestroy(right));
    CUDSS(cudssDataDestroy(handle,data)); CUDSS(cudssConfigDestroy(config)); CUDSS(cudssDestroy(handle));
    CUDA(cudaFree(dr)); CUDA(cudaFree(dc)); CUDA(cudaFree(dv)); CUDA(cudaFree(db)); CUDA(cudaFree(dx));
    return error<=1e-12 && residual<=1e-12 ? 0 : 5;
}
