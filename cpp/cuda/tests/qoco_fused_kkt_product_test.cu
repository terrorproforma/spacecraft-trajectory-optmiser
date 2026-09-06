// Compare fused sparse KKT products with the existing GPU composition and an
// independently assembled dense product, including missing/empty blocks.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {
#include "qoco.h"
#include "qoco_linalg.h"
void qoco_gpu_kkt_sparse(double*,double*,QOCOProblemData*);
int qoco_gpu_begin_reduction_scope();
void qoco_gpu_end_reduction_scope();
}
bool load_cuda_libraries();
static void require(bool ok,const char* message) {
    if (!ok) { std::fprintf(stderr,"FAIL: %s\n",message); std::exit(1); }
}
static void check(cudaError_t s) { require(s==cudaSuccess,cudaGetErrorString(s)); }
struct HostMatrix {
    int rows,cols; bool symmetric;
    std::vector<int> offsets,indices;
    std::vector<double> values;
    QOCOMatrix* device{};
    HostMatrix(int r,int c,bool sym,int seed,bool empty=false):rows(r),cols(c),symmetric(sym),offsets(c+1) {
        for (int col=0;col<c;++col) {
            offsets[col]=values.size();
            if (!empty && r) for (int j=0;j<4;++j) {
                int row=(col*7+j*13+seed)%r;
                if (sym) row%=col+1;
                indices.push_back(row);values.push_back(std::sin(col*.013+j+seed));
            }
        }
        offsets[c]=values.size();
        QOCOCscMatrix m{r,c,static_cast<int>(values.size()),indices.data(),offsets.data(),values.data()};
        device=new_qoco_matrix(&m);
    }
    ~HostMatrix() { free_qoco_matrix(device); }
    void accumulate(const std::vector<double>& x,std::vector<long double>& y,int shift,bool trans) {
        for (int c=0;c<cols;++c) for (int e=offsets[c];e<offsets[c+1];++e) {
            const int r=indices[e]; const long double v=values[e];
            if (trans) y[c]+=v*x[shift+r];
            else y[shift+r]+=v*x[c];
            if (symmetric && c!=r) y[c]+=v*x[r];
        }
    }
};
int main() {
    require(load_cuda_libraries(),"CUDA libraries");
    int cases=0;
    for (int n:{3,257,4097}) for (int variant=0;variant<8;++variant) {
        const int p=variant&1 ? 0 : n/2+1,m=variant&2 ? 0 : n+3,N=n+p+m;
        HostMatrix P(n,n,true,1),A(p,n,false,2,variant==4),G(m,n,false,3,variant==5);
        QOCOProblemData d{};d.n=n;d.p=p;d.m=m;
        d.P=variant&4 ? nullptr : P.device;d.A=A.device;d.G=G.device;
        double *x{},*y{},*reference{},*scratch{};
        for (double** ptr:{&x,&y,&reference,&scratch}) check(cudaMalloc(ptr,N*sizeof(double)));
        std::vector<double> host(N),actual(N),expected(N);
        for (int repeat=0;repeat<3;++repeat) {
            for (int i=0;i<N;++i) host[i]=std::cos(i*.03+repeat);
            check(cudaMemcpy(x,host.data(),N*sizeof(double),cudaMemcpyHostToDevice));
            require(qoco_gpu_begin_reduction_scope()==0,"scope");
            if (d.P) USpMv(d.P,x,reference); else check(cudaMemsetAsync(reference,0,n*sizeof(double)));
            if (p) { SpMtv(d.A,x+n,scratch);qoco_axpy(reference,scratch,reference,1,n);SpMv(d.A,x,reference+n); }
            if (m) { SpMtv(d.G,x+n+p,scratch);qoco_axpy(reference,scratch,reference,1,n);SpMv(d.G,x,reference+n+p); }
            qoco_gpu_kkt_sparse(x,y,&d);
            qoco_gpu_end_reduction_scope();
            check(cudaMemcpy(actual.data(),y,N*sizeof(double),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(expected.data(),reference,N*sizeof(double),cudaMemcpyDeviceToHost));
            require(std::memcmp(actual.data(),expected.data(),N*sizeof(double))==0,"bitwise GPU composition parity");
            std::vector<long double> dense(N);
            if (d.P) P.accumulate(host,dense,0,false);
            if (p) { A.accumulate(host,dense,n,true);A.accumulate(host,dense,n,false); }
            if (m) { G.accumulate(host,dense,n+p,true);G.accumulate(host,dense,n+p,false); }
            for (int i=0;i<N;++i) require(std::isfinite(actual[i]) && std::abs(actual[i]-dense[i])<=2e-12L*(1+std::abs(dense[i])),"independent dense product");
            ++cases;
        }
        for (auto* ptr:{x,y,reference,scratch}) check(cudaFree(ptr));
    }
    std::printf("Fused KKT: %d bitwise and independent dense cases PASS\n",cases);
}
