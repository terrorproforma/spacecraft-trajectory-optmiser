from pathlib import Path
root=Path('build/performance/nt-normalization-v347');root.mkdir(exist_ok=False)
header=Path('build/performance/cone-step-v345/arithmetic.cuh').read_text()
header=header.replace('// Diagnostic: compensated unregularized KKT residual, entirely on device.','// Experimental double-double Nesterov-Todd normalization on the GPU.')
header += r'''
namespace qoco_cone_arithmetic {
__device__ DD divide(DD a, DD b) {
    double q1=a.hi/b.hi;
    DD r=add(a,neg(mul(b,DD(q1))));
    double q2=r.hi/b.hi;
    r=add(r,neg(mul(b,DD(q2))));
    return add(add(DD(q1),DD(q2)),DD(r.hi/b.hi));
}
__device__ DD root(DD a) {
    double h=sqrt(a.hi);
    if(!(h>0.0) || !isfinite(h)) return DD(h);
    DD r=add(a,neg(mul(DD(h),DD(h))));
    return add(DD(h),divide(r,DD(2.0*h)));
}
__device__ DD euclidean(const double* s,const double* z,int n) {
    DD sum;
    for(int j=0;j<n;++j)sum=add(sum,mul(DD(s[j]),DD(z[j])));
    return sum;
}
}
'''
(root/'arithmetic.cuh').write_text(header)
s=Path('/home/angus/build-qoco-nonfinite-ir-v336/source/src/cone.cu').read_text()
a=s.index('__global__ void compute_nt_scaling_kernel');b=s.index('__global__ void nt_multiply_kernel',a)
kernel=s[a:b];(root/'old_kernel.cuh').write_text(kernel)
a=kernel.index('  /* --- normalize s --- */');b=kernel.index('  /* Store compact fast scaling',a)
normal=r'''
  // Keep normalization and cancellation in wbar in double-double until storage.
  using namespace qoco_cone_arithmetic;
  DD ss=root(inner(&s[idx],&s[idx],qi));
  DD zz=root(inner(&z[idx],&z[idx],qi));
  DD gamma=root(mul(DD(0.5),add(DD(1.0),
      divide(euclidean(&s[idx],&z[idx],qi),mul(ss,zz)))));
  DD denominator=mul(DD(2.0),gamma);
  for(int j=0;j<qi;++j) {
    DD sn=divide(DD(s[idx+j]),ss), zn=divide(DD(z[idx+j]),zz);
    sbar[idx+j]=rounded(divide(add(sn,j==0?zn:neg(zn)),denominator));
  }
  QOCOFloat eta=rounded(root(divide(ss,zz)));
  QOCOFloat finv=safe_div((QOCOFloat)1.0,eta);
  QOCOFloat eta2=eta*eta;
  QOCOFloat f;

'''
(root/'normalization.cuh').write_text(normal)
(root/'new_kernel.cuh').write_text(kernel[:a]+normal+kernel[b:])
build=Path('build/performance/build_cone_determinant_v344.py').read_text()
a=build.index("header=Path(");b=build.index("p.write_text(text)",a)+len("p.write_text(text)")
build=build[:a]+'''header=Path('build/performance/nt-normalization-v347/arithmetic.cuh').read_text()
(source/'src/qoco_cone_arithmetic.cuh').write_text(header)
p=source/'src/cone.cu';text=p.read_text()
a=text.index('__global__ void compute_nt_scaling_kernel');b=text.index('__global__ void nt_multiply_kernel',a)
text=text[:a]+'#include "qoco_cone_arithmetic.cuh"\\n'+Path('build/performance/nt-normalization-v347/new_kernel.cuh').read_text()+text[b:]
p.write_text(text)'''+build[b:]
build=build.replace('build-qoco-cone-determinant-v344','build-qoco-nt-normalization-v347')
Path('build/performance/build_nt_v347.py').write_text(build)
probe=r'''
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
using QOCOFloat=double;using QOCOInt=int;
#define qoco_sqrt(a) sqrt(a)
#define safe_div(a,b) (((b)!=0.0)?((a)/(b)):DBL_MAX)
#include "arithmetic.cuh"
__device__ double soc_residual2(const double* u,int n){double r=u[0]*u[0];for(int j=1;j<n;++j)r-=u[j]*u[j];return r;}
__device__ double qoco_dot_dev(const double* u,const double* v,int n){double r=0;for(int j=0;j<n;++j)r+=u[j]*v[j];return r;}
__device__ void scale_arrayf_dev(const double* u,double* v,double f,int n){for(int j=0;j<n;++j)v[j]=u[j]*f;}
namespace legacy {
#include "old_kernel.cuh"
}
namespace candidate {
#include "new_kernel.cuh"
}
#define CHECK(call) do{auto e=(call);if(e!=cudaSuccess){fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
template<class T> std::vector<T> read(FILE* f,int n){std::vector<T> a(n);if(fread(a.data(),sizeof(T),n,f)!=size_t(n))exit(2);return a;}
template<class T> T* upload(const std::vector<T>& a){T* d;CHECK(cudaMalloc(&d,a.size()*sizeof(T)));CHECK(cudaMemcpy(d,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
double* allocate(int n){double* d;CHECK(cudaMalloc(&d,n*sizeof(double)));return d;}
void output(FILE* f,double* d,int n){std::vector<double> h(n);CHECK(cudaMemcpy(h.data(),d,n*sizeof(double),cudaMemcpyDeviceToHost));fwrite(h.data(),sizeof(double),n,f);}
int main(int argc,char** argv){
 if(argc!=3)return 2;FILE* f=fopen(argv[1],"rb");if(!f)return 2;
 auto h=read<int>(f,2);auto q=read<int>(f,h[0]);auto s=read<double>(f,h[1]);auto z=read<double>(f,h[1]);fclose(f);
 int tri=0;for(auto n:q)tri+=n*(n+1)/2;
 auto dq=upload(q);auto ds=upload(s);auto dz=upload(z);
 auto W=allocate(tri),WtW=allocate(tri),Winv=allocate(tri),nt=allocate(h[1]+h[0]),sb=allocate(h[1]),zb=allocate(h[1]);
 f=fopen(argv[2],"wb");if(!f)return 2;
 legacy::compute_nt_scaling_kernel<<<(h[0]+127)/128,128>>>(W,WtW,nt,Winv,ds,dz,sb,zb,0,h[0],dq);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());output(f,nt,h[1]+h[0]);output(f,WtW,tri);
 candidate::compute_nt_scaling_kernel<<<(h[0]+127)/128,128>>>(W,WtW,nt,Winv,ds,dz,sb,zb,0,h[0],dq);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());output(f,nt,h[1]+h[0]);output(f,WtW,tri);fclose(f);
 for(auto p:{W,WtW,Winv,nt,sb,zb,ds,dz})CHECK(cudaFree(p));CHECK(cudaFree(dq));
 printf("%d paired SOC normalizations evaluated\n",h[0]);
}
'''
(root/'probe.cu').write_text(probe)
