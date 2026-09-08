// Synchronous observation-only diagnostic, forbidden during graph capture.
#include <cstdio>
#include <cstdlib>
#include <vector>
static void dump_division(const double* lambda,const double* v,const double* d,
    int l,int nsoc,const int* q) {
  const char* dir=getenv("SPACEPDHCG_DIAGNOSTIC_DIVISION_DIR");if(!dir)return;
  cudaStreamCaptureStatus capture;CUDA_CHECK(cudaStreamIsCapturing(cudaStreamPerThread,&capture));
  if(capture!=cudaStreamCaptureStatusNone)abort();
  std::vector<int> sizes(nsoc);
  CUDA_CHECK(cudaMemcpy(sizes.data(),q,nsoc*sizeof(int),cudaMemcpyDeviceToHost));
  int m=l;for(int size:sizes)m+=size;
  std::vector<double> lam(m),rhs(m),out(m);
  CUDA_CHECK(cudaMemcpy(lam.data(),lambda,m*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(rhs.data(),v,m*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.data(),d,m*sizeof(double),cudaMemcpyDeviceToHost));
  static int sequence=0;char name[4096];snprintf(name,sizeof(name),"%s/division-%04d.bin",dir,sequence++);
  FILE* f=fopen(name,"wb");if(!f)abort();int header[3]={l,nsoc,m};
  fwrite(header,sizeof(int),3,f);fwrite(sizes.data(),sizeof(int),nsoc,f);
  fwrite(lam.data(),sizeof(double),m,f);fwrite(rhs.data(),sizeof(double),m,f);fwrite(out.data(),sizeof(double),m,f);
  if(fclose(f)!=0)abort();
}
