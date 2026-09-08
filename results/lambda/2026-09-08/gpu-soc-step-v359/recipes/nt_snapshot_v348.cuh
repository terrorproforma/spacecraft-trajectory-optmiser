// Observation-only instrumentation. Synchronous, host-dispatched replay only.
#include <cstdio>
#include <cstdlib>
#include <vector>
template<class T> static void nt_dump_array(FILE* f,const T* d,int n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));
  if(fwrite(h.data(),sizeof(T),n,f)!=size_t(n))abort();
}
static void nt_dump(QOCOWorkspace* work) {
  const char* dir=getenv("SPACEPDHCG_DIAGNOSTIC_NT_DIR");if(!dir)return;
  cudaStreamCaptureStatus capture;
  CUDA_CHECK(cudaStreamIsCapturing(cudaStreamPerThread,&capture));
  if(capture!=cudaStreamCaptureStatusNone)abort();
  CUDA_CHECK(cudaDeviceSynchronize());
  auto* data=work->data;std::vector<int> q(data->nsoc);
  CUDA_CHECK(cudaMemcpy(q.data(),get_data_vectori(data->q),q.size()*sizeof(int),cudaMemcpyDeviceToHost));
  int compact=data->l,tri=data->l;
  for(auto n:q){compact+=n+1;tri+=n*(n+1)/2;}
  int header[5]={data->l,data->nsoc,data->m,compact,tri};
  static int sequence=0;char name[4096];snprintf(name,sizeof(name),"%s/nt-%04d.bin",dir,sequence++);
  FILE* f=fopen(name,"wb");if(!f)abort();
  fwrite(header,sizeof(int),5,f);fwrite(q.data(),sizeof(int),q.size(),f);
  nt_dump_array(f,get_data_vectorf(work->s),data->m);
  nt_dump_array(f,get_data_vectorf(work->z),data->m);
  nt_dump_array(f,get_data_vectorf(work->nt_scaling),compact);
  nt_dump_array(f,get_data_vectorf(work->WtW),tri);
  fclose(f);
}
