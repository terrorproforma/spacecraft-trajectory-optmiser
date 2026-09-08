// Observation only: capture the limiting cone of very small actual line searches.
#include <cstdio>
#include <cstdlib>
#include <vector>
static void dump_small_step(QOCOFloat* u,QOCOFloat* du,double factor,
    QOCOSolver* solver,QOCOFloat* output) {
  const char* dir=getenv("SPACEPDHCG_DIAGNOSTIC_STEP_DIR");if(!dir)return;
  cudaStreamCaptureStatus capture;CUDA_CHECK(cudaStreamIsCapturing(cudaStreamPerThread,&capture));
  if(capture!=cudaStreamCaptureStatusNone)abort();
  double alpha;CUDA_CHECK(cudaMemcpy(&alpha,output,sizeof(double),cudaMemcpyDeviceToHost));
  if(alpha>=1e-8)return;
  auto* d=solver->work->data;int header[3]={d->l,d->nsoc,d->m};
  std::vector<int> q(d->nsoc);std::vector<double> hu(d->m),hd(d->m);
  CUDA_CHECK(cudaMemcpy(q.data(),get_data_vectori(d->q),q.size()*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hu.data(),u,hu.size()*sizeof(double),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hd.data(),du,hd.size()*sizeof(double),cudaMemcpyDeviceToHost));
  static int sequence=0;char name[4096];snprintf(name,sizeof(name),"%s/step-%04d.bin",dir,sequence++);
  FILE* f=fopen(name,"wb");if(!f)abort();
  fwrite(header,sizeof(int),3,f);fwrite(&alpha,sizeof(double),1,f);fwrite(&factor,sizeof(double),1,f);
  fwrite(q.data(),sizeof(int),q.size(),f);fwrite(hu.data(),sizeof(double),hu.size(),f);fwrite(hd.data(),sizeof(double),hd.size(),f);fclose(f);
}
