#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>
#define CHECK(call) do {auto error=(call);if(error!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(error));exit(1);}}while(0)
struct QOCOCscMatrix {int m,n,nnz; double* x; int* i; int* p;};
namespace qoco_fused_kkt {struct Matrix {const QOCOCscMatrix* csc;const int *offsets,*entries,*columns;};}
#include "operator.cuh"
template<class T> std::vector<T> read(FILE* f,size_t count){std::vector<T> v(count);if(fread(v.data(),sizeof(T),count,f)!=count)exit(2);return v;}
struct Device {
 std::vector<void*> owned;
 ~Device(){for(void* p:owned)CHECK(cudaFree(p));}
 template<class T> T* upload(const std::vector<T>& a){T* p;CHECK(cudaMalloc(&p,std::max(size_t(1),a.size())*sizeof(T)));owned.push_back(p);if(a.size())CHECK(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
 double* zeros(int n){return upload(std::vector<double>(n));}
};
struct Entry {int row,col;double value;};
qoco_fused_kkt::Matrix matrix(Device& d,int rows,int cols,std::vector<Entry> entries){
 std::sort(entries.begin(),entries.end(),[](auto a,auto b){return a.col!=b.col?a.col<b.col:a.row<b.row;});
 std::vector<int> cp(cols+1),ri,offset(rows+1),indices,columns;std::vector<double> val;
 for(auto e:entries){++cp[e.col+1];++offset[e.row+1];ri.push_back(e.row);val.push_back(e.value);}
 for(int i=0;i<cols;++i)cp[i+1]+=cp[i];for(int i=0;i<rows;++i)offset[i+1]+=offset[i];
 indices.resize(entries.size());columns.resize(entries.size());auto cursor=offset;
 for(int i=0;i<int(entries.size());++i){auto e=entries[i];int slot=cursor[e.row]++;indices[slot]=i;columns[slot]=e.col;}
 QOCOCscMatrix c{rows,cols,int(entries.size()),d.upload(val),d.upload(ri),d.upload(cp)};
 return {d.upload(std::vector<QOCOCscMatrix>{c}),d.upload(offset),d.upload(indices),d.upload(columns)};
}
int main(int argc,char** argv){
 if(argc<3)return 2;cudaStream_t stream;CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 for(int input=2;input<argc;++input){
  FILE* f=fopen(argv[input],"rb");if(!f)return 2;
  auto h=read<int>(f,8);int N=h[0],nnz=h[1],n=h[2],p=h[3],m=h[4],l=h[5],ns=h[6];
  auto reg=read<double>(f,3);auto rp=read<int>(f,N+1),ci=read<int>(f,nnz);auto val=read<double>(f,nnz);
  auto b=read<double>(f,N),x=read<double>(f,N),res=read<double>(f,N),nt=read<double>(f,h[7]);
  auto ni=read<int>(f,ns),starts=read<int>(f,ns),q=read<int>(f,ns);if(fgetc(f)!=EOF)return 2;fclose(f);
  Device d;std::vector<Entry> pe,ae,ge;
  for(int row=0;row<n;++row)for(int k=rp[row];k<rp[row+1];++k){int col=ci[k];if(col<n)pe.push_back({row,col,val[k]});else if(col<n+p)ae.push_back({col-n,row,val[k]});else ge.push_back({col-n-p,row,val[k]});}
  auto P=matrix(d,n,n,pe),A=matrix(d,p,n,ae),G=matrix(d,m,n,ge);
  auto db=d.upload(b),dx=d.upload(x),w=d.upload(nt);auto dni=d.upload(ni),ds=d.upload(starts),dq=d.upload(q);
  auto hi=d.zeros(N),lo=d.zeros(N),temp=d.zeros(m),tail=d.zeros(m);
  CHECK(cudaDeviceSynchronize()); // Order pageable/default-stream uploads before the nonblocking stream.
  auto launch=[&]{using namespace qoco_precise_ir;
   qoco_precise_ir::nt<<<(l+ns+255)/256,256,0,stream>>>(w,dni,ds,dq,l,ns,dx+n+p,nullptr,temp,tail,false);
   sparse<<<(N+255)/256,256,0,stream>>>(P,A,G,n,p,m,reg[0],db,dx,hi,lo);
   qoco_precise_ir::nt<<<(l+ns+255)/256,256,0,stream>>>(w,dni,ds,dq,l,ns,temp,tail,hi+n+p,lo+n+p,true);
   CHECK(cudaGetLastError());};
  // Run the exact candidate kernels on a non-default stream and in a graph.
  for(int graph=0;graph<2;++graph){
   cudaGraph_t g{};cudaGraphExec_t exec{};
   if(graph){CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();CHECK(cudaStreamEndCapture(stream,&g));CHECK(cudaGraphInstantiate(&exec,g,0));CHECK(cudaGraphLaunch(exec,stream));}
   else launch();
   CHECK(cudaStreamSynchronize(stream));std::vector<double> output(N);CHECK(cudaMemcpy(output.data(),hi,N*sizeof(double),cudaMemcpyDeviceToHost));
   std::string input_path=argv[input];auto slash=input_path.find_last_of('/');std::string target=std::string(argv[1])+"/"+input_path.substr(slash+1)+"."+std::to_string(graph)+".res";
   f=fopen(target.c_str(),"wb");if(!f)return 2;fwrite(output.data(),sizeof(double),N,f);fclose(f);
   if(graph){CHECK(cudaGraphExecDestroy(exec));CHECK(cudaGraphDestroy(g));}
  }
 }
 CHECK(cudaStreamDestroy(stream));printf("%d snapshots, direct and graph execution\n",argc-2);
}
