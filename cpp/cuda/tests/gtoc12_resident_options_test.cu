#include "spacepdhcg/cuda/gtoc12_collection_c_api.h"
#include "../internal/gtoc12_collection_options.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>
using Option=spacepdhcg_gtoc12_collection_option;
using Query=spacepdhcg_gtoc12_collection_query;
using Result=spacepdhcg_gtoc12_collection_result;
using Table=spacepdhcg_gtoc12_collection_options;
void require(bool b){if(!b){std::fprintf(stderr,"resident option assertion failed\n");std::exit(2);}}
void check(cudaError_t s){if(s!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(s));std::exit(3);}}
void pass(spacepdhcg_cuda_status s){require(s==SPACEPDHCG_CUDA_SUCCESS);}
template<class T> void read(std::ifstream& f,T& value){f.read(reinterpret_cast<char*>(&value),sizeof(T));require(bool(f));}
Table* copy(const std::vector<Option>& rows,cudaStream_t stream){
    Option* device{};check(cudaMalloc(&device,std::max(size_t(1),rows.size())*sizeof(Option)));
    if(!rows.empty())check(cudaMemcpyAsync(device,rows.data(),rows.size()*sizeof(Option),cudaMemcpyHostToDevice,stream));
    Table* table{};pass(gtoc12_collection_options_copy_device(device,int(rows.size()),stream,&table));
    // Destroy producer storage before the first consumer; tables must own their rows.
    check(cudaFree(device));return table;
}
Result compare(spacepdhcg_gtoc12_collection* w,Table* table,const std::vector<Option>& rows,Query q,Option* selected){
    Result host{},resident{};Option dummy{};
    pass(spacepdhcg_gtoc12_collection_host(w,rows.empty()?&dummy:rows.data(),int(rows.size()),&q,&host));
    pass(spacepdhcg_gtoc12_collection_resident(w,table,&q,&resident,selected));
    require(std::memcmp(&host,&resident,sizeof(Result))==0);
    if(resident.index>=0)require(std::memcmp(selected,&rows[resident.index],sizeof(Option))==0);
    else require(selected->delta_v==0 && selected->departure==0 && selected->tof==0);
    return resident;
}
int main(int argc,char** argv){
    static_assert(sizeof(Query)==120 && sizeof(Option)==24 && sizeof(Result)==16);
    check(cudaSetDevice(0));cudaStream_t stream{};check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    spacepdhcg_gtoc12_collection* w{};
    if(argc==2){
        std::ifstream file(argv[1],std::ios::binary);require(bool(file));
        uint32_t nt{},nq{};read(file,nt);read(file,nq);require(nt>0 && nt<100000 && nq<1000000);
        std::vector<std::vector<Option>> rows(nt);std::vector<Table*> tables(nt);int capacity=1;
        for(uint32_t t=0;t<nt;++t){uint32_t n{};read(file,n);require(n<1000000);rows[t].resize(n);
            file.read(reinterpret_cast<char*>(rows[t].data()),size_t(n)*sizeof(Option));require(bool(file));
            tables[t]=copy(rows[t],stream);capacity=std::max(capacity,int(n));}
        pass(spacepdhcg_gtoc12_collection_create(capacity,0,&w));
        for(uint32_t i=0;i<nq;++i){uint32_t t{},found{};Query q{};double expected{};Option winner{},actual{};
            read(file,t);require(t<nt);read(file,q);read(file,expected);read(file,found);read(file,winner);
            const auto r=compare(w,tables[t],rows[t],q,&actual);require(r.status==0 && (r.index>=0)==bool(found));
            if(found){require(std::memcmp(&winner,&actual,sizeof(Option))==0);require(std::abs(r.cost-expected)<=1e-10+1e-13*std::abs(expected));}
            else require(std::isinf(r.cost)&&std::isinf(expected));
        }
        for(auto*& table:tables)pass(spacepdhcg_gtoc12_collection_options_destroy(&table));
        std::printf("%u captured queries across %u tables preserve results bitwise and captured winners/costs\n",nq,nt);
    }else{
        pass(spacepdhcg_gtoc12_collection_create(3073,0,&w));
        Query q{0,0,2000.,68000.,INFINITY,1.,1.,1.,0.,0.,1.,.6,86400.,365.25,10.,60.};
        std::vector<Option> rows(3073);for(int i=0;i<3073;++i)rows[i]={0.,66000.+i*.01,180.};
        Table* table=copy(rows,stream);Option selected{};
        auto result=compare(w,table,rows,q,&selected);require(result.index==3072);
        q.mode=1;require(compare(w,table,rows,q,&selected).index==0);
        q.mass=0.;require(compare(w,table,rows,q,&selected).status==1);
        q.mass=2000.;q.mode=0;require(compare(w,table,rows,q,&selected).index==3072);
        std::vector<Option> downloaded(rows.size());pass(spacepdhcg_gtoc12_collection_options_read(table,downloaded.data(),int(rows.size())));
        require(std::memcmp(downloaded.data(),rows.data(),rows.size()*sizeof(Option))==0);
        std::thread foreign([&]{require(spacepdhcg_gtoc12_collection_options_read(table,downloaded.data(),3073)==SPACEPDHCG_CUDA_INVALID_ARGUMENT);});foreign.join();
        require(spacepdhcg_gtoc12_collection_options_read(table,downloaded.data(),1)==SPACEPDHCG_CUDA_INVALID_ARGUMENT);
        pass(spacepdhcg_gtoc12_collection_options_destroy(&table));require(!table);
        rows.clear();table=copy(rows,stream);require(compare(w,table,rows,q,&selected).index==-1);
        pass(spacepdhcg_gtoc12_collection_options_destroy(&table));
        rows={{NAN,66000.,180.}};table=copy(rows,stream);require(compare(w,table,rows,q,&selected).status==1);
        pass(spacepdhcg_gtoc12_collection_options_destroy(&table));
        std::puts("resident ownership, scratch independence, empty/invalid recovery and selection checks pass");
    }
    pass(spacepdhcg_gtoc12_collection_destroy(&w));check(cudaStreamDestroy(stream));return 0;
}
