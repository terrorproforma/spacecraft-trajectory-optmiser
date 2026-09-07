// Isolated diagnostic only: synchronous snapshots, never used for performance.
#include <unistd.h>
template<class T> static void dump_device(FILE* f, const T* data, size_t n) {
    std::vector<T> host(n);
    if (n) CUDA_CHECK(cudaMemcpy(host.data(), data, n*sizeof(T), cudaMemcpyDeviceToHost));
    if (fwrite(host.data(),sizeof(T),n,f)!=n) { perror("snapshot write"); exit(1); }
}
static void dump_linear(LinSysData* s,QOCOWorkspace* w,const double* b,const double* x,int index,int stage) {
    const char* root=getenv("SPACEPDHCG_DIAGNOSTIC_LINEAR_DIR"); if(!root) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    compute_linsys_residual(s,w,b,x,s->d_xyz_matrix_data,false);
    CUDA_CHECK(cudaDeviceSynchronize());
    int nnz=0; CUDA_CHECK(cudaMemcpy(&nnz,s->d_csr_rows+s->Kn,sizeof(int),cudaMemcpyDeviceToHost));
    char path[4096]; snprintf(path,sizeof(path),"%s/linear-%d-%04d-%d.bin",root,getpid(),index,stage);
    FILE* f=fopen(path,"wb"); if(!f){perror(path);exit(1);}
    int header[]={s->Kn,nnz,w->data->n,w->data->p,w->data->m,w->data->l,w->data->nsoc,w->nt_scaling_nnz};
    fwrite(header,sizeof(int),8,f);
    double reg[]={s->kkt_static_reg_P,s->kkt_static_reg_A,s->kkt_static_reg_G}; fwrite(reg,sizeof(double),3,f);
    dump_device(f,s->d_csr_rows,s->Kn+1);dump_device(f,s->d_csr_columns,nnz);dump_device(f,s->d_csr_val,nnz);
    dump_device(f,b,s->Kn);dump_device(f,x,s->Kn);dump_device(f,s->d_xyz_matrix_data,s->Kn);
    dump_device(f,get_data_vectorf(w->nt_scaling),w->nt_scaling_nnz);
    dump_device(f,get_data_vectori(w->nt_scaling_soc_idx),w->data->nsoc);
    dump_device(f,get_data_vectori(w->soc_idx),w->data->nsoc);
    dump_device(f,get_data_vectori(w->data->q),w->data->nsoc);
    fclose(f);
}
