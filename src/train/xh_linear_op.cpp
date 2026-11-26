// Wan 2.1 BF16 Linear: write PyTorch outputs directly through the resident SME GEMM.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>

namespace {
using Tensor = at::Tensor;
using Native = int (*)(int,int,int,int,int,int,float,const void*,int,
                       const void*,int,float,void*,int,void*,int);
using Pool = void (*)(int,int,void (*)(int,int,void*),void*);
struct Backend { void* lib; Native gemm; Pool pool; };
const Backend& backend() {
    static Backend impl=[] {
        const char* path=std::getenv("WAN21_TRAIN_GEMM_LIB");
        TORCH_CHECK(path && *path,"set WAN21_TRAIN_GEMM_LIB before loading xhpops_biasfix");
        void* lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);
        TORCH_CHECK(lib,"failed to load SME GEMM library: ",dlerror());
        auto gemm=reinterpret_cast<Native>(dlsym(lib,"bgemm_kernel"));
        auto pool=reinterpret_cast<Pool>(dlsym(lib,"wan21_gemm_pool_run"));
        TORCH_CHECK(gemm && pool,"SME GEMM library missing bgemm_kernel or pool");
        return Backend{lib,gemm,pool};
    }();
    return impl;
}
int numa() {
    const char* value=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(value && *value)return std::atoi(value)%16;
    return 0;
}
int checked_dim(int64_t value){
    TORCH_CHECK(value>0 && value<=std::numeric_limits<int>::max(),"invalid GEMM dimension");
    return static_cast<int>(value);
}
void require_bf16(const Tensor& t) {
    TORCH_CHECK(t.defined() && t.device().is_cpu() && t.scalar_type()==at::kBFloat16 && t.is_contiguous(),
                "xh_linear expects contiguous CPU BF16 tensors");
}
void run_gemm(int trans_a,int trans_b,int m,int n,int k,
              const Tensor& a,int lda,const Tensor& b,int ldb,Tensor& c,int ldc,
              const Tensor* bias=nullptr) {
    auto ptr=backend().gemm;
    const int result=ptr(101,trans_a,trans_b,m,n,k,1.0f,
                         a.data_ptr(),lda,b.data_ptr(),ldb,0.0f,
                         c.data_ptr(),ldc,bias ? bias->data_ptr() : nullptr,bias ? 1 : 0);
    TORCH_CHECK(result==0,"xh_linear SME GEMM error ",result," (",m,",",n,",",k,")");
}
struct SumContext {const at::BFloat16* grad;at::BFloat16* db;int m,n;};
void sum_bias(int rank,int workers,void* opaque) {
    auto& c=*static_cast<SumContext*>(opaque);
    const int begin=c.n*rank/workers,end=c.n*(rank+1)/workers;
    // Read adjacent columns together. Each column still sums rows in original order.
    alignas(64) float sums[256];
    for(int start=begin;start<end;start+=256) {
        const int width=std::min(256,end-start);
        for(int j=0;j<width;++j)sums[j]=0.0f;
        for(int row=0;row<c.m;++row) {
            const at::BFloat16* grad_row=c.grad+static_cast<size_t>(row)*c.n+start;
            for(int j=0;j<width;++j)sums[j]+=static_cast<float>(grad_row[j]);
        }
        for(int j=0;j<width;++j)c.db[start+j]=static_cast<at::BFloat16>(sums[j]);
    }
}
void kp_linear_forward(const Tensor& x,const Tensor& w,
                       const c10::optional<Tensor>& bias,Tensor& y) {
    require_bf16(x);require_bf16(w);require_bf16(y);
    TORCH_CHECK(x.dim()>=2 && w.dim()==2 && y.dim()==x.dim(),"invalid Linear rank");
    const int m=checked_dim(x.numel()/x.size(-1));
    const int k=checked_dim(x.size(-1)),n=checked_dim(w.size(0));
    TORCH_CHECK(w.size(1)==k && y.numel()==static_cast<int64_t>(m)*n && y.size(-1)==n,"invalid Linear size");
    if(bias.has_value()){
        require_bf16(*bias);
        TORCH_CHECK(bias->dim()==1 && bias->numel()==n,"invalid Linear bias");
    }
    run_gemm(111,112,m,n,k,x,k,w,k,y,n,bias.has_value() ? &*bias : nullptr);
}
void kp_linear_backward(const Tensor& dy,const Tensor& x,const Tensor& w,
                        Tensor& dx,Tensor& dw,Tensor& db) {
    require_bf16(dy);require_bf16(x);require_bf16(w);
    require_bf16(dx);require_bf16(dw);require_bf16(db);
    TORCH_CHECK(x.dim()>=2 && w.dim()==2 && dy.numel()==x.numel()/x.size(-1)*w.size(0),"invalid Linear grad");
    TORCH_CHECK(dx.sizes()==x.sizes() && dw.sizes()==w.sizes(),"invalid Linear grad outputs");
    const int m=checked_dim(x.numel()/x.size(-1));
    const int k=checked_dim(x.size(-1)),n=checked_dim(w.size(0));
    TORCH_CHECK(w.size(1)==k && (db.numel()==0 || db.numel()==n),"invalid Linear grad bias");
    // dX = dY @ W; dW = dY^T @ X.  No FP32 output scratch or copy.
    run_gemm(111,111,m,k,n,dy,n,w,k,dx,k);
    run_gemm(112,111,n,k,m,dy,n,x,k,dw,k);
    if(db.numel()){
        SumContext ctx{dy.data_ptr<at::BFloat16>(),db.data_ptr<at::BFloat16>(),m,n};
        backend().pool(36,numa(),sum_bias,&ctx);
    }
}
}  // namespace
TORCH_LIBRARY(xhpops_biasfix,m) {
    m.def("kp_linear_forward(Tensor X, Tensor W, Tensor? b, Tensor(a!) Y) -> ()");
    m.def("kp_linear_backward(Tensor dY, Tensor X, Tensor W, Tensor(a!) dX, Tensor(b!) dW, Tensor(c!) db) -> ()");
}
TORCH_LIBRARY_IMPL(xhpops_biasfix,CPU,m) {
    m.impl("kp_linear_forward",&kp_linear_forward);
    m.impl("kp_linear_backward",&kp_linear_backward);
}
