// Tiled BF16 attention backward for Wan2.1 token-major Q/K/V.
// Persistent 36-worker NUMA-local dispatch; keep tile-sized workspace across calls.
// Build with wan21_gemm_thread_pool_global.cpp and wan21_attention_sme_component_pair_interleaved.cpp.
// Score/dP use the two independent SME kernels; gradients use NUMA-local SVE reduction.
#include <arm_acle.h>
#include <arm_sve.h>
#include <kblas.h>
#include <omp.h>
#include "wan21_gemm_thread_pool.h"
#include <sched.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <new>
#include <numa.h>
#include <numaif.h>
#include <cerrno>
#include <atomic>
#include <ctime>
#include <complex>

namespace ring_attn_component {
void pack_a_unchecked(CBLAS_TRANSPOSE,int,int,int,__bf16*,__bf16*);
void pack_ab_unchecked(int,int,int,int,int,__bf16*,__bf16*,__bf16*) __arm_streaming;
void pack_ab_delta_unchecked(int,int,int,int,int,__bf16*,__bf16*,__bf16*,const __bf16*,float*) __arm_streaming;
void pack_b_unchecked(CBLAS_TRANSPOSE,int,int,int,__bf16*,__bf16*);
void sbgemm_packed_unchecked(int,int,int,__bf16,__bf16*,__bf16*,__bf16,float*,int);
void sbgemm_packed_pair_unchecked(int,int,int,__bf16*,__bf16*,float*,__bf16*,__bf16*,float*,int) __arm_streaming;
void sbgemm_packed_pair_interleaved_unchecked(int,int,int,__bf16*,__bf16*,float*,__bf16*,__bf16*,float*,int) __arm_streaming;
}
namespace {
static int local_numa();
template<class T> struct HbmAllocator {
    using value_type=T;
    HbmAllocator() noexcept=default;
    template<class U> HbmAllocator(const HbmAllocator<U>&) noexcept {}
    T* allocate(std::size_t count) {
        if (count>static_cast<std::size_t>(-1)/sizeof(T))throw std::bad_alloc();
        void* ptr=numa_alloc_onnode(count*sizeof(T),local_numa()+16);
        if(!ptr)throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr,std::size_t count) noexcept {
        if(ptr) numa_free(ptr,count*sizeof(T));
    }
    template<class U> bool operator==(const HbmAllocator<U>&) const noexcept {return true;}
    template<class U> bool operator!=(const HbmAllocator<U>&) const noexcept {return false;}
};
constexpr int TQ=32,D=128,MAX_KEY=512;
struct KVPack {std::vector<__bf16,HbmAllocator<__bf16>> kt,vt,kn;};
static std::vector<KVPack> kv_cache;
static std::atomic<long long> time_ns[8]{};
static double clock_ms(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec*1e-6;}
static void add_stage(int stage,double before){time_ns[stage].fetch_add((long long)((clock_ms()-before)*1e6),std::memory_order_relaxed);}

struct Worker {
 std::vector<__bf16,HbmAllocator<__bf16>> ap;
 std::vector<float,HbmAllocator<float>> score,dp,dk,dv,dqfull,delta;
 std::vector<__bf16,HbmAllocator<__bf16>> pfull,bpqfull,bpgofull,packed_q,packed_go,packed_p,packed_ds;
 std::vector<float,HbmAllocator<float>> invden;
 Worker():ap(TQ*MAX_KEY),score(TQ*MAX_KEY),dp(TQ*MAX_KEY) {}
};
static thread_local Worker worker;
static int local_numa(){
 const char* x=getenv("OMPI_COMM_WORLD_LOCAL_RANK");
 if(x && *x) return atoi(x)%16;
 int cpu=sched_getcpu();return cpu<0?0:cpu/38;
}
static void bind_cpu(int tid,int numa){
 static thread_local int last=-1;
 int cpu=numa*38+tid;
 if(last==cpu)return;
 cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
 if(sched_setaffinity(0,sizeof(set),&set)){perror("wan21 attention backward affinity");abort();}
 last=cpu;
}
static inline svfloat32_t fast_exp_scaled(svfloat32_t values, float inv) {
    const svbool_t pg=svptrue_b32();
    // svexpa exponent quantized to ln(2)/64, followed by a second-order correction.
    svfloat32_t encoded=svmla_n_f32_x(pg,svdup_n_f32(8128.5f),values,92.33248261689366f);

    svuint32_t exponent=svcvt_u32_f32_x(pg,encoded);
    svfloat32_t delta=svmls_n_f32_x(pg,values,
      svsub_n_f32_x(pg,svcvt_f32_u32_x(pg,exponent),8128.0f),0.010830424696249145f);
    svfloat32_t correction=svmla_n_f32_x(pg,svdup_n_f32(inv),delta,inv);
    return svmul_f32_x(pg,svexpa_f32(exponent),correction);
}
static inline void pack_bf16_pair(svfloat32_t x0,svfloat32_t x1,__bf16* dst){
 const svbool_t pg32=svptrue_b32(),half=svwhilelt_b16((uint64_t)0,(uint64_t)16);
 svbfloat16_t packed=svcvt_bf16_x(pg32,x0);
 packed=svcvtnt_bf16_x(packed,pg32,x1);
 svst1_bf16(half,dst,svuzp1_bf16(packed,packed));
 svst1_bf16(half,dst+16,svuzp2_bf16(packed,packed));
}
}
extern "C" int wan21_attention_hbm_worker_node() {
    Worker& w=worker;
    if(w.score.empty())return -100;
    int node=-1;
    if(get_mempolicy(&node,nullptr,0,w.score.data(),MPOL_F_ADDR|MPOL_F_NODE)!=0)
        return -errno;
    return node;
}
// Preserve the old BF16 dQ/dK rounding before the FP64 complex inverse rotation.
// Each head writes its own final token-major gradient without a second large buffer.
static inline void inverse_rope_store(const __bf16* src, __bf16* dst,
    const std::complex<double>* freq, int token, int frames, int height, int width) {
    constexpr int C=64, A=22, B=21;
    const int hw=height*width;
    if(token >= frames*hw){std::copy(src,src+128,dst);return;}
    const int f=token/hw, h=(token%hw)/width, w=token%width;
    for(int pair=0;pair<C;++pair){
        const std::complex<double>& c = pair<A ? freq[f*C+pair] :
            (pair<A+B ? freq[h*C+pair] : freq[w*C+pair]);
        const double real=c.real(), imag=c.imag();
        const double x=static_cast<float>(src[2*pair]), y=static_cast<float>(src[2*pair+1]);
        dst[2*pair]=static_cast<__bf16>(static_cast<float>(x*real+y*imag));
        dst[2*pair+1]=static_cast<__bf16>(static_cast<float>(y*real-x*imag));
    }
}
// Accumulate one head's query chunk before dK/dV GEMM: long K instead of 32.
static int backward_impl(
 const __bf16* q,const __bf16* k,const __bf16* v,const __bf16* o,
 const __bf16* go,const float* mx,const float* denom,
 __bf16* dq,__bf16* dk,__bf16* dv,
 int tq,int tk,int heads,int dim,
 const std::complex<double>* freq,int frames,int height,int width){
 if(freq && (frames!=13 || height!=16 || width!=16))return -2;
 if(!q||!k||!v||!o||!go||!mx||!denom||!dq||!dk||!dv||
    tq!=3328||(tk!=3328&&tk!=512)||heads!=12||dim!=D)return -1;
 const bool profile=getenv("WAN21_ATTN_BATCHED_PHASES")!=nullptr;
 if(profile)for(auto& t:time_ns)t.store(0);
 double phase_start=profile?clock_ms():0;
 const int numa=local_numa(),key_tile=512,nb=(tk+key_tile-1)/key_tile;
 const float scale=1.0f/std::sqrt((float)D);
 if (kv_cache.size() < static_cast<size_t>(heads*nb)) kv_cache.resize(heads*nb);
 static std::vector<Worker*> slots;
 slots.resize(36);
  std::atomic<int> head_done[12];
  for(auto& x:head_done)x.store(0,std::memory_order_relaxed);
  std::atomic<int> pack_done[12];
  for(auto& x:pack_done)x.store(0,std::memory_order_relaxed);
 wan21_gemm_pool::run(36,numa,[&](int tid,int workers){
 const int own_head=tid/3,own_part=tid%3;
 for(int j=own_part;j<nb;j+=3){
   bind_cpu(tid,numa);
   const int h=own_head,index=h*nb+j;
   const int active_tile=std::min(key_tile,tk-j*key_tile);
   KVPack& p=kv_cache[index];
   p.kt.resize((size_t)active_tile*D);p.vt.resize((size_t)active_tile*D);
   p.kn.resize((size_t)active_tile*D);
   __bf16* kr=const_cast<__bf16*>(k)+((size_t)j*key_tile*heads+h)*D;
   __bf16* vr=const_cast<__bf16*>(v)+((size_t)j*key_tile*heads+h)*D;
   ring_attn_component::pack_b_unchecked(CblasTrans,active_tile,D,heads*D,kr,p.kt.data());
   ring_attn_component::pack_b_unchecked(CblasTrans,active_tile,D,heads*D,vr,p.vt.data());
   ring_attn_component::pack_b_unchecked(CblasNoTrans,D,active_tile,heads*D,kr,p.kn.data());
 }
  // Only these three workers consume K/V panels of their own head.
  pack_done[own_head].fetch_add(1,std::memory_order_release);
  while(pack_done[own_head].load(std::memory_order_acquire)<3)
    asm volatile("yield" ::: "memory");
  // This profiler field tracks head 0 readiness, not global K/V pack wall.
  if(profile && tid==0)add_stage(0,phase_start);
  const int h=tid/3,part=tid%3;
  double before=profile?clock_ms():0;
  bind_cpu(tid,numa);
  Worker& w=worker;
  const int tile_begin=part*(tq/TQ)/3,tile_end=(part+1)*(tq/TQ)/3;
  const int query_start=tile_begin*TQ,query_len=(tile_end-tile_begin)*TQ;
  w.dk.resize((size_t)tk*D);w.dv.resize((size_t)tk*D);
  w.packed_p.resize((size_t)query_len*key_tile);
  w.packed_ds.resize((size_t)query_len*key_tile);
  w.ap.resize((size_t)TQ*key_tile);
  w.bpqfull.resize((size_t)query_len*D);
  w.bpgofull.resize((size_t)query_len*D);
  w.dqfull.resize((size_t)query_len*D);
  w.delta.resize(query_len);
  w.invden.resize(query_len);
  w.packed_q.resize((size_t)query_len*D);
  w.packed_go.resize((size_t)query_len*D);
  slots[tid]=&w;
  __bf16* qstart=const_cast<__bf16*>(q)+((size_t)query_start*heads+h)*D;
  __bf16* gostart=const_cast<__bf16*>(go)+((size_t)query_start*heads+h)*D;
  float* delta=w.delta.data();
  for(int rel=0;rel<query_len;++rel)
    w.invden[rel]=1.0f/denom[(size_t)h*tq+query_start+rel];
  for(int off=0;off<query_len;off+=TQ){
    ring_attn_component::pack_ab_unchecked(TQ,D,heads*D,query_len,off,qstart+(size_t)off*heads*D,w.packed_q.data()+(size_t)off*D,w.bpqfull.data());
    ring_attn_component::pack_ab_delta_unchecked(TQ,D,heads*D,query_len,off,gostart+(size_t)off*heads*D,w.packed_go.data()+(size_t)off*D,w.bpgofull.data(),o+((size_t)(query_start+off)*heads+h)*D,delta);
  }
  if(profile)add_stage(1,before);
  const svbool_t pg=svptrue_b32();
  // Scratch vectors are sized before this block and retain their storage for the entire call.
  float* __restrict__ score_ptr=w.score.data();
  float* __restrict__ dp_ptr=w.dp.data();
  float* __restrict__ invden_ptr=w.invden.data();
  float* __restrict__ dq_ptr=w.dqfull.data();
  __bf16* __restrict__ ap_ptr=w.ap.data();
  __bf16* __restrict__ packed_p_ptr=w.packed_p.data();
  __bf16* __restrict__ packed_ds_ptr=w.packed_ds.data();
  __bf16* __restrict__ packed_q_ptr=w.packed_q.data();
  __bf16* __restrict__ packed_go_ptr=w.packed_go.data();
  __bf16* __restrict__ bpq_ptr=w.bpqfull.data();
  __bf16* __restrict__ bpgo_ptr=w.bpgofull.data();

  for(int turn=0;turn<nb;++turn){
    const int j=(turn+(h+part*3)%nb)%nb;
    if(profile)before=clock_ms();
    const int active_tile=std::min(key_tile,tk-j*key_tile);
    KVPack& packed=kv_cache[h*nb+j];
    for(int off=0;off<query_len;off+=TQ){
      const svbool_t full_bf16=svptrue_b16();
      alignas(64) static const uint16_t shuffle[32]={0,2,1,3,4,6,5,7,8,10,9,11,12,14,13,15,16,18,17,19,20,22,21,23,24,26,25,27,28,30,29,31};
      // SME score/dP microkernel writes 64 columns per call; every group and tail must be a multiple of 64.
      for(int colbase=0;colbase<active_tile;colbase+=128){
      const int group_cols=std::min(128,active_tile-colbase);
      ring_attn_component::sbgemm_packed_pair_unchecked(TQ,group_cols,D,
         packed_q_ptr+(size_t)off*D,packed.kt.data()+(size_t)colbase*D,score_ptr,
         packed_go_ptr+(size_t)off*D,packed.vt.data()+(size_t)colbase*D,dp_ptr,group_cols);
      if(profile){add_stage(2,before);before=clock_ms();}
      for(int r=0;r<TQ;r+=2){
        const int row0=r*group_cols,row1=(r+1)*group_cols;
        const float mr0=mx[(size_t)h*tq+query_start+off+r];
        const float mr1=mx[(size_t)h*tq+query_start+off+r+1];
        const float inv0=invden_ptr[off+r],inv1=invden_ptr[off+r+1];
        const svfloat32_t d0=svdup_n_f32(-delta[off+r]*scale);
        const svfloat32_t d1=svdup_n_f32(-delta[off+r+1]*scale);
        for(int col=colbase;col<colbase+group_cols;col+=16){
          const svfloat32_t s0=svld1_f32(pg,score_ptr+row0+(col-colbase));
          const svfloat32_t s1=svld1_f32(pg,score_ptr+row1+(col-colbase));
          const svfloat32_t p0=fast_exp_scaled(svsub_n_f32_x(pg,svmul_n_f32_x(pg,s0,scale),mr0),inv0);
          const svfloat32_t p1=fast_exp_scaled(svsub_n_f32_x(pg,svmul_n_f32_x(pg,s1,scale),mr1),inv1);
          const size_t packed_pos=(size_t)(col/16)*16*query_len+(size_t)((off+r)/2)*32;
          svbfloat16_t pair=svcvt_bf16_x(pg,p0);
          pair=svcvtnt_bf16_x(pair,pg,p1);
          svst1_bf16(full_bf16,packed_p_ptr+packed_pos,pair);
          const svfloat32_t ds0=svmul_f32_x(pg,p0,
             svmla_n_f32_x(pg,d0,svld1_f32(pg,dp_ptr+row0+(col-colbase)),scale));
          const svfloat32_t ds1=svmul_f32_x(pg,p1,
             svmla_n_f32_x(pg,d1,svld1_f32(pg,dp_ptr+row1+(col-colbase)),scale));
          pair=svcvt_bf16_x(pg,ds0);
          pair=svcvtnt_bf16_x(pair,pg,ds1);
          svst1_bf16(full_bf16,packed_ds_ptr+packed_pos,pair);
          const svuint16_t ordered=svtbl_u16(svreinterpret_u16_bf16(pair),svld1_u16(full_bf16,shuffle));
          const svuint64_t indices=svindex_u64((uint64_t)((r%16)/2),8);
          uint64_t* dest=reinterpret_cast<uint64_t*>(ap_ptr+(size_t)(r/16)*16*active_tile+(size_t)col*16);
          svst1_scatter_u64index_u64(svwhilelt_b64((uint64_t)0,(uint64_t)8),dest,indices,svreinterpret_u64_u16(ordered));
        }
      }
      if(profile){add_stage(3,before);before=clock_ms();}
      }
      ring_attn_component::sbgemm_packed_unchecked(TQ,D,active_tile,(__bf16)1,
        ap_ptr,packed.kn.data(),(__bf16)(turn==0 ? 0 : 1),dq_ptr+(size_t)off*D,D);
      if(profile){add_stage(4,before);before=clock_ms();}
    }
    if(profile)before=clock_ms();
    ring_attn_component::sbgemm_packed_unchecked(active_tile,D,query_len,(__bf16)1,
       packed_p_ptr,bpgo_ptr,(__bf16)0,w.dv.data()+(size_t)j*key_tile*D,D);
    if(profile){add_stage(5,before);before=clock_ms();}
    ring_attn_component::sbgemm_packed_unchecked(active_tile,D,query_len,(__bf16)1,
       packed_ds_ptr,bpq_ptr,(__bf16)0,w.dk.data()+(size_t)j*key_tile*D,D);
    if(profile)add_stage(6,before);
  }
  for(int rel=0;rel<query_len;++rel){
    __bf16* dest=dq+((size_t)(query_start+rel)*heads+h)*D;
    if(!freq){
      for(int x=0;x<D;++x)dest[x]=(__bf16)w.dqfull[(size_t)rel*D+x];
    }else{
      alignas(64) __bf16 rounded[D];
      for(int x=0;x<D;++x)rounded[x]=(__bf16)w.dqfull[(size_t)rel*D+x];
      inverse_rope_store(rounded,dest,freq,query_start+rel,frames,height,width);
    }
  }
  head_done[h].fetch_add(1,std::memory_order_release);
  while(head_done[h].load(std::memory_order_acquire)<3)
    asm volatile("yield" ::: "memory");
  if(profile && tid==0)before=clock_ms();
  // Each worker reduces a contiguous third of its own head, matching the producer slots.
  for(int t=part*tk/3;t<(part+1)*tk/3;++t){
     const int index=t*heads+h;
     const size_t key=(size_t)t*D;
     const Worker* a=slots[h*3+0];
     const Worker* b=slots[h*3+1];
     const Worker* c=slots[h*3+2];
     const float *ak=a->dk.data()+key,*bk=b->dk.data()+key,*ck=c->dk.data()+key;
     const float *av=a->dv.data()+key,*bv=b->dv.data()+key,*cv=c->dv.data()+key;
     __bf16* const ok=dk+(size_t)index*D;
     __bf16* const ov=dv+(size_t)index*D;
     const svbool_t pg=svptrue_b32();
     const svbool_t half=svwhilelt_b16((uint64_t)0,(uint64_t)16);
     alignas(64) __bf16 rounded[D];
     __bf16* const out_k=freq?rounded:ok;
     for(int x=0;x<D;x+=32){
       svfloat32_t k0=svadd_f32_x(pg,svadd_f32_x(pg,svld1_f32(pg,ak+x),svld1_f32(pg,bk+x)),svld1_f32(pg,ck+x));
       svfloat32_t k1=svadd_f32_x(pg,svadd_f32_x(pg,svld1_f32(pg,ak+x+16),svld1_f32(pg,bk+x+16)),svld1_f32(pg,ck+x+16));
       svfloat32_t v0=svadd_f32_x(pg,svadd_f32_x(pg,svld1_f32(pg,av+x),svld1_f32(pg,bv+x)),svld1_f32(pg,cv+x));
       svfloat32_t v1=svadd_f32_x(pg,svadd_f32_x(pg,svld1_f32(pg,av+x+16),svld1_f32(pg,bv+x+16)),svld1_f32(pg,cv+x+16));
       svbfloat16_t kk=svcvt_bf16_x(pg,k0);
       kk=svcvtnt_bf16_x(kk,pg,k1);
       svst1_bf16(half,out_k+x,svuzp1_bf16(kk,kk));
       svst1_bf16(half,out_k+x+16,svuzp2_bf16(kk,kk));
       svbfloat16_t vv=svcvt_bf16_x(pg,v0);
       vv=svcvtnt_bf16_x(vv,pg,v1);
       svst1_bf16(half,ov+x,svuzp1_bf16(vv,vv));
       svst1_bf16(half,ov+x+16,svuzp2_bf16(vv,vv));
     }
     if(freq)inverse_rope_store(rounded,ok,freq,t,frames,height,width);
  }
  // run() waits for every worker callback before returning.
  // For this variant stage 7 is only thread 0's head-local reduction time.
  if(profile && tid==0)add_stage(7,before);
 });
 if(profile)std::fprintf(stderr,"ATTN_BATCHED_PHASE tk=%d kvpack_wall=%.3f prep_thread=%.3f score_dp_thread=%.3f softmax_thread=%.3f dq_thread=%.3f dv_thread=%.3f dk_thread=%.3f reduction_wall=%.3f\n",tk,time_ns[0].load()*1e-6,time_ns[1].load()*1e-6,time_ns[2].load()*1e-6,time_ns[3].load()*1e-6,time_ns[4].load()*1e-6,time_ns[5].load()*1e-6,time_ns[6].load()*1e-6,time_ns[7].load()*1e-6);
 return 0;
}


extern "C" int wan21_attention_backward_batched_q32_hbm_prepack(
 const __bf16* q,const __bf16* k,const __bf16* v,const __bf16* o,
 const __bf16* go,const float* mx,const float* denom,
 __bf16* dq,__bf16* dk,__bf16* dv,int tq,int tk,int heads,int dim){
 return backward_impl(q,k,v,o,go,mx,denom,dq,dk,dv,tq,tk,heads,dim,nullptr,0,0,0);
}
extern "C" int wan21_attention_backward_rope_fused(
 const __bf16* q,const __bf16* k,const __bf16* v,const __bf16* o,
 const __bf16* go,const float* mx,const float* denom,
 __bf16* dq,__bf16* dk,__bf16* dv,const std::complex<double>* freq,
 int frames,int height,int width,int tq,int tk,int heads,int dim){
 if(!freq || tk!=3328)return -3;
 return backward_impl(q,k,v,o,go,mx,denom,dq,dk,dv,tq,tk,heads,dim,freq,frames,height,width);
}
