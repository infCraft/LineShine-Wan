// Wan2.1 token-major BF16 attention forward, tiled around SME FP32 GEMM.
// Prototype for the two real training shapes. Separate from the MPI ring operator.
#include <arm_acle.h>
#include <arm_sve.h>
#include <kblas.h>
#include <omp.h>
#include "wan21_gemm_thread_pool.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <atomic>
#include <ctime>
#include <cstdlib>
#include <sched.h>
#include <limits>
#include <vector>
#include <numa.h>
#include <numaif.h>
#include <cerrno>
#include <new>

namespace ring_attn_component {
void pack_a_unchecked(CBLAS_TRANSPOSE, int, int, int, __bf16*, __bf16*);
void pack_b_unchecked(CBLAS_TRANSPOSE, int, int, int, __bf16*, __bf16*);
void sbgemm_packed_unchecked(int, int, int, __bf16, __bf16*, __bf16*,
                            __bf16, float*, int);
void sbgemm_packed_colscore_unchecked(int,int,int,__bf16*,__bf16*,float*,float*);
}
namespace {
static int rank_numa();
template <class T> struct HbmAllocator {
    using value_type=T;
    HbmAllocator() noexcept=default;
    template <class U> HbmAllocator(const HbmAllocator<U>&) noexcept {}
    T* allocate(std::size_t count) {
        if(count>static_cast<std::size_t>(-1)/sizeof(T)) throw std::bad_alloc();
        void* ptr=numa_alloc_onnode(count*sizeof(T),rank_numa()+16);
        if(!ptr)throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }
    void deallocate(T* ptr,std::size_t count) noexcept {if(ptr)numa_free(ptr,count*sizeof(T));}
    template <class U> bool operator==(const HbmAllocator<U>&) const noexcept {return true;}
    template <class U> bool operator!=(const HbmAllocator<U>&) const noexcept {return false;}
};
constexpr int kQ = 128, kMaxK = 512, kD = 128;
struct KV {
    std::vector<__bf16,HbmAllocator<__bf16>> k, v;
};
struct Worker {
    std::vector<__bf16,HbmAllocator<__bf16>> qpack, ppack, prob;
    std::vector<float,HbmAllocator<float>> score, accum, scoremax;
    Worker() : qpack(kQ*kD), ppack(kQ*kMaxK), prob(kQ*kMaxK),
               score(kQ*kMaxK), accum(kQ*kD), scoremax(kQ) {}
};
static inline svfloat32_t fast_exp_f32(svfloat32_t values) {
    const svbool_t pg=svptrue_b32();
    // svexpa exponent quantized to ln(2)/64, followed by a second-order correction.
    svfloat32_t encoded=svmla_n_f32_x(pg,svdup_n_f32(8128.5f),values,92.33248261689366f);

    svuint32_t exponent=svcvt_u32_f32_x(pg,encoded);
    svfloat32_t delta=svmls_n_f32_x(pg,values,
      svsub_n_f32_x(pg,svcvt_f32_u32_x(pg,exponent),8128.0f),0.010830424696249145f);
    svfloat32_t correction=svadd_n_f32_x(pg,delta,1.0f);
    return svmul_f32_x(pg,svexpa_f32(exponent),correction);
}
static thread_local Worker worker;
static thread_local int attention_pool_tid=0;
// Retain the per-rank packed buffers across calls, but repack every new K/V input.
static std::vector<KV> shared_kv;
static int rank_numa() {
    const char* rank=std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if(rank && *rank) return std::atoi(rank)%16;
    int cpu=sched_getcpu();return cpu<0?0:cpu/38;
}
static void bind_worker(int tid, int numa) {
    const int cpu=numa*38+tid;
    static thread_local int bound=-1;
    if(bound==cpu) return;
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(cpu,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus)) { std::perror("wan21 attention CPU affinity");std::abort(); }
    bound=cpu;
}

static std::atomic<long long> stage_ns[7]{};
static double now_ms() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec*1000.0 + t.tv_nsec*1e-6; }
static void add_time(int stage, double begin) { stage_ns[stage].fetch_add((long long)((now_ms()-begin)*1e6),std::memory_order_relaxed); }

}
extern "C" int wan21_attention_thread_map_q128(int* cpus, int* counts) {
    #pragma omp parallel num_threads(36)
    {
        const int tid=omp_get_thread_num();
        bind_worker(tid,rank_numa());
        cpus[tid]=sched_getcpu();counts[tid]=omp_get_num_threads();
    }
    return 0;
}
extern "C" int wan21_attention_exp_probe_q128(const float* src, float* dst, __bf16* prob) {
    if(svcntw()!=16) return -1;
    const svbool_t pg32=svptrue_b32(),pg16=svptrue_b16();
    svfloat32_t p0=fast_exp_f32(svld1_f32(pg32,src));
    svfloat32_t p1=fast_exp_f32(svld1_f32(pg32,src+16));
    svst1_f32(pg32,dst,p0);svst1_f32(pg32,dst+16,p1);
    svbfloat16_t bf=svcvt_bf16_x(pg32,p0);
    bf=svcvtnt_bf16_x(bf,pg32,p1);
    svbool_t half=svwhilelt_b16((uint64_t)0,(uint64_t)16);
    svst1_bf16(half,prob,svuzp1_bf16(bf,bf));
    svst1_bf16(half,prob+16,svuzp2_bf16(bf,bf));
    return 0;
}
extern "C" int wan21_attention_forward_q128_hbm(const __bf16* q, const __bf16* k,
                                        const __bf16* v, __bf16* out,
                                        float* max_out, float* sum_out,
                                        int tq, int tk, int heads, int d) {
    if (!q || !k || !v || !out || !max_out || !sum_out || d!=kD || heads!=12 ||
        tq!=3328 || (tk!=3328 && tk!=512)) return -1;
    const bool profile=std::getenv("WAN21_ATTN_PHASES")!=nullptr;
    if(profile) for(auto& counter:stage_ns) counter.store(0);
    const double pack_start=now_ms();
    const int base_numa=rank_numa();
    const int key_tile=tk==512?512:256;
    const int nb=tk/key_tile;
    if (shared_kv.size() < static_cast<size_t>(heads*nb)) shared_kv.resize(heads*nb);
    const float scale=1.0f/std::sqrt((float)d);
    const char* spread_text=std::getenv("WAN21_ATTN_HEAD_SPREAD");
    const int head_spread=spread_text ? std::atoi(spread_text) : 1;
    if(head_spread<1 || heads%head_spread!=0)return -2;
    auto run_tile=[&](int tile) { 
        bind_worker(attention_pool_tid,base_numa);
        Worker& w=worker;
        const int h=tile/(tq/kQ), row=(tile%(tq/kQ))*kQ;
        double start=profile?now_ms():0;
        ring_attn_component::pack_a_unchecked(CblasNoTrans,kQ,kD,heads*kD,
            const_cast<__bf16*>(q)+(size_t)row*heads*kD+h*kD,w.qpack.data());
        if(profile) add_time(1,start);
        float mx[kQ], denom[kQ];
        std::fill_n(mx,kQ,-std::numeric_limits<float>::infinity());
        std::fill_n(denom,kQ,0.0f);
        // First PV GEMM has beta=0 and overwrites the complete accumulator.
        for(int bk=0;bk<nb;++bk) {
            if(profile) start=now_ms();
            KV& p=shared_kv[h*nb+bk];
            std::fill(w.scoremax.begin(),w.scoremax.end(),-std::numeric_limits<float>::infinity());
            ring_attn_component::sbgemm_packed_colscore_unchecked(
                kQ,key_tile,kD,w.qpack.data(),p.k.data(),w.score.data(),w.scoremax.data());
            if(profile) { add_time(2,start); start=now_ms(); }
            // Determine each row's online softmax offset before writing the
            // next A panel directly in the packed layout consumed by PV.
            for(int r=0;r<kQ;++r) {
                const float mnew=std::max(mx[r],w.scoremax[r]*scale);
                if(mnew>mx[r] && std::isfinite(mx[r])) {
                    const float correction=std::exp(mx[r]-mnew);
                    for(int x=0;x<kD;++x)w.accum[r*kD+x]*=correction;
                    denom[r]*=correction;
                }
                mx[r]=mnew;
            }
            if(profile) {add_time(3,start);start=now_ms();}
            const svbool_t pg32=svptrue_b32(),pg16=svptrue_b16();
            for(int group=0;group<kQ;group+=16){
                const float* scores=w.score.data()+(size_t)group*key_tile;
                const svfloat32_t offsets=svld1_f32(pg32,mx+group);
                const svfloat32_t negative_offsets=svneg_f32_x(pg32,offsets);
                svfloat32_t rowsums=svdup_n_f32(0.0f);
                for(int col=0;col<key_tile;col+=8){
                    const svfloat32_t s0=svld1_f32(pg32,scores+(size_t)(col+0)*16);
                    const svfloat32_t s1=svld1_f32(pg32,scores+(size_t)(col+1)*16);
                    const svfloat32_t s2=svld1_f32(pg32,scores+(size_t)(col+2)*16);
                    const svfloat32_t s3=svld1_f32(pg32,scores+(size_t)(col+3)*16);
                    const svfloat32_t s4=svld1_f32(pg32,scores+(size_t)(col+4)*16);
                    const svfloat32_t s5=svld1_f32(pg32,scores+(size_t)(col+5)*16);
                    const svfloat32_t s6=svld1_f32(pg32,scores+(size_t)(col+6)*16);
                    const svfloat32_t s7=svld1_f32(pg32,scores+(size_t)(col+7)*16);
                    const svfloat32_t p0=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s0,scale));
                    const svfloat32_t p1=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s1,scale));
                    const svfloat32_t p2=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s2,scale));
                    const svfloat32_t p3=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s3,scale));
                    const svfloat32_t p4=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s4,scale));
                    const svfloat32_t p5=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s5,scale));
                    const svfloat32_t p6=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s6,scale));
                    const svfloat32_t p7=fast_exp_f32(svmla_n_f32_x(pg32,negative_offsets,s7,scale));
                    rowsums=svadd_f32_x(pg32,rowsums,svadd_f32_x(pg32,p0,p1));
                    svbfloat16_t packed0=svcvt_bf16_x(pg32,p0);
                    packed0=svcvtnt_bf16_x(packed0,pg32,p1);
                    svst1_bf16(pg16,w.ppack.data()+(size_t)group*key_tile+(size_t)(col+0)*16,packed0);
                    rowsums=svadd_f32_x(pg32,rowsums,svadd_f32_x(pg32,p2,p3));
                    svbfloat16_t packed2=svcvt_bf16_x(pg32,p2);
                    packed2=svcvtnt_bf16_x(packed2,pg32,p3);
                    svst1_bf16(pg16,w.ppack.data()+(size_t)group*key_tile+(size_t)(col+2)*16,packed2);
                    rowsums=svadd_f32_x(pg32,rowsums,svadd_f32_x(pg32,p4,p5));
                    svbfloat16_t packed4=svcvt_bf16_x(pg32,p4);
                    packed4=svcvtnt_bf16_x(packed4,pg32,p5);
                    svst1_bf16(pg16,w.ppack.data()+(size_t)group*key_tile+(size_t)(col+4)*16,packed4);
                    rowsums=svadd_f32_x(pg32,rowsums,svadd_f32_x(pg32,p6,p7));
                    svbfloat16_t packed6=svcvt_bf16_x(pg32,p6);
                    packed6=svcvtnt_bf16_x(packed6,pg32,p7);
                    svst1_bf16(pg16,w.ppack.data()+(size_t)group*key_tile+(size_t)(col+6)*16,packed6);
                }
                svst1_f32(pg32,denom+group,svadd_f32_x(pg32,svld1_f32(pg32,denom+group),rowsums));
            }
            if(profile) {add_time(4,start);start=now_ms();}
            // The running output has already been rescaled when its row max changed.
            // Accumulate P @ V directly and avoid the FP32 partial buffer and copy.
            ring_attn_component::sbgemm_packed_unchecked(kQ,kD,key_tile,
                 (__bf16)1,w.ppack.data(),p.v.data(),(__bf16)(bk==0 ? 0 : 1),
                 w.accum.data(),kD);
            if(profile) { add_time(5,start); add_time(6,now_ms()); }
        }
        for(int r=0;r<kQ;++r) {
            max_out[(size_t)h*tq+row+r]=mx[r];
            sum_out[(size_t)h*tq+row+r]=denom[r];
            const float inv_denom=1.0f/denom[r];
            const svbool_t fp32=svptrue_b32();
            const svbool_t half=svwhilelt_b16((uint64_t)0,(uint64_t)16);
            const float* src=w.accum.data()+r*kD;
            __bf16* dst=out+((size_t)(row+r)*heads+h)*kD;
            for(int x=0;x<kD;x+=32){
                svfloat32_t v0=svmul_n_f32_x(fp32,svld1_f32(fp32,src+x),inv_denom);
                svfloat32_t v1=svmul_n_f32_x(fp32,svld1_f32(fp32,src+x+16),inv_denom);
                svbfloat16_t packed=svcvt_bf16_x(fp32,v0);
                packed=svcvtnt_bf16_x(packed,fp32,v1);
                svst1_bf16(half,dst+x,svuzp1_bf16(packed,packed));
                svst1_bf16(half,dst+x+16,svuzp2_bf16(packed,packed));
            }
        }
     };
    std::atomic<int> ready[12];
    for(auto& flag:ready)flag.store(0,std::memory_order_relaxed);
    std::atomic<int> next_tile{0};
    std::atomic<int> next_pack{0};
    wan21_gemm_pool::run(36,base_numa,[&](int tid,int workers){
      attention_pool_tid=tid;
      bind_worker(tid,base_numa);
      while(true){
          const int unit=next_pack.fetch_add(1,std::memory_order_relaxed);
          if(unit>=heads*5)break;
          const int h=unit/5,part=unit%5;
          for(int j=part;j<nb;j+=5) {
              const int block=h*nb+j,pos=j*key_tile;
              KV& p=shared_kv[block];
              p.k.resize(kD*key_tile);p.v.resize(key_tile*kD);
              ring_attn_component::pack_b_unchecked(CblasTrans,key_tile,kD,heads*kD,
                  const_cast<__bf16*>(k)+(size_t)pos*heads*kD+h*kD,p.k.data());
              ring_attn_component::pack_b_unchecked(CblasNoTrans,kD,key_tile,heads*kD,
                  const_cast<__bf16*>(v)+(size_t)pos*heads*kD+h*kD,p.v.data());
          }
          ready[h].fetch_add(1,std::memory_order_release);
          if(profile) {
              long long observed=stage_ns[0].load(std::memory_order_relaxed);
              const long long elapsed=(long long)((now_ms()-pack_start)*1e6);
              while(observed<elapsed && !stage_ns[0].compare_exchange_weak(observed,elapsed,
                        std::memory_order_relaxed)){}
          }
      }
      const int groups=tq/kQ;
      while(true){
          const int task=next_tile.fetch_add(1,std::memory_order_relaxed);
          if(task>=heads*groups)break;
          // Group adjacent heads so several workers share each head's K/V,
          // while avoiding contention across all 12 heads at once.
          const int batch=task/(head_spread*groups);
          const int offset=task%(head_spread*groups);
          const int h=batch*head_spread+offset%head_spread;
          const int group=offset/head_spread;
          while(ready[h].load(std::memory_order_acquire)<5)
              asm volatile("yield" ::: "memory");
          run_tile(h*groups+group);
      }
    });

    if(profile) {
        std::fprintf(stderr,"ATTN_PHASE tk=%d pack_wall=%.3f qpack_thread=%.3f score_thread=%.3f softmax_thread=%.3f probpack_thread=%.3f pv_thread=%.3f accum_thread=%.3f\n",tk,
            stage_ns[0].load()*1e-6, stage_ns[1].load()*1e-6,
            stage_ns[2].load()*1e-6, stage_ns[3].load()*1e-6,
            stage_ns[4].load()*1e-6, stage_ns[5].load()*1e-6,stage_ns[6].load()*1e-6);
    }
    return 0;
}


extern "C" int wan21_attention_hbm_nodes_q128(int* nodes) {
    if(!nodes)return -1;
    wan21_gemm_pool::run(36,rank_numa(),[&](int tid,int workers){
        Worker& w=worker;
        int node=-1;
        if(w.score.empty())nodes[tid]=-100;
        else if(get_mempolicy(&node,nullptr,0,w.score.data(),MPOL_F_ADDR|MPOL_F_NODE)!=0)nodes[tid]=-errno;
        else nodes[tid]=node;
    });
    return 0;
}