#include <arm_sve.h>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "wan21_gemm_thread_pool.h"

static inline svfloat32_t low(svuint16_t v) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(svptrue_b32(), svunpklo_u32(v), 16));
}
static inline svfloat32_t high(svuint16_t v) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(svptrue_b32(), svunpkhi_u32(v), 16));
}
static inline svuint32_t round_bf16(svfloat32_t x) {
    auto pg=svptrue_b32();
    auto v=svreinterpret_u32_f32(x);
    auto bias=svadd_n_u32_x(pg,svand_n_u32_x(pg,svlsr_n_u32_x(pg,v,16),1),0x7fff);
    return svlsr_n_u32_x(pg,svadd_u32_x(pg,v,bias),16);
}
static inline svfloat32_t round_fp(svfloat32_t x) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(svptrue_b32(),round_bf16(x),16));
}
static inline void store(uint16_t* ptr, svfloat32_t lo,svfloat32_t hi) {
    svst1_u16(svptrue_b16(),ptr,svuzp1_u16(svreinterpret_u16_u32(round_bf16(lo)),
                                                       svreinterpret_u16_u32(round_bf16(hi))));
}
static inline uint16_t scalar_bf16(float x) {
    uint32_t b;
    __builtin_memcpy(&b,&x,4);
    return uint16_t((b + 0x7fffU + ((b >> 16)&1)) >> 16);
}
extern "C" int wan21_rmsnorm_bf16_forward(const void* in, const void* weight,
        void* output, void* rstd, int rows, int cols, float eps) {
    if(!in || !weight || !output || !rstd || (rows!=3328 && rows!=512) || cols!=1536 || eps<=0) return -1;
    const auto* x=(const uint16_t*)in;
    const auto* w=(const uint16_t*)weight;
    auto* y=(uint16_t*)output;
    auto* invs=(float*)rstd;
    const int lanes=svcnth();
    if(cols % lanes) return -2;
    auto pg=svptrue_b32();
    wan21_gemm_pool::run(36,wan21_gemm_pool::base_numa(),[&](int rank,int count){
        const int begin=rows*rank/count, end=rows*(rank+1)/count;
        for(int row=begin;row<end;++row){
            auto sqlo=svdup_n_f32(0),sqhi=svdup_n_f32(0);
            for(int i=0;i<cols;i+=lanes){
                auto packed=svld1_u16(svptrue_b16(),x+row*cols+i);
                auto lo=low(packed),hi=high(packed);
                sqlo=svmla_f32_x(pg,sqlo,lo,lo);
                sqhi=svmla_f32_x(pg,sqhi,hi,hi);
            }
            float inv=1.0f/std::sqrt((svaddv_f32(pg,sqlo)+svaddv_f32(pg,sqhi))/cols+eps);
            invs[row]=inv;
            for(int i=0;i<cols;i+=lanes){
                auto xp=svld1_u16(svptrue_b16(),x+row*cols+i);
                auto wp=svld1_u16(svptrue_b16(),w+i);
                auto nlo=round_fp(svmul_n_f32_x(pg,low(xp),inv));
                auto nhi=round_fp(svmul_n_f32_x(pg,high(xp),inv));
                store(y+row*cols+i,svmul_f32_x(pg,nlo,low(wp)),svmul_f32_x(pg,nhi,high(wp)));
            }
        }
    });
    return 0;
}
extern "C" int wan21_rmsnorm_bf16_backward(const void* in, const void* weight,
        const void* grad, const void* rstd, void* grad_in, void* grad_weight,
        void* partial, int rows, int cols) {
    if(!in || !weight || !grad || !rstd || !grad_in || !grad_weight || !partial ||
       (rows!=3328 && rows!=512) || cols!=1536 || cols % svcnth()) return -1;
    const auto* x=(const uint16_t*)in;
    const auto* w=(const uint16_t*)weight;
    const auto* dy=(const uint16_t*)grad;
    const auto* invs=(const float*)rstd;
    auto* dx=(uint16_t*)grad_in;
    auto* dw=(uint16_t*)grad_weight;
    auto* shards=(float*)partial;
    const int lanes=svcnth();
    auto pg=svptrue_b32();
    wan21_gemm_pool::run(36,wan21_gemm_pool::base_numa(),[&](int rank,int count){
        float* shard=shards+rank*cols;
        std::fill(shard,shard+cols,0.0f);
        const int begin=rows*rank/count,end=rows*(rank+1)/count;
        for(int row=begin;row<end;++row){
            float inv=invs[row];
            auto sumlo=svdup_n_f32(0),sumhi=svdup_n_f32(0);
            for(int i=0;i<cols;i+=lanes){
                auto xp=svld1_u16(svptrue_b16(),x+row*cols+i);
                auto gp=svld1_u16(svptrue_b16(),dy+row*cols+i);
                auto wp=svld1_u16(svptrue_b16(),w+i);
                auto nlo=round_fp(svmul_n_f32_x(pg,low(xp),inv));
                auto nhi=round_fp(svmul_n_f32_x(pg,high(xp),inv));
                auto gwlo=round_fp(svmul_f32_x(pg,low(gp),low(wp)));
                auto gwhi=round_fp(svmul_f32_x(pg,high(gp),high(wp)));
                sumlo=svmla_f32_x(pg,sumlo,low(xp),gwlo);
                sumhi=svmla_f32_x(pg,sumhi,high(xp),gwhi);
                auto wdlo=round_fp(svmul_f32_x(pg,low(gp),nlo));
                auto wdhi=round_fp(svmul_f32_x(pg,high(gp),nhi));
                float* sw=shard+i;
                svst1_f32(pg,sw,svadd_f32_x(pg,svld1_f32(pg,sw),wdlo));
                svst1_f32(pg,sw+svcntw(),svadd_f32_x(pg,svld1_f32(pg,sw+svcntw()),wdhi));
            }
            float dot=(svaddv_f32(pg,sumlo)+svaddv_f32(pg,sumhi))/cols;
            float scale=inv*inv*dot;
            for(int i=0;i<cols;i+=lanes){
                auto xp=svld1_u16(svptrue_b16(),x+row*cols+i);
                auto gp=svld1_u16(svptrue_b16(),dy+row*cols+i);
                auto wp=svld1_u16(svptrue_b16(),w+i);
                auto gwlo=round_fp(svmul_f32_x(pg,low(gp),low(wp)));
                auto gwhi=round_fp(svmul_f32_x(pg,high(gp),high(wp)));
                store(dx+row*cols+i,
                    svmul_n_f32_x(pg,svsub_f32_x(pg,gwlo,svmul_n_f32_x(pg,low(xp),scale)),inv),
                    svmul_n_f32_x(pg,svsub_f32_x(pg,gwhi,svmul_n_f32_x(pg,high(xp),scale)),inv));
            }
        }
    });
    for(int col=0;col<cols;++col){
        float sum=0;
        for(int worker=0;worker<36;++worker)sum+=shards[worker*cols+col];
        dw[col]=scalar_bf16(sum);
    }
    return 0;
}
