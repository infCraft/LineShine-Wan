/* BF16 activation -> FP32 modulation -> BF16 Linear input, one pass. */
#include <stdint.h>
#include <arm_sve.h>
#include <omp.h>
static inline svfloat32_t bf16_lo(svuint16_t bits) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(svptrue_b32(),svunpklo_u32(bits),16));
}
static inline svfloat32_t bf16_hi(svuint16_t bits) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(svptrue_b32(),svunpkhi_u32(bits),16));
}
static inline svuint32_t to_bf16_bits(svfloat32_t x) {
    const svbool_t pg=svptrue_b32();
    svuint32_t bits=svreinterpret_u32_f32(x);
    svuint32_t lsb=svand_n_u32_x(pg,svlsr_n_u32_x(pg,bits,16),1);
    return svlsr_n_u32_x(pg,svadd_u32_x(pg,bits,svadd_n_u32_x(pg,lsb,0x7fff)),16);
}
void wan21_modulate_to_bf16(const uint16_t* x,const float* offset,
                             const float* scale,uint16_t* out,int rows,int cols) {
#pragma omp parallel for schedule(static)
    for(int row=0;row<rows;++row){
        const uint16_t* xr=x+(int64_t)row*cols;
        uint16_t* yr=out+(int64_t)row*cols;
        for(int i=0;i<cols;i+=svcnth()){
            const svbool_t pglo=svwhilelt_b32(i,cols);
            const svbool_t pghi=svwhilelt_b32(i+(int)svcntw(),cols);
            const svbool_t pg16=svwhilelt_b16(i,cols);
            const svuint16_t v=svld1_u16(pg16,xr+i);
            const svfloat32_t slo=svadd_n_f32_x(pglo,svld1_f32(pglo,scale+i),1.0f);
            const svfloat32_t shi=svadd_n_f32_x(pghi,svld1_f32(pghi,scale+i+svcntw()),1.0f);
            const svfloat32_t lo=svadd_f32_x(pglo,svmul_f32_x(pglo,bf16_lo(v),slo),svld1_f32(pglo,offset+i));
            const svfloat32_t hi=svadd_f32_x(pghi,svmul_f32_x(pghi,bf16_hi(v),shi),svld1_f32(pghi,offset+i+svcntw()));
            svuint16_t packed=svuzp1_u16(svreinterpret_u16_u32(to_bf16_bits(lo)),svreinterpret_u16_u32(to_bf16_bits(hi)));
            svst1_u16(pg16,yr+i,packed);
        }
    }
}

void wan21_modulate_fp32_to_bf16(const float* x,const float* offset,
                                  const float* scale,uint16_t* out,int rows,int cols) {
#pragma omp parallel for schedule(static)
    for(int row=0;row<rows;++row){
        const float* xr=x+(int64_t)row*cols;
        uint16_t* yr=out+(int64_t)row*cols;
        for(int i=0;i<cols;i+=svcnth()){
            const svbool_t pglo=svwhilelt_b32(i,cols);
            const svbool_t pghi=svwhilelt_b32(i+(int)svcntw(),cols);
            const svbool_t pg16=svwhilelt_b16(i,cols);
            const svfloat32_t slo=svadd_n_f32_x(pglo,svld1_f32(pglo,scale+i),1.0f);
            const svfloat32_t shi=svadd_n_f32_x(pghi,svld1_f32(pghi,scale+i+svcntw()),1.0f);
            const svfloat32_t lo=svadd_f32_x(pglo,svmul_f32_x(pglo,svld1_f32(pglo,xr+i),slo),svld1_f32(pglo,offset+i));
            const svfloat32_t hi=svadd_f32_x(pghi,svmul_f32_x(pghi,svld1_f32(pghi,xr+i+svcntw()),shi),svld1_f32(pghi,offset+i+svcntw()));
            svuint16_t packed=svuzp1_u16(svreinterpret_u16_u32(to_bf16_bits(lo)),svreinterpret_u16_u32(to_bf16_bits(hi)));
            svst1_u16(pg16,yr+i,packed);
        }
    }
}
/* BF16 incoming gradient, BF16/FP32 activation, BF16/FP32 dx, FP32 dOffset/dScale. */
void wan21_modulate_backward(const uint16_t* grad,const void* x,const float* scale,
                             void* dx,float* doffset,float* dscale,float* partial,
                             int rows,int cols,int activation_bf16){
#pragma omp parallel
    {
        int tid=omp_get_thread_num();
        int nt=omp_get_num_threads();
        float* offsets=partial+(int64_t)tid*cols;
        float* scales=partial+(int64_t)(tid+nt)*cols;
        for(int j=0;j<cols;++j){offsets[j]=0.0f;scales[j]=0.0f;}
#pragma omp for schedule(static)
        for(int row=0;row<rows;++row){
            const uint16_t* g=grad+(int64_t)row*cols;
            const uint16_t* xb=(const uint16_t*)x+(int64_t)row*cols;
            const float* xf=(const float*)x+(int64_t)row*cols;
            uint16_t* dxb=(uint16_t*)dx+(int64_t)row*cols;
            float* dxf=(float*)dx+(int64_t)row*cols;
            for(int i=0;i<cols;i+=svcnth()){
                const svbool_t pglo=svwhilelt_b32(i,cols);
                const svbool_t pghi=svwhilelt_b32(i+(int)svcntw(),cols);
                const svbool_t pg16=svwhilelt_b16(i,cols);
                const svuint16_t gb=svld1_u16(pg16,g+i);
                svfloat32_t gl=bf16_lo(gb),gh=bf16_hi(gb);
                svfloat32_t xl,xh;
                if(activation_bf16){
                    svuint16_t xv=svld1_u16(pg16,xb+i);
                    xl=bf16_lo(xv);xh=bf16_hi(xv);
                }else{
                    xl=svld1_f32(pglo,xf+i);
                    xh=svld1_f32(pghi,xf+i+svcntw());
                }
                svfloat32_t slo=svadd_n_f32_x(pglo,svld1_f32(pglo,scale+i),1.0f);
                svfloat32_t shi=svadd_n_f32_x(pghi,svld1_f32(pghi,scale+i+svcntw()),1.0f);
                svfloat32_t dxlo=svmul_f32_x(pglo,gl,slo);
                svfloat32_t dxhi=svmul_f32_x(pghi,gh,shi);
                if(activation_bf16){
                    svuint16_t db=svuzp1_u16(svreinterpret_u16_u32(to_bf16_bits(dxlo)),svreinterpret_u16_u32(to_bf16_bits(dxhi)));
                    svst1_u16(pg16,dxb+i,db);
                }else{
                    svst1_f32(pglo,dxf+i,dxlo);
                    svst1_f32(pghi,dxf+i+svcntw(),dxhi);
                }
                svst1_f32(pglo,offsets+i,svadd_f32_x(pglo,svld1_f32(pglo,offsets+i),gl));
                svst1_f32(pghi,offsets+i+svcntw(),svadd_f32_x(pghi,svld1_f32(pghi,offsets+i+svcntw()),gh));
                svst1_f32(pglo,scales+i,svadd_f32_x(pglo,svld1_f32(pglo,scales+i),svmul_f32_x(pglo,gl,xl)));
                svst1_f32(pghi,scales+i+svcntw(),svadd_f32_x(pghi,svld1_f32(pghi,scales+i+svcntw()),svmul_f32_x(pghi,gh,xh)));
            }
        }
#pragma omp for schedule(static)
        for(int j=0;j<cols;++j){
            float os=0.0f,ss=0.0f;
            for(int t=0;t<nt;++t){os+=partial[(int64_t)t*cols+j];ss+=partial[(int64_t)(t+nt)*cols+j];}
            doffset[j]=os;dscale[j]=ss;
        }
    }
}
