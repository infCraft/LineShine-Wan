/* BF16 tanh-GELU kernels for the 72F5 CPU.  The public entry points accept
 * contiguous BF16 tensors and preserve PyTorch's approximate='tanh' formula.
 */
#include <stdint.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#ifdef WAN21_GELU_SME
#include <arm_sme.h>
#define WAN21_STREAMING __arm_streaming
#else
#define WAN21_STREAMING
#endif

static inline svfloat32_t exp2_negative(svfloat32_t z) WAN21_STREAMING {
    const svbool_t pg = svptrue_b32();
    svfloat32_t nf = svrintm_f32_x(pg, z);
    svfloat32_t f = svsub_f32_x(pg, z, nf);
    svfloat32_t p = svdup_n_f32(0.0000013215486790144309f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.00001525273380405984f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.0001540353039338161f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.0013333558146428443f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.009618129107628477f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.05550410866482158f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.2402265069591007f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 0.6931471805599453f);
    p = svadd_n_f32_x(pg, svmul_f32_x(pg, p, f), 1.0f);
    svint32_t n = svcvt_s32_f32_x(pg, nf);
    svint32_t bits = svlsl_n_s32_x(pg, svadd_n_s32_x(pg, n, 127), 23);
    return svmul_f32_x(pg, p, svreinterpret_f32_s32(bits));
}

static inline svfloat32_t load_bf16_lo(svuint16_t packed) WAN21_STREAMING {
    svuint32_t bits = svlsl_n_u32_x(svptrue_b32(), svunpklo_u32(packed), 16);
    return svreinterpret_f32_u32(bits);
}

static inline svfloat32_t load_bf16_hi(svuint16_t packed) WAN21_STREAMING {
    svuint32_t bits = svlsl_n_u32_x(svptrue_b32(), svunpkhi_u32(packed), 16);
    return svreinterpret_f32_u32(bits);
}

static inline svuint32_t round_bf16_bits(svfloat32_t value) WAN21_STREAMING {
    const svbool_t pg = svptrue_b32();
    svuint32_t bits = svreinterpret_u32_f32(value);
    svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
    bits = svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff));
    return svlsr_n_u32_x(pg, bits, 16);
}

static inline void store_bf16(uint16_t *out, int64_t index, int64_t end,
                              svfloat32_t lo, svfloat32_t hi) WAN21_STREAMING {
    svuint32_t blo = round_bf16_bits(lo);
    svuint32_t bhi = round_bf16_bits(hi);
    svuint16_t packed = svuzp1_u16(svreinterpret_u16_u32(blo),
                                  svreinterpret_u16_u32(bhi));
    svst1_u16(svwhilelt_b16(index, end), out + index, packed);
}

static inline svfloat32_t gelu_tanh(svfloat32_t x, svfloat32_t *derivative) WAN21_STREAMING {
    const svbool_t pg = svptrue_b32();
    svfloat32_t xc = svmin_n_f32_x(pg, svmax_n_f32_x(pg, x, -10.0f), 10.0f);
    svfloat32_t x2 = svmul_f32_x(pg, xc, xc);
    svfloat32_t x3 = svmul_f32_x(pg, x2, xc);
    svfloat32_t u = svmul_n_f32_x(
        pg, svadd_f32_x(pg, xc, svmul_n_f32_x(pg, x3, 0.044715f)),
        0.7978845608028654f);
    svfloat32_t absu = svabs_f32_x(pg, u);
    svfloat32_t expv = exp2_negative(svmul_n_f32_x(pg, absu, -2.8853900817779268f));
    svfloat32_t t = svdiv_f32_x(pg, svsub_f32_x(pg, svdup_n_f32(1.0f), expv),
                               svadd_f32_x(pg, svdup_n_f32(1.0f), expv));
    t = svsel_f32(svcmplt_n_f32(pg, u, 0.0f), svneg_f32_x(pg, t), t);
    svfloat32_t one_plus_t = svadd_n_f32_x(pg, t, 1.0f);
    if (derivative != 0) {
        svfloat32_t dt = svsub_f32_x(pg, svdup_n_f32(1.0f),
                                    svmul_f32_x(pg, t, t));
        svfloat32_t du = svmul_n_f32_x(
            pg, svadd_n_f32_x(pg, svmul_n_f32_x(pg, x2, 0.134145f), 1.0f),
            0.7978845608028654f);
        svfloat32_t result = svmul_n_f32_x(pg, one_plus_t, 0.5f);
        result = svadd_f32_x(pg, result,
            svmul_n_f32_x(pg, svmul_f32_x(pg, svmul_f32_x(pg, x, dt), du), 0.5f));
        *derivative = result;
    }
    return svmul_n_f32_x(pg, svmul_f32_x(pg, x, one_plus_t), 0.5f);
}

static void forward_block(const uint16_t *input, uint16_t *output,
                          int64_t start, int64_t end) WAN21_STREAMING {
    for (int64_t i = start; i < end; i += svcnth()) {
        svuint16_t raw = svld1_u16(svwhilelt_b16(i, end), input + i);
        svfloat32_t lo = gelu_tanh(load_bf16_lo(raw), 0);
        svfloat32_t hi = gelu_tanh(load_bf16_hi(raw), 0);
        store_bf16(output, i, end, lo, hi);
    }
}

void wan21_gelu_bf16_forward(const uint16_t *input, uint16_t *output, int64_t n) {
    const int64_t block = 32768;
#pragma omp parallel for schedule(static)
    for (int64_t start = 0; start < n; start += block) {
        int64_t end = start + block < n ? start + block : n;
        forward_block(input, output, start, end);
    }
}

#ifndef WAN21_GELU_FORWARD_ONLY
static void backward_block(const uint16_t *gradient, const uint16_t *input,
                           uint16_t *output, int64_t start, int64_t end) WAN21_STREAMING {
    for (int64_t i = start; i < end; i += svcnth()) {
        const svbool_t pg = svptrue_b32();
        svbool_t active = svwhilelt_b16(i, end);
        svuint16_t raw_x = svld1_u16(active, input + i);
        svuint16_t raw_g = svld1_u16(active, gradient + i);
        svfloat32_t lo, hi;
        gelu_tanh(load_bf16_lo(raw_x), &lo);
        gelu_tanh(load_bf16_hi(raw_x), &hi);
        lo = svmul_f32_x(pg, lo, load_bf16_lo(raw_g));
        hi = svmul_f32_x(pg, hi, load_bf16_hi(raw_g));
        store_bf16(output, i, end, lo, hi);
    }
}

void wan21_gelu_bf16_backward(const uint16_t *gradient, const uint16_t *input,
                              uint16_t *output, int64_t n) {
    const int64_t block = 32768;
#pragma omp parallel for schedule(static)
    for (int64_t start = 0; start < n; start += block) {
        int64_t end = start + block < n ? start + block : n;
        backward_block(gradient, input, output, start, end);
    }
}
#endif

#else
#error "wan21_gelu_bf16 requires SVE"
#endif
