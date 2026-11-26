/* BF16 input/output, FP32 row statistics. No affine parameters. */
#include <arm_sve.h>
#include <math.h>
#include <stdint.h>

static inline svfloat32_t bf16_lo(svuint16_t packed) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(
        svptrue_b32(), svunpklo_u32(packed), 16));
}

static inline svfloat32_t bf16_hi(svuint16_t packed) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(
        svptrue_b32(), svunpkhi_u32(packed), 16));
}

static inline svuint32_t rne_bf16(svfloat32_t x) {
    const svbool_t pg = svptrue_b32();
    svuint32_t bits = svreinterpret_u32_f32(x);
    svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
    return svlsr_n_u32_x(pg,
        svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff)), 16);
}

static inline void store_bf16(uint16_t *out, int64_t i, int64_t end,
                              svfloat32_t lo, svfloat32_t hi) {
    svuint16_t packed = svuzp1_u16(
        svreinterpret_u16_u32(rne_bf16(lo)),
        svreinterpret_u16_u32(rne_bf16(hi)));
    svst1_u16(svwhilelt_b16(i, end), out + i, packed);
}

void wan21_layernorm_bf16_forward(const uint16_t *input, uint16_t *output,
                                  float *means, float *rstds,
                                  int64_t rows, int64_t cols, float eps) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const svbool_t pg = svptrue_b32();
        const uint16_t *x = input + row * cols;
        uint16_t *y = output + row * cols;
        svfloat32_t sum_lo = svdup_n_f32(0), sum_hi = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcnth()) {
            svuint16_t packed = svld1_u16(svwhilelt_b16(i, cols), x + i);
            sum_lo = svadd_f32_x(pg, sum_lo, bf16_lo(packed));
            sum_hi = svadd_f32_x(pg, sum_hi, bf16_hi(packed));
        }
        float mean = (svaddv_f32(pg, sum_lo) + svaddv_f32(pg, sum_hi)) / (float)cols;
        svfloat32_t sq_lo = svdup_n_f32(0), sq_hi = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcnth()) {
            svbool_t active = svwhilelt_b16(i, cols);
            svuint16_t packed = svld1_u16(active, x + i);
            svfloat32_t lo = svsub_n_f32_x(pg, bf16_lo(packed), mean);
            svfloat32_t hi = svsub_n_f32_x(pg, bf16_hi(packed), mean);
            /* Unused tail lanes must not contribute mean^2. */
            svbool_t lo_active = svwhilelt_b32(i, cols);
            svbool_t hi_active = svwhilelt_b32(i + (int64_t)svcntw(), cols);
            lo = svsel_f32(lo_active, lo, svdup_n_f32(0));
            hi = svsel_f32(hi_active, hi, svdup_n_f32(0));
            sq_lo = svmla_f32_x(pg, sq_lo, lo, lo);
            sq_hi = svmla_f32_x(pg, sq_hi, hi, hi);
        }
        float var = (svaddv_f32(pg, sq_lo) + svaddv_f32(pg, sq_hi)) / (float)cols;
        float inv = 1.0f / sqrtf(var + eps);
        means[row] = mean;
        rstds[row] = inv;
        for (int64_t i = 0; i < cols; i += svcnth()) {
            svuint16_t packed = svld1_u16(svwhilelt_b16(i, cols), x + i);
            svfloat32_t lo = svmul_n_f32_x(pg, svsub_n_f32_x(pg, bf16_lo(packed), mean), inv);
            svfloat32_t hi = svmul_n_f32_x(pg, svsub_n_f32_x(pg, bf16_hi(packed), mean), inv);
            store_bf16(y, i, cols, lo, hi);
        }
    }
}

void wan21_layernorm_bf16_backward(const uint16_t *gradient,
                                   const uint16_t *input, uint16_t *output,
                                   const float *means, const float *rstds,
                                   int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const svbool_t pg = svptrue_b32();
        const uint16_t *x = input + row * cols;
        const uint16_t *g = gradient + row * cols;
        uint16_t *dx = output + row * cols;
        float mean = means[row], inv = rstds[row];
        svfloat32_t sum_glo = svdup_n_f32(0), sum_ghi = svdup_n_f32(0);
        svfloat32_t sum_xglo = svdup_n_f32(0), sum_xghi = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcnth()) {
            svbool_t active = svwhilelt_b16(i, cols);
            svuint16_t xp = svld1_u16(active, x + i);
            svuint16_t gp = svld1_u16(active, g + i);
            svfloat32_t xlo = svsub_n_f32_x(pg, bf16_lo(xp), mean);
            svfloat32_t xhi = svsub_n_f32_x(pg, bf16_hi(xp), mean);
            svfloat32_t glo = bf16_lo(gp), ghi = bf16_hi(gp);
            sum_glo = svadd_f32_x(pg, sum_glo, glo);
            sum_ghi = svadd_f32_x(pg, sum_ghi, ghi);
            sum_xglo = svmla_f32_x(pg, sum_xglo, xlo, glo);
            sum_xghi = svmla_f32_x(pg, sum_xghi, xhi, ghi);
        }
        float grad_mean = (svaddv_f32(pg, sum_glo) + svaddv_f32(pg, sum_ghi)) / (float)cols;
        float xgrad_mean = (svaddv_f32(pg, sum_xglo) + svaddv_f32(pg, sum_xghi)) / (float)cols;
        float scale = inv * inv * xgrad_mean;
        for (int64_t i = 0; i < cols; i += svcnth()) {
            svbool_t active = svwhilelt_b16(i, cols);
            svuint16_t xp = svld1_u16(active, x + i);
            svuint16_t gp = svld1_u16(active, g + i);
            svfloat32_t xlo = svsub_n_f32_x(pg, bf16_lo(xp), mean);
            svfloat32_t xhi = svsub_n_f32_x(pg, bf16_hi(xp), mean);
            svfloat32_t lo = svmul_n_f32_x(pg,
                svsub_f32_x(pg, svsub_n_f32_x(pg, bf16_lo(gp), grad_mean),
                            svmul_n_f32_x(pg, xlo, scale)), inv);
            svfloat32_t hi = svmul_n_f32_x(pg,
                svsub_f32_x(pg, svsub_n_f32_x(pg, bf16_hi(gp), grad_mean),
                            svmul_n_f32_x(pg, xhi, scale)), inv);
            store_bf16(dx, i, cols, lo, hi);
        }
    }
}

/* FP32 LayerNorm is the common path after the first Wan residual block. */
void wan21_layernorm_fp32_forward(const float *input, float *output,
                                  float *means, float *rstds,
                                  int64_t rows, int64_t cols, float eps) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const svbool_t pg = svptrue_b32();
        const float *x = input + row * cols;
        float *y = output + row * cols;
        float origin = x[0];
        svfloat32_t sumv = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcntw()) {
            svbool_t active = svwhilelt_b32(i, cols);
            svfloat32_t centered = svsel_f32(active,
                svsub_n_f32_x(active, svld1_f32(active, x + i), origin),
                svdup_n_f32(0.0f));
            sumv = svadd_f32_x(pg, sumv, centered);
        }
        float mean = origin + svaddv_f32(pg, sumv) / (float)cols;
        svfloat32_t variance = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcntw()) {
            svbool_t active = svwhilelt_b32(i, cols);
            svfloat32_t centered = svsel_f32(active,
                svsub_n_f32_x(active, svld1_f32(active, x + i), mean),
                svdup_n_f32(0.0f));
            variance = svmla_f32_x(pg, variance, centered, centered);
        }
        float inv = 1.0f / sqrtf(svaddv_f32(pg, variance) / (float)cols + eps);
        means[row] = mean;
        rstds[row] = inv;
        for (int64_t i = 0; i < cols; i += svcntw()) {
            svbool_t active = svwhilelt_b32(i, cols);
            svfloat32_t v = svsub_n_f32_x(active, svld1_f32(active, x + i), mean);
            svst1_f32(active, y + i, svmul_n_f32_x(active, v, inv));
        }
    }
}

void wan21_layernorm_fp32_backward(const float *gradient,
                                   const float *input, float *output,
                                   const float *means, const float *rstds,
                                   int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const svbool_t pg = svptrue_b32();
        const float *x = input + row * cols;
        const float *g = gradient + row * cols;
        float *dx = output + row * cols;
        float mean = means[row], inv = rstds[row];
        svfloat32_t sum_g = svdup_n_f32(0), sum_xg = svdup_n_f32(0);
        for (int64_t i = 0; i < cols; i += svcntw()) {
            svbool_t active = svwhilelt_b32(i, cols);
            svfloat32_t xc = svsel_f32(active,
                svsub_n_f32_x(active, svld1_f32(active, x + i), mean),
                svdup_n_f32(0.0f));
            svfloat32_t gv = svld1_f32(active, g + i);
            sum_g = svadd_f32_x(pg, sum_g, gv);
            sum_xg = svmla_f32_x(pg, sum_xg, xc, gv);
        }
        float grad_mean = svaddv_f32(pg, sum_g) / (float)cols;
        float scale = inv * inv * svaddv_f32(pg, sum_xg) / (float)cols;
        for (int64_t i = 0; i < cols; i += svcntw()) {
            svbool_t active = svwhilelt_b32(i, cols);
            svfloat32_t xc = svsub_n_f32_x(active, svld1_f32(active, x + i), mean);
            svfloat32_t v = svsub_f32_x(active,
                svsub_n_f32_x(active, svld1_f32(active, g + i), grad_mean),
                svmul_n_f32_x(active, xc, scale));
            svst1_f32(active, dx + i, svmul_n_f32_x(active, v, inv));
        }
    }
}
