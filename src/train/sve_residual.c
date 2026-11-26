/* Fused FP32 residual + BF16 gated branch and its broadcast gradient.
 * x: [rows, cols] FP32; y: [rows, cols] BF16; gate: [cols] FP32.
 */
#include <arm_sve.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline svfloat32_t load_bf16_lo(svuint16_t packed) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(
        svptrue_b32(), svunpklo_u32(packed), 16));
}

static inline svfloat32_t load_bf16_hi(svuint16_t packed) {
    return svreinterpret_f32_u32(svlsl_n_u32_x(
        svptrue_b32(), svunpkhi_u32(packed), 16));
}

static inline svuint32_t round_bf16_bits(svfloat32_t value) {
    const svbool_t pg = svptrue_b32();
    svuint32_t bits = svreinterpret_u32_f32(value);
    svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
    return svlsr_n_u32_x(pg, svadd_u32_x(pg, bits,
        svadd_n_u32_x(pg, lsb, 0x7fff)), 16);
}

static inline void store_bf16(uint16_t *out, int64_t index, int64_t end,
                              svfloat32_t lo, svfloat32_t hi) {
    svuint16_t packed = svuzp1_u16(
        svreinterpret_u16_u32(round_bf16_bits(lo)),
        svreinterpret_u16_u32(round_bf16_bits(hi)));
    svst1_u16(svwhilelt_b16(index, end), out + index, packed);
}

int wan21_residual_omp_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void wan21_residual_bf16_forward(const float *x, const uint16_t *y,
                                 const float *gate, float *output,
                                 int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const float *xr = x + row * cols;
        const uint16_t *yr = y + row * cols;
        float *out = output + row * cols;
        for (int64_t i = 0; i < cols; i += svcnth()) {
            const svbool_t pglo = svwhilelt_b32(i, cols);
            const svbool_t pghi = svwhilelt_b32(i + (int64_t)svcntw(), cols);
            svuint16_t packed = svld1_u16(svwhilelt_b16(i, cols), yr + i);
            svfloat32_t mlo = svmul_f32_x(pglo, load_bf16_lo(packed),
                                         svld1_f32(pglo, gate + i));
            svfloat32_t mhi = svmul_f32_x(pghi, load_bf16_hi(packed),
                                         svld1_f32(pghi, gate + i + svcntw()));
            svst1_f32(pglo, out + i,
                       svadd_f32_x(pglo, svld1_f32(pglo, xr + i), mlo));
            svst1_f32(pghi, out + i + svcntw(),
                       svadd_f32_x(pghi, svld1_f32(pghi, xr + i + svcntw()), mhi));
        }
    }
}

/* Thread-local scratch: [omp_max_threads, cols] FP32, initialized here. */
void wan21_residual_bf16_backward(const float *grad, const uint16_t *y,
                                  const float *gate, uint16_t *grad_y,
                                  float *grad_gate, float *scratch,
                                  int64_t rows, int64_t cols) {
    const int thread_count = wan21_residual_omp_threads();
#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float *sums = scratch + tid * cols;
        for (int64_t i = 0; i < cols; ++i) sums[i] = 0.0f;
#pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            const float *g = grad + row * cols;
            const uint16_t *yr = y + row * cols;
            uint16_t *dy = grad_y + row * cols;
            for (int64_t i = 0; i < cols; i += svcnth()) {
                const svbool_t pglo = svwhilelt_b32(i, cols);
                const svbool_t pghi = svwhilelt_b32(i + (int64_t)svcntw(), cols);
                svuint16_t packed = svld1_u16(svwhilelt_b16(i, cols), yr + i);
                svfloat32_t yl = load_bf16_lo(packed), yh = load_bf16_hi(packed);
                svfloat32_t gl = svld1_f32(pglo, g + i);
                svfloat32_t gh = svld1_f32(pghi, g + i + svcntw());
                svfloat32_t gate_l = svld1_f32(pglo, gate + i);
                svfloat32_t gate_h = svld1_f32(pghi, gate + i + svcntw());
                store_bf16(dy, i, cols, svmul_f32_x(pglo, gl, gate_l),
                           svmul_f32_x(pghi, gh, gate_h));
                svst1_f32(pglo, sums + i,
                    svmla_f32_x(pglo, svld1_f32(pglo, sums + i), gl, yl));
                svst1_f32(pghi, sums + i + svcntw(),
                    svmla_f32_x(pghi, svld1_f32(pghi, sums + i + svcntw()), gh, yh));
            }
        }
#pragma omp for schedule(static)
        for (int64_t i = 0; i < cols; ++i) {
            float total = 0.0f;
            for (int t = 0; t < thread_count; ++t) total += scratch[t * cols + i];
            grad_gate[i] = total;
        }
    }
}


void wan21_residual_bf16_input_forward(const uint16_t *x, const uint16_t *y,
                                      const float *gate, float *output,
                                      int64_t rows, int64_t cols) {
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < rows; ++row) {
        const uint16_t *xr = x + row * cols;
        const uint16_t *yr = y + row * cols;
        float *out = output + row * cols;
        for (int64_t i = 0; i < cols; i += svcnth()) {
            const svbool_t pglo = svwhilelt_b32(i, cols);
            const svbool_t pghi = svwhilelt_b32(i + (int64_t)svcntw(), cols);
            svuint16_t packed_x = svld1_u16(svwhilelt_b16(i, cols), xr + i);
            svuint16_t packed_y = svld1_u16(svwhilelt_b16(i, cols), yr + i);
            svfloat32_t yl = svmul_f32_x(pglo, load_bf16_lo(packed_y),
                                        svld1_f32(pglo, gate + i));
            svfloat32_t yh = svmul_f32_x(pghi, load_bf16_hi(packed_y),
                                        svld1_f32(pghi, gate + i + svcntw()));
            svst1_f32(pglo, out + i, svadd_f32_x(pglo, load_bf16_lo(packed_x), yl));
            svst1_f32(pghi, out + i + svcntw(),
                      svadd_f32_x(pghi, load_bf16_hi(packed_x), yh));
        }
    }
}

void wan21_residual_bf16_input_backward(const float *grad, const uint16_t *y,
                                  const float *gate, uint16_t *grad_x, uint16_t *grad_y,
                                  float *grad_gate, float *scratch,
                                  int64_t rows, int64_t cols) {
    const int thread_count = wan21_residual_omp_threads();
#pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float *sums = scratch + tid * cols;
        for (int64_t i = 0; i < cols; ++i) sums[i] = 0.0f;
#pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            const float *g = grad + row * cols;
            const uint16_t *yr = y + row * cols;
            uint16_t *dx = grad_x + row * cols;
            uint16_t *dy = grad_y + row * cols;
            for (int64_t i = 0; i < cols; i += svcnth()) {
                const svbool_t pglo = svwhilelt_b32(i, cols);
                const svbool_t pghi = svwhilelt_b32(i + (int64_t)svcntw(), cols);
                svuint16_t packed = svld1_u16(svwhilelt_b16(i, cols), yr + i);
                svfloat32_t yl = load_bf16_lo(packed), yh = load_bf16_hi(packed);
                svfloat32_t gl = svld1_f32(pglo, g + i);
                svfloat32_t gh = svld1_f32(pghi, g + i + svcntw());
                svfloat32_t gate_l = svld1_f32(pglo, gate + i);
                svfloat32_t gate_h = svld1_f32(pghi, gate + i + svcntw());
                store_bf16(dx, i, cols, gl, gh);
                store_bf16(dy, i, cols, svmul_f32_x(pglo, gl, gate_l),
                           svmul_f32_x(pghi, gh, gate_h));
                svst1_f32(pglo, sums + i,
                    svmla_f32_x(pglo, svld1_f32(pglo, sums + i), gl, yl));
                svst1_f32(pghi, sums + i + svcntw(),
                    svmla_f32_x(pghi, svld1_f32(pghi, sums + i + svcntw()), gh, yh));
            }
        }
#pragma omp for schedule(static)
        for (int64_t i = 0; i < cols; ++i) {
            float total = 0.0f;
            for (int t = 0; t < thread_count; ++t) total += scratch[t * cols + i];
            grad_gate[i] = total;
        }
    }
}

