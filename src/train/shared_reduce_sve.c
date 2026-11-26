#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif

void wan21_store_release_i32(int32_t *address, int32_t value) {
    __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

int32_t wan21_load_acquire_i32(const int32_t *address) {
    return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

static inline void reduce_range(float *dst, const float *src0, const float *src1,
                                const float *src2, const float *src3,
                                int64_t begin, int64_t end, float scale) {
#if defined(__ARM_FEATURE_SVE)
    int64_t i = begin;
    const svfloat32_t scale_vec = svdup_n_f32(scale);
    while (i < end) {
        svbool_t active = svwhilelt_b32(i, end);
        svfloat32_t value = svadd_x(active, svld1(active, src0 + i),
                                     svld1(active, src1 + i));
        value = svadd_x(active, value, svld1(active, src2 + i));
        value = svmul_x(active, svadd_x(active, value, svld1(active, src3 + i)),
                        scale_vec);
        svst1(active, dst + i, value);
        i += svcntw();
    }
#else
    for (int64_t i = begin; i < end; ++i)
        dst[i] = (src0[i] + src1[i] + src2[i] + src3[i]) * scale;
#endif
}

/* Reduce four contiguous FP32 rank slices into the selected rank slice. */
void wan21_reduce4_f32(float *all, int64_t total_elements, int64_t offset,
                       int64_t elements, int64_t destination_rank, float scale) {
    float *dst = all + destination_rank * total_elements + offset;
    const float *src0 = all + offset;
    const float *src1 = all + total_elements + offset;
    const float *src2 = all + 2 * total_elements + offset;
    const float *src3 = all + 3 * total_elements + offset;
#ifdef _OPENMP
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        int thread = omp_get_thread_num();
        int64_t begin = elements * thread / threads;
        int64_t end = elements * (thread + 1) / threads;
        reduce_range(dst, src0, src1, src2, src3, begin, end, scale);
    }
#else
    reduce_range(dst, src0, src1, src2, src3, 0, elements, scale);
#endif
}

/* Serial SVE/SVE2 path for the communication thread's dedicated CPU. */
void wan21_reduce4_f32_serial(float *all, int64_t total_elements, int64_t offset,
                              int64_t elements, int64_t destination_rank,
                              float scale) {
    float *dst = all + destination_rank * total_elements + offset;
    reduce_range(dst, all + offset, all + total_elements + offset,
                 all + 2 * total_elements + offset,
                 all + 3 * total_elements + offset, 0, elements, scale);
}

/* Four die-local BF16 gradients: FP32 pairwise sum, then one BF16 rounding.
 * Source and destination may alias, but each output index reads all sources
 * before writing that index. No cross-die mapping or temporary allocation. */
void wan21_reduce4_bf16_serial(uint16_t *all, int64_t total_elements,
                                int64_t offset, int64_t elements,
                                int64_t destination_rank, float scale) {
    uint16_t *dst = all + destination_rank * total_elements + offset;
    const uint16_t *a = all + offset;
    const uint16_t *b = a + total_elements;
    const uint16_t *c = b + total_elements;
    const uint16_t *d = c + total_elements;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    for (int64_t i = 0; i < elements; i += svcntw()) {
        const svbool_t pg32 = svwhilelt_b32(i, elements);
        const int64_t lanes = elements - i < svcntw() ? elements - i : svcntw();
        const svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)lanes);
        const svuint16_t av = svld1_u16(pg16, a + i);
        const svuint16_t bv = svld1_u16(pg16, b + i);
        const svuint16_t cv = svld1_u16(pg16, c + i);
        const svuint16_t dv = svld1_u16(pg16, d + i);
        const svfloat32_t fa = svreinterpret_f32_u32(svlsl_n_u32_z(pg32, svunpklo_u32(av), 16));
        const svfloat32_t fb = svreinterpret_f32_u32(svlsl_n_u32_z(pg32, svunpklo_u32(bv), 16));
        const svfloat32_t fc = svreinterpret_f32_u32(svlsl_n_u32_z(pg32, svunpklo_u32(cv), 16));
        const svfloat32_t fd = svreinterpret_f32_u32(svlsl_n_u32_z(pg32, svunpklo_u32(dv), 16));
        svfloat32_t sum = svadd_f32_x(pg32, fa, fb);
        sum = svadd_f32_x(pg32, sum, fc);
        sum = svmul_n_f32_x(pg32, svadd_f32_x(pg32, sum, fd), scale);
        const svbfloat16_t rounded = svcvt_bf16_x(pg32, sum);
        svst1_u16(pg16, dst + i,
                  svreinterpret_u16_bf16(svuzp1_bf16(rounded, rounded)));
    }
#else
    for (int64_t i = 0; i < elements; ++i) {
        union { uint32_t bits; float value; } v;
        v.bits = (uint32_t)a[i] << 16;
        float sum = v.value;
        v.bits = (uint32_t)b[i] << 16;
        sum += v.value;
        v.bits = (uint32_t)c[i] << 16;
        sum += v.value;
        v.bits = (uint32_t)d[i] << 16;
        v.value = (sum + v.value) * scale;
        dst[i] = (uint16_t)((v.bits + 0x7fffU + ((v.bits >> 16) & 1U)) >> 16);
    }
#endif
}

/* Inter-die BF16 shards: reduce in FP32 in the same rank order as the
 * previous scratch-buffer path, then round once for BF16 all-gather. */
void wan21_reduce_replicas_bf16_serial(const uint16_t *inbound, uint16_t *outbound,
                                        int64_t stride, int64_t elements,
                                        int64_t replicas, float scale) {
    if (replicas < 2 || elements > stride) return;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    for (int64_t i = 0; i < elements; i += svcntw()) {
        const svbool_t pg32 = svwhilelt_b32(i, elements);
        const int64_t lanes = elements - i < svcntw() ? elements - i : svcntw();
        const svbool_t pg16 = svwhilelt_b16((uint64_t)0, (uint64_t)lanes);
        svuint16_t bits = svld1_u16(pg16, inbound + i);
        svfloat32_t sum = svreinterpret_f32_u32(
            svlsl_n_u32_z(pg32, svunpklo_u32(bits), 16));
        for (int64_t r = 1; r < replicas; ++r) {
            bits = svld1_u16(pg16, inbound + r * stride + i);
            const svfloat32_t next = svreinterpret_f32_u32(
                svlsl_n_u32_z(pg32, svunpklo_u32(bits), 16));
            sum = svadd_f32_x(pg32, sum, next);
        }
        sum = svmul_n_f32_x(pg32, sum, scale);
        const svbfloat16_t rounded = svcvt_bf16_x(pg32, sum);
        svst1_u16(pg16, outbound + i,
                  svreinterpret_u16_bf16(svuzp1_bf16(rounded, rounded)));
    }
#else
    for (int64_t i = 0; i < elements; ++i) {
        union { uint32_t bits; float value; } v;
        v.bits = (uint32_t)inbound[i] << 16;
        float sum = v.value;
        for (int64_t r = 1; r < replicas; ++r) {
            v.bits = (uint32_t)inbound[r * stride + i] << 16;
            sum += v.value;
        }
        v.value = sum * scale;
        outbound[i] = (uint16_t)((v.bits + 0x7fffU + ((v.bits >> 16) & 1U)) >> 16);
    }
#endif
}
