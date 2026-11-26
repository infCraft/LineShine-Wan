#define BGEMM
#include <fcntl.h>
#include <sys/ipc.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <limits>
#include <numeric>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_acle.h>
#include <kblas.h>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <numaif.h>
#include <numa.h>
#include <sched.h>
#include <cerrno>
#include <omp.h>
#include <mutex>

#ifdef RING_ATTN_STANDARD_KBLAS
// The original private KBLAS header exposes the packed-matrix identifiers as
// CblasA/CblasB.  Public HPCKit headers omit these unused extensions; the ring
// kernel only needs stable identifiers for its own pack_get_size helper.
constexpr int CblasA = 101;
constexpr int CblasB = 102;
#endif

// dim = h_num * h_len
// t_num = batch_size*token_num
// Q: (t_num, h_num, h_len)
// grad_Q: (t_num, h_num, h_len)
// K: (t_num, h_num, h_len)
// dK: (t_num, h_num, h_len)
// V: (t_num, h_num, h_len)
// dV: (t_num, h_num, h_len)
// out: (t_num, h_num, h_len)
// grad_out: (t_num, h_num, h_len)
// D: (h_num, t_num)
// mi: (h_num, t_num)
// li: (h_num, t_num)

namespace ring_attn_component {

constexpr int atm_magic = 16;
constexpr int atpacked_m = 2 * atm_magic;
constexpr int btn_magic = 32;
constexpr int btpacked_n = 64;
constexpr int m_magic = 32;
constexpr int m_pack = 16;
constexpr int n_magic = 32;
constexpr int n_pack = 64;

__attribute__((noinline)) void pack_A_T(__bf16* A, int k, int m, int lda, __bf16* A_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = m < atpacked_m ? m : atpacked_m;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    const int block = k * atm_magic;
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t row_vec0 = svld1_bf16(pg_tail, A + i * lda);
        svbfloat16_t row_vec1 = svld1_bf16(pg_tail, A + (i + 1) * lda);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, A_p + i * atm_magic, a1);
        svst1(pg, A_p + i * atm_magic + block, a2);
    }
}

__attribute__((noinline)) void pack_B_T(__bf16* B, int n, int k, int ldb, __bf16* B_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    const int dst_panel_stride = btn_magic * btpacked_n;
    const int dst_pair_stride = 2 * btpacked_n;
    __bf16* src_panel = B;
    __bf16* dst_panel = B_p;
    for (int kk = 0; kk < k; kk += btn_magic) {
        const int rows0 = n > btn_magic ? btn_magic : n;
        const int rows1 = n > btn_magic ? (n - btn_magic) : 0;
        __bf16* src0 = src_panel;
        __bf16* src1 = src_panel + ldb * btn_magic;
        svzero_za();
        for (int mm = 0; mm < rows0; ++mm) {
            svld1_hor_za16(0, mm, pg, src0);
            src0 += ldb;
        }
        for (int mm = 0; mm < rows1; ++mm) {
            svld1_hor_za16(1, mm, pg, src1);
            src1 += ldb;
        }
        __bf16* dst = dst_panel;
        for (int nn = 0; nn < btn_magic; nn += 2) {
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst, a1);
            svst1(pg, dst + btn_magic, a2);
            row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn);
            row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 1, nn + 1);
            a1 = svzip1(row_vec0, row_vec1);
            a2 = svzip2(row_vec0, row_vec1);
            svst1(pg, dst + 2 * btn_magic, a1);
            svst1(pg, dst + 3 * btn_magic, a2);
            dst += dst_pair_stride;
        }
        src_panel += btn_magic;
        dst_panel += dst_panel_stride;
    }
}

__attribute__((noinline)) void pack_A(__bf16* A, int m, int k, int lda, __bf16* A_p)
__arm_streaming __arm_inout("za")
{
    svbool_t pg = svptrue_b16();
    svzero_za();
    for (int kk = 0; kk < k; kk += m_magic) {
        for (int mm = 0; mm < m; ++mm) {
            svld1_hor_za16(0, mm, pg, A + mm * lda + kk);
        }
        for (int nn = 0; nn < m_magic; nn += 2) {
            svbfloat16_t row_vec0 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn);
            svbfloat16_t row_vec1 = svread_ver_za16_bf16_m(svundef_bf16(), pg, 0, nn + 1);
            svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
            svst1(pg, A_p + m_pack * nn + kk * m_pack, a1);
        }
    }
}

__attribute__((noinline)) void pack_B(__bf16* B, int k, int n, int ldb, __bf16* B_p)
{
    svbool_t pg = svptrue_b16();
    const int valid_cols = (n - n_magic) < n_magic ? (n - n_magic) : n_magic;
    svbool_t pg_tail = svwhilelt_b16((uint64_t)0, (uint64_t)valid_cols);
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t row_vec0 = svld1_bf16(pg, B + i * ldb);
        svbfloat16_t row_vec1 = svld1_bf16(pg, B + (i + 1) * ldb);
        svbfloat16_t a1 = svzip1(row_vec0, row_vec1);
        svbfloat16_t a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, B_p + i * n_pack, a1);
        svst1(pg, B_p + i * n_pack + n_magic, a2);
        row_vec0 = svld1_bf16(pg_tail, B + i * ldb + n_magic);
        row_vec1 = svld1_bf16(pg_tail, B + (i + 1) * ldb + n_magic);
        a1 = svzip1(row_vec0, row_vec1);
        a2 = svzip2(row_vec0, row_vec1);
        svst1(pg, B_p + i * n_pack + 2 * n_magic, a1);
        svst1(pg, B_p + i * n_pack + 3 * n_magic, a2);
    }
}

__attribute__((noinline)) void sme_kernel_bf16(__bf16* A, __bf16* B, __bf16* C_out,
                int ldc, int k, int m_use, int n_use,
                __bf16 alpha, __bf16 beta)
__arm_streaming __arm_inout("za")
{
    (void)n_use;
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (beta == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            svuint16_t vb16 = svld1_u16(pg_bf16, reinterpret_cast<const uint16_t*>(C_out + (size_t)h * ldc));
            svuint32_t lo32 = svunpklo_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(0, h, pg, lo32);
            lo32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(1, h, pg, lo32);

            vb16 = svld1_u16(pg_bf16, reinterpret_cast<const uint16_t*>(C_out + (size_t)h * ldc + 2 * wl));
            lo32 = svunpklo_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(2, h, pg, lo32);
            lo32 = svunpkhi_u32(vb16);
            lo32 = svlsl_n_u32_x(pg, lo32, 16);
            svwrite_hor_za32_u32_m(3, h, pg, lo32);
        }
    } else if (beta == 0.0f) {
        svzero_za();
    }
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t a0 = svld1_bf16(pg_bf16, a_ptr);
        a_ptr += tl;
        svbfloat16_t b0 = svldnt1_bf16(pg_bf16, b_ptr);
        svbfloat16_t b1 = svldnt1_bf16(pg_bf16, b_ptr + tl);
        svbfloat16_t b2 = svldnt1_bf16(pg_bf16, b_ptr + 2 * tl);
        svbfloat16_t b3 = svldnt1_bf16(pg_bf16, b_ptr + 3 * tl);
        b_ptr += 4 * tl;
        svmopa_za32_bf16_m(0, pg_bf16, pg_bf16, a0, b0);
        svmopa_za32_bf16_m(1, pg_bf16, pg_bf16, a0, b1);
        svmopa_za32_bf16_m(2, pg_bf16, pg_bf16, a0, b2);
        svmopa_za32_bf16_m(3, pg_bf16, pg_bf16, a0, b3);
    }
    svbfloat16_t old_bf16 = svdup_n_bf16((__bf16)0);
    svfloat32_t old_f32 = svdup_n_f32(0.0f);
    svfloat32_t scale = svdup_n_f32((float)alpha);
    if (alpha == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            svbfloat16_t z0 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 0, 2 * h);
            svbfloat16_t z1 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 1, 2 * h);
            z1 = svuzp2_bf16(z0, z1);
            svst1_bf16(pg_bf16, C_out + h * ldc, z1);
            z0 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 0, 2 * h + 1);
            z1 = svread_hor_za16_bf16_m(old_bf16, svptrue_b16(), 1, 2 * h + 1);
            z1 = svuzp2_bf16(z0, z1);
            svst1_bf16(pg_bf16, C_out + h * ldc + 2 * wl, z1);
        }
    } else {
        for (int h = 0; h < m_use; ++h) {
            svfloat32_t z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 0, h);
            svfloat32_t z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 1, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svbfloat16_t u0 = svreinterpret_bf16_f32(z0);
            svbfloat16_t u1 = svreinterpret_bf16_f32(z1);
            u1 = svuzp2_bf16(u0, u1);
            svst1_bf16(pg_bf16, C_out + h * ldc, u1);

            z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 2, h);
            z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 3, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            u0 = svreinterpret_bf16_f32(z0);
            u1 = svreinterpret_bf16_f32(z1);
            u1 = svuzp2_bf16(u0, u1);
            svst1_bf16(pg_bf16, C_out + h * ldc + 2 * wl, u1);
        }
    }
}

__attribute__((noinline)) void sme_kernel_f32(__bf16* A, __bf16* B, float* C_out,
                int ldc, int k, int m_use, int n_use,
                __bf16 alpha, __bf16 beta)
__arm_streaming __arm_inout("za")
{
    (void)n_use;
    const int tl = svcnth();
    const int wl = svcntw();
    svbool_t pg = svptrue_b32();
    svbool_t pg_bf16 = svptrue_b16();
    __bf16* a_ptr = A;
    __bf16* b_ptr = B;
    if (beta == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            float* c0 = C_out + h * ldc;
            float* c1 = C_out + h * ldc + wl;
            float* c2 = C_out + h * ldc + 2 * wl;
            float* c3 = C_out + h * ldc + 3 * wl;
            svld1_hor_za32(0, h, pg, c0);
            svld1_hor_za32(1, h, pg, c1);
            svld1_hor_za32(2, h, pg, c2);
            svld1_hor_za32(3, h, pg, c3);
        }
    } else if (beta == 0.0f) {
        svzero_za();
    }
    for (int i = 0; i < k; i += 2) {
        svbfloat16_t a0 = svld1_bf16(pg_bf16, a_ptr);
        a_ptr += tl;
        svbfloat16_t b0 = svld1_bf16(pg_bf16, b_ptr);
        svbfloat16_t b1 = svld1_bf16(pg_bf16, b_ptr + tl);
        svbfloat16_t b2 = svld1_bf16(pg_bf16, b_ptr + 2 * tl);
        svbfloat16_t b3 = svld1_bf16(pg_bf16, b_ptr + 3 * tl);
        b_ptr += 4 * tl;
        svmopa_za32_bf16_m(0, pg_bf16, pg_bf16, a0, b0);
        svmopa_za32_bf16_m(1, pg_bf16, pg_bf16, a0, b1);
        svmopa_za32_bf16_m(2, pg_bf16, pg_bf16, a0, b2);
        svmopa_za32_bf16_m(3, pg_bf16, pg_bf16, a0, b3);
    }
    svbfloat16_t old_bf16 = svdup_n_bf16((__bf16)0);
    svfloat32_t old_f32 = svdup_n_f32(0.0f);
    svfloat32_t scale = svdup_n_f32((float)alpha);
    if (alpha == 1.0f) {
        for (int h = 0; h < m_use; ++h) {
            float* c0 = C_out + h * ldc;
            float* c1 = C_out + h * ldc + wl;
            float* c2 = C_out + h * ldc + 2 * wl;
            float* c3 = C_out + h * ldc + 3 * wl;
            svst1_hor_za32(0, h, pg, c0);
            svst1_hor_za32(1, h, pg, c1);
            svst1_hor_za32(2, h, pg, c2);
            svst1_hor_za32(3, h, pg, c3);
        }
    } else {
        for (int h = 0; h < m_use; ++h) {
            svfloat32_t z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 0, h);
            svfloat32_t z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 1, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svst1_f32(pg, C_out + h * ldc, z0);
            svst1_f32(pg, C_out + h * ldc + wl, z1);
            z0 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 2, h);
            z1 = svread_hor_za32_f32_m(old_f32, svptrue_b32(), 3, h);
            z0 = svmul_f32_x(pg, z0, scale);
            z1 = svmul_f32_x(pg, z1, scale);
            svst1_f32(pg, C_out + h * ldc + 2 * wl, z0);
            svst1_f32(pg, C_out + h * ldc + 3 * wl, z1);
        }
    }
}

constexpr int magic_n = 64;
constexpr int magic_m_nt = 16;
constexpr int magic_m_t = 32;

__arm_new("za")
__attribute__((noinline)) void pack_B_whole(const enum CBLAS_TRANSPOSE transB, int n, int k, int ldb, __bf16* b, __bf16* b_p)
__arm_streaming
{
    if (transB == CblasNoTrans) {
        for (int i = 0; i < n; i += magic_n) {
            pack_B(b + i, k, magic_n, ldb, b_p + (size_t)i * k);
        }
    } else {
        for (int i = 0; i < n; i += magic_n) {
            pack_B_T(b + (size_t)i * ldb, magic_n, k, ldb, b_p + (size_t)i * k);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void pack_A_whole(const enum CBLAS_TRANSPOSE transA, int m, int k, int lda, __bf16* a, __bf16* a_p)
__arm_streaming
{
    if (transA == CblasNoTrans) {
        for (int i = 0; i < m; i += magic_m_nt) {
            pack_A(a + (size_t)i * lda, magic_m_nt, k, lda, a_p + (size_t)i * k);
        }
    } else {
        for (int i = 0; i < m; i += magic_m_t) {
            pack_A_T(a + (size_t)i, k, magic_m_t, lda, a_p + (size_t)i * k);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void easy_bgemm(int m, int n, int k, __bf16* a, __bf16* b, __bf16* c, int ldc, __bf16 alpha, __bf16 beta)
__arm_streaming
{
    for (int i = 0; i < n; i += magic_n) {
        for (int mm = 0; mm < m; mm += magic_m_nt) {
            sme_kernel_bf16(a + (size_t)mm * k,
                       b + (size_t)i * k,
                       c + (size_t)mm * ldc + i,
                       ldc, k, magic_m_nt, magic_n,
                       alpha, beta);
        }
    }
}

__arm_new("za")
__attribute__((noinline)) void easy_sbgemm(int m, int n, int k, __bf16* a, __bf16* b, float* c, int ldc, __bf16 alpha, __bf16 beta)
__arm_streaming
{
    for (int i = 0; i < n; i += magic_n) {
        for (int mm = 0; mm < m; mm += magic_m_nt) {
            sme_kernel_f32(a + (size_t)mm * k,
                       b + (size_t)i * k,
                       c + (size_t)mm * ldc + i,
                       ldc, k, magic_m_nt, magic_n,
                       alpha, beta);
        }
    }
}

static inline bool supports_pack_a(int m, int k)
{
    return m > 0 && k > 0 &&
           (m % magic_m_nt) == 0 &&
           (k % m_magic) == 0;
}

static inline bool supports_pack_b(int n, int k)
{
    return n > 0 && k > 0 &&
           (n % magic_n) == 0 &&
           (k % m_magic) == 0;
}

static inline bool supports_gemm(int m, int n, int k)
{
    return m > 0 && n > 0 && k > 0 &&
           (m % magic_m_nt) == 0 &&
           (n % magic_n) == 0 &&
           (k % m_magic) == 0;
}

size_t pack_get_size(int kind, int m, int n, int k)
{
    if (kind == CblasA) {
        return supports_pack_a(m, k) ? (size_t)m * (size_t)k : 0;
    }
    if (kind == CblasB) {
        return supports_pack_b(n, k) ? (size_t)n * (size_t)k : 0;
    }
    return 0;
}

__attribute__((noinline)) void pack_a_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                const int m,
                                                const int k,
                                                const int lda,
                                                __bf16* a,
                                                __bf16* a_p)
{
    pack_A_whole(transA, m, k, lda, a, a_p);
}

__attribute__((noinline)) void pack_b_unchecked(const enum CBLAS_TRANSPOSE transB,
                                                const int n,
                                                const int k,
                                                const int ldb,
                                                __bf16* b,
                                                __bf16* b_p)
{
    pack_B_whole(transB, n, k, ldb, b, b_p);
}

__attribute__((noinline)) void bgemm_packed_unchecked(const int m,
                                                      const int n,
                                                      const int k,
                                                      const __bf16 alpha,
                                                      __bf16* packed_a,
                                                      __bf16* packed_b,
                                                      const __bf16 beta,
                                                      __bf16* c,
                                                      const int ldc)
{
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) void sbgemm_packed_unchecked(const int m,
                                                       const int n,
                                                       const int k,
                                                       const __bf16 alpha,
                                                       __bf16* packed_a,
                                                       __bf16* packed_b,
                                                       const __bf16 beta,
                                                       float* c,
                                                       const int ldc)
{
    easy_sbgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) bool bgemm_kernel(const enum CBLAS_ORDER order,
                  const enum CBLAS_TRANSPOSE transA,
                  const enum CBLAS_TRANSPOSE transB,
                  const int m,
                  const int n,
                  const int k,
                  const __bf16 alpha,
                  __bf16* a,
                  const int lda,
                  __bf16* b,
                  const int ldb,
                  const __bf16 beta,
                  __bf16* c,
                  const int ldc,
                  __bf16* packed_a,
                  __bf16* packed_b)
{
    if (order != CblasRowMajor) {
        return false;
    }
    if (!supports_pack_a(m, k) ||
        !supports_pack_b(n, k) ||
        !supports_gemm(m, n, k)) {
        return false;
    }
    if (a == nullptr || b == nullptr || c == nullptr ||
        packed_a == nullptr || packed_b == nullptr) {
        return false;
    }
    if (!(beta == (__bf16)0.0f || beta == (__bf16)1.0f)) {
        return false;
    }

    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
    return true;
}

__attribute__((noinline)) void bgemm_kernel_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                      const enum CBLAS_TRANSPOSE transB,
                                                      const int m,
                                                      const int n,
                                                      const int k,
                                                      const __bf16 alpha,
                                                      __bf16* a,
                                                      const int lda,
                                                      __bf16* b,
                                                      const int ldb,
                                                      const __bf16 beta,
                                                      __bf16* c,
                                                      const int ldc,
                                                      __bf16* packed_a,
                                                      __bf16* packed_b)
{
    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_bgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}

__attribute__((noinline)) void sbgemm_kernel_unchecked(const enum CBLAS_TRANSPOSE transA,
                                                       const enum CBLAS_TRANSPOSE transB,
                                                       const int m,
                                                       const int n,
                                                       const int k,
                                                       const __bf16 alpha,
                                                       __bf16* a,
                                                       const int lda,
                                                       __bf16* b,
                                                       const int ldb,
                                                       const __bf16 beta,
                                                       float* c,
                                                       const int ldc,
                                                       __bf16* packed_a,
                                                       __bf16* packed_b)
{
    pack_A_whole(transA, m, k, lda, a, packed_a);
    pack_B_whole(transB, n, k, ldb, b, packed_b);
    easy_sbgemm(m, n, k, packed_a, packed_b, c, ldc, alpha, beta);
}


// Attention score: emit 16 query rows per key column, so softmax reads
// full SVE vectors without gathering across row-major scores.
__attribute__((noinline)) void sme_kernel_score_columns(
    __bf16* a, __bf16* b, float* score, float* block_max, int k)
__arm_streaming __arm_inout("za")
{
    const svbool_t pg16=svptrue_b16(), pg32=svptrue_b32();
    const int lanes=svcnth();
    svzero_za();
    for(int kk=0;kk<k;kk+=2) {
        const svbfloat16_t va=svld1_bf16(pg16,a);
        const svbfloat16_t vb0=svld1_bf16(pg16,b);
        const svbfloat16_t vb1=svld1_bf16(pg16,b+lanes);
        const svbfloat16_t vb2=svld1_bf16(pg16,b+2*lanes);
        const svbfloat16_t vb3=svld1_bf16(pg16,b+3*lanes);
        a+=lanes;b+=4*lanes;
        svmopa_za32_bf16_m(0,pg16,pg16,va,vb0);
        svmopa_za32_bf16_m(1,pg16,pg16,va,vb1);
        svmopa_za32_bf16_m(2,pg16,pg16,va,vb2);
        svmopa_za32_bf16_m(3,pg16,pg16,va,vb3);
    }
    svfloat32_t max0=svdup_n_f32(-__builtin_inff());
    svfloat32_t max1=max0,max2=max0,max3=max0;
    for(int col=0;col<16;++col) {
        svfloat32_t z0=svread_ver_za32_f32_m(svundef_f32(),pg32,0,col);
        svfloat32_t z1=svread_ver_za32_f32_m(svundef_f32(),pg32,1,col);
        svfloat32_t z2=svread_ver_za32_f32_m(svundef_f32(),pg32,2,col);
        svfloat32_t z3=svread_ver_za32_f32_m(svundef_f32(),pg32,3,col);
        svst1_f32(pg32,score+col*16,z0);
        svst1_f32(pg32,score+(col+16)*16,z1);
        svst1_f32(pg32,score+(col+32)*16,z2);
        svst1_f32(pg32,score+(col+48)*16,z3);
        max0=svmax_f32_x(pg32,max0,z0);
        max1=svmax_f32_x(pg32,max1,z1);
        max2=svmax_f32_x(pg32,max2,z2);
        max3=svmax_f32_x(pg32,max3,z3);
    }
    svfloat32_t best=svmax_f32_x(pg32,svmax_f32_x(pg32,max0,max1),
                                         svmax_f32_x(pg32,max2,max3));
    svst1_f32(pg32,block_max,svmax_f32_x(pg32,svld1_f32(pg32,block_max),best));
}
__arm_new("za")
__attribute__((noinline)) void easy_sbgemm_score_columns(
    int m,int n,int k,__bf16* a,__bf16* b,float* score,float* block_max)
__arm_streaming
{
    for(int col=0;col<n;col+=64)
        for(int row=0;row<m;row+=16)
            sme_kernel_score_columns(a+(size_t)row*k,b+(size_t)col*k,
                score+(size_t)row*n+(size_t)col*16,block_max+row,k);
}
__attribute__((noinline)) void sbgemm_packed_colscore_unchecked(
    int m,int n,int k,__bf16* a,__bf16* b,float* score,float* block_max)
{
    easy_sbgemm_score_columns(m,n,k,a,b,score,block_max);
}

}  // namespace ring_attn_component

