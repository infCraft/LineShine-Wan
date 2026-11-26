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
        svbfloat16_t b0 = svld1_vnum_bf16(pg_bf16, b_ptr, 0);
        svbfloat16_t b1 = svld1_vnum_bf16(pg_bf16, b_ptr, 1);
        svbfloat16_t b2 = svld1_vnum_bf16(pg_bf16, b_ptr, 2);
        svbfloat16_t b3 = svld1_vnum_bf16(pg_bf16, b_ptr, 3);
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

// Generate SME A tiles and both B feature panels from one read of each Q/dO row.
// Rows are split into 16 row A panels; B keeps its full query stride.
__arm_new("za")
__attribute__((noinline)) void pack_ab_unchecked(
 int rows,int cols,int lda,int full_k,int off,__bf16* source,__bf16* dest_a,__bf16* dest_b)
__arm_streaming
{
 const svbool_t pg=svptrue_b16();
 for(int rbase=0;rbase<rows;rbase+=16){
  for(int kk=0;kk<cols;kk+=32){
   svzero_za();
   for(int row=0;row<16;++row)
    svld1_hor_za16(0,row,pg,source+(size_t)(rbase+row)*lda+kk);
   for(int row=0;row<16;row+=2){
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,row);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,row+1);
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+row)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   for(int col=0;col<32;col+=2){
    const svbfloat16_t a=svread_ver_za16_bf16_m(svundef_bf16(),pg,0,col);
    const svbfloat16_t b=svread_ver_za16_bf16_m(svundef_bf16(),pg,0,col+1);
    svst1_bf16(pg,dest_a+(size_t)rbase*cols+(size_t)kk*16+col*16,svzip1_bf16(a,b));
   }
  }
 }
}

// While reusing dO values for both GEMM layouts, accumulate dO*O in the
// original four 32-BF16 chunks and the same FP32 reduction order.
__arm_new("za")
__attribute__((noinline)) void pack_ab_delta_unchecked(
 int rows,int cols,int lda,int full_k,int off,
 __bf16* source,__bf16* dest_a,__bf16* dest_b,
 const __bf16* output,float* delta)
__arm_streaming
{
 const svbool_t pg=svptrue_b16(),pg32=svptrue_b32();
 for(int rbase=0;rbase<rows;rbase+=16){
  svfloat32_t dot0=svdup_n_f32(0.0f);
  svfloat32_t dot1=svdup_n_f32(0.0f);
  svfloat32_t dot2=svdup_n_f32(0.0f);
  svfloat32_t dot3=svdup_n_f32(0.0f);
  svfloat32_t dot4=svdup_n_f32(0.0f);
  svfloat32_t dot5=svdup_n_f32(0.0f);
  svfloat32_t dot6=svdup_n_f32(0.0f);
  svfloat32_t dot7=svdup_n_f32(0.0f);
  svfloat32_t dot8=svdup_n_f32(0.0f);
  svfloat32_t dot9=svdup_n_f32(0.0f);
  svfloat32_t dot10=svdup_n_f32(0.0f);
  svfloat32_t dot11=svdup_n_f32(0.0f);
  svfloat32_t dot12=svdup_n_f32(0.0f);
  svfloat32_t dot13=svdup_n_f32(0.0f);
  svfloat32_t dot14=svdup_n_f32(0.0f);
  svfloat32_t dot15=svdup_n_f32(0.0f);
  for(int kk=0;kk<cols;kk+=32){
   svzero_za();
   for(int row=0;row<16;++row)
    svld1_hor_za16(0,row,pg,source+(size_t)(rbase+row)*lda+kk);
   { // query rows 0,1
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,0);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,1);
    dot0=svbfdot_f32(dot0,a,svld1_bf16(pg,output+(size_t)(rbase+0)*lda+kk));
    dot1=svbfdot_f32(dot1,b,svld1_bf16(pg,output+(size_t)(rbase+1)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+0)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 2,3
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,2);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,3);
    dot2=svbfdot_f32(dot2,a,svld1_bf16(pg,output+(size_t)(rbase+2)*lda+kk));
    dot3=svbfdot_f32(dot3,b,svld1_bf16(pg,output+(size_t)(rbase+3)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+2)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 4,5
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,4);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,5);
    dot4=svbfdot_f32(dot4,a,svld1_bf16(pg,output+(size_t)(rbase+4)*lda+kk));
    dot5=svbfdot_f32(dot5,b,svld1_bf16(pg,output+(size_t)(rbase+5)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+4)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 6,7
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,6);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,7);
    dot6=svbfdot_f32(dot6,a,svld1_bf16(pg,output+(size_t)(rbase+6)*lda+kk));
    dot7=svbfdot_f32(dot7,b,svld1_bf16(pg,output+(size_t)(rbase+7)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+6)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 8,9
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,8);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,9);
    dot8=svbfdot_f32(dot8,a,svld1_bf16(pg,output+(size_t)(rbase+8)*lda+kk));
    dot9=svbfdot_f32(dot9,b,svld1_bf16(pg,output+(size_t)(rbase+9)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+8)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 10,11
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,10);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,11);
    dot10=svbfdot_f32(dot10,a,svld1_bf16(pg,output+(size_t)(rbase+10)*lda+kk));
    dot11=svbfdot_f32(dot11,b,svld1_bf16(pg,output+(size_t)(rbase+11)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+10)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 12,13
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,12);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,13);
    dot12=svbfdot_f32(dot12,a,svld1_bf16(pg,output+(size_t)(rbase+12)*lda+kk));
    dot13=svbfdot_f32(dot13,b,svld1_bf16(pg,output+(size_t)(rbase+13)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+12)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   { // query rows 14,15
    const svbfloat16_t a=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,14);
    const svbfloat16_t b=svread_hor_za16_bf16_m(svundef_bf16(),pg,0,15);
    dot14=svbfdot_f32(dot14,a,svld1_bf16(pg,output+(size_t)(rbase+14)*lda+kk));
    dot15=svbfdot_f32(dot15,b,svld1_bf16(pg,output+(size_t)(rbase+15)*lda+kk));
    __bf16* out=dest_b+(size_t)(kk/64)*64*full_k+(size_t)(off+rbase+14)*64+(kk%64)*2;
    svst1_bf16(pg,out,svzip1_bf16(a,b));
    svst1_bf16(pg,out+32,svzip2_bf16(a,b));
   }
   for(int col=0;col<32;col+=2){
    const svbfloat16_t a=svread_ver_za16_bf16_m(svundef_bf16(),pg,0,col);
    const svbfloat16_t b=svread_ver_za16_bf16_m(svundef_bf16(),pg,0,col+1);
    svst1_bf16(pg,dest_a+(size_t)rbase*cols+(size_t)kk*16+col*16,svzip1_bf16(a,b));
   }
  }
  delta[off+rbase+0]=svaddv_f32(pg32,dot0);
  delta[off+rbase+1]=svaddv_f32(pg32,dot1);
  delta[off+rbase+2]=svaddv_f32(pg32,dot2);
  delta[off+rbase+3]=svaddv_f32(pg32,dot3);
  delta[off+rbase+4]=svaddv_f32(pg32,dot4);
  delta[off+rbase+5]=svaddv_f32(pg32,dot5);
  delta[off+rbase+6]=svaddv_f32(pg32,dot6);
  delta[off+rbase+7]=svaddv_f32(pg32,dot7);
  delta[off+rbase+8]=svaddv_f32(pg32,dot8);
  delta[off+rbase+9]=svaddv_f32(pg32,dot9);
  delta[off+rbase+10]=svaddv_f32(pg32,dot10);
  delta[off+rbase+11]=svaddv_f32(pg32,dot11);
  delta[off+rbase+12]=svaddv_f32(pg32,dot12);
  delta[off+rbase+13]=svaddv_f32(pg32,dot13);
  delta[off+rbase+14]=svaddv_f32(pg32,dot14);
  delta[off+rbase+15]=svaddv_f32(pg32,dot15);
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


// Compute QK^T and dO V^T in one streaming region.  Both calls write FP32
// outputs and use independent ZA accumulators exactly like the separate path.
// Reuse a packed B panel's full query stride when contracting one query chunk.
__arm_new("za")
__attribute__((noinline)) void sbgemm_packed_bstrided_unchecked(
 int m,int n,int k,int full_k,__bf16 alpha,__bf16* a,__bf16* b,
 __bf16 beta,float* c,int ldc)
__arm_streaming
{
 for(int col=0;col<n;col+=magic_n)
  for(int row=0;row<m;row+=magic_m_nt)
   sme_kernel_f32(a+(size_t)row*k,b+(size_t)col*full_k,
       c+(size_t)row*ldc+col,ldc,k,magic_m_nt,magic_n,alpha,beta);
}

__arm_new("za")
__attribute__((noinline)) void sbgemm_packed_pair_unchecked(
        int m, int n, int k, __bf16* a0, __bf16* b0, float* c0,
        __bf16* a1, __bf16* b1, float* c1, int ldc)
__arm_streaming
{
    for(int col=0;col<n;col+=magic_n) {
        for(int row=0;row<m;row+=magic_m_nt) {
            sme_kernel_f32(a0+(size_t)row*k,b0+(size_t)col*k,
                c0+(size_t)row*ldc+col,ldc,k,magic_m_nt,magic_n,
                (__bf16)1,(__bf16)0);
            sme_kernel_f32(a1+(size_t)row*k,b1+(size_t)col*k,
                c1+(size_t)row*ldc+col,ldc,k,magic_m_nt,magic_n,
                (__bf16)1,(__bf16)0);
        }
    }
}

// Interleave the independent score and dP dot products across ZA tiles:
// each 16x32 output takes two tiles, so both results share one K traversal.
__attribute__((noinline)) void sme_kernel_pair_interleaved(
 __bf16* a0,__bf16* b0,__bf16* a1,__bf16* b1,
 float* c0,float* c1,int ldc,int k)
__arm_streaming __arm_inout("za")
{
 const int tl=svcnth(),wl=svcntw();
 const svbool_t pg16=svptrue_b16(),pg32=svptrue_b32();
 svzero_za();
 for(int i=0;i<k;i+=2){
  svbfloat16_t q=svld1_bf16(pg16,a0),go=svld1_bf16(pg16,a1);
  svbfloat16_t k0=svld1_bf16(pg16,b0),k1=svld1_bf16(pg16,b0+tl);
  svbfloat16_t v0=svld1_bf16(pg16,b1),v1=svld1_bf16(pg16,b1+tl);
  a0+=tl;a1+=tl;b0+=4*tl;b1+=4*tl;
  svmopa_za32_bf16_m(0,pg16,pg16,q,k0);
  svmopa_za32_bf16_m(2,pg16,pg16,go,v0);
  svmopa_za32_bf16_m(1,pg16,pg16,q,k1);
  svmopa_za32_bf16_m(3,pg16,pg16,go,v1);
 }
 for(int h=0;h<16;++h){
  svst1_hor_za32(0,h,pg32,c0+(size_t)h*ldc);
  svst1_hor_za32(1,h,pg32,c0+(size_t)h*ldc+wl);
  svst1_hor_za32(2,h,pg32,c1+(size_t)h*ldc);
  svst1_hor_za32(3,h,pg32,c1+(size_t)h*ldc+wl);
 }
}
__arm_new("za")
__attribute__((noinline)) void sbgemm_packed_pair_interleaved_unchecked(
 int m,int n,int k,__bf16* a0,__bf16* b0,float* c0,
 __bf16* a1,__bf16* b1,float* c1,int ldc)
__arm_streaming
{
 for(int col=0;col<n;col+=64){
  for(int sub=0;sub<64;sub+=32){
   for(int row=0;row<m;row+=16){
    sme_kernel_pair_interleaved(a0+(size_t)row*k,
      b0+(size_t)col*k+2*sub,
      a1+(size_t)row*k,b1+(size_t)col*k+2*sub,
      c0+(size_t)row*ldc+col+sub,
      c1+(size_t)row*ldc+col+sub,ldc,k);
   }
  }
 }
}
}  // namespace ring_attn_component

