#include "wan21_gemm_thread_pool.h"
#include "bgemm.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace last_gemm {

enum class ImplKind {
    Demo1,
    Demo2,
    Outer,
    Vector,
};

struct GemmCall {
    XHBLAS_ORDER order;
    XHBLAS_TRANSPOSE trans_a;
    XHBLAS_TRANSPOSE trans_b;
    int M;
    int N;
    int K;
    float alpha;
    const __bf16* A;
    int lda;
    const __bf16* B;
    int ldb;
    float beta;
    __bf16* C;
    int ldc;
    __bf16* ba;
    int sign;
};

static XHBLAS_TRANSPOSE normalize_transpose(XHBLAS_TRANSPOSE trans)
{
    return trans == XhblasConjTrans ? XhblasTrans : trans;
}

static const char* order_name(XHBLAS_ORDER order)
{
    switch (order) {
        case XhblasRowMajor: return "RowMajor";
        case XhblasColMajor: return "ColMajor";
    }
    return "UnknownOrder";
}

static const char* transpose_name(XHBLAS_TRANSPOSE trans)
{
    switch (trans) {
        case XhblasNoTrans: return "NoTrans";
        case XhblasTrans: return "Trans";
        case XhblasConjTrans: return "ConjTrans";
    }
    return "UnknownTrans";
}

extern "C" int last_gemm_demo1_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                    int M, int N, int K, float alpha,
                                    const __bf16* A, int lda,
                                    const __bf16* B, int ldb,
                                    float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int last_gemm_demo2_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                    int M, int N, int K, float alpha,
                                    const __bf16* A, int lda,
                                    const __bf16* B, int ldb,
                                    float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int last_gemm_outer_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                    int M, int N, int K, float alpha,
                                    const __bf16* A, int lda,
                                    const __bf16* B, int ldb,
                                    float beta, __bf16* C, int ldc, __bf16* ba, int sign);

extern "C" int last_gemm_vector_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                     int M, int N, int K, float alpha,
                                     const __bf16* A, int lda,
                                     const __bf16* B, int ldb,
                                     float beta, __bf16* C, int ldc, __bf16* ba, int sign);

static int run_impl(ImplKind impl, const GemmCall& call)
{
    switch (impl) {
        case ImplKind::Demo1:
            return last_gemm_demo1_impl(call.order, call.trans_a, call.trans_b,
                                        call.M, call.N, call.K, call.alpha,
                                        call.A, call.lda, call.B, call.ldb,
                                        call.beta, call.C, call.ldc, call.ba, call.sign);
        case ImplKind::Demo2:
            return last_gemm_demo2_impl(call.order, call.trans_a, call.trans_b,
                                        call.M, call.N, call.K, call.alpha,
                                        call.A, call.lda, call.B, call.ldb,
                                        call.beta, call.C, call.ldc, call.ba, call.sign);
        case ImplKind::Outer:
            return last_gemm_outer_impl(call.order, call.trans_a, call.trans_b,
                                        call.M, call.N, call.K, call.alpha,
                                        call.A, call.lda, call.B, call.ldb,
                                        call.beta, call.C, call.ldc, call.ba, call.sign);
        case ImplKind::Vector:
            return last_gemm_vector_impl(call.order, call.trans_a, call.trans_b,
                                         call.M, call.N, call.K, call.alpha,
                                         call.A, call.lda, call.B, call.ldb,
                                         call.beta, call.C, call.ldc, call.ba, call.sign);
    }
    return -6;
}

static int scale_only(float beta, __bf16* C, int M, int N, int ldc)
{
    if (beta == 0.0f) {
        memset(C, 0, (size_t)M * (size_t)ldc * sizeof(__bf16));
        return 0;
    }
    return -8;
}

static int getenv_int_or(const char* name, int default_value)
{
    const char* value = getenv(name);
    if (!value || !*value) {
        return default_value;
    }
    const int parsed = atoi(value);
    return parsed > 0 ? parsed : default_value;
}

static int choose_magic_k(int K)
{
    const char* override_k = getenv("WAN21_REF_KBLOCK");
    if (override_k && *override_k) {
        const int requested = atoi(override_k);
        if (requested > 0 && requested <= 4096 && K % requested == 0 && requested % 2 == 0)
            return requested;
    }
    if (K <= 4096) {
        return K;
    }

    const int min_parts = (K + 4095) / 4096;
    int best_even = -1;
    int best_any = -1;
    for (int parts = min_parts; parts <= K; ++parts) {
        if (K % parts != 0) {
            continue;
        }
        const int candidate = K / parts;
        if (candidate > 4096) {
            continue;
        }
        if ((candidate % 64) == 0) {
            return candidate;
        }
        if (best_even < 0 && (candidate % 2) == 0) {
            best_even = candidate;
        }
        if (best_any < 0) {
            best_any = candidate;
        }
    }
    if (best_even > 0) {
        return best_even;
    }
    return best_any;
}

static int choose_demo1_thread_tile(int packed_dim, int bench_threads)
{
    for (int tile = 64; tile >= 48; --tile) {
        if (packed_dim % tile != 0) {
            continue;
        }
        return tile;
    }
    return -1;
}

static bool demo1_can_handle_directly(const GemmCall& call)
{
    const int magic_k = choose_magic_k(call.K);
    if (magic_k <= 0 || (call.K % magic_k) != 0) {
        return false;
    }

    const int bench_threads = getenv_int_or("FINAL_GEMM_THREADS", 32);
    const int packed_dim = call.M < call.N ? call.N : call.M;
    return choose_demo1_thread_tile(packed_dim, bench_threads) >= 0;
}

static bool demo2_can_handle_directly(const GemmCall& call)
{
    return call.trans_a == XhblasNoTrans &&
           call.trans_b == XhblasNoTrans &&
           call.K <= 4096 &&
           (call.M % 16) == 0 &&
           (call.N % 64) == 0 &&
           (call.K % 2) == 0;
}

static GemmCall normalize_m1_call(const GemmCall& call)
{
    GemmCall normalized = call;
    normalized.trans_a = XhblasNoTrans;
    normalized.lda = call.K;
    return normalized;
}

static int dispatch_call(const GemmCall& call)
{
    if (call.N == 1) {
        return -6;
    }

    if (call.M == 1) {
        const GemmCall normalized = normalize_m1_call(call);
        return run_impl(ImplKind::Vector, normalized);
    }

    if (call.K == 1) {
        return run_impl(ImplKind::Outer, call);
    }

    if (!demo1_can_handle_directly(call)) {
        if (demo2_can_handle_directly(call)) {
            return run_impl(ImplKind::Demo2, call);
        }
        return -6;
    }

    return run_impl(ImplKind::Demo1, call);
}

static int xhblas_gemm_status_impl(XHBLAS_ORDER order, XHBLAS_TRANSPOSE trans_a, XHBLAS_TRANSPOSE trans_b,
                                   int M, int N, int K, float alpha,
                                   const __bf16* A, int lda,
                                   const __bf16* B, int ldb,
                                   float beta, __bf16* C, int ldc, __bf16* ba, int sign)
{
    trans_a = normalize_transpose(trans_a);
    trans_b = normalize_transpose(trans_b);

    if (order != XhblasRowMajor) {
        return -1;
    }
    if (trans_a != XhblasNoTrans && trans_a != XhblasTrans) {
        return -2;
    }
    if (trans_b != XhblasNoTrans && trans_b != XhblasTrans) {
        return -3;
    }
    if (M < 0 || N < 0 || K < 0) {
        return -4;
    }

    const int a_cols = trans_a == XhblasNoTrans ? K : M;
    const int b_cols = trans_b == XhblasNoTrans ? N : K;
    if (lda != a_cols || ldb != b_cols || ldc != N) {
        return -5;
    }

    if (beta != 0.0f) {
        return -8;
    }
    if (K == 0 || alpha == 0.0f) {
        return scale_only(beta, C, M, N, ldc);
    }

    const GemmCall call = {
        order, trans_a, trans_b, M, N, K, alpha,
        A, lda, B, ldb, beta, C, ldc, ba, sign
    };
    return dispatch_call(call);
}

static const char* status_message_impl(int status)
{
    switch (status) {
        case 0: return "success";
        case -1: return "only row-major layout is supported";
        case -2: return "unsupported TransA";
        case -3: return "unsupported TransB";
        case -4: return "invalid dimensions";
        case -5: return "invalid leading dimensions";
        case -6: return "unsupported shape";
        case -7: return "workspace allocation failed";
        case -8: return "non-zero beta is unsupported in the bf16-output path";
        case -9: return "backend rejected request";
        default: return "unknown error";
    }
}

}  // namespace last_gemm

extern "C" int bgemm_kernel(int order, int trans_a, int trans_b,
                            int M, int N, int K, float alpha,
                            const __bf16* A, int lda,
                            const __bf16* B, int ldb,
                            float beta, __bf16* C, int ldc,
                            __bf16* ba, int sign)
{
    const int result = last_gemm::xhblas_gemm_status_impl((XHBLAS_ORDER)order,
                                                          (XHBLAS_TRANSPOSE)trans_a,
                                                          (XHBLAS_TRANSPOSE)trans_b,
                                                          M, N, K, alpha,
                                                          A, lda, B, ldb, beta, C, ldc, ba, sign);
    if (result != 0) {
        if (result == -6) {
            fprintf(stderr,
                    "[last_gemm] unsupported shape: order=%s(%d) M=%d N=%d K=%d transA=%s(%d) transB=%s(%d) lda=%d ldb=%d ldc=%d sign=%d\n",
                    last_gemm::order_name((XHBLAS_ORDER)order), order,
                    M, N, K,
                    last_gemm::transpose_name((XHBLAS_TRANSPOSE)trans_a), trans_a,
                    last_gemm::transpose_name((XHBLAS_TRANSPOSE)trans_b), trans_b,
                    lda, ldb, ldc, sign);
        } else {
            fprintf(stderr, "last_gemm failed: %s\n", xhblas_gemm_status_message(result));
        }
    }
    if (result != 0 || sign == 0 || ba == NULL) {
        return result;
    }
    if (sign == 1) {
        return result;
    }
    if (sign == 2) {
        wan21_gemm_pool::run(32, wan21_gemm_pool::base_numa(),
            [&](int rank, int total) {
                for (int j = rank; j < M; j += total) {
                    float sum = 0.0f;
                    for (int i = 0; i < K; ++i) sum += A[i * M + j];
                    ba[j] = sum;
                }
            });
    }
    return result;
}

extern "C" int xhblas_gemm(int order, int trans_a, int trans_b,
                           int M, int N, int K, float alpha,
                           const __bf16* A, int lda,
                           const __bf16* B, int ldb,
                           float beta, __bf16* C, int ldc,
                           __bf16* ba, int sign)
{
    return bgemm_kernel(order, trans_a, trans_b,
                        M, N, K, alpha,
                        A, lda, B, ldb, beta, C, ldc, ba, sign);
}

extern "C" const char* xhblas_gemm_status_message(int status)
{
    return last_gemm::status_message_impl(status);
}
