#include <algorithm>
#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>
#include "wan21_gemm_thread_pool.h"

static inline float unpack(uint16_t v) { uint32_t bits = uint32_t(v) << 16; float result; std::memcpy(&result, &bits, 4); return result; }
static inline uint16_t pack(float x) {
    uint32_t v; std::memcpy(&v, &x, 4);
    uint32_t bias = 0x7fffU + ((v >> 16) & 1U);
    // Preserve NaNs instead of rounding the payload into an infinity.
    if ((v & 0x7fffffffU) > 0x7f800000U) return uint16_t((v >> 16) | 0x40U);
    return uint16_t((v + bias) >> 16);
}

// Match Wan rope_apply: contiguous complex pairs, and split 64 frequencies
// into the frame / height / width axes. Inverse rotation implements the VJP.
extern "C" int wan21_rope_pair_bf16(const void *q_ptr, const void *k_ptr,
        void *oq_ptr, void *ok_ptr, const void *freq_ptr,
        int frames, int height, int width, int tokens, int heads, int dim, int inverse) {
    if (!q_ptr || !k_ptr || !oq_ptr || !ok_ptr || !freq_ptr ||
        frames < 1 || height < 1 || width < 1 ||
        frames > 1024 || height > 1024 || width > 1024 ||
        tokens < frames * height * width || heads != 12 || dim != 128 ||
        (inverse != 0 && inverse != 1)) return -1;
    const auto *q = static_cast<const uint16_t *>(q_ptr);
    const auto *k = static_cast<const uint16_t *>(k_ptr);
    auto *oq = static_cast<uint16_t *>(oq_ptr);
    auto *ok = static_cast<uint16_t *>(ok_ptr);
    const auto *freq = static_cast<const std::complex<double> *>(freq_ptr);
    constexpr int c = 64, a = c - 2 * (c / 3), b = c / 3;
    const int hw = height * width, valid = frames * hw;
    wan21_gemm_pool::run(36, wan21_gemm_pool::base_numa(), [&](int rank, int total) {
    for (int th = rank; th < tokens * heads; th += total) {
        int tok = th / heads;
        int f = tok / hw, h = (tok % hw) / width, w = tok % width;
        int base = th * dim;
        for (int pair = 0; pair < c; ++pair) {
            std::complex<double> rot(1., 0.);
            if (tok < valid) {
                if (pair < a) rot = freq[f * c + pair];
                else if (pair < a + b) rot = freq[h * c + pair];
                else rot = freq[w * c + pair];
            }
            double re = rot.real(), im = inverse ? -rot.imag() : rot.imag();
            int at = base + pair * 2;
            const double qr = unpack(q[at]), qi = unpack(q[at + 1]);
            const double kr = unpack(k[at]), ki = unpack(k[at + 1]);
            oq[at] = pack(float(qr * re - qi * im));
            oq[at + 1] = pack(float(qr * im + qi * re));
            ok[at] = pack(float(kr * re - ki * im));
            ok[at + 1] = pack(float(kr * im + ki * re));
        }
    }
    });
    return 0;
}
