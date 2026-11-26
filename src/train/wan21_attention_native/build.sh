#!/usr/bin/env bash
set -euo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT="${1:?provide a node-local /tmp build directory}"
[[ "$OUTPUT" == /tmp/* ]] || { echo "build output must be under /tmp" >&2; exit 2; }
mkdir -p "$OUTPUT"
CXX="${WAN21_BISHENG_ROOT:?set the BiSheng compiler root}/bin/clang++"
INC="${WAN21_KBLAS_INCLUDE:?set the directory containing kblas.h}"
COMMON=(-std=c++20 -g -fno-omit-frame-pointer -fPIC -shared -fopenmp -pthread
        -march=armv9-a+sme+bf16+sve -Wl,-Bsymbolic -DRING_ATTN_STANDARD_KBLAS
        -I"$INC" -I"$D")
LIBS=("-Wl,-rpath,$WAN21_BISHENG_ROOT/lib"
      --rtlib=compiler-rt -lnuma -lm)
"$CXX" "${COMMON[@]}" -O2 "$D/component_forward.cpp" "$D/forward.cpp" "$D/rope.cpp" "$D/rmsnorm.cpp" \
  "$D/wan21_gemm_thread_pool_global.cpp" -o "$OUTPUT/forward.so" "${LIBS[@]}"
"$CXX" "${COMMON[@]}" -O3 -mcpu=native "$D/component_backward.cpp" "$D/backward.cpp" \
  "$D/wan21_gemm_thread_pool_global.cpp" -o "$OUTPUT/backward.so" "${LIBS[@]}"
