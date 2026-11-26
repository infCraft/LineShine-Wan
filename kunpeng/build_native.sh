#!/usr/bin/env bash
# Build on an allocated compute node; install only the finished libraries.
set -euo pipefail
grep -qi '0xd22' /proc/cpuinfo || { echo 'Build on a 72F5 compute node' >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd -P)
source "${WAN21_ENV_ACTIVATE:?set the runtime activation script}"
: "${WAN21_BISHENG_ROOT:?set the BiSheng compiler root}" "${WAN21_KBLAS_INCLUDE:?set the kblas.h directory}"
[[ -x "$WAN21_BISHENG_ROOT/bin/clang++" && -r "$WAN21_KBLAS_INCLUDE/kblas.h" ]] || { echo 'Invalid BiSheng or KBLAS development paths' >&2; exit 2; }
: "${WAN21_PYTHON:?activation must export the runtime Python executable}"
export WAN21_BISHENG_ROOT WAN21_KBLAS_INCLUDE
work=$(mktemp -d /tmp/lineshine-native.XXXXXXXX)
trap 'rm -r -- "$work"' EXIT
export TMPDIR="$work" TORCH_EXTENSIONS_DIR="$work/extensions" PYTHONDONTWRITEBYTECODE=1
gemm="$root/kunpeng/wan21_gemm_library"
attention="$root/src/train/wan21_attention_native"
cp -R "$gemm/src" "$work/gemm"
ln -s "${WAN21_LIBSTDCXX:-/usr/lib64/libstdc++.so.6}" "$work/gemm/libstdc++.so"
make -C "$work/gemm" -j4 CXX="$WAN21_BISHENG_ROOT/bin/clang++" lib
"$WAN21_PYTHON" "$root/src/train/build_xh_linear.py" --build-dir "$work/linear"
"$attention/build.sh" "$work/attention"
install -m 755 "$work/gemm/libsme_gemm.so" "$work/linear/xhpops_biasfix.so" "$gemm/"
install -m 755 "$work/attention/forward.so" "$work/attention/backward.so" "$attention/"
sha256sum "$gemm/libsme_gemm.so" "$gemm/xhpops_biasfix.so" "$attention/forward.so" "$attention/backward.so"
