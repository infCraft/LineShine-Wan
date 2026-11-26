#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
die() { echo "$*" >&2; exit 1; }
: "${HMPI_ROOT:?}" "${KBLAS_SME_MULTI_DIR:?}" "${HPCKIT_GCC_ROOT:?}"
EXPECTED_TORCH_PREFIX="${WAN21_TORCH_PREFIX:-${VIRTUAL_ENV:?}}"
EXPECTED_TORCH_VERSION="${WAN21_EXPECT_TORCH_VERSION:?}"
WHEEL_CHECKSUMS="${WAN21_WHEEL_CHECKSUMS:?set the absolute path of the trusted wheel checksum list}"
(
    cd "${WAN21_WHEEL_ROOT:?set the base for paths in the checksum list}"
    sha256sum -c "$WHEEL_CHECKSUMS"
)

LIBTORCH_CPU="$("${WAN21_PYTHON:?}" - "$EXPECTED_TORCH_VERSION" <<'PY'
from pathlib import Path
import sys
import torch

assert torch.__version__ == sys.argv[1], (torch.__version__, sys.argv[1])
path = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
if not path.is_file():
    raise SystemExit(f"missing {path}")
print(path)
PY
)"
[[ "$LIBTORCH_CPU" == "$EXPECTED_TORCH_PREFIX"/* ]] || \
    die "Torch 未从已验收的 hmpi-kblas 载荷加载：$LIBTORCH_CPU"
READELF_OUTPUT="$(readelf -d "$LIBTORCH_CPU")"
LDD_OUTPUT="$(ldd "$LIBTORCH_CPU")"
if printf '%s\n' "$LDD_OUTPUT" | grep -F '=> not found' >/dev/null; then
    die "ldd 存在未解析的动态依赖"
fi
for needed in 'libmpi\.so\.40' 'libkblas\.so' 'libgomp\.so\.1'; do
    printf '%s\n' "$READELF_OUTPUT" | grep -Eq "NEEDED.*\[$needed" || \
        die "readelf 未发现直接依赖：$needed"
done

resolve_dependency() {
    local pattern="$1"
    local expected_root="$2"
    local label="$3"
    local path
    local resolved
    path="$(printf '%s\n' "$LDD_OUTPUT" | awk -v pattern="$pattern" '$1 ~ pattern && $2 == "=>" {print $3; exit}')"
    [[ -n "$path" && "$path" != "not" ]] || die "ldd 未解析出 $label"
    resolved="$(realpath -e "$path")"
    [[ "$resolved" == "$expected_root"/* ]] || die "$label 解析到错误路径：$resolved"
    printf '%s=%s\n' "$label" "$resolved"
}

MPI_PATH="$(resolve_dependency '^libmpi[.]so[.]40' "$HMPI_ROOT/lib" mpi)"
KBLAS_PATH="$(resolve_dependency '^libkblas[.]so' "$KBLAS_SME_MULTI_DIR" kblas)"
GOMP_PATH="$(resolve_dependency '^libgomp[.]so[.]1' "$HPCKIT_GCC_ROOT/lib64" libgomp)"
[[ "$KBLAS_PATH" == "kblas=$KBLAS_SME_MULTI_DIR/libkblas.so.25.2.1" ]] || \
    die "KBLAS 不是固定的 25.2.1 SME multi 库：$KBLAS_PATH"
KBLAS_LIBRARY="${KBLAS_PATH#kblas=}"
for symbol in sgemm_ sbgemm_ dgemm_; do
    readelf -Ws "$LIBTORCH_CPU" | \
        awk -v symbol="$symbol" '$7 == "UND" && $8 == symbol {found=1} END {exit !found}' || \
        die "libtorch_cpu.so 没有引用 KBLAS GEMM 符号：$symbol"
    readelf -Ws "$KBLAS_LIBRARY" | \
        awk -v symbol="$symbol" '$7 != "UND" && $8 == symbol {found=1} END {exit !found}' || \
        die "固定 KBLAS 没有导出 GEMM 符号：$symbol"
done
if printf '%s\n' "$LDD_OUTPUT" | grep -Ei 'openblas|/bisheng/|(^|[[:space:]/])libomp\.so' >/dev/null; then
    die "动态依赖混入 OpenBLAS、BiSheng 或 libomp"
fi

TMP_PARENT="/tmp"
TMP_DIR="$(mktemp -d "$TMP_PARENT/shine-wan21-links.XXXXXX")"
cleanup() {
    if [[ -n "${TMP_DIR:-}" && "$TMP_DIR" == "$TMP_PARENT/shine-wan21-links."* ]]; then
        rm -r -- "$TMP_DIR"
    fi
}
trap cleanup EXIT INT TERM
LD_DEBUG=bindings "$WAN21_PYTHON" "$SCRIPT_DIR/wan21_kblas.py" \
    > "$TMP_DIR/matmul.out" 2> "$TMP_DIR/bindings.log"
cat "$TMP_DIR/matmul.out"
for symbol in sgemm_ sbgemm_; do
    grep -m1 -E "libtorch_cpu[.]so.*libkblas[.]so[.]25[.]2[.]1.*symbol [\`']$symbol" \
        "$TMP_DIR/bindings.log" || die "运行时未观察到 $symbol 绑定到 KBLAS"
done

if [[ "${1:-}" == "--skip-pip-check" ]]; then
    printf 'Skipping environment-wide pip check; caller must validate its application imports.\n'
else
    "$WAN21_PYTHON" -m pip check
fi
printf '%s\n' "$MPI_PATH" "$KBLAS_PATH" "$GOMP_PATH"
printf 'WAN21_LINKS_OK torch=%s\n' "$LIBTORCH_CPU"
