#!/usr/bin/env bash
set -euo pipefail

die() { echo "$*" >&2; exit 1; }
: "${HMPI_ROOT:?}" "${HPCKIT_GCC_ROOT:?}" "${OPENBLAS_LIBRARY_SHA256:?}"

[[ $# -eq 1 ]] || die "用法：$0 <isolated-python-prefix>"
TEST_PREFIX="$(cd "$1" && pwd -P)"
TEST_PYTHON="$TEST_PREFIX/bin/python"
VENDOR_PREFIX="${WAN21_OPENBLAS_PREFIX:?}"
OPENBLAS_LIBRARY="$VENDOR_PREFIX/lib/${OPENBLAS_LIBRARY_REALNAME:?}"

[[ -x "$TEST_PYTHON" && -f "$OPENBLAS_LIBRARY" ]] || die 'Missing Python or OpenBLAS library'

LIBTORCH_CPU="$("$TEST_PYTHON" - <<'PY'
from pathlib import Path
import torch

path = Path(torch.__file__).resolve().parent / "lib" / "libtorch_cpu.so"
if not path.is_file():
    raise SystemExit(f"missing {path}")
print(path)
PY
)"

printf '%s\n' '[readelf -d libtorch_cpu.so]'
READELF_OUTPUT="$(readelf -d "$LIBTORCH_CPU")"
printf '%s\n' "$READELF_OUTPUT"
printf '%s\n' '[ldd libtorch_cpu.so]'
LDD_OUTPUT="$(ldd "$LIBTORCH_CPU")"
printf '%s\n' "$LDD_OUTPUT"

if printf '%s\n' "$LDD_OUTPUT" | grep -F '=> not found' >/dev/null; then
    die "ldd 存在未解析的动态依赖"
fi

for needed in 'libmpi[.]so[.]40' 'libopenblas[.]so[.]0' 'libgomp[.]so[.]1'; do
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
    [[ "$resolved" == "$expected_root"/* ]] || \
        die "$label 解析到错误路径：$resolved"
    printf '%s=%s\n' "$label" "$resolved"
}

MPI_PATH="$(resolve_dependency '^libmpi[.]so[.]40' "$HMPI_ROOT/lib" mpi)"
OPENBLAS_PATH="$(resolve_dependency '^libopenblas[.]so[.]0' "$VENDOR_PREFIX/lib" openblas)"
GOMP_PATH="$(resolve_dependency '^libgomp[.]so[.]1' "$HPCKIT_GCC_ROOT/lib64" libgomp)"

[[ "$OPENBLAS_PATH" == "openblas=$OPENBLAS_LIBRARY" ]] || \
    die "OpenBLAS 不是固定 vendor 主库：$OPENBLAS_PATH"
[[ "$(sha256sum "$OPENBLAS_LIBRARY" | awk '{print $1}')" == "$OPENBLAS_LIBRARY_SHA256" ]] || \
    die "OpenBLAS 主库 SHA256 不匹配"

for symbol in sgemm_ dgemm_; do
    readelf -Ws "$LIBTORCH_CPU" | \
        awk -v symbol="$symbol" '$7 == "UND" && $8 == symbol {found=1} END {exit !found}' || \
        die "libtorch_cpu.so 没有引用 OpenBLAS GEMM 符号：$symbol"
done
for symbol in sgemm_ dgemm_ cheev_ cgesdd_ sgesv_; do
    readelf -Ws "$OPENBLAS_LIBRARY" | \
        awk -v symbol="$symbol" '$7 != "UND" && $8 == symbol {found=1} END {exit !found}' || \
        die "固定 OpenBLAS 没有导出符号：$symbol"
done

if printf '%s\n' "$LDD_OUTPUT" | grep -Ei 'kblas|/bisheng/|(^|[[:space:]/])libomp[.]so' >/dev/null; then
    die "动态依赖混入 KBLAS、BiSheng 或 libomp"
fi

OPENBLAS_LDD="$(ldd "$OPENBLAS_LIBRARY")"
printf '%s\n' '[ldd vendor libopenblas.so.0]' "$OPENBLAS_LDD"
printf '%s\n' "$OPENBLAS_LDD" | grep -F "$HPCKIT_GCC_ROOT/lib64/libgomp.so.1" >/dev/null || \
    die "vendor OpenBLAS 未解析到 HPCKit libgomp"
if printf '%s\n' "$OPENBLAS_LDD" | grep -Ei '=> not found|libomp[.]so|/bisheng/' >/dev/null; then
    die "vendor OpenBLAS 存在错误动态依赖"
fi

printf '%s\n' "$MPI_PATH" "$OPENBLAS_PATH" "$GOMP_PATH"
printf 'OpenBLAS GEMM/LAPACK 符号检查通过\n'
printf '动态链接检查通过：%s\n' "$LIBTORCH_CPU"

