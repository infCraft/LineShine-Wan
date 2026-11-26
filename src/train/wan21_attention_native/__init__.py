"""Stride-safe CPU BF16 autograd adapter for the fixed Wan2.1 attention shapes."""
import ctypes
import math
import os
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent
_LIBRARIES = None


def _native():
    global _LIBRARIES
    if _LIBRARIES is None:
        f_lib = ctypes.CDLL(str(_ROOT / "forward.so"))
        b_lib = ctypes.CDLL(str(_ROOT / "backward.so"))
        forward = f_lib.wan21_attention_forward_q128_hbm
        backward = b_lib.wan21_attention_backward_batched_q32_hbm_prepack
        forward.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4
        backward.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 4
        forward.restype = backward.restype = ctypes.c_int
        _LIBRARIES = (f_lib, b_lib, forward, backward)
    return _LIBRARIES[2:]


def _layout(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
        raise TypeError("attention tensors must be CPU BF16")
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-1] != 128:
        raise ValueError("expected [1, T, 12, 128] or [1, 12, T, 128]")
    if tensor.shape[2] == 12:
        return "BTHD", tensor.shape[1]
    if tensor.shape[1] == 12:
        return "BHTD", tensor.shape[2]
    raise ValueError("expected exactly 12 heads")


def _validate(q, k, v):
    q_layout, tq = _layout(q)
    k_layout, tk = _layout(k)
    v_layout, tv = _layout(v)
    if tq != 3328 or tk not in (512, 3328) or tk != tv or k_layout != v_layout:
        raise ValueError("only Tq=3328, Tk=3328/512 with consistent K/V layouts are supported")
    return q_layout, k_layout, tk


def _token_major(tensor):
    layout, seq = _layout(tensor)
    # A BHTD view made by transpose(1,2) of BTHD token-major storage has
    # strides [*,128,1536,1] and needs no copy. Check actual strides, not just shape.
    logical = tensor if layout == "BTHD" else tensor.transpose(1, 2)
    if logical.stride()[1:] != (1536, 128, 1):
        logical = logical.contiguous()
    return logical.view(seq, 12, 128)


def _restore_layout(tensor, layout):
    token_major = tensor.view(1, tensor.shape[0], 12, 128)
    return token_major if layout == "BTHD" else token_major.transpose(1, 2)


class _WanAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q_layout, k_layout, tk = _validate(q, k, v)
        q0, k0, v0 = (_token_major(x) for x in (q, k, v))
        tq, h, d = 3328, 12, 128
        out = torch.empty_like(q0)
        row_max = torch.empty((h, tq), dtype=torch.float32)
        row_sum = torch.empty_like(row_max)
        f, _ = _native()
        rc = f(*(x.data_ptr() for x in (q0, k0, v0, out, row_max, row_sum)), tq, tk, h, d)
        if rc:
            raise RuntimeError(f"Wan attention forward returned {rc}")
        ctx.save_for_backward(q0, k0, v0, out, row_max, row_sum)
        ctx.layouts = (q_layout, k_layout)
        return _restore_layout(out, q_layout)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, row_max, row_sum = ctx.saved_tensors
        if _layout(grad_out)[0] != ctx.layouts[0]:
            raise ValueError("output gradient layout changed")
        go = _token_major(grad_out)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        _, b = _native()
        tq, tk, h, d = 3328, k.shape[0], 12, 128
        rc = b(*(x.data_ptr() for x in (q, k, v, out, go, row_max, row_sum, dq, dk, dv)), tq, tk, h, d)
        if rc:
            raise RuntimeError(f"Wan attention backward returned {rc}")
        return _restore_layout(dq, ctx.layouts[0]), _restore_layout(dk, ctx.layouts[1]), _restore_layout(dv, ctx.layouts[1])


def attention(q, k, v, *, dropout_p=0.0, is_causal=False, scale=None):
    """Batch-one CPU BF16 attention. Accept token-major tensors and their BHTD views."""
    _validate(q, k, v)
    if dropout_p != 0 or is_causal or (scale is not None and not math.isclose(scale, 1 / math.sqrt(128), rel_tol=1e-7)):
        raise ValueError("this optimized kernel requires dropout=0, causal=False, scale=1/sqrt(128)")
    os.environ.setdefault("WAN21_ATTN_HEAD_SPREAD", "2")
    return _WanAttention.apply(q, k, v)


def _rope_native():
    _native()
    f_lib = _LIBRARIES[0]
    function = f_lib.wan21_rope_pair_bf16
    function.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 7
    function.restype = ctypes.c_int
    return function


def _fused_backward():
    _native()
    function = _LIBRARIES[1].wan21_attention_backward_rope_fused
    function.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 7
    function.restype = ctypes.c_int
    return function


def _rope_pair(q, k, freqs, grid, inverse=False):
    result_q, result_k = torch.empty_like(q), torch.empty_like(k)
    f, h, w = (int(x) for x in grid)
    rc = _rope_native()(*[x.data_ptr() for x in (q, k, result_q, result_k, freqs)],
                        f, h, w, q.shape[0], 12, 128, int(inverse))
    if rc:
        raise RuntimeError(f"Wan RoPE kernel returned {rc}")
    return result_q, result_k


class _WanAttentionRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, grid_sizes, freqs):
        q_layout, k_layout, tk = _validate(q, k, v)
        if tk != 3328 or q_layout != k_layout:
            raise ValueError("fused RoPE only supports Wan self attention")
        if grid_sizes.shape != (1, 3) or freqs.shape[1] != 64 or not freqs.is_contiguous():
            raise ValueError("unexpected Wan RoPE grid/frequency shape or strides")
        if freqs.device.type != 'cpu' or freqs.dtype != torch.complex128:
            raise TypeError("RoPE frequencies must be CPU complex128")
        grid = tuple(int(x) for x in grid_sizes[0])
        q0, k0, v0 = (_token_major(x) for x in (q, k, v))
        rq, rk = _rope_pair(q0, k0, freqs, grid)
        tq, h, d = 3328, 12, 128
        out = torch.empty_like(q0)
        row_max = torch.empty((h, tq), dtype=torch.float32)
        row_sum = torch.empty_like(row_max)
        f, _ = _native()
        rc = f(*(x.data_ptr() for x in (rq, rk, v0, out, row_max, row_sum)), tq, tk, h, d)
        if rc:
            raise RuntimeError(f"Wan attention forward returned {rc}")
        ctx.save_for_backward(rq, rk, v0, out, row_max, row_sum, freqs)
        ctx.layouts = (q_layout, k_layout)
        ctx.grid = grid
        return _restore_layout(out, q_layout)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, row_max, row_sum, freqs = ctx.saved_tensors
        go = _token_major(grad_out)
        dq, dk, dv = (torch.empty_like(x) for x in (q, k, v))
        b = _fused_backward()
        tq, tk, h, d = 3328, k.shape[0], 12, 128
        rc = b(*(x.data_ptr() for x in (q, k, v, out, go, row_max, row_sum, dq, dk, dv, freqs)),
               *ctx.grid, tq, tk, h, d)
        if rc:
            raise RuntimeError(f"Wan attention fused backward returned {rc}")
        return (_restore_layout(dq, ctx.layouts[0]),
                _restore_layout(dk, ctx.layouts[1]),
                _restore_layout(dv, ctx.layouts[1]), None, None)


def rope_attention(q, k, v, grid_sizes, freqs, *, dropout_p=0.0, is_causal=False):
    """Wan2.1 batch-one self attention with BF16 RoPE folded into the operator."""
    if dropout_p != 0 or is_causal:
        raise ValueError("fused RoPE supports only full, noncausal, no-dropout attention")
    os.environ.setdefault("WAN21_ATTN_HEAD_SPREAD", "2")
    return _WanAttentionRope.apply(q, k, v, grid_sizes, freqs)
