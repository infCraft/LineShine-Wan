"""SVE2 BF16 LayerNorm with saved FP32 row statistics for CPU training."""
from __future__ import annotations

import ctypes
import functools
import os

import torch


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    path = os.environ["WAN21_SVE_LAYERNORM_LIB"]
    lib = ctypes.CDLL(path)
    lib.wan21_layernorm_bf16_forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_float]
    lib.wan21_layernorm_bf16_backward.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 2
    lib.wan21_layernorm_bf16_forward.restype = None
    lib.wan21_layernorm_bf16_backward.restype = None
    lib.wan21_layernorm_fp32_forward.argtypes = lib.wan21_layernorm_bf16_forward.argtypes
    lib.wan21_layernorm_fp32_backward.argtypes = lib.wan21_layernorm_bf16_backward.argtypes
    lib.wan21_layernorm_fp32_forward.restype = None
    lib.wan21_layernorm_fp32_backward.restype = None
    print(f"WAN21_SVE_LAYERNORM_LOADED path={path}", flush=True)
    return lib


class _FusedLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        *lead, columns = x.shape
        rows = x.numel() // columns
        output = torch.empty_like(x)
        means = torch.empty(rows, dtype=torch.float32, device=x.device)
        rstds = torch.empty_like(means)
        kernel = (_library().wan21_layernorm_fp32_forward if x.dtype == torch.float32
                  else _library().wan21_layernorm_bf16_forward)
        kernel(x.data_ptr(), output.data_ptr(), means.data_ptr(), rstds.data_ptr(),
               rows, columns, eps)
        ctx.save_for_backward(x, means, rstds)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, means, rstds = ctx.saved_tensors
        grad = grad_output.contiguous()
        if grad.dtype != x.dtype:
            grad = grad.to(x.dtype)
        output = torch.empty_like(x)
        kernel = (_library().wan21_layernorm_fp32_backward if x.dtype == torch.float32
                  else _library().wan21_layernorm_bf16_backward)
        kernel(grad.data_ptr(), x.data_ptr(), output.data_ptr(),
               means.data_ptr(), rstds.data_ptr(), x.numel() // x.size(-1),
               x.size(-1))
        return output, None


def layer_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type != "cpu" or x.dtype not in (torch.bfloat16, torch.float32) or not x.is_contiguous():
        return torch.nn.functional.layer_norm(x.float(), (x.size(-1),), eps=eps).type_as(x)
    return _FusedLayerNorm.apply(x, eps)
