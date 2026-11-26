"""Optional BF16 SVE2 GELU for the Wan feed-forward blocks."""
from __future__ import annotations

import ctypes
import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=1)
def _library() -> ctypes.CDLL:
    path = os.environ["WAN21_SVE_GELU_LIB"]
    lib = ctypes.CDLL(path)
    lib.wan21_gelu_bf16_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_int64]
    lib.wan21_gelu_bf16_forward.restype = None
    lib.wan21_gelu_bf16_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_void_p, ctypes.c_int64]
    lib.wan21_gelu_bf16_backward.restype = None
    print(f"WAN21_SVE_GELU_LOADED path={path}", flush=True)
    return lib


class _SVEGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _library().wan21_gelu_bf16_forward(x.data_ptr(), out.data_ptr(), x.numel())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        if grad_output.dtype != torch.bfloat16 or grad_output.device.type != "cpu":
            return torch.ops.aten.gelu_backward(grad_output, x, approximate="tanh")
        grad = grad_output.contiguous()
        out = torch.empty_like(x)
        _library().wan21_gelu_bf16_backward(
            grad.data_ptr(), x.data_ptr(), out.data_ptr(), x.numel())
        return out


class SVEGELU(nn.Module):
    """Match nn.GELU(approximate='tanh'); use SVE2 only for contiguous CPU BF16."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu" and x.dtype == torch.bfloat16 and x.is_contiguous():
            return _SVEGELUFunction.apply(x)
        return F.gelu(x, approximate="tanh")
