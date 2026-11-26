"""Fused SVE2 broadcast residual with BF16 branch and FP32 output."""
from __future__ import annotations
import ctypes
import functools
import os
import torch


@functools.lru_cache(maxsize=1)
def _library():
    path = os.environ['WAN21_SVE_RESIDUAL_LIB']
    lib = ctypes.CDLL(path)
    lib.wan21_residual_bf16_forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
    lib.wan21_residual_bf16_backward.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 2
    lib.wan21_residual_omp_threads.restype = ctypes.c_int
    if hasattr(lib,'wan21_residual_bf16_input_forward'):
        lib.wan21_residual_bf16_input_forward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2
        lib.wan21_residual_bf16_input_forward.restype = None
        lib.wan21_residual_bf16_input_backward.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int64] * 2
        lib.wan21_residual_bf16_input_backward.restype = None
    print(f'WAN21_SVE_RESIDUAL_LOADED path={path}', flush=True)
    return lib


class _Residual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gate):
        rows = x.numel() // x.size(-1)
        cols = x.size(-1)
        result = torch.empty_like(x)
        _library().wan21_residual_bf16_forward(
            x.data_ptr(), branch.data_ptr(), gate.data_ptr(),
            result.data_ptr(), rows, cols)
        ctx.save_for_backward(branch, gate)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        branch, gate = ctx.saved_tensors
        grad = grad_output.contiguous()
        rows = branch.numel() // branch.size(-1)
        cols = branch.size(-1)
        if grad.dtype != torch.float32:
            grad = grad.float()
        dy = torch.empty_like(branch)
        de = torch.empty_like(gate)
        scratch = torch.empty((_library().wan21_residual_omp_threads(), cols),
                              dtype=torch.float32, device=branch.device)
        _library().wan21_residual_bf16_backward(
            grad.data_ptr(), branch.data_ptr(), gate.data_ptr(),
            dy.data_ptr(), de.data_ptr(), scratch.data_ptr(), rows, cols)
        return grad_output, dy, de


class _ResidualBF16Input(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, branch, gate):
        rows = x.numel() // x.size(-1)
        cols = x.size(-1)
        output = torch.empty(x.shape,dtype=torch.float32,device=x.device)
        _library().wan21_residual_bf16_input_forward(
            x.data_ptr(),branch.data_ptr(),gate.data_ptr(),output.data_ptr(),rows,cols)
        ctx.save_for_backward(branch,gate)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        branch,gate = ctx.saved_tensors
        grad=grad_output.contiguous()
        if grad.dtype != torch.float32:grad=grad.float()
        dx=torch.empty_like(branch)
        dy=torch.empty_like(branch)
        de=torch.empty_like(gate)
        scratch=torch.empty((_library().wan21_residual_omp_threads(),branch.size(-1)),
                            dtype=torch.float32,device=branch.device)
        _library().wan21_residual_bf16_input_backward(
            grad.data_ptr(),branch.data_ptr(),gate.data_ptr(),
            dx.data_ptr(),dy.data_ptr(),de.data_ptr(),scratch.data_ptr(),
            branch.numel()//branch.size(-1),branch.size(-1))
        return dx,dy,de


def residual(x, branch, gate):
    if (os.environ.get('WAN21_SVE_RESIDUAL_LIB') and x.device.type == 'cpu'
            and x.is_contiguous() and branch.dtype == torch.bfloat16
            and branch.is_contiguous() and gate.dtype == torch.float32
            and gate.is_contiguous() and x.shape == branch.shape
            and gate.numel() == x.size(-1)):
        if x.dtype == torch.float32:
            return _Residual.apply(x, branch, gate)
        if (x.dtype == torch.bfloat16
                and hasattr(_library(),'wan21_residual_bf16_input_forward')):
            return _ResidualBF16Input.apply(x, branch, gate)
    return x + branch * gate
