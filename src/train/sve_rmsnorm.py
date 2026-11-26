"""Wan BF16 RMSNorm with the original two BF16 roundings."""
import ctypes
import functools
import os
from pathlib import Path
import torch

@functools.lru_cache(maxsize=1)
def _library():
    path=os.environ.get('WAN21_RMSNORM_LIB')
    if path is None:
        path=str(Path(__file__).resolve().parent/'wan21_attention_native'/'forward.so')
    lib=ctypes.CDLL(path)
    f=lib.wan21_rmsnorm_bf16_forward
    b=lib.wan21_rmsnorm_bf16_backward
    f.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_int,ctypes.c_float]
    b.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_int]
    f.restype=b.restype=ctypes.c_int
    return f,b

class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,eps):
        rows=x.numel()//1536
        y=torch.empty_like(x)
        rstd=torch.empty((rows,),dtype=torch.float32,device=x.device)
        f,_=_library()
        code=f(x.data_ptr(),w.data_ptr(),y.data_ptr(),rstd.data_ptr(),rows,1536,eps)
        if code: raise RuntimeError(f'RMSNorm forward returned {code}')
        ctx.save_for_backward(x,w,rstd)
        return y

    @staticmethod
    def backward(ctx,grad):
        x,w,rstd=ctx.saved_tensors
        grad=grad.contiguous()
        if grad.dtype!=torch.bfloat16: grad=grad.to(torch.bfloat16)
        dx=torch.empty_like(x)
        dw=torch.empty_like(w)
        partial=torch.empty((36,1536),dtype=torch.float32,device=x.device)
        _,b=_library()
        code=b(x.data_ptr(),w.data_ptr(),grad.data_ptr(),rstd.data_ptr(),
               dx.data_ptr(),dw.data_ptr(),partial.data_ptr(),x.numel()//1536,1536)
        if code: raise RuntimeError(f'RMSNorm backward returned {code}')
        return dx,dw,None

def rms_norm(x,weight,eps):
    if (x.device.type!='cpu' or weight.device.type!='cpu' or
        x.dtype!=torch.bfloat16 or weight.dtype!=torch.bfloat16 or
        not x.is_contiguous() or not weight.is_contiguous() or
        x.shape[-1]!=1536 or weight.shape!=(1536,) or
        x.numel()//1536 not in (512,3328)):
        raise ValueError('RMSNorm native requires contiguous CPU BF16 [512/3328,1536]')
    return _RMSNorm.apply(x,weight,eps)
