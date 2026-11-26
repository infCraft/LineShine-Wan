"""Fused Wan2.1 modulation BF16 output with exact logical gradients."""
import ctypes
import os
import torch

from functools import lru_cache

@lru_cache(maxsize=1)
def _library():
    lib = ctypes.CDLL(os.environ["WAN21_SVE_MODULATION_LIB"])
    bf16 = lib.wan21_modulate_to_bf16
    fp32 = lib.wan21_modulate_fp32_to_bf16
    backward = lib.wan21_modulate_backward
    bf16.argtypes = fp32.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
    backward.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3
    bf16.restype = fp32.restype = backward.restype = None
    return bf16, fp32, backward

class _Modulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,offset,scale):
        out=torch.empty(x.shape,dtype=torch.bfloat16,device=x.device)
        (_library()[0] if x.dtype==torch.bfloat16 else _library()[1])(x.data_ptr(),offset.data_ptr(),scale.data_ptr(),out.data_ptr(),
                x.numel()//x.size(-1),x.size(-1))
        ctx.save_for_backward(x,scale)
        return out

    @staticmethod
    def backward(ctx,grad):
        x,scale=ctx.saved_tensors
        g=grad.contiguous()
        if g.dtype!=torch.bfloat16:g=g.to(torch.bfloat16)
        dx=torch.empty_like(x)
        doffset=torch.empty_like(scale)
        dscale=torch.empty_like(scale)
        scratch=torch.empty((72,x.shape[-1]),dtype=torch.float32,device=x.device)
        _library()[2](g.data_ptr(),x.data_ptr(),scale.data_ptr(),dx.data_ptr(),
                         doffset.data_ptr(),dscale.data_ptr(),scratch.data_ptr(),
                         x.numel()//x.shape[-1],x.shape[-1],int(x.dtype==torch.bfloat16))
        return dx,doffset,dscale

def modulate(x, offset, scale):
    if (os.environ.get("WAN21_SVE_MODULATION_LIB") and x.device.type == "cpu"
            and torch.is_autocast_enabled("cpu")
            and torch.get_autocast_dtype("cpu") == torch.bfloat16
            and x.dtype in (torch.bfloat16, torch.float32) and x.is_contiguous()
            and offset.device.type == scale.device.type == "cpu"
            and offset.dtype == scale.dtype == torch.float32
            and offset.is_contiguous() and scale.is_contiguous()
            and offset.numel() == scale.numel() == x.size(-1)):
        return _Modulation.apply(x, offset, scale)
    return x.float() * (1 + scale) + offset
