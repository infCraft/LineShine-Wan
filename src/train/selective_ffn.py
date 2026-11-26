"""Native fixed-shape FFN with a standard PyTorch path for other inputs."""
import os
import torch
import torch.nn as nn
from src.train.sve_gelu import _library

if os.environ.get('WAN21_FFN_HBM', '0') == '1':
    from src.train.rank_hbm import bf16 as _allocate_bf16
else:
    def _allocate_bf16(shape, slot=None):
        return torch.empty(shape, dtype=torch.bfloat16)
class _RecomputeGeluFFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w1,b1,w2,b2,slot):
        xb=x.to(torch.bfloat16).contiguous()
        raw=_allocate_bf16((*x.shape[:-1],w1.shape[0]),slot=slot)
        torch.ops.xhpops_biasfix.kp_linear_forward(xb,w1,b1,raw)
        if slot == 0 and os.environ.get('WAN21_FFN_HBM_POOL') == '1':
            from src.train.rank_hbm import verify_residency
            verify_residency(raw)
        act=_allocate_bf16(raw.shape,slot=30)
        _library().wan21_gelu_bf16_forward(raw.data_ptr(),act.data_ptr(),raw.numel())
        y=torch.empty((*x.shape[:-1],w2.shape[0]),dtype=torch.bfloat16)
        torch.ops.xhpops_biasfix.kp_linear_forward(act,w2,b2,y)
        ctx.save_for_backward(xb,w1,w2,raw)
        ctx.input_dtype=x.dtype
        ctx.slot=slot
        ctx.has_b1=b1 is not None
        ctx.has_b2=b2 is not None
        return y
    @staticmethod
    def backward(ctx,dy):
        xb,w1,w2,raw=ctx.saved_tensors
        lib=_library()
        act=_allocate_bf16(raw.shape,slot=30)
        lib.wan21_gelu_bf16_forward(raw.data_ptr(),act.data_ptr(),raw.numel())
        dw2=torch.empty_like(w2)
        db2=torch.empty((w2.shape[0],) if ctx.has_b2 else (0,),dtype=torch.bfloat16)
        grad_raw=_allocate_bf16(raw.shape,slot=31)
        torch.ops.xhpops_biasfix.kp_linear_backward(dy.contiguous(),act,w2,grad_raw,dw2,db2)
        lib.wan21_gelu_bf16_backward(grad_raw.data_ptr(),raw.data_ptr(),grad_raw.data_ptr(),raw.numel())
        grad_x=torch.empty_like(xb)
        dw1=torch.empty_like(w1)
        db1=torch.empty((w1.shape[0],) if ctx.has_b1 else (0,),dtype=torch.bfloat16)
        torch.ops.xhpops_biasfix.kp_linear_backward(grad_raw,xb,w1,grad_x,dw1,db1)
        if ctx.input_dtype!=torch.bfloat16:grad_x=grad_x.to(ctx.input_dtype)
        return grad_x,dw1,db1 if ctx.has_b1 else None,dw2,db2 if ctx.has_b2 else None,None

def native_supported(module, x):
    parameters = (module[0].weight, module[0].bias, module[2].weight, module[2].bias)
    return (os.environ.get("WAN21_NATIVE_OPS") == "1"
            and bool(os.environ.get("WAN21_XH_LINEAR_OP"))
            and bool(os.environ.get("WAN21_SVE_GELU_LIB"))
            and x.device.type == "cpu" and x.shape == (1, 3328, 1536)
            and x.dtype in (torch.float32, torch.bfloat16)
            and module[0].weight.shape == (8960, 1536)
            and module[2].weight.shape == (1536, 8960)
            and all(p is None or (p.device.type == "cpu" and p.dtype == torch.bfloat16
                                  and p.is_contiguous()) for p in parameters))


def recompute(self, x):
    if not native_supported(self, x):
        return nn.Sequential.forward(self, x)
    return _RecomputeGeluFFN.apply(x, self[0].weight, self[0].bias,
                                 self[2].weight, self[2].bias,
                                 getattr(self, "_ffn_hbm_slot", None))
