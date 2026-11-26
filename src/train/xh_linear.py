"""Opt-in Wan BF16 Linear bridge, with native SME forward and backward."""
import atexit
import os
import types
import torch
import torch.nn.functional as F

_NATIVE_FORWARD={(3328,1536,1536),(512,1536,1536),
                 (3328,8960,1536),(3328,1536,8960)}

def load_native(path=None):
    torch.ops.load_library(path or os.environ['WAN21_XH_LINEAR_OP'])

def _supported(x,w,bias):
    if x.ndim < 2 or w.ndim!=2 or x.device.type!='cpu' or w.device.type!='cpu':
        return False
    if x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or not x.is_contiguous() or not w.is_contiguous():
        return False
    if w.size(1)!=x.size(-1) or (bias is not None and
        (bias.dtype!=torch.bfloat16 or not bias.is_contiguous() or bias.device.type!='cpu')):
        return False
    return (x.numel()//x.size(-1),w.size(0),w.size(1)) in _NATIVE_FORWARD

class _XhLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,bias):
        y=torch.empty((*x.shape[:-1],w.shape[0]),dtype=x.dtype,device=x.device)
        torch.ops.xhpops_biasfix.kp_linear_forward(x,w,bias,y)
        ctx.save_for_backward(x,w)
        ctx.has_bias=bias is not None
        return y
    @staticmethod
    def backward(ctx,dy):
        x,w=ctx.saved_tensors
        dx=torch.empty_like(x)
        dw=torch.empty_like(w)
        db=torch.empty((w.size(0),) if ctx.has_bias else (0,),dtype=w.dtype,device=w.device)
        torch.ops.xhpops_biasfix.kp_linear_backward(dy.contiguous(),x,w,dx,dw,db)
        return dx,dw,db if ctx.has_bias else None

_selected=0
_fallback=0

def _report():
    if os.environ.get('WAN21_XH_STATS'):
        print(f'XH_LINEAR_STATS selected={_selected} fallback={_fallback}',flush=True)
atexit.register(_report)

def _linear_module_forward(self,x):
    return xh_linear(x,self.weight,self.bias)

def install_on_model(model,library_path=None):
    """Replace module forwards after model construction, preserving parameters and state_dict."""
    load_native(library_path)
    count=0
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):
            module.forward=types.MethodType(_linear_module_forward,module)
            count+=1
    return count

def xh_linear(x,w,bias=None):
    global _selected,_fallback
    if not _supported(x, w, bias):
        # F.linear under CPU BF16 autocast casts FP32 activations to BF16.
        # Keep that cast in the autograd graph so dX returns as FP32.
        if (x.dtype == torch.float32 and x.device.type == 'cpu' and
                torch.is_autocast_enabled('cpu') and
                torch.get_autocast_dtype('cpu') == torch.bfloat16):
            x_bf16 = x.to(torch.bfloat16)
            if _supported(x_bf16, w, bias):
                _selected += 1
                return _XhLinear.apply(x_bf16, w, bias)
        _fallback += 1
        return F.linear(x, w, bias)
    _selected += 1
    return _XhLinear.apply(x, w, bias)
