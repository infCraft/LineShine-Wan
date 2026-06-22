"""Hybrid Muon + AdamW optimizer."""

from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    # Orthogonalize G via quintic Newton-Schulz iteration, run in bfloat16.
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    X = X / (X.norm() + 1e-7)  # ensure spectral norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Combined optimizer with Muon groups and auxiliary AdamW groups."""

    def __init__(self, param_groups):
        param_groups = list(param_groups)
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("param group must set use_muon")
            group["params"] = list(group["params"])
            if group["use_muon"]:
                for key in ("lr", "momentum"):
                    if key not in group:
                        raise ValueError(f"muon param group missing {key}")
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
                for p in group["params"]:
                    if p.ndim != 2:
                        raise ValueError("Muon parameters must be 2D tensors")
            else:
                for key in ("lr", "betas", "eps", "weight_decay"):
                    if key not in group:
                        raise ValueError(f"AdamW param group missing {key}")
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                lr = group["lr"]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, ns_steps)
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if weight_decay:
                        p.mul_(1 - lr * weight_decay)
                    p.add_(u, alpha=-lr * scale)
            else:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step = state["step"]
                    if weight_decay:
                        p.mul_(1 - lr * weight_decay)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                    p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        return loss
