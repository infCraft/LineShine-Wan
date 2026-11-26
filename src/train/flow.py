"""Flow-matching utilities aligned with Wan's flow scheduler."""

from __future__ import annotations

from typing import Sequence

import torch


def sample_logit_normal_sigmas(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    logits = torch.randn(batch_size, device=device, dtype=dtype, generator=generator) * std + mean
    return torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)


def expand_sigmas(sigmas: torch.Tensor, ndim: int) -> torch.Tensor:
    return sigmas.reshape(sigmas.shape[0], *([1] * (ndim - 1)))


def add_flow_noise(
    clean_latents: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = torch.randn(clean_latents.shape, device=clean_latents.device, dtype=clean_latents.dtype, generator=generator)
    sigma_view = expand_sigmas(sigmas.to(clean_latents.dtype), clean_latents.ndim)
    noisy = (1.0 - sigma_view) * clean_latents + sigma_view * eps
    target = eps - clean_latents
    timesteps = sigmas.to(torch.float32) * 1000.0
    return noisy, target, timesteps


def apply_cfg_dropout(
    contexts: Sequence[torch.Tensor],
    empty_context: torch.Tensor,
    *,
    dropout_prob: float,
    generator: torch.Generator | None = None,
) -> list[torch.Tensor]:
    if dropout_prob <= 0.0:
        return [ctx for ctx in contexts]
    device = contexts[0].device if contexts else empty_context.device
    draws = torch.rand(len(contexts), device=device, generator=generator)
    out: list[torch.Tensor] = []
    for ctx, draw in zip(contexts, draws):
        out.append(empty_context.to(device=ctx.device, dtype=ctx.dtype) if float(draw.item()) < dropout_prob else ctx)
    return out

