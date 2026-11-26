"""Adapters around WanModel's list-based forward interface."""

from __future__ import annotations

import math
from typing import Sequence

import torch


def latent_seq_len(latent_shape: Sequence[int], patch_size: Sequence[int] = (1, 2, 2), sp_size: int = 1) -> int:
    if len(latent_shape) == 5:
        _, _, frames, height, width = latent_shape
    elif len(latent_shape) == 4:
        _, frames, height, width = latent_shape
    else:
        raise ValueError(f"bad latent shape {latent_shape}")
    seq = math.ceil((frames / patch_size[0]) * (height / patch_size[1]) * (width / patch_size[2]) / sp_size) * sp_size
    return int(seq)


def call_wan_model(
    model,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    contexts: Sequence[torch.Tensor],
    *,
    seq_len: int | None = None,
) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"latents must be [B,C,F,H,W], got {tuple(latents.shape)}")
    if len(contexts) != latents.shape[0]:
        raise ValueError(f"context count {len(contexts)} does not match batch {latents.shape[0]}")
    if seq_len is None:
        seq_len = latent_seq_len(latents.shape, getattr(model, "patch_size", (1, 2, 2)))
    outputs = model([x for x in latents], t=timesteps, context=list(contexts), seq_len=seq_len)
    return torch.stack(outputs, dim=0)

