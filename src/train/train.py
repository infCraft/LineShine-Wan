"""Optimizer, schedule and checkpoint retention helpers for online training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.train.ckpt import checkpoint_step


def param_groups(model, weight_decay: float):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "modulation" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_lambda(args: argparse.Namespace):
    total_steps = args.lr_total_steps or args.max_steps

    def fn(step: int) -> float:
        if step < args.warmup_steps:
            return max(1, step + 1) / max(1, args.warmup_steps)
        if total_steps <= args.warmup_steps:
            return 1.0
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    return fn


def parse_keep_checkpoint_steps(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part) for part in value.replace(",", " ").split() if part}


def prune_checkpoints(run_dir: Path, *, keep_last: int, keep_steps: set[int]) -> None:
    if keep_last <= 0 and not keep_steps:
        return
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return
    checkpoints: list[tuple[int, Path]] = []
    for path in ckpt_dir.glob("step_*"):
        if path.is_dir() and not (path / "COMPLETED").is_file():
            continue
        try:
            checkpoints.append((checkpoint_step(path), path))
        except ValueError:
            continue
    checkpoints.sort()
    if not checkpoints:
        return
    keep = set(keep_steps)
    if keep_last > 0:
        keep.update(step for step, _ in checkpoints[-keep_last:])
    keep.add(checkpoints[-1][0])
    for step, path in checkpoints:
        if step not in keep and path.exists():
            if path.is_dir():
                for child in path.iterdir():
                    child.unlink()
                path.rmdir()
            else:
                path.unlink()
