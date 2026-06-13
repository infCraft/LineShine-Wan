"""Checkpoint helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from src.common import ensure_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:08d}.pt"


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = []
    for path in ckpt_dir.glob("step_*.pt"):
        match = re.search(r"step_(\d+)\.pt$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    model,
    optimizer,
    scheduler=None,
    scaler: Any | None,
    args: Any,
) -> Path:
    path = checkpoint_path(run_dir, step)
    ensure_dir(path.parent)
    module = model.module if hasattr(model, "module") else model
    raw_args = _jsonable(vars(args) if hasattr(args, "__dict__") else args)
    torch.save(
        {
            "step": step,
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": raw_args,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )
    latest = path.parent / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> int:
    data = torch.load(path, map_location=map_location, weights_only=False)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(data["model"])
    if optimizer is not None and data.get("optimizer") is not None:
        optimizer.load_state_dict(data["optimizer"])
    if scheduler is not None and data.get("scheduler") is not None:
        scheduler.load_state_dict(data["scheduler"])
    if scaler is not None and data.get("scaler") is not None:
        scaler.load_state_dict(data["scaler"])
    if data.get("rng_cpu") is not None:
        torch.set_rng_state(data["rng_cpu"].cpu())
    if torch.cuda.is_available() and data.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in data["rng_cuda"]])
    return int(data.get("step", 0))
