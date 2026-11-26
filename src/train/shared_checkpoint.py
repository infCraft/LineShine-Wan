"""Atomic checkpoints for shared parameters and four AdamW owners."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from src.train.ckpt import (
    SCHEMA_VERSION, _collect_rng_state, _collect_data_state, _jsonable,
    _sha256_file, _replace_latest_symlink, validate_checkpoint,
)


def owner_signature(owners):
    return hashlib.sha256(json.dumps(owners, sort_keys=True).encode()).hexdigest()


def collective_errors(error):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(next(value for value in errors if value))


def save_shared_checkpoint(run_dir, step, model, optimizer, scheduler, owners, args, data_state):
    rank, world = dist.get_rank(), dist.get_world_size()
    path = Path(run_dir) / "checkpoints" / f"step_{step:08d}"
    temporary = path.with_name(f".{path.name}.tmp.{args.job_token}")
    rng = _collect_rng_state()
    data = _collect_data_state(data_state)
    error = None
    if rank == 0:
        try:
            if path.exists():
                raise FileExistsError(f"checkpoint already exists: {path}")
            temporary.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            error = str(exc)
    collective_errors(error)
    try:
        error = None
        try:
            if rank < 4:
                torch.save(optimizer.state_dict(), temporary / f"owner_{rank:02d}.pt")
            if rank == 0:
                save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()},
                          str(temporary / "model.safetensors"))
                torch.save({
                    "schema_version": SCHEMA_VERSION, "parallel_mode": "shared_hierarchical",
                    "step": step, "world_size": world, "owners_sha256": owner_signature(owners),
                    "optimizer": None, "optimizer_owners": 4, "scheduler": scheduler.state_dict(),
                    "rng_by_rank": rng, "data_state_by_rank": data,
                    "args": _jsonable(vars(args)),
                }, temporary / "trainer.pt")
        except Exception as exc:
            error = f"checkpoint payload write: {type(exc).__name__}: {exc}"
        collective_errors(error)
        error = None
        if rank == 0:
            try:
                names = {"model": "model.safetensors", "trainer": "trainer.pt"}
                names.update({f"owner_{i}": f"owner_{i:02d}.pt" for i in range(4)})
                files = {key: {"name": name, "bytes": (temporary / name).stat().st_size,
                               "sha256": _sha256_file(temporary / name)} for key, name in names.items()}
                manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "step": step,
                            "world_size": world, "parallel_mode": "shared_hierarchical", "files": files}
                (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                with (temporary / "COMPLETED").open("x") as handle:
                    handle.write("manifest_sha256=" + _sha256_file(temporary / "manifest.json") + "\n")
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, path)
                _replace_latest_symlink(path)
            except Exception as exc:
                error = f"checkpoint publication: {type(exc).__name__}: {exc}"
        collective_errors(error)
    finally:
        if rank == 0 and temporary.exists():
            shutil.rmtree(temporary)
    return path


def read_shared_checkpoint(path: Path, owners, args):
    rank, world = dist.get_rank(), dist.get_world_size()
    record = [None]
    if rank == 0:
        try:
            manifest = validate_checkpoint(path)
            if manifest.get("parallel_mode") != "shared_hierarchical":
                raise ValueError("use --init-checkpoint for weights-only migration from a DDP checkpoint")
            if manifest["world_size"] != world:
                raise ValueError("shared checkpoint world size differs from the current job")
            for i in range(4):
                item = manifest["files"][f"owner_{i}"]
                expected = f"owner_{i:02d}.pt"
                if item["name"] != expected or _sha256_file(path / expected) != item["sha256"]:
                    raise ValueError(f"invalid optimizer owner file {expected}")
            trainer = torch.load(path / "trainer.pt", map_location="cpu", weights_only=False)
            if trainer["owners_sha256"] != owner_signature(owners):
                raise ValueError("parameter ownership changed since the checkpoint")
            saved = trainer["args"]
            for key in ("lr", "min_lr", "lr_total_steps", "warmup_steps", "batch_size", "grad_accum",
                        "seed", "cfg_dropout", "weight_decay", "clip_grad_norm"):
                if saved[key] != getattr(args, key):
                    raise ValueError(f"resume changes training setting {key}")
            for key in ("stored_precision", "text_precision"):
                if saved.get(key, "fp32") != getattr(args, key, "fp32"):
                    raise ValueError(f"resume changes {key}")
            record[0] = {"trainer": trainer, "error": None}
        except Exception as exc:
            record[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(record, src=0)
    if record[0]["error"]:
        raise RuntimeError(record[0]["error"])
    return record[0]["trainer"]


def restore_shared_state(path, trainer, optimizer, scheduler, online, local_rank):
    rank = dist.get_rank()
    error = None
    try:
        optimizer.load_state_dict(torch.load(Path(path) / f"owner_{local_rank:02d}.pt",
                                              map_location="cpu", weights_only=False))
        scheduler.load_state_dict(trainer["scheduler"])
        if online is not None:
            online.load_state_dict(trainer["data_state_by_rank"][rank])
        torch.set_rng_state(trainer["rng_by_rank"][rank]["cpu"])
    except Exception as exc:
        error = f"restore owner {local_rank}: {type(exc).__name__}: {exc}"
    collective_errors(error)
