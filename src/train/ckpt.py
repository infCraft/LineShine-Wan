"""Atomic, integrity-checked training checkpoints with legacy .pt support."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP

from src.common import ensure_dir, utc_now, write_json


SCHEMA_VERSION = 2
_STEP_RE = re.compile(r"step_(\d+)(?:\.pt)?$")


@dataclass(frozen=True)
class LoadedCheckpoint:
    step: int
    data_state: dict[str, Any] | None
    args: dict[str, Any] | None
    world_size: int


@dataclass(frozen=True)
class _TensorReference:
    index: int
    shape: tuple[int, ...]
    dtype: str


def unwrap_model(model):
    """Strip DDP and torch.compile wrappers from a module."""
    module = model
    while True:
        if isinstance(module, DDP):
            module = module.module
        elif hasattr(module, "_orig_mod"):
            module = module._orig_mod
        else:
            return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return repr(value)


def checkpoint_step(path: Path) -> int:
    match = _STEP_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer checkpoint step from {path}")
    return int(match.group(1))


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step_{step:08d}"


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step_*"):
        if not (path.is_dir() or path.suffix == ".pt"):
            continue
        try:
            step = checkpoint_step(path)
        except ValueError:
            continue
        if path.is_dir() and not (path / "COMPLETED").is_file():
            continue
        candidates.append((step, path))
    return max(candidates, key=lambda item: (item[0], item[1].is_dir()))[1] if candidates else None


def has_checkpoint_entries(run_dir: Path) -> bool:
    checkpoint_dir = run_dir / "checkpoints"
    return checkpoint_dir.is_dir() and any(checkpoint_dir.glob("step_*"))


def _replace_latest_symlink(path: Path) -> None:
    latest_name = "latest" if path.is_dir() else "latest.pt"
    latest = path.parent / latest_name
    temporary = latest.with_name(f".{latest.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(path.name)
    os.replace(temporary, latest)


def atomic_torch_save(obj: Any, path: Path) -> Path:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_rng_state() -> list[dict[str, torch.Tensor]]:
    local = {"cpu": torch.get_rng_state().cpu()}
    if torch.cuda.is_available():
        local["cuda"] = torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [local]

    backend = dist.get_backend()
    collective_device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    collected: dict[str, list[torch.Tensor]] = {}
    for name in sorted(local):
        value = local[name].to(collective_device)
        gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, value)
        collected[name] = [item.cpu() for item in gathered]
    return [
        {name: values[rank] for name, values in collected.items()}
        for rank in range(dist.get_world_size())
    ]


def _collect_data_state(value: dict[str, Any] | None) -> list[dict[str, Any] | None]:
    normalized = _jsonable(value)
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [normalized]
    backend = dist.get_backend()
    device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
    payload = json.dumps(normalized, sort_keys=True).encode("utf-8")
    length = torch.tensor([len(payload)], dtype=torch.int64, device=device)
    lengths = [torch.empty_like(length) for _ in range(dist.get_world_size())]
    dist.all_gather(lengths, length)
    maximum = max(int(item.item()) for item in lengths)
    encoded = torch.zeros(maximum, dtype=torch.uint8, device=device)
    if payload:
        encoded[: len(payload)] = torch.tensor(list(payload), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(encoded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, encoded)
    return [
        json.loads(bytes(item[: int(size.item())].cpu().tolist()).decode("utf-8"))
        for item, size in zip(gathered, lengths)
    ]


def _checkpoint_files(path: Path) -> dict[str, Path]:
    return {
        "model": path / "model.safetensors",
        "trainer": path / "trainer.pt",
        "manifest": path / "manifest.json",
        "completed": path / "COMPLETED",
    }


def _share_rank0_error(error: str | None, *, device: torch.device = torch.device("cpu")) -> str | None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return error
    objects = [error if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(objects, src=0, device=device)
    return objects[0]


def _raise_rank0_error(error: str | None, original: Exception | None = None) -> None:
    if error is None:
        return
    if original is not None:
        raise RuntimeError(error) from original
    raise RuntimeError(error)


def _model_state_for_save(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, tensor in unwrap_model(model).state_dict().items():
        state[name] = tensor.detach().to(device="cpu").contiguous()
    return state


def save_checkpoint(
    run_dir: Path,
    *,
    step: int,
    model,
    optimizer,
    scheduler=None,
    scaler: Any | None,
    args: Any,
    data_state: dict[str, Any] | None = None,
) -> Path:
    path = checkpoint_path(run_dir, step)
    ensure_dir(path.parent)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    collective_device = torch.device("cpu")
    if dist.is_initialized() and dist.get_backend() == "nccl":
        collective_device = torch.device("cuda", torch.cuda.current_device())

    existing = False
    existing_error: str | None = None
    existing_exception: Exception | None = None
    if rank == 0 and path.exists():
        try:
            manifest = validate_checkpoint(path)
            if int(manifest.get("step", -1)) != step:
                raise RuntimeError(f"checkpoint step mismatch: {path}")
            if int(manifest.get("world_size", -1)) != world_size:
                raise RuntimeError(
                    f"existing checkpoint world_size={manifest.get('world_size')} does not match {world_size}"
                )
            _replace_latest_symlink(path)
            existing = True
        except Exception as exc:  # noqa: BLE001 - propagate rank-0 filesystem failures to every rank.
            existing_exception = exc
            existing_error = f"rank 0 could not reuse checkpoint {path}: {type(exc).__name__}: {exc}"
    if dist.is_initialized() and world_size > 1:
        state = [{"existing": existing, "error": existing_error} if rank == 0 else None]
        dist.broadcast_object_list(state, src=0, device=collective_device)
        existing = bool(state[0]["existing"])
        existing_error = state[0]["error"]
    _raise_rank0_error(existing_error, existing_exception)
    if existing:
        return path

    rng_by_rank = _collect_rng_state()
    data_state_by_rank = _collect_data_state(data_state)

    publish_error: str | None = None
    publish_exception: Exception | None = None
    if rank == 0:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            if temporary.exists():
                raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
            temporary.mkdir()
            files = _checkpoint_files(temporary)
            save_file(_model_state_for_save(model), str(files["model"]))
            trainer_payload = {
                "schema_version": SCHEMA_VERSION,
                "step": step,
                "optimizer": optimizer.state_dict() if optimizer is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": _jsonable(vars(args) if hasattr(args, "__dict__") else args),
                "data_state_by_rank": data_state_by_rank,
                "rng_by_rank": rng_by_rank,
                "world_size": world_size,
            }
            torch.save(trainer_payload, files["trainer"])
            payload_records = {
                name: {
                    "name": file_path.name,
                    "bytes": file_path.stat().st_size,
                    "sha256": _sha256_file(file_path),
                }
                for name, file_path in (("model", files["model"]), ("trainer", files["trainer"]))
            }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "created_at": utc_now(),
                "step": step,
                "world_size": world_size,
                "files": payload_records,
            }
            write_json(files["manifest"], manifest)
            manifest_sha256 = _sha256_file(files["manifest"])
            with files["completed"].open("x", encoding="ascii") as handle:
                handle.write(f"manifest_sha256={manifest_sha256}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            _replace_latest_symlink(path)
        except Exception as exc:  # noqa: BLE001 - every rank must leave the collective on failure.
            publish_exception = exc
            publish_error = f"rank 0 could not publish checkpoint {path}: {type(exc).__name__}: {exc}"
        finally:
            if temporary.exists():
                for child in temporary.iterdir():
                    child.unlink()
                temporary.rmdir()
    publish_error = _share_rank0_error(publish_error, device=collective_device)
    _raise_rank0_error(publish_error, publish_exception)
    return path


def validate_checkpoint(
    path: Path,
    *,
    payloads: tuple[str, ...] = ("model", "trainer"),
) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"v2 checkpoint is not a directory: {path}")
    files = _checkpoint_files(path)
    if not all(file_path.is_file() for file_path in files.values()):
        raise RuntimeError(f"checkpoint is incomplete: {path}")
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise RuntimeError(f"unsupported or incomplete checkpoint manifest: {path}")
    if int(manifest.get("step", -1)) != checkpoint_step(path):
        raise RuntimeError(f"checkpoint manifest step mismatch: {path}")
    if int(manifest.get("world_size", 0)) <= 0:
        raise RuntimeError(f"checkpoint manifest has invalid world_size: {path}")
    completed = files["completed"].read_text(encoding="ascii").strip()
    expected_marker = f"manifest_sha256={_sha256_file(files['manifest'])}"
    if completed != expected_marker:
        raise RuntimeError(f"checkpoint completion marker mismatch: {path}")
    if not set(payloads) <= {"model", "trainer"}:
        raise ValueError(f"unsupported checkpoint payload selection: {payloads}")
    for name in payloads:
        record = manifest["files"][name]
        file_path = path / record["name"]
        if file_path != files[name]:
            raise RuntimeError(f"checkpoint manifest has unexpected {name} filename: {file_path}")
        if file_path.stat().st_size != int(record["bytes"]) or _sha256_file(file_path) != record["sha256"]:
            raise RuntimeError(f"checkpoint payload checksum mismatch: {file_path}")
    return manifest


def _split_tensors(value: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        reference = _TensorReference(len(tensors), tuple(value.shape), str(value.dtype).removeprefix("torch."))
        tensors.append(value)
        return reference
    if isinstance(value, dict):
        return {key: _split_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_split_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_split_tensors(item, tensors) for item in value)
    return value


def _join_tensors(value: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(value, _TensorReference):
        return tensors[value.index]
    if isinstance(value, dict):
        return {key: _join_tensors(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_join_tensors(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_join_tensors(item, tensors) for item in value)
    return value


def _broadcast_nested(value: Any, *, src: int, device: torch.device) -> Any:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    rank = dist.get_rank()
    tensors: list[torch.Tensor] = []
    skeleton = _split_tensors(value, tensors) if rank == src else None
    objects = [skeleton]
    dist.broadcast_object_list(objects, src=src, device=device)
    skeleton = objects[0]
    references: list[_TensorReference] = []

    def collect_references(item: Any) -> None:
        if isinstance(item, _TensorReference):
            references.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect_references(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect_references(child)

    collect_references(skeleton)
    references.sort(key=lambda item: item.index)
    received: list[torch.Tensor] = []
    for reference in references:
        if rank == src:
            tensor = tensors[reference.index].to(device=device)
        else:
            tensor = torch.empty(reference.shape, dtype=getattr(torch, reference.dtype), device=device)
        dist.broadcast(tensor, src=src)
        received.append(tensor)
    return _join_tensors(skeleton, received)


def _broadcast_module(model, *, src: int) -> None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for tensor in unwrap_model(model).state_dict().values():
        dist.broadcast(tensor, src=src)


def _restore_rng(trainer: dict[str, Any], rank: int, restore_rng: bool) -> None:
    if not restore_rng:
        return
    rng_by_rank = trainer.get("rng_by_rank")
    if rng_by_rank is None:
        if trainer.get("rng_cpu") is not None:
            torch.set_rng_state(trainer["rng_cpu"].cpu())
        if torch.cuda.is_available() and trainer.get("rng_cuda") is not None:
            cuda_states = [state.cpu() for state in trainer["rng_cuda"]]
            if len(cuda_states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(cuda_states)
        return
    if rank >= len(rng_by_rank):
        raise RuntimeError("checkpoint has no RNG state for this rank")
    torch.set_rng_state(rng_by_rank[rank]["cpu"].cpu())
    if torch.cuda.is_available() and "cuda" in rng_by_rank[rank]:
        torch.cuda.set_rng_state(rng_by_rank[rank]["cuda"].cpu(), torch.cuda.current_device())


def load_checkpoint(
    path: Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    return_state: bool = False,
) -> int | LoadedCheckpoint:
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    target_device = torch.device(map_location)
    collective_device = target_device
    if dist.is_initialized() and dist.get_backend() != "nccl":
        collective_device = torch.device("cpu")
    trainer: dict[str, Any] | None = None
    model_state: dict[str, torch.Tensor] | None = None
    load_error: str | None = None
    load_exception: Exception | None = None
    if rank == 0:
        try:
            if path.is_dir():
                manifest = validate_checkpoint(path)
                if int(manifest["world_size"]) != world_size:
                    raise RuntimeError(
                        f"checkpoint world_size={manifest['world_size']} cannot resume with world_size={world_size}"
                    )
                model_state = load_file(str(path / "model.safetensors"), device="cpu")
                trainer = torch.load(path / "trainer.pt", map_location=target_device, weights_only=False)
            else:
                legacy = torch.load(path, map_location=target_device, weights_only=False)
                model_state = legacy["model"]
                trainer = {key: value for key, value in legacy.items() if key != "model"}
                trainer.setdefault("world_size", world_size)
                trainer.setdefault("data_state_by_rank", [trainer.get("data_state")] * world_size)
                trainer.setdefault("args", None)
            unwrap_model(model).load_state_dict(model_state)
        except Exception as exc:  # noqa: BLE001 - avoid leaving peer ranks in a broadcast.
            load_exception = exc
            load_error = f"rank 0 could not load checkpoint {path}: {type(exc).__name__}: {exc}"
    load_error = _share_rank0_error(load_error, device=collective_device)
    _raise_rank0_error(load_error, load_exception)
    _broadcast_module(model, src=0)

    trainer = _broadcast_nested(trainer, src=0, device=collective_device)
    assert trainer is not None
    saved_world_size = int(trainer.get("world_size", 1))
    if saved_world_size != world_size:
        raise RuntimeError(
            f"checkpoint world_size={saved_world_size} cannot resume with world_size={world_size}"
        )
    if trainer.get("parallel_mode") == "shared_hierarchical":
        raise RuntimeError("Shared-owner checkpoints must resume through the online shared training entry.")
    if optimizer is not None and trainer.get("optimizer") is not None:
        optimizer.load_state_dict(trainer["optimizer"])
    if scheduler is not None and trainer.get("scheduler") is not None:
        scheduler.load_state_dict(trainer["scheduler"])
    if scaler is not None and trainer.get("scaler") is not None:
        scaler.load_state_dict(trainer["scaler"])
    _restore_rng(trainer, rank, restore_rng)
    loaded_step = trainer.get("step")
    if loaded_step is None:
        loaded_step = checkpoint_step(path)
    data_state_by_rank = trainer.get("data_state_by_rank") or [None] * world_size
    if len(data_state_by_rank) != world_size:
        raise RuntimeError(
            f"checkpoint has {len(data_state_by_rank)} data states for world_size={world_size}"
        )
    loaded = LoadedCheckpoint(
        step=int(loaded_step),
        data_state=data_state_by_rank[rank],
        args=trainer.get("args"),
        world_size=saved_world_size,
    )
    return loaded if return_state else loaded.step


def load_model_state_only(
    path: Path,
    model,
    *,
    map_location: str | torch.device = "cpu",
) -> int | None:
    if path.is_dir():
        validate_checkpoint(path, payloads=("model",))
        state = load_file(str(path / "model.safetensors"), device="cpu")
        unwrap_model(model).load_state_dict(state)
        return checkpoint_step(path)
    if path.suffix == ".safetensors":
        state = load_file(str(path), device="cpu")
        unwrap_model(model).load_state_dict(state)
        return None
    data = torch.load(path, map_location=map_location, weights_only=False)
    state = data["model"] if isinstance(data, dict) and "model" in data else data
    unwrap_model(model).load_state_dict(state)
    if isinstance(data, dict) and "step" in data:
        return int(data["step"])
    return None


def load_model_state_distributed(
    path: Path,
    model,
    *,
    map_location: str | torch.device = "cpu",
) -> int | None:
    rank = dist.get_rank() if dist.is_initialized() else 0
    step: int | None = None
    collective_device = (
        torch.device(map_location)
        if dist.is_initialized() and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    load_error: str | None = None
    load_exception: Exception | None = None
    if rank == 0:
        try:
            step = load_model_state_only(path, model, map_location=map_location)
        except Exception as exc:  # noqa: BLE001 - avoid leaving peer ranks in a broadcast.
            load_exception = exc
            load_error = f"rank 0 could not load model state {path}: {type(exc).__name__}: {exc}"
    load_error = _share_rank0_error(load_error, device=collective_device)
    _raise_rank0_error(load_error, load_exception)
    _broadcast_module(model, src=0)
    if dist.is_initialized() and dist.get_world_size() > 1:
        objects = [step]
        dist.broadcast_object_list(objects, src=0, device=collective_device)
        step = objects[0]
    return step
