"""Device, precision, and distributed runtime helpers."""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


_MPI_WORLD_KEYS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE")
_MPI_RANK_KEYS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")


def _first_env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


def resolve_device(requested: str = "auto", *, force_cpu: bool = False) -> torch.device:
    if force_cpu or requested == "cpu":
        return torch.device("cpu")
    if requested == "auto":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        local_rank = _first_env_int(("LOCAL_RANK",), 0)
        return torch.device("cuda", local_rank)
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {requested}")
    return device


def resolve_dtype(
    precision: str,
    device: torch.device,
    *,
    allow_cpu_bf16: bool = False,
) -> torch.dtype:
    if precision == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if device.type == "cpu" and not allow_cpu_bf16:
            raise RuntimeError("CPU BF16 is disabled for this path; use --precision fp32")
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision}")


def autocast_context(device: torch.device, dtype: torch.dtype):
    """Return the mixed-precision context used by model forward passes.

    CPU BF16 is deliberately opt-in at resolve_dtype/init_runtime.
    Inputs, parameters, gradients, and optimizer state remain FP32; this
    context only lets eligible operators choose BF16 kernels.
    """
    if dtype == torch.bfloat16 and device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def configure_torch_threads(threads: int | None, interop_threads: int | None) -> None:
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
    if interop_threads is not None:
        if interop_threads <= 0:
            raise ValueError("interop_threads must be positive")
        torch.set_num_interop_threads(interop_threads)


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    dtype: torch.dtype
    backend: str | None
    rank: int
    world_size: int
    local_rank: int

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def ddp_kwargs(self, *, bucket_cap_mb: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "bucket_cap_mb": bucket_cap_mb,
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
        }
        if self.device.type == "cuda":
            kwargs["device_ids"] = [self.local_rank]
            kwargs["output_device"] = self.local_rank
        else:
            kwargs["device_ids"] = None
            kwargs["output_device"] = None
        return kwargs


def _auto_backend(device: torch.device) -> str | None:
    mpi_world = _first_env_int(_MPI_WORLD_KEYS, 1)
    if mpi_world > 1:
        if not dist.is_mpi_available():
            raise RuntimeError("MPI launch detected but this Torch build has no MPI backend")
        return "mpi"
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return "nccl" if device.type == "cuda" else "gloo"
    return None


def init_runtime(
    *,
    requested_device: str = "auto",
    force_cpu: bool = False,
    precision: str = "auto",
    backend: str = "auto",
    threads: int | None = None,
    interop_threads: int = 1,
    allow_cpu_bf16: bool = False,
) -> RuntimeContext:
    device = resolve_device(requested_device, force_cpu=force_cpu)
    dtype = resolve_dtype(precision, device, allow_cpu_bf16=allow_cpu_bf16)
    configure_torch_threads(threads, interop_threads)

    selected_backend = _auto_backend(device) if backend == "auto" else backend
    if selected_backend == "none":
        selected_backend = None
    if selected_backend not in {None, "mpi", "gloo", "nccl"}:
        raise ValueError(f"unsupported distributed backend: {selected_backend}")
    if selected_backend == "mpi" and not dist.is_mpi_available():
        raise RuntimeError("Torch MPI backend is unavailable")
    if selected_backend == "nccl" and device.type != "cuda":
        raise RuntimeError("NCCL requires a CUDA device")
    detected_world_size = max(
        _first_env_int(_MPI_WORLD_KEYS, 1),
        _first_env_int(("WORLD_SIZE",), 1),
    )
    if selected_backend is None and detected_world_size > 1:
        raise RuntimeError(
            f"distributed launch with world_size={detected_world_size} cannot use backend=none"
        )

    local_rank = _first_env_int(("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"), 0)
    if device.type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    if selected_backend is not None:
        dist.init_process_group(selected_backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = _first_env_int(("RANK",) + _MPI_RANK_KEYS, 0)
        world_size = 1
    return RuntimeContext(
        device=device,
        dtype=dtype,
        backend=selected_backend,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )


def all_reduce_sum(values: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values
