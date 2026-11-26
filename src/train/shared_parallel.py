"""Shared FP32/BF16 parameters, owner gradients and overlapped HMPI reduction.

Shared owner updates with FP32 or BF16 storage and optional node-aware exchange.
"""
from __future__ import annotations
import ctypes
import json
import mmap
import os
import resource
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn


def env_int(names: tuple[str, ...], default: int) -> int:
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    return default



def rank_coordinates(
    rank: int, world_size: int, local_world_size: int,
    dies_per_node: int, node_local_rank: int, node_local_size: int,
) -> tuple[int, int, int, int, int]:
    """Map a node-major MPI rank to node, die, replica, and die-local rank."""
    ranks_per_node = local_world_size * dies_per_node
    if local_world_size < 1 or dies_per_node < 1:
        raise ValueError("local world size and dies per node must be positive")
    if node_local_size != ranks_per_node:
        raise RuntimeError(
            f"expected {ranks_per_node} ranks per node, got {node_local_size}")
    if world_size % ranks_per_node:
        raise RuntimeError(
            f"world size {world_size} is not divisible by {ranks_per_node} ranks per node")
    if rank % ranks_per_node != node_local_rank:
        raise RuntimeError("hierarchical rank layout must be node-major and contiguous")
    node_index = rank // ranks_per_node
    die_index = node_local_rank // local_world_size
    local_rank = node_local_rank % local_world_size
    replica_index = node_index * dies_per_node + die_index
    replica_count = world_size // local_world_size
    return node_index, die_index, replica_index, local_rank, replica_count



def hierarchy_rank_groups(
    world_size: int, local_world_size: int
) -> tuple[list[list[int]], list[list[int]]]:
    if world_size % local_world_size:
        raise RuntimeError(
            f"world size {world_size} is not divisible by local size {local_world_size}")
    replica_count = world_size // local_world_size
    die_groups = [
        list(range(replica * local_world_size, (replica + 1) * local_world_size))
        for replica in range(replica_count)
    ]
    owner_groups = [
        [replica * local_world_size + local for replica in range(replica_count)]
        for local in range(local_world_size)
    ]
    return die_groups, owner_groups



def thread_affinities(thread_ids: set[int]) -> list[dict[str, object]]:
    """Read Linux thread names and affinity masks for startup validation."""
    records = []
    for thread_id in sorted(thread_ids):
        task = Path(f"/proc/self/task/{thread_id}")
        try:
            records.append({
                "tid": thread_id,
                "name": (task / "comm").read_text(encoding="ascii").strip(),
                "affinity": sorted(os.sched_getaffinity(thread_id)),
            })
        except (FileNotFoundError, ProcessLookupError):
            continue
    return records



def hierarchy_groups(
    world_size: int, local_world_size: int, dies_per_node: int,
    communication_cpu: int | None,
):
    node_local_rank = env_int(("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"), 0)
    node_local_size = env_int(
        ("OMPI_COMM_WORLD_LOCAL_SIZE",), local_world_size * dies_per_node)
    coordinates = rank_coordinates(
        dist.get_rank(), world_size, local_world_size, dies_per_node,
        node_local_rank, node_local_size,
    )
    node_index, die_index, replica_index, local_rank, replica_count = coordinates
    die_ranks, owner_ranks = hierarchy_rank_groups(world_size, local_world_size)
    original_affinity = os.sched_getaffinity(0)
    before_threads = {
        int(task.name) for task in Path("/proc/self/task").iterdir()
        if task.name.isdigit()
    }
    if communication_cpu is not None:
        os.sched_setaffinity(0, {communication_cpu})
    try:
        # ProcessGroupMPI creates one backend worker for every group this rank
        # joins. New threads inherit the creator's affinity mask.
        die_groups = [dist.new_group(ranks=ranks) for ranks in die_ranks]
        owner_groups = [dist.new_group(ranks=ranks) for ranks in owner_ranks]
        node_owner_group = die_owner_group = None
        if os.environ.get("WAN21_HIERARCHICAL_COMM", "0") == "1":
            if dies_per_node != 4:
                raise ValueError("node-aware gradient exchange requires four dies per node")
            nodes = world_size // (local_world_size * dies_per_node)
            node_owner_groups = [dist.new_group(ranks=[
                node * local_world_size * dies_per_node + die * local_world_size + owner
                for die in range(dies_per_node)])
                for node in range(nodes) for owner in range(local_world_size)]
            die_owner_groups = [dist.new_group(ranks=[
                node * local_world_size * dies_per_node + die * local_world_size + owner
                for node in range(nodes)])
                for die in range(dies_per_node) for owner in range(local_world_size)]
            node_owner_group = node_owner_groups[node_index * local_world_size + local_rank]
            die_owner_group = die_owner_groups[die_index * local_world_size + local_rank]
    finally:
        if communication_cpu is not None:
            os.sched_setaffinity(0, original_affinity)
    after_threads = {
        int(task.name) for task in Path("/proc/self/task").iterdir()
        if task.name.isdigit()
    }
    group_threads = thread_affinities(after_threads - before_threads)
    if communication_cpu is not None and any(
        record["affinity"] != [communication_cpu] for record in group_threads
    ):
        raise RuntimeError(
            f"MPI group workers escaped communication CPU {communication_cpu}: "
            f"{group_threads}"
        )
    return (
        node_index, die_index, replica_index, local_rank, replica_count,
        die_groups[replica_index], owner_groups[local_rank],
        node_owner_group, die_owner_group, group_threads,
    )



def mpi_thread_level() -> int | None:
    """Return MPI's provided thread level (2 serialized, 3 multiple)."""
    try:
        library = ctypes.CDLL("libmpi.so")
        query = library.MPI_Query_thread
        query.argtypes = [ctypes.POINTER(ctypes.c_int)]
        query.restype = ctypes.c_int
        provided = ctypes.c_int()
        if query(ctypes.byref(provided)) != 0:
            return None
        return provided.value
    except (OSError, AttributeError):
        return None



def contiguous_owners_for_model(model: nn.Module, world_size: int) -> dict[str, int]:
    """Assign whole parameters in flat order to approximately equal contiguous shards."""
    parameters = list(model.named_parameters())
    if os.environ.get("WAN21_COMM_OWNER_MODE", "contiguous") == "interleave":
        block_ids = {}
        for name, parameter in parameters:
            parts = name.split(".")
            if "blocks" in parts:
                index = parts.index("blocks") + 1
                if index < len(parts) and parts[index].isdigit():
                    block_ids[name] = int(parts[index])
        if not block_ids:
            raise RuntimeError("interleaved owner mode needs numbered model.blocks")
        last_block = max(block_ids.values())
        owners = {name: (last_block - block_id) % world_size
                  for name, block_id in block_ids.items()}
        totals = [0] * world_size
        for name, parameter in parameters:
            if name in owners:
                totals[owners[name]] += parameter.numel()
        # Assign non-block tensors by size after balancing the repeating blocks.
        for name, parameter in sorted((entry for entry in parameters if entry[0] not in owners),
                                      key=lambda entry: entry[1].numel(), reverse=True):
            owner = min(range(world_size), key=lambda rank: (totals[rank], rank))
            owners[name] = owner
            totals[owner] += parameter.numel()
        return owners
    if os.environ.get("WAN21_COMM_OWNER_MODE", "contiguous") not in ("contiguous", "swap_ends"):
        raise ValueError("WAN21_COMM_OWNER_MODE must be contiguous, interleave, or swap_ends")
    total = sum(parameter.numel() for _, parameter in parameters)
    owners: dict[str, int] = {}
    offset = 0
    for name, parameter in parameters:
        midpoint = offset + parameter.numel() // 2
        owner = min(world_size - 1, midpoint * world_size // max(total, 1))
        owners[name] = owner
        offset += parameter.numel()
    if os.environ.get("WAN21_COMM_OWNER_MODE", "contiguous") == "swap_ends":
        block_ids = {}
        for name, _parameter in parameters:
            parts = name.split(".")
            if "blocks" in parts:
                index = parts.index("blocks") + 1
                if index < len(parts) and parts[index].isdigit():
                    block_ids[name] = int(parts[index])
        if not block_ids:
            raise RuntimeError("swap_ends requires numbered model.blocks")
        first, last = min(block_ids.values()), max(block_ids.values())
        edge = {name: owner for name, owner in owners.items()
                if block_ids.get(name) in (first, last)}
        old_first = {owner for name, owner in edge.items() if block_ids[name] == first}
        old_last = {owner for name, owner in edge.items() if block_ids[name] == last}
        if len(old_first) != 1 or len(old_last) != 1 or old_first == old_last:
            raise RuntimeError(f"edge blocks must have one different owner each: {old_first} {old_last}")
        low, high = old_first.pop(), old_last.pop()
        for name in edge:
            owners[name] = high if block_ids[name] == first else low
    return owners



def contiguous_gradient_buckets(
    model: nn.Module, owners: dict[str, int], owner: int, cap_elements: int
) -> list[tuple[int, int, list[nn.Parameter]]]:
    """Split an owner's contiguous flat segments into fixed-size ranges."""
    if cap_elements < 1:
        raise ValueError("gradient bucket must contain at least one element")
    entries: list[tuple[int, int, nn.Parameter]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        if owners[name] == owner:
            entries.append((offset, parameter.numel(), parameter))
        offset += parameter.numel()
    if not entries:
        raise RuntimeError(f"owner {owner} has no parameters")
    segments: list[list[tuple[int, int, nn.Parameter]]] = []
    segment: list[tuple[int, int, nn.Parameter]] = []
    previous_end = -1
    for entry in entries:
        start, elements, _ = entry
        if segment and start != previous_end:
            segments.append(segment)
            segment = []
        segment.append(entry)
        previous_end = start + elements
    if segment:
        segments.append(segment)

    buckets = []
    for segment in segments:
        start = segment[0][0]
        end = segment[-1][0] + segment[-1][1]
        cursor = start
        while cursor < end:
            bucket_end = min(end, cursor + cap_elements)
            dependencies = [
                parameter
                for parameter_start, elements, parameter in segment
                if parameter_start < bucket_end
                and parameter_start + elements > cursor
            ]
            if not dependencies:
                raise RuntimeError("gradient bucket has no parameter dependency")
            buckets.append((cursor, bucket_end - cursor, dependencies))
            cursor = bucket_end
    return buckets



def share_parameters(model: nn.Module, path: Path, rank: int, group=None) -> torch.Tensor:
    """Map the same file on all four local ranks; no copy-on-write parameters."""
    if not path.is_absolute() or not path.parent.is_dir():
        raise ValueError(f"shared-memory directory is unavailable: {path}")
    parameters = list(model.parameters())
    dtype = parameters[0].dtype
    if dtype not in (torch.float32, torch.bfloat16) or any(p.dtype != dtype for p in parameters):
        raise ValueError("the shared model must have uniform FP32 or BF16 parameters")
    elements = sum(parameter.numel() for parameter in parameters)
    if rank == 0:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(descriptor, elements * dtype.itemsize)
        finally:
            os.close(descriptor)
    dist.barrier(group=group)
    flat = torch.from_file(str(path), shared=True, size=elements, dtype=dtype)
    hbm_node = None
    if os.environ.get("WAN21_SHARED_WEIGHTS_HBM", "0") == "1":
        if (os.environ.get("WAN21_STRIPED_SHARED_WEIGHTS", "1") != "1"
                or int(os.environ.get("WAN21_WEIGHT_STRIPE_MB", "2")) not in (2, 8)):
            raise ValueError("HBM shared weights require 2 or 8 MiB first-touch stripes")
        node_local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        if node_local_rank % 4 != rank:
            raise RuntimeError("shared-weight HBM NUMA does not match die rank")
        hbm_node = 16 + node_local_rank
        hbm_mask = ctypes.c_ulong(1 << hbm_node)
        hbm_libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
        hbm_libnuma.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
                                     ctypes.c_uint]
        hbm_libnuma.mbind.restype = ctypes.c_int

    def bind_stripe(first: int, count: int) -> None:
        if hbm_node is None:
            return
        if hbm_libnuma.mbind(
                ctypes.c_void_p(flat.data_ptr() + first * dtype.itemsize),
                ctypes.c_ulong(count * dtype.itemsize), ctypes.c_int(2),
                ctypes.byref(hbm_mask), ctypes.c_ulong(64), ctypes.c_uint(0)) != 0:
            raise OSError(ctypes.get_errno(),
                          f"cannot bind shared weight stripe to HBM NUMA {hbm_node}")

    if os.environ.get("WAN21_STRIPED_SHARED_WEIGHTS", "1") == "1":
        world_size = dist.get_world_size(group=group)
        # Each rank faults one disjoint, 2 MiB-aligned portion before rank 0 copies
        # the checkpoint. The job's rank wrapper restricts all four to this die's
        # four DDR NUMA nodes; this spreads shared weight traffic across them.
        stripe_mb = int(os.environ.get("WAN21_WEIGHT_STRIPE_MB", "2"))
        if stripe_mb not in (0, 2, 8):
            raise ValueError("shared weight stripe must be 0, 2 or 8 MiB")
        if stripe_mb:
            stripe_elements = stripe_mb * 1024 * 1024 // dtype.itemsize
            # Assign neighboring stripes to different DDR NUMAs so every large
            # GEMM reads from all four memory controllers, rather than one.
            for first in range(rank * stripe_elements, elements, world_size * stripe_elements):
                count = min(stripe_elements, elements - first)
                bind_stripe(first, count)
                flat.narrow(0, first, count).zero_()
            if hbm_node is not None:
                print(f"WAN21_SHARED_WEIGHT_HBM_BIND rank={node_local_rank} "
                      f"node={hbm_node} stripes={len(range(rank * stripe_elements, elements, world_size * stripe_elements))}", flush=True)
        else:
            # Whole quarters are the first, already tested NUMA layout.
            page_elements = max(1, (2 * 1024 * 1024) // dtype.itemsize)
            span = ((elements + world_size * page_elements - 1)
                    // (world_size * page_elements)) * page_elements
            first, last = min(elements, rank * span), min(elements, (rank + 1) * span)
            if first < last:
                flat.narrow(0, first, last - first).zero_()
        dist.barrier(group=group)
    if rank == 0:
        offset = 0
        with torch.no_grad():
            for parameter in parameters:
                end = offset + parameter.numel()
                flat[offset:end].copy_(parameter.detach().reshape(-1))
                offset = end
    if rank == 0 and os.environ.get("WAN21_STRIPED_SHARED_WEIGHTS", "1") == "1":
        numa_lines = [line for line in Path("/proc/self/numa_maps").read_text().splitlines()
                      if "file=" + str(path) in line]
        nodes = {}
        for line in numa_lines:
            for node, pages in re.findall(r"\bN(\d+)=(\d+)", line):
                nodes[int(node)] = nodes.get(int(node), 0) + int(pages)
        print("SHARED_DIE_WEIGHT_NUMA " + json.dumps({"nodes": nodes, "elements": elements, "stripe_mb": int(os.environ.get("WAN21_WEIGHT_STRIPE_MB", "2"))}), flush=True)
    dist.barrier(group=group)
    offset = 0
    for parameter in parameters:
        end = offset + parameter.numel()
        parameter.data = flat[offset:end].view_as(parameter)
        offset = end
    dist.barrier(group=group)
    mapping = {
        "rank": rank,
        "inode": os.stat(path).st_ino,
        "elements": elements,
        "storage_shared": bool(flat.untyped_storage().is_shared()),
    }
    mappings = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(mappings, mapping, group=group)
    if rank == 0:
        if len({item["inode"] for item in mappings}) != 1:
            raise RuntimeError(f"ranks mapped different files: {mappings}")
        if not all(item["storage_shared"] for item in mappings):
            raise RuntimeError(f"mapping is not process-shared: {mappings}")
        print("SHARED_DIE_MAPPING " + json.dumps(mappings, sort_keys=True), flush=True)
    if rank == 0:
        path.unlink()
    dist.barrier(group=group)
    return flat  # Keep the mapping alive for the entire training run.



class SharedGradients:
    """Process-shared per-rank gradient slices with owner-side reduction."""

    def __init__(self, model: nn.Module, path: Path, rank: int, world_size: int,
                 group=None):
        if world_size != 4:
            raise ValueError("shared gradients currently require four ranks")
        if not path.is_absolute() or not path.parent.is_dir():
            raise ValueError(f"shared-gradient directory is unavailable: {path}")
        self.parameters = list(model.parameters())
        self.dtype = self.parameters[0].dtype
        if self.dtype not in (torch.float32, torch.bfloat16) or any(
                p.dtype != self.dtype for p in self.parameters):
            raise ValueError("shared gradients require uniform FP32 or BF16 parameters")
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.total_elements = sum(parameter.numel() for parameter in self.parameters)
        self.offsets: dict[int, int] = {}
        offset = 0
        for parameter in self.parameters:
            self.offsets[id(parameter)] = offset
            offset += parameter.numel()
        if rank == 0:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(descriptor, self.total_elements * world_size * self.dtype.itemsize)
            finally:
                os.close(descriptor)
        dist.barrier(group=group)
        self.flat = torch.from_file(
            str(path), shared=True, size=self.total_elements * world_size,
            dtype=self.dtype,
        )
        self.local = self.flat.narrow(0, rank * self.total_elements, self.total_elements)
        self.local.zero_()  # First-touch each rank's pages on its bound NUMA node.
        dist.barrier(group=group)
        for parameter in self.parameters:
            start = self.offsets[id(parameter)]
            parameter.grad = self.local.narrow(0, start, parameter.numel()).view_as(parameter)
        dist.barrier(group=group)
        mapping = {
            "rank": rank,
            "inode": os.stat(path).st_ino,
            "elements": self.total_elements * world_size,
            "storage_shared": bool(self.flat.untyped_storage().is_shared()),
        }
        mappings = [None] * world_size
        dist.all_gather_object(mappings, mapping, group=group)
        if rank == 0:
            if len({item["inode"] for item in mappings}) != 1:
                raise RuntimeError(f"gradient ranks mapped different files: {mappings}")
            if not all(item["storage_shared"] for item in mappings):
                raise RuntimeError(f"gradient mapping is not process-shared: {mappings}")
            print("SHARED_DIE_GRADIENT_MAPPING " + json.dumps(mappings, sort_keys=True), flush=True)
            path.unlink()
        dist.barrier(group=group)

    def zero_local(self) -> None:
        self.local.zero_()

    def reduce(self, model: nn.Module, owners: dict[str, int], rank: int,
               vector_reducer=None) -> float:
        """Average four rank slices in-place; only each owner rank writes its range."""
        dist.barrier()  # All four backward passes must have finished before reading.
        norm_squared = torch.zeros(1, dtype=torch.float64)
        for name, parameter in model.named_parameters():
            if owners[name] != rank:
                continue
            if parameter.grad is None:
                raise RuntimeError("all model parameters must participate in backward")
            start = self.offsets[id(parameter)]
            elements = parameter.numel()
            if vector_reducer is not None:
                vector_reducer(self.flat, self.total_elements, start, elements, rank, 0.25)
            else:
                destination = self.flat.narrow(
                    0, rank * self.total_elements + start, elements
                )
                destination.mul_(0.25)
                for source_rank in range(self.world_size):
                    if source_rank != rank:
                        destination.add_(self.flat.narrow(
                            0, source_rank * self.total_elements + start, elements
                        ), alpha=0.25)
            norm = parameter.grad.float().norm()
            norm_squared += norm.double().square()
        dist.all_reduce(norm_squared, op=dist.ReduceOp.SUM)
        norm = float(norm_squared.sqrt().item())
        if not torch.isfinite(norm_squared).all():
            raise RuntimeError("non-finite global gradient norm")
        factor = min(1.0, 1.0 / (norm + 1e-6))
        if factor < 1.0:
            for name, parameter in model.named_parameters():
                if owners[name] == rank:
                    parameter.grad.mul_(factor)
        return norm



class HierarchicalGradients(SharedGradients):
    """Die-local shared reduction followed by asynchronous inter-die all-reduce."""

    def __init__(self, model: nn.Module, path: Path, rank: int, world_size: int,
                 local_world_size: int, dies_per_node: int,
                 communication_cpu: int | None,
                 local_group, inter_node_group,
                 node_owner_group=None, die_owner_group=None):
        (
            self.node_index, self.die_index, self.replica_index,
            self.local_rank, self.replica_count,
        ) = rank_coordinates(
            rank, world_size, local_world_size, dies_per_node,
            env_int(("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"), 0),
            env_int(
                ("OMPI_COMM_WORLD_LOCAL_SIZE",),
                local_world_size * dies_per_node,
            ),
        )
        self.global_rank = rank
        self.global_world_size = world_size
        self.local_group = local_group
        self.inter_node_group = inter_node_group
        self.node_owner_group = node_owner_group
        self.die_owner_group = die_owner_group
        self.dies_per_node = dies_per_node
        self.communication_cpu = communication_cpu
        super().__init__(model, path, self.local_rank, local_world_size, group=local_group)
        self.owner_parameters: dict[int, list[nn.Parameter]] = {}
        self.buckets: list[tuple[int, int, list[nn.Parameter]]] = []
        self.owner_bucket_ids: dict[int, list[int]] = {}
        self.ready_path = path.with_name(path.name + ".ready")
        self.ready_flags: torch.Tensor | None = None
        self._generation = 0
        self.publish_ready = True
        self._counts: list[int] = []
        self._expected: list[int] = []
        self._worker: threading.Thread | None = None
        self._complete = threading.Event()
        self._worker_error: BaseException | None = None
        self._reduce_worker: threading.Thread | None = None
        self._reduce_error: BaseException | None = None
        self._reduce_events: dict[int, threading.Event] = {}
        self._reduced: dict[int, tuple[float, float, float]] = {}
        self._native_reducer = None
        self._local_reduce_pool: ThreadPoolExecutor | None = None

    def set_owners(
        self, model: nn.Module, owners: dict[str, int], cap_elements: int
    ) -> None:
        if self.ready_flags is not None:
            raise RuntimeError("gradient owners are already configured")
        ranges: dict[int, list[tuple[int, int, nn.Parameter]]] = {
            owner: [] for owner in range(self.world_size)
        }
        for name, parameter in model.named_parameters():
            ranges[owners[name]].append(
                (self.offsets[id(parameter)], parameter.numel(), parameter)
            )
        for owner, entries in ranges.items():
            if not entries:
                raise RuntimeError(f"owner {owner} has no parameters")
            self.owner_parameters[owner] = [p for _, _, p in entries]
            owner0_mb = int(os.environ.get("WAN21_OWNER0_BUCKET_MB", "0"))
            if owner0_mb not in (0, 32, 64, 128):
                raise ValueError("owner0 bucket size must be 0, 32, 64 or 128 MiB")
            owner_cap = min(cap_elements, owner0_mb * 1024 * 1024 // self.parameters[0].element_size()) if owner == 0 and owner0_mb else cap_elements
            owner_buckets = contiguous_gradient_buckets(
                model, owners, owner, owner_cap)
            ids = list(range(len(self.buckets), len(self.buckets) + len(owner_buckets)))
            self.owner_bucket_ids[owner] = ids
            self.buckets.extend(owner_buckets)
        self._counts = [0] * len(self.buckets)
        self._expected = [len(parameters) for _, _, parameters in self.buckets]
        self._published_seconds = [None] * len(self.buckets)
        parameter_buckets: dict[int, list[int]] = {id(p): [] for p in self.parameters}
        for bucket_id, (_, _, parameters) in enumerate(self.buckets):
            for parameter in parameters:
                parameter_buckets[id(parameter)].append(bucket_id)
        for parameter in self.parameters:
            bucket_ids = parameter_buckets[id(parameter)]
            if not bucket_ids:
                raise RuntimeError("model parameter is missing from gradient buckets")
            parameter.register_post_accumulate_grad_hook(
                self._ready_hook(tuple(bucket_ids)))

        flag_elements = self.world_size * len(self.buckets)
        if self.local_rank == 0:
            descriptor = os.open(
                self.ready_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(descriptor, flag_elements * 4)
            finally:
                os.close(descriptor)
        dist.barrier(group=self.local_group)
        self.ready_flags = torch.from_file(
            str(self.ready_path), shared=True, size=flag_elements, dtype=torch.int32,
        ).view(self.world_size, len(self.buckets))
        self.ready_flags[self.local_rank].zero_()
        dist.barrier(group=self.local_group)
        if self.local_rank == 0:
            self.ready_path.unlink()
        dist.barrier(group=self.local_group)

    def _ready_hook(self, bucket_ids: tuple[int, ...]):
        def hook(_parameter: nn.Parameter) -> None:
            if not self.publish_ready:
                return
            assert self.ready_flags is not None
            assert self._native_reducer is not None
            for bucket_id in bucket_ids:
                self._counts[bucket_id] += 1
                if self._counts[bucket_id] == self._expected[bucket_id]:
                    index = self.local_rank * len(self.buckets) + bucket_id
                    self._published_seconds[bucket_id] = time.monotonic() - self.timing["step_start"]
                    self._native_reducer.store_ready(
                        self.ready_flags, index, self._generation)
        return hook

    def begin_step(self, generation: int, vector_reducer) -> None:
        if self.ready_flags is None:
            raise RuntimeError("gradient owners must be configured before training")
        if vector_reducer is None:
            raise RuntimeError(
                "hierarchical gradients require the native atomic-ready helper")
        self._native_reducer = vector_reducer
        self.zero_local()
        self._generation = generation
        self._counts = [0] * len(self.buckets)
        self._published_seconds = [None] * len(self.buckets)
        self.ready_flags[self.local_rank].zero_()
        dist.barrier(group=self.local_group)
        self._worker_error = None
        self.timing = {
            "rank": self.global_rank,
            "node": self.node_index,
            "die": self.die_index,
            "replica": self.replica_index,
            "local_rank": self.local_rank,
            "generation": generation,
            "hierarchical_comm": self.node_owner_group is not None,
            "step_start": time.monotonic(),
        }
        self._complete.clear()
        self._reduce_worker = None
        self._reduce_error = None
        self._reduced = {}
        pipeline = (os.environ.get("WAN21_PIPELINE_LOCAL_REDUCE", "1") == "1"
                    and self.dtype == torch.bfloat16 and self.replica_count > 1
                    and vector_reducer.direct_enabled)
        self.timing["pipeline_local_reduce"] = pipeline
        self._reduce_events = ({bucket_id: threading.Event()
                                for bucket_id in self.owner_bucket_ids[self.local_rank]}
                               if pipeline else {})
        if pipeline:
            self._reduce_worker = threading.Thread(
                target=self._local_reduction_worker, args=(generation, vector_reducer),
                name="wan21-gradient-local-reduce")
            self._reduce_worker.start()
        self._worker = threading.Thread(
            target=self._communication_worker,
            args=(generation, vector_reducer),
            name="wan21-gradient-communication",
        )
        self._worker.start()

    def _local_reduction_worker(self, generation: int, vector_reducer) -> None:
        """Prepare the next ready bucket while the MPI worker transfers this one."""
        try:
            if self.communication_cpu is not None:
                # Keep the reserved system CPU unused. Sharing one compute core
                # costs at most one of the rank's 36 lanes during backward.
                os.sched_setaffinity(0, {self.communication_cpu - 1})
            self.timing["local_reduction_affinity"] = sorted(os.sched_getaffinity(0))
            lanes = int(os.environ.get("WAN21_LOCAL_REDUCE_LANES", "1"))
            if lanes not in (1, 2):
                raise ValueError("local BF16 reduction requires one or two lanes")
            if lanes == 2:
                if self.communication_cpu is None or not vector_reducer.direct_enabled:
                    raise RuntimeError("two-lane reduction needs pinned CPUs and direct BF16 SVE")
                if self._local_reduce_pool is None:
                    helper_cpu = self.communication_cpu - 2
                    self._local_reduce_pool = ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="wan21-local-reduce-sve",
                        initializer=lambda: os.sched_setaffinity(0, {helper_cpu}),
                    )
                self.timing["local_reduce_helper_affinity"] = self._local_reduce_pool.submit(
                    lambda: sorted(os.sched_getaffinity(0))).result()
            self.timing["local_reduce_lanes"] = lanes
            assert self.ready_flags is not None
            for bucket_id in reversed(self.owner_bucket_ids[self.local_rank]):
                while not vector_reducer.all_ready(
                    self.ready_flags, self.world_size, len(self.buckets),
                    bucket_id, generation,
                ):
                    if self._complete.is_set():
                        return
                    time.sleep(0.0001)
                ready = time.monotonic()
                start, elements, _ = self.buckets[bucket_id]
                reduce_start = time.monotonic()
                scale = 1.0 / self.world_size
                # The native per-element reduction does not mix neighboring
                # elements. Split contiguous index ranges so both pinned SVE
                # threads can work without a copy or a change of sum order.
                split = (elements // 2 // 256) * 256 if lanes == 2 else 0
                if split:
                    assert self._local_reduce_pool is not None
                    other = self._local_reduce_pool.submit(
                        vector_reducer, self.flat, self.total_elements, start, split,
                        self.local_rank, scale, False)
                    vector_reducer(self.flat, self.total_elements, start + split,
                                   elements - split, self.local_rank, scale,
                                   parallel=False)
                    other.result()
                else:
                    vector_reducer(self.flat, self.total_elements, start, elements,
                                   self.local_rank, scale, parallel=False)
                reduce_end = time.monotonic()
                self._reduced[bucket_id] = (ready, reduce_start, reduce_end)
                self._reduce_events[bucket_id].set()
        except BaseException as error:
            self._reduce_error = error
            for event in self._reduce_events.values():
                event.set()

    def _communication_worker(self, generation: int, vector_reducer) -> None:
        try:
            if self.communication_cpu is not None:
                os.sched_setaffinity(0, {self.communication_cpu})
            self.timing["communication_cpu"] = self.communication_cpu
            self.timing["communication_affinity"] = sorted(os.sched_getaffinity(0))
            assert self.ready_flags is not None
            bucket_timings = []
            step_start = self.timing["step_start"]
            # Early owners start transferring while backward is still busy.
            # Use larger chunks only for them, preserving the late owner's
            # small chunks and the identical per-element reduction order.
            early_chunk = int(os.environ.get("WAN21_EARLY_COMM_CHUNK_MB", "0"))
            if early_chunk not in (0, 128):
                raise ValueError("early-owner communication chunk must be 0 or 128 MiB")
            if self.local_rank >= 2 and early_chunk:
                vector_reducer.bf16_comm_chunk_mb = early_chunk
            self.timing["bf16_comm_chunk_mb"] = vector_reducer.bf16_comm_chunk_mb
            for bucket_id in reversed(self.owner_bucket_ids[self.local_rank]):
                start, elements, _ = self.buckets[bucket_id]
                if self._reduce_worker is not None:
                    self._reduce_events[bucket_id].wait()
                    if self._reduce_error is not None:
                        raise RuntimeError("pipeline local reduction failed") from self._reduce_error
                    ready, local_reduce_start, local_reduce_end = self._reduced[bucket_id]
                else:
                    while not vector_reducer.all_ready(
                        self.ready_flags, self.world_size, len(self.buckets),
                        bucket_id, generation,
                    ):
                        time.sleep(0.0001)
                    ready = time.monotonic()
                    local_reduce_start = time.monotonic()
                    # Round the die mean once to BF16 before BF16 communication.
                    scale = (1.0 / self.world_size if self.dtype == torch.bfloat16
                             else 1.0 / self.global_world_size)
                    vector_reducer(
                        self.flat, self.total_elements, start, elements,
                        self.local_rank, scale, parallel=False,
                    )
                    local_reduce_end = time.monotonic()
                destination = self.flat.narrow(
                    0, self.local_rank * self.total_elements + start, elements)
                # Test whether early owner3 network traffic stalls ranks that
                # are still in backward. This rank retains identical buckets,
                # reduction order and MPI calls; only its first launch moves.
                delay_ms = int(os.environ.get("WAN21_OWNER3_FIRST_COMM_DELAY_MS", "0"))
                if delay_ms not in (0, 500, 1000):
                    raise ValueError("owner3 first-communication delay must be 0, 500 or 1000 ms")
                if self.local_rank == 3 and not bucket_timings and delay_ms:
                    delay_start = time.monotonic()
                    time.sleep(delay_ms / 1000.0)
                    self.timing["owner3_first_comm_delay_seconds"] = time.monotonic() - delay_start
                collective_start = time.monotonic()
                if self.dtype == torch.bfloat16:
                    if self.replica_count > 1:
                        # HMPI transfers BF16 only; the receive-side sum is FP32/SVE.
                        vector_reducer.comm_breakdown = {}
                        if self.node_owner_group is not None:
                            vector_reducer.reduce_bf16_hierarchical(
                                destination, self.node_owner_group, self.die_owner_group,
                                self.replica_count // self.dies_per_node,
                                self.node_index, self.die_index, self.dies_per_node)
                        else:
                            vector_reducer.reduce_bf16_replicas(
                                destination, self.inter_node_group, self.replica_count,
                                self.replica_index)
                else:
                    work = dist.all_reduce(
                        destination, op=dist.ReduceOp.SUM, group=self.inter_node_group,
                        async_op=True)
                    work.wait()
                collective_end = time.monotonic()
                bucket_timings.append({
                    "bucket_id": bucket_id,
                    "offset": start,
                    "elements": elements,
                    "local_reduce_start": local_reduce_start - step_start,
                    "local_reduce_end": local_reduce_end - step_start,
                    "local_reduce_seconds": local_reduce_end - local_reduce_start,
                    "ready": ready - step_start,
                    "collective_start": collective_start - step_start,
                    "collective_seconds": collective_end - collective_start,
                    "collective_phases": vector_reducer.comm_breakdown.copy() if self.dtype == torch.bfloat16 else {},
                    "collective_end": collective_end - step_start,
                })
                if "owner_ready" not in self.timing:
                    self.timing["owner_ready"] = ready
                    self.timing["collective_start"] = collective_start
                self.timing["collective_end"] = collective_end
            self.timing["bucket_count"] = len(bucket_timings)
            self.timing["bucket_timings"] = bucket_timings
            self.timing["local_reduce_seconds"] = sum(
                bucket["local_reduce_seconds"] for bucket in bucket_timings)
            self.timing["collective_seconds"] = sum(
                bucket["collective_seconds"] for bucket in bucket_timings)
        except BaseException as error:
            self._worker_error = error
        finally:
            self._complete.set()

    def finish_step(self) -> float:
        """Wait for this owner's inter-node work, then clip the global norm."""
        self._complete.wait()
        assert self._worker is not None
        self._worker.join()
        self._worker = None
        if self._reduce_worker is not None:
            self._reduce_worker.join()
            self._reduce_worker = None
        if self._reduce_error is not None:
            raise RuntimeError("pipeline local reduction failed") from self._reduce_error
        if self._worker_error is not None:
            raise RuntimeError("hierarchical gradient communication failed") from self._worker_error
        self.timing["published_seconds"] = self._published_seconds.copy()
        # The communication thread is joined before the main thread resumes MPI,
        # satisfying HMPI's MPI_THREAD_SERIALIZED level.
        barrier_start = time.monotonic()
        if os.environ.get("WAN21_COMM_SKIP_CLIP_BARRIER", "1") != "1":
            dist.barrier()
        self.timing["clip_barrier_seconds"] = time.monotonic() - barrier_start
        norm_start = time.monotonic()
        norm_squared = torch.zeros(1, dtype=torch.float64)
        for parameter in self.owner_parameters[self.local_rank]:
            value = parameter.grad.float().norm()
            norm_squared += value.double().square() / self.replica_count
        self.timing["clip_local_norm_seconds"] = time.monotonic() - norm_start
        global_norm_start = time.monotonic()
        dist.all_reduce(norm_squared, op=dist.ReduceOp.SUM)
        self.timing["clip_global_norm_seconds"] = time.monotonic() - global_norm_start
        norm = float(norm_squared.sqrt().item())
        if not torch.isfinite(norm_squared).all():
            raise RuntimeError("non-finite global gradient norm")
        factor = min(1.0, getattr(self, "clip_norm", 1.0) / (norm + 1e-6))
        clip_start = time.monotonic()
        if factor < 1.0:
            for parameter in self.owner_parameters[self.local_rank]:
                parameter.grad.mul_(factor)
        self.timing["clip_apply_seconds"] = time.monotonic() - clip_start
        return norm



class VectorReducer:
    """Small optional C/OpenMP/SVE bridge; torch fallback remains available."""

    def __init__(self, library: Path, reduce_enabled: bool = True):
        self.library = ctypes.CDLL(str(library))
        self.reduce_enabled = reduce_enabled
        self.function = self.library.wan21_reduce4_f32
        self.serial_function = self.library.wan21_reduce4_f32_serial
        signature = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_float,
        ]
        self.function.argtypes = signature
        self.function.restype = None
        self.serial_function.argtypes = signature
        self.serial_function.restype = None
        self.direct_function = self.library.wan21_reduce4_bf16_serial
        self.direct_function.argtypes = signature
        self.direct_function.restype = None
        self.direct_replicas_function = self.library.wan21_reduce_replicas_bf16_serial
        self.direct_replicas_function.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
            ctypes.c_int64, ctypes.c_float]
        self.direct_replicas_function.restype = None
        self.direct_enabled = os.environ.get("WAN21_BF16_DIRECT_REDUCE", "1") == "1"
        self.store_function = self.library.wan21_store_release_i32
        self.store_function.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        self.store_function.restype = None
        self.load_function = self.library.wan21_load_acquire_i32
        self.load_function.argtypes = [ctypes.c_void_p]
        self.load_function.restype = ctypes.c_int32
        # Reused by the one communication thread; first-touched inside its die.
        self.bf16_chunk_elements = 2 * 1024 * 1024
        self._bf16_scratch: torch.Tensor | None = None
        self._bf16_receives: list[torch.Tensor] = []
        self._bf16_padding: torch.Tensor | None = None
        self._bf16_inbound: torch.Tensor | None = None
        self._bf16_outbound: torch.Tensor | None = None
        self._bf16_replica_count = 0
        self._hbm_scratch_enabled = os.environ.get("WAN21_COMM_SCRATCH_HBM", "0") == "1"
        self._hbm_scratch_mmaps: list[mmap.mmap] = []
        self._hier_node_inbound: torch.Tensor | None = None
        self._hier_node_outbound: torch.Tensor | None = None
        self._hier_node_padding: torch.Tensor | None = None
        self._hier_node_receives: list[torch.Tensor] = []
        self.comm_breakdown = {}
        self.direct_gather = os.environ.get("WAN21_BF16_DIRECT_GATHER", "0") == "1"
        self.bf16_comm_chunk_mb = int(os.environ.get("WAN21_BF16_COMM_CHUNK_MB", "64"))
        if self.bf16_comm_chunk_mb not in (16, 32, 64, 128):
            raise ValueError("BF16 communication chunk must be 16, 32, 64 or 128 MiB")

    def _allocate_int16(self, elements: int) -> torch.Tensor:
        """Allocate persistent communication buffers on this rank's HBM NUMA."""
        if not self._hbm_scratch_enabled:
            return torch.empty(elements, dtype=torch.int16)
        if elements < 1:
            raise ValueError("communication buffer must be nonempty")
        node = 16 + int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        backing = mmap.mmap(-1, elements * 2,
                            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        address = ctypes.addressof(ctypes.c_char.from_buffer(backing))
        mask = ctypes.c_ulong(1 << node)
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
        libnuma.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong,
                                 ctypes.c_uint]
        libnuma.mbind.restype = ctypes.c_int
        if libnuma.mbind(ctypes.c_void_p(address), ctypes.c_ulong(elements * 2),
                        ctypes.c_int(2), ctypes.byref(mask), ctypes.c_ulong(64),
                        ctypes.c_uint(0)) != 0:
            error = ctypes.get_errno()
            backing.close()
            raise OSError(error, f"cannot bind communication buffer to HBM NUMA {node}")
        tensor = torch.frombuffer(backing, dtype=torch.int16, count=elements)
        if tensor.data_ptr() != address:
            raise RuntimeError("HBM communication buffer was copied")
        self._hbm_scratch_mmaps.append(backing)
        print(f"WAN21_COMM_HBM_SCRATCH rank={os.environ['OMPI_COMM_WORLD_RANK']} "
              f"node={node} bytes={elements * 2}", flush=True)
        return tensor

    def validate_bf16_direct(self) -> None:
        """Check exact BF16 values and untouched slices before enabling the native path."""
        if not self.direct_enabled:
            return
        generator = torch.Generator().manual_seed(44531)
        total, offset, elements = 10007, 19, 9803
        for scale in (0.25, 0.0625, 1.0):
            for owner in range(4):
                sample = (torch.randn((4, total), generator=generator) * 3).to(torch.bfloat16)
                reference, result = sample.flatten().clone(), sample.flatten().clone()
                self.direct_enabled = False
                try:
                    self(reference, total, offset, elements, owner, scale, parallel=False)
                finally:
                    self.direct_enabled = True
                self(result, total, offset, elements, owner, scale, parallel=False)
                if not torch.equal(reference, result):
                    mismatches = int((reference != result).sum())
                    raise RuntimeError(
                        f"native BF16 four-rank reduction mismatch: "
                        f"scale={scale}, owner={owner}, elements={mismatches}")
        for replicas in (2, 4, 16):
            samples = (torch.randn((replicas, total), generator=generator) * 3).to(torch.bfloat16)
            reference = samples[0].float().clone()
            for replica in range(1, replicas):
                reference.add_(samples[replica].float())
            reference = (reference / replicas).to(torch.bfloat16)
            result = torch.empty_like(reference)
            self.direct_replicas_function(
                ctypes.c_void_p(samples.data_ptr()), ctypes.c_void_p(result.data_ptr()),
                ctypes.c_int64(total), ctypes.c_int64(total),
                ctypes.c_int64(replicas), ctypes.c_float(1.0 / replicas))
            if not torch.equal(reference, result):
                mismatches = int((reference != result).sum())
                raise RuntimeError(
                    f"native BF16 replica reduction mismatch: "
                    f"replicas={replicas}, elements={mismatches}")

    def _scratch(self) -> torch.Tensor:
        if self._bf16_scratch is None:
            self._bf16_scratch = torch.empty(
                (4, self.bf16_chunk_elements), dtype=torch.float32)
        return self._bf16_scratch

    def reduce_bf16_replicas(self, destination: torch.Tensor, group,
                             replicas: int, replica_index: int) -> None:
        """BF16 all-to-all shards, FP32/SVE sum, BF16 all-gather of results."""
        if destination.dtype != torch.bfloat16 or replicas < 2:
            raise ValueError("BF16 inter-die reduction requires at least two replicas")
        if not 0 <= replica_index < replicas:
            raise ValueError("invalid replica index")
        # Four old chunks per call; the FP32 reducer scratch stays <= 2 Mi elements.
        block_elements = (self.bf16_comm_chunk_mb * 1024 * 1024 // 2 if self.direct_enabled
                          else self.bf16_chunk_elements * min(4, replicas))
        shard_capacity = (block_elements + replicas - 1) // replicas
        # Allocate once per rank. Each owner reuses its die-local buffers on
        # every gradient bucket and training step; MPI calls are serialized.
        if (self._bf16_replica_count != replicas or self._bf16_inbound is None
                or self._bf16_inbound.numel() != block_elements):
            self._bf16_padding = self._allocate_int16(block_elements)
            self._bf16_inbound = self._allocate_int16(block_elements)
            self._bf16_outbound = self._allocate_int16(shard_capacity)
            self._bf16_receives = [self._allocate_int16(shard_capacity)
                                   for _ in range(replicas)]
            self._bf16_replica_count = replicas
        assert self._bf16_padding is not None and self._bf16_inbound is not None
        assert self._bf16_outbound is not None
        padding, inbound = self._bf16_padding, self._bf16_inbound
        outbound, received = self._bf16_outbound, self._bf16_receives
        scratch = None if self.direct_enabled else self._scratch()
        function = self.serial_function
        for offset in range(0, destination.numel(), block_elements):
            count = min(block_elements, destination.numel() - offset)
            shard_elements = (count + replicas - 1) // replicas
            padded = shard_elements * replicas
            pack_start = time.monotonic()
            if count == padded:
                send_bits = destination.narrow(0, offset, count).view(torch.int16)
            else:
                padding[:padded].zero_()
                padding[:count].copy_(destination.narrow(0, offset, count).view(torch.int16))
                send_bits = padding[:padded]
            alltoall_start = time.monotonic()
            dist.all_to_all_single(inbound[:padded], send_bits, group=group)
            replica_start = time.monotonic()
            if self.direct_enabled and self.reduce_enabled:
                self.direct_replicas_function(
                    ctypes.c_void_p(inbound.data_ptr()),
                    ctypes.c_void_p(outbound.data_ptr()),
                    ctypes.c_int64(shard_elements), ctypes.c_int64(shard_elements),
                    ctypes.c_int64(replicas), ctypes.c_float(1.0 / replicas))
            else:
                assert scratch is not None
                cursor = 0
                while cursor < replicas:
                    if cursor == 0:
                        width = min(4, replicas)
                        for index in range(width):
                            scratch[index, :shard_elements].copy_(
                                inbound[index * shard_elements:(index + 1) * shard_elements].view(torch.bfloat16))
                        for index in range(width, 4):
                            scratch[index, :shard_elements].zero_()
                        cursor = width
                    else:
                        width = min(3, replicas - cursor)
                        for index in range(width):
                            src = cursor + index
                            scratch[index + 1, :shard_elements].copy_(
                                inbound[src * shard_elements:(src + 1) * shard_elements].view(torch.bfloat16))
                        for index in range(width + 1, 4):
                            scratch[index, :shard_elements].zero_()
                        cursor += width
                    if self.reduce_enabled:
                        function(ctypes.c_void_p(scratch.data_ptr()),
                                 ctypes.c_int64(self.bf16_chunk_elements), ctypes.c_int64(0),
                                 ctypes.c_int64(shard_elements), ctypes.c_int64(0), ctypes.c_float(1.0))
                    else:
                        scratch[0, :shard_elements].copy_(scratch[:, :shard_elements].sum(dim=0))

                outbound[:shard_elements].view(torch.bfloat16).copy_(
                    scratch[0, :shard_elements] / replicas)
            allgather_start = time.monotonic()
            direct_output = self.direct_gather and count == padded
            if direct_output:
                gathered = [destination.narrow(
                    0, offset + source * shard_elements, shard_elements).view(torch.int16)
                    for source in range(replicas)]
            else:
                gathered = [tensor[:shard_elements] for tensor in received]
            dist.all_gather(gathered, outbound[:shard_elements], group=group)
            unpack_start = time.monotonic()
            if not direct_output:
                for source in range(replicas):
                    relative = source * shard_elements
                    valid = min(shard_elements, count - relative)
                    if valid > 0:
                        destination.narrow(0, offset + relative, valid).copy_(
                            gathered[source][:valid].view(torch.bfloat16))
            unpack_end = time.monotonic()
            for key, duration in (("pack", alltoall_start - pack_start),
                                  ("all_to_all", replica_start - alltoall_start),
                                  ("replica_reduce", allgather_start - replica_start),
                                  ("all_gather", unpack_start - allgather_start),
                                  ("unpack", unpack_end - unpack_start)):
                self.comm_breakdown[key] = self.comm_breakdown.get(key, 0.0) + duration

    def reduce_bf16_hierarchical(self, destination: torch.Tensor, node_group,
                                 die_group, nodes: int, node_index: int,
                                 die_index: int, dies: int) -> None:
        """Reduce-scatter among local dies, reduce BF16 shards across nodes, gather dies.

        The node-local stage rounds once to BF16 before the inter-node exchange.
        This is an experimental numerical order and stays opt-in until compared
        with the original full-replica FP32/SVE accumulation.
        """
        if not self.direct_enabled or not self.reduce_enabled or dies != 4:
            raise RuntimeError("node-aware BF16 exchange needs the native four-rank reducer")
        if destination.dtype != torch.bfloat16 or not 0 <= die_index < dies or not 0 <= node_index < nodes:
            raise ValueError("invalid node-aware gradient exchange layout")
        block = self.bf16_comm_chunk_mb * 1024 * 1024 // 2
        capacity = (block + dies - 1) // dies
        if self._hier_node_inbound is None or self._hier_node_inbound.numel() != block:
            self._hier_node_inbound = self._allocate_int16(block)
            self._hier_node_outbound = self._allocate_int16(capacity)
            self._hier_node_padding = self._allocate_int16(block)
            self._hier_node_receives = [self._allocate_int16(capacity)
                                        for _ in range(dies)]
        assert self._hier_node_inbound is not None and self._hier_node_outbound is not None
        assert self._hier_node_padding is not None
        for offset in range(0, destination.numel(), block):
            count = min(block, destination.numel() - offset)
            shard = (count + dies - 1) // dies
            padded = shard * dies
            if count == padded:
                send = destination.narrow(0, offset, count).view(torch.int16)
            else:
                self._hier_node_padding[:padded].zero_()
                self._hier_node_padding[:count].copy_(
                    destination.narrow(0, offset, count).view(torch.int16))
                send = self._hier_node_padding[:padded]
            start = time.monotonic()
            dist.all_to_all_single(self._hier_node_inbound[:padded], send,
                                   group=node_group)
            reduced_start = time.monotonic()
            self.direct_replicas_function(
                ctypes.c_void_p(self._hier_node_inbound.data_ptr()),
                ctypes.c_void_p(self._hier_node_outbound.data_ptr()),
                ctypes.c_int64(shard), ctypes.c_int64(shard),
                ctypes.c_int64(dies), ctypes.c_float(1.0 / dies))
            cross_node_start = time.monotonic()
            local_shard = self._hier_node_outbound[:shard].view(torch.bfloat16)
            if nodes > 1:
                self.reduce_bf16_replicas(local_shard, die_group, nodes, node_index)
            cross_node_end = time.monotonic()
            if count == padded:
                gathered = [destination.narrow(0, offset + die * shard, shard).view(torch.int16)
                            for die in range(dies)]
            else:
                gathered = [receive[:shard] for receive in self._hier_node_receives]
            dist.all_gather(gathered, local_shard.view(torch.int16), group=node_group)
            unpack_start = time.monotonic()
            if count != padded:
                for die, receive in enumerate(gathered):
                    valid = min(shard, count - die * shard)
                    if valid > 0:
                        destination.narrow(0, offset + die * shard, valid).copy_(
                            receive[:valid].view(torch.bfloat16))
            end = time.monotonic()
            for name, duration in (("node_all_to_all", reduced_start - start),
                                   ("node_reduce", cross_node_start - reduced_start),
                                   ("node_all_gather", unpack_start - cross_node_end),
                                   ("node_unpack", end - unpack_start)):
                self.comm_breakdown[name] = self.comm_breakdown.get(name, 0.0) + duration

    @staticmethod
    def _int32_address(tensor: torch.Tensor, index: int) -> ctypes.c_void_p:
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError("atomic ready flags must be contiguous int32")
        if not 0 <= index < tensor.numel():
            raise IndexError("atomic ready flag index is out of range")
        return ctypes.c_void_p(tensor.data_ptr() + index * 4)

    def store_ready(self, tensor: torch.Tensor, index: int, value: int) -> None:
        self.store_function(self._int32_address(tensor, index), ctypes.c_int32(value))

    def all_ready(
        self, tensor: torch.Tensor, rows: int, columns: int,
        column: int, value: int,
    ) -> bool:
        return all(
            self.load_function(self._int32_address(tensor, row * columns + column))
            == value
            for row in range(rows)
        )

    def __call__(self, flat: torch.Tensor, total: int, offset: int,
                 elements: int, destination_rank: int, scale: float,
                 parallel: bool = True) -> None:
        if flat.dtype == torch.bfloat16:
            if self.direct_enabled and self.reduce_enabled:
                self.direct_function(
                    ctypes.c_void_p(flat.data_ptr()), ctypes.c_int64(total),
                    ctypes.c_int64(offset), ctypes.c_int64(elements),
                    ctypes.c_int64(destination_rank), ctypes.c_float(scale))
                return
            scratch = self._scratch()
            function = self.function if parallel else self.serial_function
            for chunk_offset in range(offset, offset + elements, self.bf16_chunk_elements):
                count = min(self.bf16_chunk_elements, offset + elements - chunk_offset)
                for source_rank in range(4):
                    scratch[source_rank, :count].copy_(flat.narrow(
                        0, source_rank * total + chunk_offset, count))
                if self.reduce_enabled:
                    function(ctypes.c_void_p(scratch.data_ptr()),
                             ctypes.c_int64(self.bf16_chunk_elements), ctypes.c_int64(0),
                             ctypes.c_int64(count), ctypes.c_int64(destination_rank),
                             ctypes.c_float(scale))
                else:
                    scratch[destination_rank, :count].copy_(
                        scratch[:, :count].sum(dim=0) * scale)
                flat.narrow(0, destination_rank * total + chunk_offset, count).copy_(
                    scratch[destination_rank, :count])
            return
        if flat.dtype != torch.float32:
            raise ValueError("SVE reduction requires FP32 or BF16 storage")
        if not self.reduce_enabled:
            destination = flat.narrow(
                0, destination_rank * total + offset, elements)
            destination.mul_(scale)
            for source_rank in range(4):
                if source_rank != destination_rank:
                    destination.add_(flat.narrow(
                        0, source_rank * total + offset, elements), alpha=scale)
            return
        function = self.function if parallel else self.serial_function
        function(
            ctypes.c_void_p(flat.data_ptr()), ctypes.c_int64(total), ctypes.c_int64(offset),
            ctypes.c_int64(elements), ctypes.c_int64(destination_rank), ctypes.c_float(scale),
        )



def build_vector_reducer(run_dir: Path, rank: int) -> VectorReducer | None:
    """Build once per job; use SVE when supported and fall back to generic C."""
    reduce_enabled = os.environ.get("WAN21_SHARED_DIE_VECTOR_REDUCE", "1") != "0"
    source = Path(__file__).with_name("shared_reduce_sve.c")
    library = run_dir / "shared_reduce_sve.so"
    status = run_dir / "shared_reduce_sve.status"
    if rank == 0:
        commands = [
            ("sve2bf16", ["cc", "-O3", "-fopenmp", "-fPIC", "-shared",
                           "-march=armv8.2-a+sve2+bf16", str(source), "-o", str(library)]),
            ("sve2", ["cc", "-O3", "-fopenmp", "-fPIC", "-shared",
                      "-march=armv8.2-a+sve2", str(source), "-o", str(library)]),
            ("sve", ["cc", "-O3", "-fopenmp", "-fPIC", "-shared",
                     "-march=armv8.2-a+sve", str(source), "-o", str(library)]),
            ("generic", ["cc", "-O3", "-fopenmp", "-fPIC", "-shared",
                         str(source), "-o", str(library)]),
        ]
        result = None
        selected = None
        for name, command in commands:
            if shutil.which(command[0]) is None:
                continue
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                selected = name
                break
        if result is not None and result.returncode == 0:
            status.write_text(f"ok:{selected}\n", encoding="ascii")
            state = "enabled" if reduce_enabled else "disabled"
            print(
                f"SHARED_DIE_VECTOR_REDUCE {state} arch={selected} atomic_ready=enabled",
                flush=True,
            )
        else:
            detail = "compiler unavailable" if result is None else result.stderr[-500:]
            status.write_text("fallback\n", encoding="utf-8")
            print(f"SHARED_DIE_VECTOR_REDUCE fallback: {detail}", flush=True)
    dist.barrier()
    if not status.is_file() or not status.read_text(encoding="utf-8").startswith("ok:"):
        return None
    reducer = VectorReducer(library, reduce_enabled=reduce_enabled)
    reducer.validate_bf16_direct()
    if reducer.direct_enabled and rank == 0:
        print("BF16_DIRECT_NUMERIC_OK local4_and_replica2_4_16", flush=True)
    return reducer



def memory_record(rank: int) -> dict[str, int]:
    values = {"rank": rank, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, separator, raw = line.partition(":")
        if separator and name in {"VmRSS", "RssAnon", "RssFile", "RssShmem"}:
            values[f"{name.lower()}_kib"] = int(raw.split()[0])
    return values


