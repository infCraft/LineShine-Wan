"""Allocate temporary BF16 tensors on the current rank's corresponding HBM NUMA."""
import atexit
import ctypes
import mmap
import os
import re
from pathlib import Path
import threading
import weakref

import torch

_libnuma = ctypes.CDLL('libnuma.so.1', use_errno=True)
_libnuma.mbind.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong, ctypes.c_uint]
_libnuma.mbind.restype = ctypes.c_int
_lock = threading.Lock()
_live = 0
_peak = 0
_calls = 0
_MAX_BYTES = 3 * 1024**3
_BLOCK_BYTES = 3328 * 8960 * 2
_POOL_BLOCKS = 32
_pool = None
_pool_bytes = 0
_verified = False


def _release(size):
    global _live
    with _lock:
        _live -= size


def _pooled(shape, slot):
    """Reuse one first-touched HBM slab over every training step.

    Raw slots 0..29 belong to Wan's 30 sequential blocks. Slot 30 is
    scratch GELU activation, slot 31 is backward's second scratch tensor.
    Both scratch tensors have no graph consumers once their operator returns.
    """
    global _pool, _pool_bytes
    if slot is None or not 0 <= slot < _POOL_BLOCKS:
        raise ValueError(f'invalid FFN HBM slot {slot}')
    shape = tuple(shape)
    count = 1
    for value in shape:
        count *= value
    if count * 2 != _BLOCK_BYTES:
        raise ValueError(f'pooled FFN HBM shape changed: {shape}')
    with _lock:
        if _pool is None:
            size = _POOL_BLOCKS * _BLOCK_BYTES
            backing = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE)
            address = ctypes.addressof(ctypes.c_char.from_buffer(backing))
            node = 16 + int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            mask = ctypes.c_ulong(1 << node)
            if _libnuma.mbind(ctypes.c_void_p(address), ctypes.c_ulong(size), ctypes.c_int(2),
                             ctypes.byref(mask), ctypes.c_ulong(64), ctypes.c_uint(0)) != 0:
                error = ctypes.get_errno()
                backing.close()
                raise OSError(error, f'cannot bind FFN slab to HBM NUMA {node}')
            slots = []
            for i in range(_POOL_BLOCKS):
                view = memoryview(backing)[i * _BLOCK_BYTES:(i + 1) * _BLOCK_BYTES]
                tensor = torch.frombuffer(view, dtype=torch.bfloat16, count=count).reshape(shape)
                if tensor.data_ptr() != address + i * _BLOCK_BYTES:
                    raise RuntimeError('FFN HBM slab tensor storage was copied')
                slots.append(tensor)
            _pool = (backing, slots)
            _pool_bytes = size
            print(f'WAN21_FFN_HBM_POOL local_rank={os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]} '
                  f'node={node} bytes={size} block_bytes={_BLOCK_BYTES}', flush=True)
        return _pool[1][slot]


def verify_residency(tensor):
    """Verify pages actually touched by the GEMM reside on the rank's HBM."""
    global _verified
    if _verified or os.environ.get('WAN21_FFN_HBM_POOL') != '1':
        return
    address = tensor.data_ptr()
    mapping = None
    for line in Path('/proc/self/numa_maps').read_text().splitlines():
        try:
            begin = int(line.split(' ', 1)[0], 16)
        except ValueError:
            continue
        if begin <= address and (mapping is None or begin > mapping[0]):
            mapping = (begin, line)
    if mapping is None:
        raise RuntimeError('FFN HBM allocation not found in numa_maps')
    pages = {int(node): int(count) for node, count in re.findall(r'\bN(\d+)=(\d+)', mapping[1])}
    node = 16 + int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    if pages.get(node, 0) == 0 or any(other != node for other in pages):
        raise RuntimeError(f'FFN HBM pages on unexpected NUMA nodes: expected {node}, got {pages}')
    _verified = True
    print(f'WAN21_FFN_HBM_RESIDENCY local_rank={os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]} '
          f'node={node} pages={pages[node]}', flush=True)


def bf16(shape, slot=None):
    """Return BF16 storage from a private HBM mapping with autograd-safe ownership.

    PyTorch's frombuffer tensor owns a reference to its mmap. A reshaped view
    retains that base tensor until no user of the storage remains.
    """
    global _live, _peak, _calls
    if os.environ.get('WAN21_FFN_HBM_POOL') == '1':
        return _pooled(shape, slot)
    shape = tuple(shape)
    count = 1
    for value in shape:
        count *= value
    size = count * 2
    if size <= 0:
        raise ValueError('HBM tensor size must be positive')
    with _lock:
        if _live + size > _MAX_BYTES:
            raise MemoryError(f'FFN HBM allocation exceeds {_MAX_BYTES} bytes: {_live}+{size}')
        _live += size
        _peak = max(_peak, _live)
        _calls += 1
    backing = None
    registered = False
    try:
        backing = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
        address = ctypes.addressof(ctypes.c_char.from_buffer(backing))
        node = 16 + int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        mask = ctypes.c_ulong(1 << node)
        if _libnuma.mbind(ctypes.c_void_p(address), ctypes.c_ulong(size), ctypes.c_int(2),
                         ctypes.byref(mask), ctypes.c_ulong(64), ctypes.c_uint(0)) != 0:
            raise OSError(ctypes.get_errno(), f'cannot bind FFN tensor to HBM NUMA {node}')
        base = torch.frombuffer(backing, dtype=torch.bfloat16, count=count)
        if base.data_ptr() != address:
            raise RuntimeError('HBM tensor storage was copied')
        weakref.finalize(base, _release, size)
        registered = True
        return base.reshape(shape)
    except BaseException:
        if not registered:
            with _lock:
                _live -= size
            if backing is not None:
                backing.close()
        raise


def _report():
    if _pool is not None:
        print(f'WAN21_FFN_HBM_POOL_STATS local_rank={os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")} '
              f'reserved_bytes={_pool_bytes} residency_verified={_verified}', flush=True)
    if _calls:
        print(f'WAN21_FFN_HBM_STATS local_rank={os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")} '
              f'allocations={_calls} peak_live_bytes={_peak} final_live_bytes={_live}', flush=True)


atexit.register(_report)
