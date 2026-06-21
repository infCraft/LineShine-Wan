#!/usr/bin/env python3
"""Single-GPU training-throughput profiler for the Wan DiT flow-matching loop.

Purpose: attribute per-step time to dataloader vs compute, estimate MFU, and
quantify the A-class data-pipeline / batch-size wins *before* changing train.py.

It is read-only with respect to training state: it builds a model (random or
from a checkpoint) and never writes checkpoints or metrics. Run inside a Slurm
GPU job, e.g.:

  srun -p compute --gres=gpu:1 --cpus-per-gpu=8 --mem=64G \
    bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate lineshine-wan \
      && cd $CODE && PYTHONPATH=. python src/train/profile_speed.py \
      --cache-dir $ROOT/cache/train --cache-pattern "train-*.tar" --max-shards 8 \
      --report $ROOT/reports/profile/w5_profile.json'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train.dataset import CacheDataset, collate_samples
from src.train.flow import add_flow_noise, sample_logit_normal_sigmas
from src.train.model_adapter import call_wan_model
from src.train.train import create_model

# A800 / A100 dense bf16 tensor-core peak (FLOP/s). Used only for an approximate MFU.
A800_BF16_PEAK = 312e12
TOKENS_PER_SAMPLE = 13 * 16 * 16  # (1+48/4) x 256/16 x 256/16 for 49f / 256^2 / patch (1,2,2)


def build_dataset(cache_dir: Path, pattern: str, max_shards: int) -> CacheDataset:
    shards = sorted(cache_dir.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"no shards matched {cache_dir}/{pattern}")
    if max_shards > 0:
        shards = shards[:max_shards]
    return CacheDataset(shards)


def make_loader(ds, batch_size: int, num_workers: int, pin: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_samples,
        drop_last=True,
        pin_memory=pin,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)


def to_device(batch, device):
    latents = batch["latents"].to(device, dtype=torch.bfloat16, non_blocking=True)
    contexts = [c.to(device, dtype=torch.bfloat16, non_blocking=True) for c in batch["contexts"]]
    return latents, contexts


def fwd_bwd(model, latents, contexts):
    sigmas = sample_logit_normal_sigmas(latents.shape[0], device=latents.device)
    noisy, target, timesteps = add_flow_noise(latents, sigmas=sigmas)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = call_wan_model(model, noisy, timesteps, contexts)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
    loss.backward()
    return float(loss.detach())


def compute_only(model, optimizer, device, batch_size, grad_accum, warmup, iters):
    """Throughput with one batch resident on GPU (no dataloader in the loop)."""
    latents = torch.randn(batch_size, 16, 13, 32, 32, device=device, dtype=torch.bfloat16)
    contexts = [torch.randn(512, 4096, device=device, dtype=torch.bfloat16) for _ in range(batch_size)]
    micro = 0
    for _ in range(warmup):
        fwd_bwd(model, latents, contexts)
        micro += 1
        if micro % grad_accum == 0:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for _ in range(iters):
        fwd_bwd(model, latents, contexts)
        micro += 1
        n += batch_size
        if micro % grad_accum == 0:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    optimizer.zero_grad(set_to_none=True)
    return n / dt


def data_only(loader, device, max_batches):
    """Pure dataloader throughput: fetch + H2D, no compute."""
    it = iter(loader)
    # warmup one batch (spins up workers)
    b = next(it); to_device(b, device); torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    for i, b in enumerate(it):
        lat, _ = to_device(b, device)
        n += lat.shape[0]
        if i + 1 >= max_batches:
            break
    torch.cuda.synchronize()
    return n / (time.perf_counter() - t0)


def end_to_end(model, optimizer, loader, device, grad_accum, warmup, iters):
    """Full step throughput including dataloader."""
    micro = 0
    seen = 0
    t0 = None
    n = 0
    for b in loader:
        lat, ctx = to_device(b, device)
        fwd_bwd(model, lat, ctx)
        micro += 1
        if micro % grad_accum == 0:
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        seen += 1
        if seen == warmup:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        elif seen > warmup:
            n += lat.shape[0]
            if seen >= warmup + iters:
                break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    optimizer.zero_grad(set_to_none=True)
    return n / dt


def oom_safe(fn, *a, **k):
    """Run fn; on CUDA OOM clean up and return None instead of crashing the sweep."""
    try:
        return fn(*a, **k)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def mfu(samples_per_sec: float, n_params: int) -> float:
    flops = samples_per_sec * TOKENS_PER_SAMPLE * 6 * n_params  # ~6ND fwd+bwd
    return flops / A800_BF16_PEAK


def implied_8gpu_step_s(samples_per_sec_1gpu: float, eff_batch: int = 64, gpus: int = 8) -> float:
    return (eff_batch / gpus) / samples_per_sec_1gpu


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--cache-pattern", default="train-*.tar")
    p.add_argument("--max-shards", type=int, default=8, help="limit shards scanned at init for fast profiling")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--compile", action="store_true", help="also profile torch.compile compute path")
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a GPU (run inside Slurm)"
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    class _A:  # minimal namespace for create_model
        tiny_model = False
        text_len = 512
        latent_channels = 16
        text_dim = 4096
    model = create_model(_A()).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    ds = build_dataset(args.cache_dir, args.cache_pattern, args.max_shards)
    result = {
        "n_params": n_params,
        "tokens_per_sample": TOKENS_PER_SAMPLE,
        "dataset_samples_scanned": len(ds),
        "gpu": torch.cuda.get_device_name(0),
        "compute_only": {},
        "data_only": {},
        "end_to_end": {},
        "notes": "MFU is approximate (6*N*tokens, ignores attention quadratic). data_only may be optimistic due to OS page cache.",
    }

    # 1) compute-only upper bound: sweep batch sizes, stop at first OOM (memory ceiling)
    for bs in [1, 2, 3, 4]:
        ga = max(1, 8 // bs)
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        sps = oom_safe(compute_only, model, optimizer, device, bs, ga, args.warmup, args.iters)
        if sps is None:
            result["compute_only"][f"bs{bs}"] = "OOM"
            print(f"[compute-only] bs={bs}: OOM (memory ceiling without grad-checkpointing)")
            break
        peak = torch.cuda.max_memory_allocated() / 1e9
        result["compute_only"][f"bs{bs}_ga{ga}"] = {
            "samples_per_sec": round(sps, 3),
            "mfu": round(mfu(sps, n_params), 4),
            "implied_8gpu_step_s_eff64": round(implied_8gpu_step_s(sps), 3),
            "peak_mem_gb": round(peak, 2),
        }
        print(f"[compute-only] bs={bs} ga={ga}: {sps:.2f} samp/s  MFU~{mfu(sps,n_params)*100:.1f}%  "
              f"8gpu eff64 step~{implied_8gpu_step_s(sps):.2f}s  peak={peak:.1f}GB")
        torch.cuda.reset_peak_memory_stats()

    if args.compile:
        cmodel = torch.compile(model)
        sps = oom_safe(compute_only, cmodel, optimizer, device, 2, 4, max(args.warmup, 8), args.iters)
        if sps is not None:
            result["compute_only"]["bs2_compiled"] = {
                "samples_per_sec": round(sps, 3),
                "mfu": round(mfu(sps, n_params), 4),
            }
            print(f"[compute-only] bs=2 compiled: {sps:.2f} samp/s  MFU~{mfu(sps,n_params)*100:.1f}%")

    # 2) dataloader sweep: data-only + end-to-end (batch capped at memory ceiling)
    sweep = [
        ("baseline_bs1_nw0", 1, 0, False, 8),
        ("bs1_nw4_pin", 1, 4, True, 8),
        ("bs2_nw4_pin", 2, 4, True, 4),
        ("bs3_nw6_pin", 3, 6, True, 3),
    ]
    for name, bs, nw, pin, ga in sweep:
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        loader = make_loader(ds, bs, nw, pin)
        d_sps = oom_safe(data_only, loader, device, max(args.iters, 16))
        del loader
        loader = make_loader(ds, bs, nw, pin)
        e_sps = oom_safe(end_to_end, model, optimizer, loader, device, ga, args.warmup, args.iters)
        del loader
        result["data_only"][name] = None if d_sps is None else round(d_sps, 3)
        if e_sps is None:
            result["end_to_end"][name] = "OOM"
            print(f"[sweep] {name:18s} data-only={d_sps}  e2e=OOM")
            continue
        result["end_to_end"][name] = {
            "samples_per_sec": round(e_sps, 3),
            "implied_8gpu_step_s_eff64": round(implied_8gpu_step_s(e_sps), 3),
            "mfu": round(mfu(e_sps, n_params), 4),
        }
        print(f"[sweep] {name:18s} data-only={d_sps:7.2f} samp/s  e2e={e_sps:7.2f} samp/s  "
              f"8gpu eff64 step~{implied_8gpu_step_s(e_sps):.2f}s  MFU~{mfu(e_sps,n_params)*100:.1f}%")

    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
