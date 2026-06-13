#!/usr/bin/env python3
"""Wan DiT flow-matching training loop for cache smoke and baseline runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, WAN_ROOT, ensure_dir, write_json
from src.train.ckpt import latest_checkpoint, load_checkpoint, save_checkpoint
from src.train.dataset import CacheDataset, SyntheticDataset, collate_samples
from src.train.flow import add_flow_noise, apply_cfg_dropout, sample_logit_normal_sigmas
from src.train.metrics import JsonlMetrics, TbMetrics
from src.train.model_adapter import call_wan_model


def add_wan_to_path() -> None:
    wan = str(WAN_ROOT)
    if wan not in sys.path:
        sys.path.insert(0, wan)


def init_distributed() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return True, rank, world, local_rank


def create_model(args: argparse.Namespace):
    add_wan_to_path()
    from wan.configs.wan_t2v_1_3B import t2v_1_3B
    from wan.modules.model import WanModel

    if args.tiny_model:
        return WanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=args.text_len,
            in_dim=args.latent_channels,
            dim=args.tiny_dim,
            ffn_dim=args.tiny_ffn_dim,
            freq_dim=64,
            text_dim=args.text_dim,
            out_dim=args.latent_channels,
            num_heads=args.tiny_heads,
            num_layers=args.tiny_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
        )
    return WanModel(
        model_type="t2v",
        patch_size=t2v_1_3B.patch_size,
        text_len=t2v_1_3B.text_len,
        in_dim=16,
        dim=t2v_1_3B.dim,
        ffn_dim=t2v_1_3B.ffn_dim,
        freq_dim=t2v_1_3B.freq_dim,
        text_dim=4096,
        out_dim=16,
        num_heads=t2v_1_3B.num_heads,
        num_layers=t2v_1_3B.num_layers,
        window_size=t2v_1_3B.window_size,
        qk_norm=t2v_1_3B.qk_norm,
        cross_attn_norm=t2v_1_3B.cross_attn_norm,
        eps=t2v_1_3B.eps,
    )


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
    def fn(step: int) -> float:
        if step < args.warmup_steps:
            return max(1, step + 1) / max(1, args.warmup_steps)
        if args.max_steps <= args.warmup_steps:
            return 1.0
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        min_ratio = args.min_lr / args.lr
        return min_ratio + (1.0 - min_ratio) * cosine

    return fn


def load_empty_context(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if args.empty_context is not None:
        from safetensors.torch import load_file

        tensors = load_file(args.empty_context)
        return tensors["context"].to(device=device, dtype=dtype)
    return torch.zeros((1, args.text_dim), device=device, dtype=dtype)


def make_dataset(args: argparse.Namespace):
    if args.synthetic:
        dataset = SyntheticDataset(
            count=args.synthetic_count,
            latent_shape=(args.latent_channels, args.latent_frames, args.latent_height, args.latent_width),
            text_len=args.synthetic_text_len,
            text_dim=args.text_dim,
        )
    else:
        dataset = CacheDataset.from_dir(args.cache_dir, args.cache_pattern)
    if args.dataset_limit is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(min(args.dataset_limit, len(dataset)))))
    return dataset


@torch.no_grad()
def evaluate(model, loader, args: argparse.Namespace, device: torch.device, empty_context: torch.Tensor, max_batches: int) -> float:
    model.eval()
    losses = []
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
        contexts = [ctx.to(device=device, dtype=torch.bfloat16) for ctx in batch["contexts"]]
        sigmas = sample_logit_normal_sigmas(latents.shape[0], device=device)
        noisy, target, timesteps = add_flow_noise(latents, sigmas=sigmas)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = call_wan_model(model, noisy, timesteps, contexts)
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train(args: argparse.Namespace) -> None:
    distributed, rank, world, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed + rank)

    dataset = make_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None and args.shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
        drop_last=True,
    )

    model = create_model(args).to(device)
    if args.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args))
    empty_context = load_empty_context(args, device, torch.bfloat16)
    metrics = JsonlMetrics(args.run_dir / "metrics.jsonl") if rank == 0 else None
    tb = TbMetrics(args.run_dir / "tb", enabled=(rank == 0 and args.tensorboard))
    fixed_sigmas = None
    fixed_noise = None
    if args.fixed_noise:
        gen = torch.Generator(device=device).manual_seed(args.seed + 999)
        fixed_sigmas = sample_logit_normal_sigmas(args.batch_size, device=device, generator=gen)
        fixed_noise = torch.randn(
            (args.batch_size, args.latent_channels, args.latent_frames, args.latent_height, args.latent_width),
            device=device,
            dtype=torch.bfloat16,
            generator=gen,
        )
    ensure_dir(args.run_dir)
    if rank == 0:
        write_json(args.run_dir / "train_config.json", vars(args))

    start_step = 0
    ckpt = args.resume or latest_checkpoint(args.run_dir)
    if ckpt is not None and ckpt.exists():
        start_step = load_checkpoint(ckpt, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)

    step = start_step
    accum = 0
    model.train()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            if step >= args.max_steps:
                break
            latents = batch["latents"].to(device=device, dtype=torch.bfloat16)
            contexts = [ctx.to(device=device, dtype=torch.bfloat16) for ctx in batch["contexts"]]
            contexts = apply_cfg_dropout(contexts, empty_context, dropout_prob=args.cfg_dropout)
            if args.fixed_noise:
                if latents.shape[0] != args.batch_size:
                    continue
                sigmas = fixed_sigmas
                eps = fixed_noise.to(device=device, dtype=latents.dtype)
                sigma_view = sigmas.to(latents.dtype).reshape(sigmas.shape[0], 1, 1, 1, 1)
                noisy = (1.0 - sigma_view) * latents + sigma_view * eps
                target = eps - latents
                timesteps = sigmas.to(torch.float32) * 1000.0
            else:
                sigmas = sample_logit_normal_sigmas(latents.shape[0], device=device)
                noisy, target, timesteps = add_flow_noise(latents, sigmas=sigmas)

            sync_context = model.no_sync if distributed and ((accum + 1) % args.grad_accum_steps != 0) else None
            if sync_context is None:
                from contextlib import nullcontext

                sync_context = nullcontext
            with sync_context():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred = call_wan_model(model, noisy, timesteps, contexts)
                    loss = torch.nn.functional.mse_loss(pred.float(), target.float()) / args.grad_accum_steps
                loss.backward()
            accum += 1
            if accum % args.grad_accum_steps != 0:
                continue
            raw_loss = float(loss.item() * args.grad_accum_steps)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            step += 1

            if rank == 0 and args.val_every > 0 and step % args.val_every == 0:
                val_loss = evaluate(model.module if hasattr(model, "module") else model, loader, args, device, empty_context, args.val_batches)
                assert metrics is not None
                metrics.write({"step": step, "val_loss": val_loss})
                tb.scalar("val/loss", val_loss, step)

            if rank == 0 and (step == 1 or step % args.log_every == 0):
                assert metrics is not None
                metrics.write({"step": step, "loss": raw_loss, "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"]})
                tb.scalar("train/loss", raw_loss, step)
                tb.scalar("train/grad_norm", float(grad_norm), step)
                tb.scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            if rank == 0 and args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(args.run_dir, step=step, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, args=args)

    if rank == 0:
        save_checkpoint(args.run_dir, step=step, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, args=args)
        module = model.module if hasattr(model, "module") else model
        torch.save(module.state_dict(), args.run_dir / "wan_model_latest_state_dict.pt")
        tb.close()
        print(json.dumps({"step": step, "run_dir": str(args.run_dir)}, sort_keys=True))
    if distributed:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/smoke")
    parser.add_argument("--cache-pattern", default="smoke-*.tar")
    parser.add_argument("--run-dir", type=Path, default=root / "runs/smoke_train")
    parser.add_argument("--empty-context", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--cfg-dropout", type=float, default=0.10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--synthetic-count", type=int, default=16)
    parser.add_argument("--synthetic-text-len", type=int, default=8)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-frames", type=int, default=13)
    parser.add_argument("--latent-height", type=int, default=32)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--tiny-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tiny-dim", type=int, default=128)
    parser.add_argument("--tiny-ffn-dim", type=int, default=512)
    parser.add_argument("--tiny-heads", type=int, default=4)
    parser.add_argument("--tiny-layers", type=int, default=2)
    parser.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed-noise", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(func=train)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
