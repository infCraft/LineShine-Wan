#!/usr/bin/env python3
"""Wan CPU BF16 training with shared state and online raw-media encoders."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import resource
import socket
import sys
import time
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
from torch import nn
from torch.profiler import record_function

from src.runtime import autocast_context, init_runtime
from src.data.online_training import OnlineInput, validate_media_options
from src.train.node_memory import NodeMemoryMonitor
from src.train.ckpt import latest_checkpoint, load_model_state_only
from src.train.flow import add_flow_noise, apply_cfg_dropout, sample_logit_normal_sigmas
from src.train.model_adapter import call_wan_model
from src.train.model_factory import create_model
from src.train.train import lr_lambda, param_groups, prune_checkpoints, parse_keep_checkpoint_steps
from src.train.shared_parallel import (
    hierarchy_groups, mpi_thread_level, share_parameters, contiguous_owners_for_model,
    HierarchicalGradients, build_vector_reducer, memory_record, thread_affinities,
)
from src.train.shared_checkpoint import (
    collective_errors, save_shared_checkpoint, read_shared_checkpoint, restore_shared_state,
)


def toy_batch(rank, index):
    generator = torch.Generator().manual_seed(7000 + 100 * index + rank)
    return torch.randn(1, 4, generator=generator), torch.randn(1, 4, generator=generator)


def check_updates(optimizer, rank):
    dtypes = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.dtype not in (torch.float32, torch.bfloat16) or parameter.grad.dtype != parameter.dtype:
                raise RuntimeError("owner parameter/gradient dtype mismatch")
            dtypes.add(str(parameter.dtype))
            if not torch.isfinite(parameter).all():
                raise RuntimeError("non-finite owner parameter after AdamW")
            for key in ("exp_avg", "exp_avg_sq"):
                value = optimizer.state[parameter][key]
                if value.dtype != parameter.dtype or not torch.isfinite(value).all():
                    raise RuntimeError(f"invalid AdamW state {key}")
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    allowed = set(range(local_rank * 38, local_rank * 38 + 37))
    records = thread_affinities({int(p.name) for p in Path("/proc/self/task").iterdir()})
    if any(not set(record["affinity"]) <= allowed for record in records):
        raise RuntimeError(f"rank {rank} has a thread outside its assigned CPUs")
    return {"rank": rank, "threads": len(records), "parameter_dtypes": sorted(dtypes),
            "optimizer_state_finite": True, "thread_masks_ok": True}


def verify_toy(model, initial, args, world):
    optimizer = torch.optim.AdamW(param_groups(initial, args.weight_decay), lr=args.lr,
                                   betas=(0.9, 0.95), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args))
    for step in range(args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            for rank in range(world):
                x, target = toy_batch(rank, step * args.grad_accum + micro)
                (nn.functional.mse_loss(initial(x), target) / (world * args.grad_accum)).backward()
        nn.utils.clip_grad_norm_(initial.parameters(), args.clip_grad_norm)
        optimizer.step(); scheduler.step()
    error = max((a - b).abs().max().item() for a, b in zip(model.parameters(), initial.parameters()))
    tolerance = 2e-6 if args.stored_precision == "fp32" else 1e-2
    if error > tolerance:
        raise AssertionError(f"shared updates disagree with the serial reference: {error}")
    print("SHARED_TRAIN_TOY_OK " + json.dumps({"steps": args.max_steps, "max_error": error,
                                              "grad_accum": args.grad_accum}), flush=True)


def _export_cpu_profile(prof, output_dir, rank, first_step, steps, detailed=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
    rows = [dict(operator=e.key, calls=e.count,
                 self_cpu_time_us=e.self_cpu_time_total, total_cpu_time_us=e.cpu_time_total)
            for e in events]
    (output_dir / f"operators_rank{rank}.json").write_text(json.dumps(dict(
        rank=rank, first_step=first_step, steps=steps, operators=rows,
        note="Operator CPU times can overlap across threads and nested calls."
    ), indent=2) + "\n")
    prof.export_chrome_trace(str(output_dir / f"trace_rank{rank}.json.gz"))


def _cpu_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _bound_cpu_frequencies():
    """Sample the per-CPU kernel frequency of four assigned compute CPUs (kHz)."""
    # OMP pins the Python main thread to one CPU; sample the whole 36-core
    # range assigned by run_wan21_online_rank.sh instead of this one affinity.
    local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    cpus = (list(range(int(local_rank) * 38, int(local_rank) * 38 + 36))
            if local_rank is not None else sorted(os.sched_getaffinity(0)))
    probes = [cpus[index] for index in (0, len(cpus) // 3, 2 * len(cpus) // 3, len(cpus) - 1)]
    readings = {}
    for cpu in dict.fromkeys(probes):
        try:
            readings[str(cpu)] = int(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read_text().strip())
        except (OSError, ValueError):
            readings[str(cpu)] = None
    return readings


def train(args):
    with ExitStack() as cleanup:
        _train(args, cleanup)


def _train(args, cleanup):
    startup = time.perf_counter()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.job_token):
        raise ValueError("invalid job token")
    if args.max_steps < 1 or args.grad_accum < 1 or args.batch_size < 1:
        raise ValueError("steps, batch size and gradient accumulation must be positive")
    if args.profile_steps < 0 or args.profile_start_step < 0 or (args.profile_steps and
            (args.profile_start_step < 2 or args.profile_start_step + args.profile_steps > args.max_steps)):
        raise ValueError("profile window must fit within training steps and start at step 2 or later")
    if args.lr_total_steps < args.max_steps or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("invalid LR schedule")
    if not 0 <= args.cfg_dropout <= 1 or args.clip_grad_norm <= 0:
        raise ValueError("invalid dropout or gradient clipping")
    validate_media_options(image_size=args.image_size, video_frames=args.video_frames,
                           video_fps=args.video_fps, training_task=args.training_task)
    if not args.toy and args.data_index is None:
        raise ValueError("--data-index requires a raw-image catalog directory or raw-media JSONL manifest")
    if not args.toy and args.encoder_weights is None:
        raise ValueError("--encoder-weights is required for online encoding")
    if args.init_checkpoint and (args.resume or args.auto_resume):
        raise ValueError("weights-only initialization and resume are mutually exclusive")
    if args.stored_precision == "bf16" and args.precision != "bf16":
        raise ValueError("BF16 stored parameters require BF16 autocast")
    runtime = init_runtime(requested_device="cpu", force_cpu=True,
                           precision="fp32" if args.toy else args.precision, allow_cpu_bf16=True,
                           backend="mpi", threads=36, interop_threads=1)
    rank, world = runtime.rank, runtime.world_size
    if world != args.nnodes * args.dies_per_node * 4:
        raise ValueError("MPI world size does not match nodes and four-rank sharing groups")
    if not 0 <= args.profile_rank < world:
        raise ValueError("profile rank is outside the MPI world")
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    if len(set(hosts)) != args.nnodes or (mpi_thread_level() or 0) < 2:
        raise RuntimeError("unexpected host layout or insufficient MPI thread support")
    communication_cpu = int(os.environ["WAN21_COMMUNICATION_CPU"])
    (node, die, replica, local_rank, replicas, local_group, owner_group,
     node_owner_group, die_owner_group, workers) = hierarchy_groups(
        world, 4, args.dies_per_node, communication_cpu)
    resume = args.resume or (latest_checkpoint(args.run_dir) if args.auto_resume else None)
    error = None
    if rank == 0:
        try:
            if args.auto_resume and resume is None:
                raise FileNotFoundError("no completed checkpoint to resume")
            if resume is None and (args.run_dir / "metrics.jsonl").exists():
                raise FileExistsError("scratch training refuses an existing metrics file")
            args.run_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            error = str(exc)
    collective_errors(error)
    if args.monitor_node_memory:
        processes = [None] * world
        dist.all_gather_object(processes, {"host": socket.gethostname(), "pid": os.getpid()})
        error = None
        if int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]) == 0:
            try:
                pids = [item["pid"] for item in processes if item["host"] == socket.gethostname()]
                cleanup.enter_context(NodeMemoryMonitor(
                    args.run_dir / f"node_memory_{socket.gethostname()}.jsonl", pids,
                    cpu=min(os.sched_getaffinity(0))))
            except Exception as exc:
                error = f"memory monitor: {type(exc).__name__}: {exc}"
        collective_errors(error)
    torch.manual_seed(args.seed)
    model = (nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4)) if args.toy
             else create_model(SimpleNamespace(tiny_model=False)))
    if not args.toy and os.environ.get("WAN21_NATIVE_OPS", "0") == "1":
        import types as _types
        from wan.modules import model as _wan
        from src.train.modulation import modulate as _modulate
        from src.train.selective_ffn import recompute as _recompute
        def _optimized_block(self,x,e,seq_lens,grid_sizes,freqs,context,context_lens):
            with torch.autocast(device_type=x.device.type,enabled=False):
                modulation=(self.modulation+e).chunk(6,dim=1)
            y=self.self_attn(_modulate(self.norm1(x),modulation[0],modulation[1]),
                             seq_lens,grid_sizes,freqs)
            with torch.autocast(device_type=x.device.type,enabled=False):
                x=_wan._wan_residual(x,y,modulation[2])
            x=x+self.cross_attn(self.norm3(x),context,context_lens)
            y=self.ffn(_modulate(self.norm2(x),modulation[3],modulation[4]))
            with torch.autocast(device_type=x.device.type,enabled=False):
                x=_wan._wan_residual(x,y,modulation[5])
            return x
        _wan.WanAttentionBlock.forward=_optimized_block
        for _slot, _block in enumerate(model.blocks):
            if os.environ.get('WAN21_FFN_HBM_POOL') == '1':
                if os.environ.get('WAN21_RECOMPUTE_BLOCKS', '0') == '1':
                    raise ValueError('pooled FFN scratch requires disabled whole-block recompute')
                _block.ffn._ffn_hbm_slot = _slot
            _block.ffn.forward=_types.MethodType(_recompute,_block.ffn)
        assert len(model.blocks)==30
        print('WAN21_ISOLATED_CANDIDATE rank='+str(rank)+' fused_blocks=30 selective_gelu=1 biasfix=1',flush=True)
        model.gradient_checkpointing = os.environ.get("WAN21_RECOMPUTE_BLOCKS", "0") == "1"
    if not args.toy and os.environ.get("WAN21_XH_LINEAR_OP"):
        from src.train.xh_linear import install_on_model
        installed = install_on_model(model, os.environ["WAN21_XH_LINEAR_OP"])
        print(f"XH_LINEAR_MODULES rank={rank} modules={installed}", flush=True)
    reference = copy.deepcopy(model) if args.toy and rank == 0 else None
    owners = contiguous_owners_for_model(model, 4)
    trainer = read_shared_checkpoint(resume, owners, args) if resume is not None else None
    initial_weights = resume or args.init_checkpoint
    error = None
    if initial_weights is not None and local_rank == 0:
        try:
            load_model_state_only(initial_weights, model)
        except Exception as exc:
            error = f"model load: {type(exc).__name__}: {exc}"
    collective_errors(error)
    if args.stored_precision == "bf16":
        model.to(dtype=torch.bfloat16)
    shm_root = Path(os.environ.get("WAN21_SHARED_ROOT", "/tmp"))
    prefix = f"wan21-online.{os.getuid()}.{args.job_token}.die{die}"
    flat = share_parameters(model, shm_root / (prefix + ".weights"), local_rank, group=local_group)
    reducer = build_vector_reducer(args.run_dir, rank)
    gradients = HierarchicalGradients(model, shm_root / (prefix + ".gradients"), rank, world, 4,
                                     args.dies_per_node, communication_cpu, local_group, owner_group,
                                     node_owner_group, die_owner_group)
    gradients.set_owners(model, owners, args.bucket_mb * 1024 * 1024 //
                         next(model.parameters()).element_size())
    gradients.clip_norm = args.clip_grad_norm
    owned_ids = {id(p) for name, p in model.named_parameters() if owners[name] == local_rank}
    groups = [{**group, "params": [p for p in group["params"] if id(p) in owned_ids]}
              for group in param_groups(model, args.weight_decay)]
    groups = [group for group in groups if group["params"]]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(args))
    online = None if args.toy else OnlineInput(args, local_group, local_rank)
    torch.manual_seed(args.seed + rank)
    start_step = 0
    if trainer is not None:
        restore_shared_state(resume, trainer, optimizer, scheduler, online, local_rank)
        start_step = int(trainer["step"])
    if start_step >= args.max_steps:
        raise ValueError("max steps must exceed the resumed checkpoint step")
    if rank == 0:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        input_mode = "toy" if args.toy else "raw_online"
        encoder_dtype = "float32" if args.toy else f"vae=float32,t5={args.text_precision}"
        config.update(world_size=world, parallel_mode="shared_hierarchical", input_mode=input_mode,
                      compute_dtype="float32" if args.toy else str(runtime.dtype), encoder_dtype=encoder_dtype,
                      parameter_dtype=str(next(model.parameters()).dtype),
                      optimizer_dtype=str(next(model.parameters()).dtype), encoder_groups=replicas,
                      activation_recompute=bool(getattr(model, "gradient_checkpointing", False)),
                      bf16_comm_chunk_mb=int(os.environ.get("WAN21_BF16_COMM_CHUNK_MB", "64")),
                      owner0_bucket_mb=int(os.environ.get("WAN21_OWNER0_BUCKET_MB", "0")),
                      bf16_direct_reduce=os.environ.get("WAN21_BF16_DIRECT_REDUCE", "1") == "1",
                      input_fingerprint=online.data.catalog_fingerprint if online else None)
        (args.run_dir / "train_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print("SHARED_TRAIN_READY " + json.dumps({"startup_seconds": time.perf_counter() - startup,
                                                   "start_step": start_step, "world_size": world}), flush=True)
    model.train()
    last_saved = start_step if resume else None
    profile = None
    profile_end = args.profile_start_step + args.profile_steps
    for step in range(start_step, args.max_steps):
        if args.profile_steps and (rank == args.profile_rank or os.environ.get("WAN21_TRAIN_PROFILE_ALL_RANKS") == "1") and step + 1 == args.profile_start_step:
            output_dir = args.run_dir / "profile"
            profile = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=args.profile_steps, repeat=1),
                record_shapes=args.profile_detailed, profile_memory=args.profile_memory,
                with_stack=args.profile_detailed,
                on_trace_ready=lambda prof: _export_cpu_profile(
                    prof, output_dir, rank, args.profile_start_step + 1, args.profile_steps, args.profile_detailed))
            profile.__enter__()
        dist.barrier()
        began = time.perf_counter()
        step_cpu_start = _cpu_usage()
        frequencies_start = _bound_cpu_frequencies()
        local_loss = 0.0
        phases = {"read_seconds": 0.0, "vae_seconds": 0.0, "umt5_seconds": 0.0,
                  "samples": 0, "data_seconds": 0.0, "forward_seconds": 0.0, "backward_seconds": 0.0, "context_tokens": 0}
        for micro in range(args.grad_accum):
            if args.toy:
                x, target = toy_batch(rank, step * args.grad_accum + micro)
            else:
                batch, data_timing = online.next_batch()
                for key, value in data_timing.items():
                    phases[key] += value
            if micro == 0:
                gradients.begin_step(step + 1, reducer)
            gradients.publish_ready = micro == args.grad_accum - 1
            forward_start = time.perf_counter()
            forward_cpu_start = _cpu_usage()
            if args.toy:
                loss = nn.functional.mse_loss(model(x.to(next(model.parameters()).dtype)).float(), target)
            else:
                latents = batch["latents"]
                contexts = apply_cfg_dropout(batch["contexts"], online.empty_context, dropout_prob=args.cfg_dropout)
                phases["context_tokens"] += sum(int(context.size(0)) for context in contexts)
                sigma = sample_logit_normal_sigmas(latents.size(0), device=torch.device("cpu"))
                noisy, target, timesteps = add_flow_noise(latents, sigmas=sigma)
                with autocast_context(torch.device("cpu"), runtime.dtype), record_function("wan21/forward"):
                    predicted = call_wan_model(model, noisy, timesteps, contexts)
                    loss = nn.functional.mse_loss(predicted.float(), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step + 1} rank {rank}")
            local_loss += float(loss.detach()) / args.grad_accum
            backward_start = time.perf_counter()
            phases["forward_seconds"] += backward_start - forward_start
            phases["forward_cpu_seconds"] = phases.get("forward_cpu_seconds", 0.0) + _cpu_usage() - forward_cpu_start
            backward_cpu_start = _cpu_usage()
            with record_function("wan21/backward"):
                (loss / args.grad_accum).backward()
            phases["backward_seconds"] += time.perf_counter() - backward_start
            phases["backward_cpu_seconds"] = phases.get("backward_cpu_seconds", 0.0) + _cpu_usage() - backward_cpu_start
            if not args.toy:
                del batch, latents, contexts, sigma, noisy, target, timesteps, predicted
            del loss
        finish_start = time.perf_counter()
        finish_cpu_start = _cpu_usage()
        with record_function("wan21/communication_tail_and_clip"):
            grad_norm = gradients.finish_step()
        phases["communication_tail_and_clip_seconds"] = time.perf_counter() - finish_start
        phases["communication_tail_and_clip_cpu_seconds"] = _cpu_usage() - finish_cpu_start
        optimizer_start = time.perf_counter()
        optimizer_cpu_start = _cpu_usage()
        with record_function("wan21/optimizer_and_scheduler"):
            optimizer.step(); scheduler.step()
        phases["optimizer_seconds"] = time.perf_counter() - optimizer_start
        phases["optimizer_cpu_seconds"] = _cpu_usage() - optimizer_cpu_start
        dist.barrier()
        local_seconds = time.perf_counter() - began
        if args.verify_updates:
            update = check_updates(optimizer, rank)
            updates = [None] * world
            dist.all_gather_object(updates, update)
            if rank == 0:
                print("SHARED_OWNER_UPDATES_OK " + json.dumps({"step": step + 1, "ranks": updates}), flush=True)
        local = {"rank": rank, "loss": local_loss, "step_seconds": local_seconds, **phases}
        local["process_cpu_seconds"] = _cpu_usage() - step_cpu_start
        local["compute_cpu_frequencies_khz"] = {"start": frequencies_start, "end": _bound_cpu_frequencies()}
        # Observability only; preserve existing barrier, reduction and optimizer order.
        local["communication"] = {key: value for key, value in gradients.timing.items()
                                  if key not in ("step_start", "generation")}
        if os.environ.get("WAN21_XH_STATS"):
            import src.train.xh_linear as linear_bridge
            local["xh_linear_selected_total"] = linear_bridge._selected
            local["xh_linear_fallback_total"] = linear_bridge._fallback
        records = [None] * world
        dist.all_gather_object(records, local)
        if rank == 0:
            total_samples = world * args.batch_size * args.grad_accum
            seconds = max(r["step_seconds"] for r in records)
            metrics = {"step": step + 1, "loss": sum(r["loss"] for r in records) / world,
                       "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"],
                       "step_seconds": seconds, "samples_per_second": total_samples / seconds,
                       "global_batch": total_samples, "input_mode": "toy" if args.toy else "raw_online",
                       "training_task": args.training_task,
                       "profiled_step": args.profile_steps > 0 and args.profile_start_step < step + 1 <= profile_end,
                       "profiler_warmup_step": args.profile_steps > 0 and step + 1 == args.profile_start_step,
                       "encoded_samples": sum(r["samples"] for r in records),
                       "vae_seconds_per_sample": sum(r["vae_seconds"] for r in records) / total_samples,
                       "umt5_seconds_per_sample": sum(r["umt5_seconds"] for r in records) / total_samples}
            for key in phases:
                if key != "samples":
                    metrics[key + "_max"] = max(r[key] for r in records)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            with (args.run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n"); handle.flush()
            with (args.run_dir / "rank_timing.jsonl").open("a") as handle:
                handle.write(json.dumps({"step": step + 1, "ranks": records}) + "\n")
        if profile is not None:
            profile.step()
            if step + 1 == profile_end:
                profile.__exit__(None, None, None)
                profile = None
        if args.save_every and (step + 1) % args.save_every == 0:
            save_shared_checkpoint(args.run_dir, step + 1, model, optimizer, scheduler, owners, args,
                                   online.state_dict() if online else None)
            last_saved = step + 1
            if rank == 0:
                prune_checkpoints(args.run_dir, keep_last=args.keep_last_checkpoints,
                                  keep_steps=parse_keep_checkpoint_steps(args.keep_checkpoint_steps))
    if args.save_final and last_saved != args.max_steps:
        save_shared_checkpoint(args.run_dir, args.max_steps, model, optimizer, scheduler, owners, args,
                               online.state_dict() if online else None)
    if args.toy and rank == 0:
        verify_toy(model, reference, args, world)
    memory = [None] * world
    dist.all_gather_object(memory, memory_record(rank))
    if rank == 0:
        print("SHARED_TRAIN_MEMORY " + json.dumps(memory), flush=True)
        (args.run_dir / "TRAINING_COMPLETED").write_text(f"steps={args.max_steps}\n")
        print(f"WAN21_SHARED_TRAIN_OK steps={args.max_steps} run_dir={args.run_dir}", flush=True)
    dist.barrier()
    runtime.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-index", type=Path)
    p.add_argument("--encoder-weights", type=Path)
    p.add_argument("--text-precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--image-bucket", default="256/256x256")
    p.add_argument("--video-frames", type=int, default=49)
    p.add_argument("--video-fps", type=float, default=16.0)
    p.add_argument("--video-root", type=Path, help="Root for OpenVid video filenames; explicit path fields take precedence")
    p.add_argument("--training-task", choices=("t2i", "t2v"), default="t2i")
    p.add_argument("--monitor-node-memory", action="store_true")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--stored-precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--nnodes", type=int, default=4)
    p.add_argument("--dies-per-node", type=int, default=4)
    p.add_argument("--bucket-mb", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--profile-start-step", type=int, default=0)
    p.add_argument("--profile-steps", type=int, default=0)
    p.add_argument("--profile-rank", type=int, default=0)
    p.add_argument("--profile-detailed", action="store_true",
                   help="Record operator shapes and Python stacks")
    p.add_argument("--profile-memory", action="store_true",
                   help="Record per-operator memory events; increases profiler memory use")
    p.add_argument("--lr-total-steps", type=int, default=2000)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--cfg-dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep-last-checkpoints", type=int, default=0)
    p.add_argument("--keep-checkpoint-steps", default="")
    p.add_argument("--resume", type=Path)
    p.add_argument("--auto-resume", action="store_true")
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--verify-updates", action="store_true")
    p.add_argument("--toy", action="store_true", help="small deterministic MPI arithmetic regression only")
    p.add_argument("--job-token", default="manual")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
