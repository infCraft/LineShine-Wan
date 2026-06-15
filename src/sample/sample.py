#!/usr/bin/env python3
"""Sample videos from a trained or random WanModel checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import av
import numpy as np
import torch
from safetensors.torch import load_file

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, WAN_ROOT, ensure_dir, utc_now, write_json
from src.train.ckpt import latest_checkpoint, load_checkpoint
from src.train.model_adapter import call_wan_model
from src.train.train import create_model


def add_wan_to_path() -> None:
    wan = str(WAN_ROOT)
    if wan not in sys.path:
        sys.path.insert(0, wan)


def load_prompt_context(prompt_dir: Path, prompt_name: str, device: torch.device) -> torch.Tensor:
    path = prompt_dir / f"{prompt_name}.safetensors"
    tensors = load_file(path)
    return tensors["context"].to(device=device, dtype=torch.bfloat16)


def write_video(path: Path, video: torch.Tensor, fps: int = 16) -> None:
    ensure_dir(path.parent)
    array = video.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    array = array.permute(1, 2, 3, 0).cpu().numpy()
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(array.shape[2])
        stream.height = int(array.shape[1])
        stream.pix_fmt = "yuv420p"
        for frame_arr in array:
            frame = av.VideoFrame.from_ndarray(np.asarray(frame_arr), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


@torch.no_grad()
def sample(args: argparse.Namespace) -> None:
    add_wan_to_path()
    from wan.modules.vae import WanVAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    device = torch.device(args.device)
    ensure_dir(args.output_dir)
    model = create_model(args).to(device).eval()
    ckpt = args.checkpoint or latest_checkpoint(args.run_dir)
    if ckpt is not None and ckpt.exists():
        load_checkpoint(ckpt, model=model, map_location=device, restore_rng=False)

    context = load_prompt_context(args.prompt_dir, args.prompt_name, device)
    empty = load_prompt_context(args.prompt_dir, args.empty_name, device)
    target_shape = (args.latent_channels, args.latent_frames, args.latent_height, args.latent_width)
    seq_len = args.latent_frames * (args.latent_height // 2) * (args.latent_width // 2)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = torch.randn(target_shape, device=device, dtype=torch.float32, generator=generator)

    scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(args.steps, device=device, shift=args.shift)
    for timestep in scheduler.timesteps:
        t = torch.stack([timestep]).to(device=device)
        model_input = latent.unsqueeze(0).to(torch.bfloat16)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred_cond = call_wan_model(model, model_input, t, [context], seq_len=seq_len)[0]
            pred_uncond = call_wan_model(model, model_input, t, [empty], seq_len=seq_len)[0]
            pred = pred_uncond + args.guidance_scale * (pred_cond - pred_uncond)
        latent = scheduler.step(pred.unsqueeze(0), timestep, latent.unsqueeze(0), return_dict=False, generator=generator)[0].squeeze(0)

    latent_path = args.output_dir / f"{args.output_prefix}_latent.pt"
    torch.save({"latent": latent.cpu(), "seed": args.seed, "steps": args.steps}, latent_path)
    video_path = None
    if not args.skip_decode:
        vae = WanVAE(vae_pth=str(args.weights_dir / "Wan2.1_VAE.pth"), dtype=torch.float32, device=device)
        video = vae.decode([latent.to(torch.float32)])[0].cpu()
        video_path = args.output_dir / f"{args.output_prefix}.mp4"
        write_video(video_path, video, fps=args.fps)

    report = {
        "created_at": utc_now(),
        "checkpoint": str(ckpt) if ckpt is not None else None,
        "latent": str(latent_path),
        "video": str(video_path) if video_path else None,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "prompt_name": args.prompt_name,
        "empty_name": args.empty_name,
    }
    write_json(args.output_dir / f"{args.output_prefix}_metadata.json", report)
    print(json.dumps(report, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=root / "runs/smoke_train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=root / "weights/wan2.1_t2v_1.3b")
    parser.add_argument("--prompt-dir", type=Path, default=root / "cache/prompts")
    parser.add_argument("--prompt-name", default="00_a_cinematic_shot_of_a_red_car_driving_along_a_co")
    parser.add_argument("--empty-name", default="empty")
    parser.add_argument("--output-dir", type=Path, default=root / "reports/W4.2_sample_smoke")
    parser.add_argument("--output-prefix", default="smoke_sample")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--skip-decode", action=argparse.BooleanOptionalAction, default=False)
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
    parser.set_defaults(func=sample)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
