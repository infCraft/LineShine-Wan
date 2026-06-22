#!/usr/bin/env python3
"""Decode OpenVid clips and build Wan latent/T5 WebDataset cache shards."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from safetensors.torch import save as save_safetensors

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, WAN_ROOT, ensure_dir, read_jsonl, sample_id, utc_now, write_json


PROMPTS = [
    "A cinematic shot of a red car driving along a coastal road at sunset.",
    "A close-up video of fresh flowers moving gently in the wind.",
    "A person walking through a busy city street at night with neon signs.",
    "A small boat crossing a calm lake surrounded by mountains.",
]


def add_wan_to_path() -> None:
    wan = str(WAN_ROOT)
    if wan not in sys.path:
        sys.path.insert(0, wan)


def parse_rate(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        a, b = text.split("/", 1)
        try:
            denom = float(b)
            return None if denom == 0 else float(a) / denom
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def resize_center_crop(frame: np.ndarray, size: int) -> torch.Tensor:
    image = Image.fromarray(frame)
    width, height = image.size
    scale = size / min(width, height)
    resized = (round(width * scale), round(height * scale))
    image = image.resize(resized, Image.Resampling.BICUBIC)
    left = max(0, (resized[0] - size) // 2)
    top = max(0, (resized[1] - size) // 2)
    image = image.crop((left, top, left + size, top + size))
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor.div(127.5).sub(1.0)


def decode_video_grid(
    path: Path,
    *,
    frames: int = 49,
    target_fps: float = 16.0,
    image_size: int = 256,
    clip_start_sec: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        source_fps = parse_rate(stream.average_rate) or parse_rate(stream.base_rate)
        decoded: list[tuple[float, np.ndarray]] = []
        fallback_idx = 0
        for frame in container.decode(stream):
            if frame.pts is not None and stream.time_base is not None:
                ts = float(frame.pts * stream.time_base)
            elif source_fps:
                ts = fallback_idx / source_fps
            else:
                ts = float(fallback_idx)
            fallback_idx += 1
            decoded.append((ts, frame.to_rgb().to_ndarray()))
    finally:
        container.close()

    if len(decoded) < frames:
        raise ValueError(f"{path} has only {len(decoded)} decoded frames; need {frames}")

    timestamps = np.asarray([x[0] for x in decoded], dtype=np.float64)
    duration = float(max(timestamps[-1] - timestamps[0], 0.0))
    window = (frames - 1) / target_fps
    if duration + 1e-6 < window:
        raise ValueError(f"{path} duration {duration:.3f}s is shorter than target window {window:.3f}s")

    if clip_start_sec is None:
        start = timestamps[0] + max(0.0, duration - window) / 2.0
        sampling_mode = "center"
    else:
        start = timestamps[0] + float(clip_start_sec)
        sampling_mode = "fixed_start"
        if start < timestamps[0] - 1e-6 or start + window > timestamps[-1] + 1e-6:
            raise ValueError(
                f"{path} clip_start_sec={clip_start_sec:.3f}s cannot fit target window {window:.3f}s "
                f"in decoded duration {duration:.3f}s"
            )
    target_times = start + np.arange(frames, dtype=np.float64) / target_fps
    idxs = np.searchsorted(timestamps, target_times, side="left")
    chosen = []
    for target, idx in zip(target_times, idxs):
        candidates = []
        if idx < len(timestamps):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(float(timestamps[i] - target)))
        chosen.append(best)

    video = torch.stack([resize_center_crop(decoded[i][1], image_size) for i in chosen], dim=1)
    meta = {
        "source_fps": source_fps,
        "decoded_frames": len(decoded),
        "sampled_frames": frames,
        "target_fps": target_fps,
        "sampling_mode": sampling_mode,
        "requested_clip_start_sec": clip_start_sec,
        "sample_start_sec": start,
        "sample_end_sec": float(target_times[-1]),
        "resize_short_side": image_size,
        "crop": "center",
    }
    return video, meta


def load_components(weights_dir: Path, device: torch.device, dtype: torch.dtype):
    add_wan_to_path()
    from wan.configs.wan_t2v_1_3B import t2v_1_3B
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE

    vae = WanVAE(vae_pth=str(weights_dir / t2v_1_3B.vae_checkpoint), dtype=torch.float32, device=device)
    text_encoder = T5EncoderModel(
        text_len=t2v_1_3B.text_len,
        dtype=dtype,
        device=device,
        checkpoint_path=str(weights_dir / t2v_1_3B.t5_checkpoint),
        tokenizer_path=str(weights_dir / t2v_1_3B.t5_tokenizer),
        shard_fn=None,
    )
    return vae, text_encoder


def fake_encode(video: torch.Tensor, caption: str) -> tuple[torch.Tensor, torch.Tensor]:
    seed = abs(hash((caption, tuple(video.shape)))) % (2**31)
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn((16, 13, 32, 32), dtype=torch.float32, generator=generator)
    text_len = min(32, max(2, len(caption.split()) + 2))
    context = torch.randn((text_len, 4096), dtype=torch.float32, generator=generator)
    return latent.to(torch.bfloat16), context.to(torch.bfloat16)


@torch.no_grad()
def encode_sample(
    row: dict[str, Any],
    *,
    vae,
    text_encoder,
    device: torch.device,
    dtype: torch.dtype,
    fake_encoders: bool,
    frames: int,
    target_fps: float,
    image_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = Path(row["local_path"])
    clip_start = row.get("clip_start_sec")
    clip_start_sec = float(clip_start) if clip_start is not None else None
    video, video_meta = decode_video_grid(
        path,
        frames=frames,
        target_fps=target_fps,
        image_size=image_size,
        clip_start_sec=clip_start_sec,
    )
    caption = str(row.get("caption_clean") or row.get("caption") or row.get("raw_caption") or "")

    if fake_encoders:
        latent, context = fake_encode(video, caption)
    else:
        video_dev = video.to(device=device, dtype=torch.float32, non_blocking=True)
        latent = vae.encode([video_dev])[0].to(torch.bfloat16).cpu()
        context = text_encoder([caption], device)[0].to(torch.bfloat16).cpu()

    tensors = {
        "latent": latent.contiguous(),
        "context": context.contiguous(),
        "text_len": torch.tensor([context.shape[0]], dtype=torch.int32),
    }
    meta = {
        "sample_id": sample_id(row),
        "video": row.get("video"),
        "source_id": row.get("source_id"),
        "part": row.get("part"),
        "local_path": str(path),
        "caption": caption,
        "raw_caption": row.get("raw_caption"),
        "quality_score": row.get("quality_score"),
        "clip_index": row.get("clip_index"),
        "clip_start_sec": row.get("clip_start_sec"),
        "clip_duration_sec": row.get("clip_duration_sec"),
        "segment_source_sample_id": row.get("segment_source_sample_id"),
        "video_preprocess": video_meta,
    }
    return tensors, meta


def shard_name(cache_dir: Path, prefix: str, shard_idx: int) -> Path:
    return cache_dir / f"{prefix}-{shard_idx:06d}.tar"


def write_cache(args: argparse.Namespace) -> None:
    ensure_dir(args.cache_dir)
    rows = list(read_jsonl(args.manifest))
    if args.num_shards is not None:
        if args.shard_index is None:
            raise ValueError("--num-shards requires --shard-index")
        rows = [row for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.start is not None or args.end is not None:
        rows = rows[args.start or 0 : args.end]
    if not rows:
        raise RuntimeError("no manifest rows selected")

    device = torch.device(args.device)
    dtype = torch.bfloat16
    vae = text_encoder = None
    if not args.fake_encoders:
        vae, text_encoder = load_components(args.weights_dir, device, dtype)

    initial_shard_idx = args.shard_start_index + (args.shard_index or 0)
    shard_idx = initial_shard_idx
    manifest_path = args.cache_dir / f"{args.prefix}_{initial_shard_idx:06d}_cache_manifest.json"
    if args.skip_existing and manifest_path.exists():
        print(json.dumps({"skipped": True, "manifest": str(manifest_path)}))
        return

    shard_count = 0
    sample_count = 0
    failures: list[dict[str, Any]] = []
    current_path = shard_name(args.cache_dir, args.prefix, shard_idx)
    sink = wds.TarWriter(str(current_path))
    try:
        for row in rows:
            if shard_count >= args.shard_size:
                sink.close()
                shard_idx += args.num_shards or 1
                shard_count = 0
                current_path = shard_name(args.cache_dir, args.prefix, shard_idx)
                sink = wds.TarWriter(str(current_path))
            sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id(row))
            try:
                tensors, meta = encode_sample(
                    row,
                    vae=vae,
                    text_encoder=text_encoder,
                    device=device,
                    dtype=dtype,
                    fake_encoders=args.fake_encoders,
                    frames=args.frames,
                    target_fps=args.target_fps,
                    image_size=args.image_size,
                )
                sink.write(
                    {
                        "__key__": sid,
                        "safetensors": save_safetensors(tensors),
                        "json": json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"),
                    }
                )
                shard_count += 1
                sample_count += 1
            except Exception as exc:  # noqa: BLE001 - failures are reported per sample.
                failures.append({"sample_id": sample_id(row), "video": row.get("video"), "error": repr(exc)})
    finally:
        sink.close()

    manifest = {
        "created_at": utc_now(),
        "source_manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "prefix": args.prefix,
        "sample_count": sample_count,
        "failure_count": len(failures),
        "shards": sorted(str(p) for p in args.cache_dir.glob(f"{args.prefix}-*.tar") if p.stat().st_size > 0),
        "fake_encoders": args.fake_encoders,
        "frames": args.frames,
        "target_fps": args.target_fps,
        "image_size": args.image_size,
    }
    write_json(manifest_path, manifest)
    write_json(args.cache_dir / f"{args.prefix}_cache_manifest.json", manifest)
    with (args.cache_dir / f"{args.prefix}_{initial_shard_idx:06d}_failed.jsonl").open("w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"sample_count": sample_count, "failure_count": len(failures), "cache_dir": str(args.cache_dir)}))


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/openvid/meta/openvid_smoke_extracted.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/smoke")
    parser.add_argument("--weights-dir", type=Path, default=root / "weights/wan2.1_t2v_1.3b")
    parser.add_argument("--prefix", default="smoke")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fake-encoders", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(func=write_cache)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
