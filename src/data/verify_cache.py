#!/usr/bin/env python3
"""Verify LineShine WebDataset cache shards."""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from safetensors.torch import load as load_safetensors

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, WAN_ROOT, ensure_dir, utc_now, write_json


def add_wan_to_path() -> None:
    wan = str(WAN_ROOT)
    if wan not in sys.path:
        sys.path.insert(0, wan)


def iter_tar_samples(path: Path):
    grouped: dict[str, dict[str, bytes]] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            name = Path(member.name)
            key = name.stem
            grouped.setdefault(key, {})[name.suffix.lstrip(".")] = fileobj.read()
    for key, parts in grouped.items():
        yield key, parts


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


def maybe_add_decode_sample(
    reservoir: list[tuple[str, torch.Tensor]],
    *,
    key: str,
    latent: torch.Tensor,
    seen_count: int,
    limit: int,
    rng: random.Random,
) -> None:
    if limit <= 0:
        return
    item = (key, latent.cpu().float())
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    idx = rng.randint(0, seen_count - 1)
    if idx < limit:
        reservoir[idx] = item


def decode_samples(reservoir: list[tuple[str, torch.Tensor]], args: argparse.Namespace) -> list[dict[str, str]]:
    if not reservoir:
        return []
    add_wan_to_path()
    from wan.modules.vae import WanVAE

    device = torch.device(args.device)
    vae = WanVAE(vae_pth=str(args.weights_dir / "Wan2.1_VAE.pth"), dtype=torch.float32, device=device)
    ensure_dir(args.decode_dir)
    outputs = []
    for idx, (key, latent) in enumerate(reservoir):
        video = vae.decode([latent.to(device)])[0].cpu()
        safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key)
        out_path = args.decode_dir / f"{idx:03d}_{safe_key}.mp4"
        write_video(out_path, video, fps=args.decode_fps)
        outputs.append({"key": key, "path": str(out_path)})
    return outputs


def verify(args: argparse.Namespace) -> None:
    shards = sorted(args.cache_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching {args.cache_dir / args.pattern}")

    seen: set[str] = set()
    errors: list[dict[str, Any]] = []
    sample_count = 0
    total_tensor_bytes = 0
    context_lens: list[int] = []
    rng = random.Random(args.seed)
    decode_reservoir: list[tuple[str, torch.Tensor]] = []
    for shard in shards:
        shard_samples = 0
        for key, parts in iter_tar_samples(shard):
            shard_samples += 1
            sample_count += 1
            if "safetensors" not in parts or "json" not in parts:
                errors.append({"key": key, "shard": str(shard), "error": "missing parts"})
                continue
            try:
                tensors = load_safetensors(parts["safetensors"])
                meta = json.loads(parts["json"].decode("utf-8"))
                sid = str(meta.get("sample_id", key))
                if sid in seen:
                    errors.append({"key": key, "shard": str(shard), "error": "duplicate sample_id"})
                seen.add(sid)
                latent = tensors["latent"]
                context = tensors["context"]
                text_len = int(tensors["text_len"].item())
                if tuple(latent.shape) != tuple(args.latent_shape):
                    errors.append({"key": key, "error": f"bad latent shape {tuple(latent.shape)}"})
                if latent.dtype != torch.bfloat16:
                    errors.append({"key": key, "error": f"bad latent dtype {latent.dtype}"})
                if context.ndim != 2 or context.shape[1] != args.text_dim:
                    errors.append({"key": key, "error": f"bad context shape {tuple(context.shape)}"})
                if context.dtype != torch.bfloat16:
                    errors.append({"key": key, "error": f"bad context dtype {context.dtype}"})
                if text_len != context.shape[0] or text_len <= 0 or text_len > args.text_len:
                    errors.append({"key": key, "error": f"bad text_len {text_len}"})
                context_lens.append(text_len)
                total_tensor_bytes += len(parts["safetensors"])
                maybe_add_decode_sample(
                    decode_reservoir,
                    key=key,
                    latent=latent,
                    seen_count=sample_count,
                    limit=args.decode_samples,
                    rng=rng,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append({"key": key, "shard": str(shard), "error": repr(exc)})
        if shard_samples == 0:
            errors.append({"shard": str(shard), "error": "empty shard"})

    decoded = decode_samples(decode_reservoir, args) if args.decode_samples > 0 and not errors else []
    report = {
        "created_at": utc_now(),
        "cache_dir": str(args.cache_dir),
        "pattern": args.pattern,
        "shards": [str(p) for p in shards],
        "sample_count": sample_count,
        "unique_sample_count": len(seen),
        "error_count": len(errors),
        "errors": errors[: args.max_errors],
        "tensor_bytes_total": total_tensor_bytes,
        "tensor_bytes_per_sample": total_tensor_bytes / sample_count if sample_count else None,
        "context_len_min": min(context_lens) if context_lens else None,
        "context_len_max": max(context_lens) if context_lens else None,
        "context_len_mean": sum(context_lens) / len(context_lens) if context_lens else None,
        "decoded_samples": decoded,
    }
    write_json(args.report, report)
    print(json.dumps({k: report[k] for k in ["sample_count", "error_count", "tensor_bytes_per_sample"]}, sort_keys=True))
    if errors:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/smoke")
    parser.add_argument("--pattern", default="smoke-*.tar")
    parser.add_argument("--report", type=Path, default=root / "reports/W3.2_smoke_cache/verify_cache.json")
    parser.add_argument("--latent-shape", type=int, nargs=4, default=[16, 13, 32, 32])
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--max-errors", type=int, default=50)
    parser.add_argument("--decode-samples", type=int, default=0)
    parser.add_argument("--decode-dir", type=Path, default=root / "reports/W3.5_verify_cache_decode")
    parser.add_argument("--decode-fps", type=int, default=16)
    parser.add_argument("--weights-dir", type=Path, default=root / "weights/wan2.1_t2v_1.3b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.set_defaults(func=verify)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
