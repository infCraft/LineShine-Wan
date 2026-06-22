#!/usr/bin/env python3
"""Audit fixed-shape LineShine cache buckets and emit clean shard links."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, ensure_dir, utc_now, write_json, write_jsonl


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


def visual_token_count(latent_shape: tuple[int, int, int, int], patch_size: tuple[int, int, int]) -> int | None:
    _, frames, height, width = latent_shape
    pt, ph, pw = patch_size
    if frames % pt != 0 or height % ph != 0 or width % pw != 0:
        return None
    return (frames // pt) * (height // ph) * (width // pw)


def close_float(value: Any, expected: float, atol: float = 1e-6) -> bool:
    try:
        return abs(float(value) - expected) <= atol
    except (TypeError, ValueError):
        return False


def validate_sample(
    *,
    key: str,
    parts: dict[str, bytes],
    args: argparse.Namespace,
    seen_sample_ids: set[str],
) -> tuple[bool, dict[str, Any], list[str]]:
    errors: list[str] = []
    clean_row: dict[str, Any] = {"key": key}
    if "safetensors" not in parts or "json" not in parts:
        missing = sorted({"safetensors", "json"} - set(parts))
        return False, clean_row, [f"missing parts: {missing}"]

    try:
        tensors = load_safetensors(parts["safetensors"])
    except Exception as exc:  # noqa: BLE001
        return False, clean_row, [f"bad safetensors: {exc!r}"]
    try:
        meta = json.loads(parts["json"].decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, clean_row, [f"bad json: {exc!r}"]

    sample_id = str(meta.get("sample_id", key))
    clean_row.update(
        {
            "sample_id": sample_id,
            "video": meta.get("video"),
            "source_id": meta.get("source_id"),
            "part": meta.get("part"),
            "local_path": meta.get("local_path"),
            "quality_score": meta.get("quality_score"),
        }
    )
    if sample_id in seen_sample_ids:
        errors.append("duplicate sample_id")
    seen_sample_ids.add(sample_id)

    latent = tensors.get("latent")
    if latent is None:
        errors.append("missing latent")
    else:
        latent_shape = tuple(int(x) for x in latent.shape)
        clean_row["latent_shape"] = list(latent_shape)
        if latent_shape != tuple(args.latent_shape):
            errors.append(f"bad latent shape {latent_shape}")
        if latent.dtype != torch.bfloat16:
            errors.append(f"bad latent dtype {latent.dtype}")
        token_count = visual_token_count(latent_shape, tuple(args.patch_size)) if len(latent_shape) == 4 else None
        clean_row["visual_token_count"] = token_count
        if token_count != args.visual_tokens:
            errors.append(f"bad visual token count {token_count}")

    context = tensors.get("context")
    if context is None:
        errors.append("missing context")
    else:
        context_shape = tuple(int(x) for x in context.shape)
        clean_row["context_shape"] = list(context_shape)
        if context.ndim != 2 or context.shape[1] != args.text_dim:
            errors.append(f"bad context shape {context_shape}")
        if context.dtype != torch.bfloat16:
            errors.append(f"bad context dtype {context.dtype}")

    text_len_tensor = tensors.get("text_len")
    if text_len_tensor is None:
        errors.append("missing text_len")
    elif context is not None:
        text_len = int(text_len_tensor.item())
        clean_row["text_len"] = text_len
        if text_len != context.shape[0] or text_len <= 0 or text_len > args.text_len:
            errors.append(f"bad text_len {text_len}")

    preprocess = meta.get("video_preprocess") or {}
    clean_row["video_preprocess"] = preprocess
    if preprocess.get("sampled_frames") != args.frames:
        errors.append(f"bad sampled_frames {preprocess.get('sampled_frames')}")
    if not close_float(preprocess.get("target_fps"), args.target_fps):
        errors.append(f"bad target_fps {preprocess.get('target_fps')}")
    if preprocess.get("resize_short_side") != args.image_size:
        errors.append(f"bad resize_short_side {preprocess.get('resize_short_side')}")
    if preprocess.get("crop") != args.crop:
        errors.append(f"bad crop {preprocess.get('crop')}")
    return not errors, clean_row, errors


def link_clean_shards(clean_shards: list[Path], output_dir: Path, *, replace: bool) -> list[str]:
    ensure_dir(output_dir)
    linked: list[str] = []
    for shard in clean_shards:
        target = output_dir / shard.name
        if target.exists() or target.is_symlink():
            if not replace:
                linked.append(str(target))
                continue
            if target.is_dir() and not target.is_symlink():
                raise IsADirectoryError(target)
            target.unlink()
        rel = os.path.relpath(shard, target.parent)
        target.symlink_to(rel)
        linked.append(str(target))
    return linked


def audit(args: argparse.Namespace) -> None:
    shards = sorted(args.cache_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching {args.cache_dir / args.pattern}")

    seen_sample_ids: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    clean_shards: list[Path] = []
    dirty_shards: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []
    text_lens: list[int] = []
    token_counts: dict[str, int] = {}
    total_samples = 0

    for shard in shards:
        shard_samples = 0
        shard_bad = 0
        shard_clean = 0
        shard_bad_examples: list[dict[str, Any]] = []
        for key, parts in iter_tar_samples(shard):
            shard_samples += 1
            total_samples += 1
            ok, clean_row, errors = validate_sample(key=key, parts=parts, args=args, seen_sample_ids=seen_sample_ids)
            clean_row["shard"] = str(shard)
            if ok:
                shard_clean += 1
                clean_rows.append(clean_row)
                if "text_len" in clean_row:
                    text_lens.append(int(clean_row["text_len"]))
                token = clean_row.get("visual_token_count")
                token_counts[str(token)] = token_counts.get(str(token), 0) + 1
            else:
                shard_bad += 1
                bad = {
                    "key": key,
                    "sample_id": clean_row.get("sample_id", key),
                    "shard": str(shard),
                    "errors": errors,
                }
                bad_rows.append(bad)
                if len(shard_bad_examples) < args.max_bad_samples:
                    shard_bad_examples.append(bad)
        if shard_samples == 0:
            shard_bad += 1
            bad = {"key": None, "sample_id": None, "shard": str(shard), "errors": ["empty shard"]}
            bad_rows.append(bad)
            shard_bad_examples.append(bad)
        if shard_bad == 0:
            clean_shards.append(shard)
        else:
            dirty_shards.append({"shard": str(shard), "samples": shard_samples, "bad_samples": shard_bad})
        shard_summaries.append(
            {
                "shard": str(shard),
                "samples": shard_samples,
                "clean_samples": shard_clean,
                "bad_samples": shard_bad,
                "bad_examples": shard_bad_examples,
            }
        )

    linked_shards: list[str] = []
    if args.link_clean_shards is not None:
        linked_shards = link_clean_shards(clean_shards, args.link_clean_shards, replace=args.replace_links)

    write_jsonl(args.clean_manifest, clean_rows)
    write_jsonl(args.bad_manifest, bad_rows)
    report = {
        "created_at": utc_now(),
        "cache_dir": str(args.cache_dir),
        "pattern": args.pattern,
        "spec": {
            "frames": args.frames,
            "target_fps": args.target_fps,
            "image_size": args.image_size,
            "crop": args.crop,
            "latent_shape": args.latent_shape,
            "patch_size": args.patch_size,
            "visual_tokens": args.visual_tokens,
            "text_dim": args.text_dim,
            "text_len_max": args.text_len,
        },
        "sample_count": total_samples,
        "clean_sample_count": len(clean_rows),
        "bad_sample_count": len(bad_rows),
        "unique_sample_count": len(seen_sample_ids),
        "shard_count": len(shards),
        "clean_shard_count": len(clean_shards),
        "dirty_shard_count": len(dirty_shards),
        "dirty_shards": dirty_shards,
        "linked_clean_shards": linked_shards,
        "clean_manifest": str(args.clean_manifest),
        "bad_manifest": str(args.bad_manifest),
        "text_len_min": min(text_lens) if text_lens else None,
        "text_len_max": max(text_lens) if text_lens else None,
        "text_len_mean": sum(text_lens) / len(text_lens) if text_lens else None,
        "visual_token_counts": token_counts,
        "shards": shard_summaries,
        "bad_examples": bad_rows[: args.max_bad_samples],
    }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "sample_count": total_samples,
                "clean_sample_count": len(clean_rows),
                "bad_sample_count": len(bad_rows),
                "clean_shard_count": len(clean_shards),
                "dirty_shard_count": len(dirty_shards),
                "linked_clean_shard_count": len(linked_shards),
            },
            sort_keys=True,
        )
    )
    if args.fail_on_bad and bad_rows:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root / "cache/train")
    parser.add_argument("--pattern", default="*.tar")
    parser.add_argument("--report", type=Path, default=root / "reports/stage1_bucket_audit/train_audit.json")
    parser.add_argument("--clean-manifest", type=Path, default=root / "data/openvid/meta/stage1_clean_existing_cache.jsonl")
    parser.add_argument("--bad-manifest", type=Path, default=root / "data/openvid/meta/stage1_bad_existing_cache.jsonl")
    parser.add_argument("--link-clean-shards", type=Path)
    parser.add_argument("--replace-links", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--latent-shape", type=int, nargs=4, default=[16, 13, 32, 32])
    parser.add_argument("--patch-size", type=int, nargs=3, default=[1, 2, 2])
    parser.add_argument("--visual-tokens", type=int, default=3328)
    parser.add_argument("--text-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=512)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--target-fps", type=float, default=16.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--crop", default="center")
    parser.add_argument("--max-bad-samples", type=int, default=50)
    parser.add_argument("--fail-on-bad", action=argparse.BooleanOptionalAction, default=False)
    parser.set_defaults(func=audit)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
