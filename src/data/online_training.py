"""Raw image/video batches encoded online by one VAE/UMT5 per four ranks."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from src.data.raw_index import ImageTrainingDataset, SynchronizedBucketBatchSampler


def validate_media_options(*, image_size, video_frames, video_fps, training_task="t2i"):
    if image_size <= 0 or image_size % 16:
        raise ValueError("image size must be positive and divisible by 16")
    if video_frames < 1 or (video_frames - 1) % 4:
        raise ValueError("video frames must be 4k+1")
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError("video fps must be finite and positive")
    if training_task == "t2v" and video_frames < 5:
        raise ValueError("T2V requires at least 5 video frames")


def image_tensor(payload: bytes, size: tuple[int, int]) -> torch.Tensor:
    width, height = size
    with Image.open(io.BytesIO(payload)) as image:
        image.draft("RGB", size)
        image = image.convert("RGB")
        scale = max(width / image.width, height / image.height)
        image = image.resize((max(width, round(image.width * scale)),
                              max(height, round(image.height * scale))), Image.Resampling.BICUBIC)
        left, top = (image.width - width) // 2, (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
        pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1).float().div_(127.5).sub_(1)


class ManifestBucket:
    def __init__(self, size):
        self.size = size
        self.parts = [{"sample_count": size}]
        self.part_starts = [0]

    def __len__(self):
        return self.size


class RawMedia:
    """Metadata index only; no latent/context files are read or written."""
    def __init__(self, path: Path, *, image_size: int, image_bucket: str | None,
                 video_frames: int, video_fps: float, seed: int,
                 video_root: Path | None = None, training_task: str = "t2i"):
        validate_media_options(image_size=image_size, video_frames=video_frames,
                               video_fps=video_fps, training_task=training_task)
        self.path = path.resolve()
        video_root = video_root.resolve() if video_root is not None else None
        self.image_size, self.video_frames, self.video_fps = image_size, video_frames, video_fps
        self.index = None
        self.rows = defaultdict(list)
        if self.path.is_dir():
            if training_task == "t2v":
                raise ValueError("T2V requires a video JSONL manifest, not an image catalog")
            self.index = ImageTrainingDataset(self.path, resolutions=(image_size,),
                                              caption_seed=seed, index_cache_size=1)
            if image_bucket:
                if image_bucket not in self.index.buckets:
                    raise ValueError(f"image bucket {image_bucket} is absent from {path}")
                self.index.buckets = {image_bucket: self.index.buckets[image_bucket]}
                self.index.bucket_keys = [image_bucket]
            self.buckets = self.index.buckets
            self.bucket_keys = self.index.bucket_keys
            fingerprint = self.index.catalog_fingerprint
        else:
            digest = hashlib.sha256()
            with self.path.open("rb") as handle:
                for line_number, line in enumerate(handle, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    raw_path = row.get("path") or row.get("local_path") or row.get("video")
                    if not raw_path:
                        raise ValueError(f"raw sample has no path at line {line_number}")
                    media_path = Path(raw_path)
                    kind = row.get("input_type") or row.get("type")
                    if kind is None:
                        kind = "image" if media_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} else "video"
                    if kind not in {"image", "video"} or (training_task == "t2v" and kind != "video"):
                        raise ValueError(f"unsupported raw media type for {training_task}: {kind}")
                    if kind == "video" and video_root is not None and not row.get("path"):
                        if not row.get("video"):
                            raise ValueError(f"--video-root requires the OpenVid video filename at line {line_number}")
                        media_path = video_root / Path(row["video"]).name
                    if not media_path.is_absolute():
                        media_path = self.path.parent / media_path
                    caption = str(row.get("caption") or row.get("text") or row.get("caption_clean")
                                  or row.get("raw_caption") or "").strip()
                    if not caption:
                        raise ValueError(f"raw sample has no caption: {media_path}")
                    start = row.get("clip_start_sec")
                    if start is not None:
                        start = float(start)
                        if not math.isfinite(start) or start < 0:
                            raise ValueError(f"invalid clip start for {media_path}: {start}")
                    frames = 1 if kind == "image" else video_frames
                    key = f"{image_size}/{kind}_{frames}"
                    self.rows[key].append({"path": str(media_path), "input_type": kind,
                                           "caption": caption, "clip_start_sec": start,
                                           "key": str(row.get("sample_id") or media_path)})
            fingerprint = digest.hexdigest()
            self.bucket_keys = sorted(self.rows)
            self.buckets = {key: ManifestBucket(len(rows)) for key, rows in self.rows.items()}
        identity = [fingerprint, image_size, image_bucket, video_frames, video_fps]
        # Preserve existing T2I checkpoint fingerprints when new options are unused.
        if video_root is not None or training_task != "t2i":
            identity += [str(video_root) if video_root is not None else None, training_task]
        self.catalog_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if not self.bucket_keys:
            raise ValueError("raw input has no matching samples")

    @property
    def bucket_sizes(self):
        if self.index is not None:
            return self.index.bucket_sizes
        return {key: len(rows) for key, rows in self.rows.items()}

    def read(self, request):
        if self.index is not None:
            row = self.index[request]
            with Path(row["source_tar"]).open("rb") as handle:
                handle.seek(int(row["image_offset_data"]))
                payload = handle.read(int(row["image_size"]))
            if len(payload) != int(row["image_size"]):
                raise EOFError(f"short original-image read: {row['key']}")
            pixels = image_tensor(payload, (int(row["target_width"]), int(row["target_height"])))
        else:
            row = self.rows[request[0]][request[1]]
            path = Path(row["path"])
            if row["input_type"] == "image":
                pixels = image_tensor(path.read_bytes(), (self.image_size, self.image_size))
            else:
                from src.video_io import read_video_grid
                frames, _ = read_video_grid(path, frames=self.video_frames, target_fps=self.video_fps,
                                             image_size=self.image_size,
                                             clip_start_sec=row.get("clip_start_sec"))
                pixels = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(127.5).sub_(1)
        return row["key"], row["caption"], pixels


class FrozenEncoders:
    def __init__(self, weights: Path, *, text_dtype=torch.float32):
        wan_root = Path(__file__).resolve().parents[2] / "third_party" / "Wan2.1"
        if str(wan_root) not in sys.path:
            sys.path.insert(0, str(wan_root))
        from wan.configs.wan_t2v_1_3B import t2v_1_3B
        from wan.modules.vae import WanVAE
        from wan.modules.t5 import T5EncoderModel
        started = time.perf_counter()
        self.vae = WanVAE(vae_pth=str(weights / t2v_1_3B.vae_checkpoint),
                          dtype=torch.float32, device=torch.device("cpu"))
        vae_loaded = time.perf_counter()
        self.text = T5EncoderModel(text_len=t2v_1_3B.text_len, dtype=text_dtype,
                                   device=torch.device("cpu"),
                                   checkpoint_path=str(weights / t2v_1_3B.t5_checkpoint),
                                   tokenizer_path=str(weights / t2v_1_3B.t5_tokenizer), shard_fn=None)
        text_loaded = time.perf_counter()
        with torch.no_grad():
            self.empty_context = self.text([""], torch.device("cpu"))[0].detach().float().clone()
        self.load_timing = {"vae_load_seconds": vae_loaded - started,
                            "umt5_load_seconds": text_loaded - vae_loaded,
                            "empty_context_seconds": time.perf_counter() - text_loaded}

    @torch.no_grad()
    def encode(self, pixels, caption):
        started = time.perf_counter()
        latent = self.vae.encode([pixels])[0].detach().float().clone()
        vae_done = time.perf_counter()
        context = self.text([caption], torch.device("cpu"))[0].detach().float().clone()
        finished = time.perf_counter()
        expected = (16, (pixels.shape[1] - 1) // 4 + 1, pixels.shape[2] // 8, pixels.shape[3] // 8)
        if tuple(latent.shape) != expected or context.ndim != 2 or context.shape[1] != 4096:
            raise ValueError(f"invalid encoder shapes: {latent.shape}, {context.shape}")
        if not torch.isfinite(latent).all() or not torch.isfinite(context).all():
            raise RuntimeError("online encoder produced non-finite values")
        return latent, context, {"vae_seconds": vae_done - started, "umt5_seconds": finished - vae_done}


class OnlineInput:
    def __init__(self, args, local_group, local_rank):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.local_rank, self.group = local_rank, local_group
        self.log_groups = args.verify_updates
        self.leader = self.rank - local_rank
        error = None
        try:
            self.data = RawMedia(args.data_index, image_size=args.image_size, image_bucket=args.image_bucket,
                                 video_frames=args.video_frames, video_fps=args.video_fps, seed=args.seed,
                                 video_root=args.video_root, training_task=args.training_task)
            self.sampler = SynchronizedBucketBatchSampler(self.data, batch_size=args.batch_size,
                                                           num_replicas=self.world, rank=self.rank,
                                                           seed=args.seed, shuffle=True)
        except Exception as exc:
            error = f"raw input: {type(exc).__name__}: {exc}"
        self._check_errors(error)
        self.encoder = None
        initial = [None]
        if local_rank == 0:
            print("ONLINE_ENCODER_LOAD_START " + json.dumps({"rank": self.rank}), flush=True)
            try:
                text_dtype = torch.bfloat16 if getattr(args, "text_precision", "fp32") == "bf16" else torch.float32
                self.encoder = FrozenEncoders(args.encoder_weights, text_dtype=text_dtype)
                initial[0] = {"context": self.encoder.empty_context, "error": None}
                print("ONLINE_ENCODER_LOAD_OK " + json.dumps(
                    {"rank": self.rank, **self.encoder.load_timing}), flush=True)
            except Exception as error:
                initial[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(initial, src=self.leader, group=self.group)
        self._check_errors(initial[0].get("error"))
        self.empty_context = initial[0]["context"]
        self.iterator = iter(self.sampler)

    def _check_errors(self, error):
        errors = [None] * self.world
        dist.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError(next(value for value in errors if value))

    def next_batch(self):
        started = time.perf_counter()
        try:
            requests = next(self.iterator)
        except StopIteration:
            self.sampler.next_epoch()
            self.iterator = iter(self.sampler)
            requests = next(self.iterator)
        requests_by_rank = [None] * 4
        dist.all_gather_object(requests_by_rank, requests, group=self.group)
        payload = [None]
        if self.local_rank == 0:
            stats = {"read_seconds": 0.0, "vae_seconds": 0.0, "umt5_seconds": 0.0, "samples": 0}
            batches = []
            try:
                for batch_requests in requests_by_rank:
                    keys, latents, contexts = [], [], []
                    for request in batch_requests:
                        read_start = time.perf_counter()
                        key, caption, pixels = self.data.read(request)
                        stats["read_seconds"] += time.perf_counter() - read_start
                        latent, context, timing = self.encoder.encode(pixels, caption)
                        if self.log_groups:
                            print("ONLINE_ENCODED_SAMPLE " + json.dumps({
                                "rank": self.rank, "sample_id": key, "pixels_shape": list(pixels.shape),
                                "latent_shape": list(latent.shape), "context_shape": list(context.shape),
                                **timing}), flush=True)
                        keys.append(key); latents.append(latent); contexts.append(context)
                        stats["vae_seconds"] += timing["vae_seconds"]
                        stats["umt5_seconds"] += timing["umt5_seconds"]
                        stats["samples"] += 1
                    batches.append({"keys": keys, "latents": torch.stack(latents), "contexts": contexts})
                payload[0] = {"batches": batches, "stats": stats, "error": None}
                if self.log_groups:
                    print("ONLINE_ENCODER_BATCH " + json.dumps({"rank": self.rank, **stats}), flush=True)
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(payload, src=self.leader, group=self.group)
        self._check_errors(payload[0].get("error"))
        batch = payload[0]["batches"][self.local_rank]
        stats = dict(payload[0]["stats"]) if self.local_rank == 0 else {
            "read_seconds": 0.0, "vae_seconds": 0.0, "umt5_seconds": 0.0, "samples": 0}
        stats["data_seconds"] = time.perf_counter() - started
        self.sampler.advance()
        return batch, stats

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state)
        self.iterator = iter(self.sampler)
