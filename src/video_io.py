"""Video I/O backed by imageio-ffmpeg's bundled static executable."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import torch


class _ResizeWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith("The frame size for reading")


logging.getLogger("imageio_ffmpeg").addFilter(_ResizeWarningFilter())


def _read_metadata(path: Path) -> dict[str, Any]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        return dict(next(reader))
    finally:
        reader.close()


def read_video_grid(
    path: Path,
    *,
    frames: int,
    target_fps: float,
    image_size: int,
    clip_start_sec: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if frames <= 0 or target_fps <= 0 or image_size <= 0:
        raise ValueError("frames, target_fps, and image_size must be positive")
    metadata = _read_metadata(path)
    source_fps = float(metadata.get("fps") or 0.0) or None
    source_duration = float(metadata.get("duration") or 0.0) or None
    source_size = tuple(int(value) for value in metadata.get("source_size") or metadata.get("size") or ())
    window = (frames - 1) / target_fps
    if clip_start_sec is None:
        if source_duration is None:
            raise ValueError(f"cannot center-sample video without duration metadata: {path}")
        start = max(0.0, source_duration - window) / 2.0
        sampling_mode = "center"
    else:
        start = float(clip_start_sec)
        sampling_mode = "fixed_start"
    if start < 0:
        raise ValueError(f"negative clip start for {path}: {start}")
    tolerance = max(1.0 / target_fps, 1.0 / source_fps if source_fps else 0.0)
    if source_duration is not None and start + window > source_duration + tolerance:
        raise ValueError(
            f"{path} sampling window [{start:.3f}, {start + window:.3f}] exceeds "
            f"duration {source_duration:.3f}"
        )

    video_filter = (
        f"setpts=PTS-STARTPTS,fps={target_fps:.12g}:round=near,"
        f"scale={image_size}:{image_size}:force_original_aspect_ratio=increase:flags=bicubic,"
        f"crop={image_size}:{image_size}"
    )
    input_params = ["-ss", f"{start:.9f}"] if start > 0 else None
    output_params = ["-vf", video_filter, "-frames:v", str(frames)]
    reader = imageio_ffmpeg.read_frames(
        str(path),
        pix_fmt="rgb24",
        input_params=input_params,
        output_params=output_params,
    )
    decoded: list[np.ndarray] = []
    expected_bytes = image_size * image_size * 3
    try:
        output_metadata = dict(next(reader))
        for raw in reader:
            if len(raw) != expected_bytes:
                raise RuntimeError(
                    f"ffmpeg returned {len(raw)} bytes for {path}; expected {expected_bytes}"
                )
            decoded.append(
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(image_size, image_size, 3)
                .copy()
            )
    finally:
        reader.close()
    if len(decoded) != frames:
        raise ValueError(f"ffmpeg decoded {len(decoded)} frames from {path}; expected {frames}")
    array = np.stack(decoded, axis=0)
    return array, {
        "backend": "imageio-ffmpeg",
        "ffmpeg_version": metadata.get("ffmpeg_version"),
        "source_fps": source_fps,
        "source_duration_sec": source_duration,
        "source_size": list(source_size),
        "output_size": list(output_metadata.get("size") or (image_size, image_size)),
        "decoded_frames_in_window": len(decoded),
        "sampled_frames": frames,
        "target_fps": target_fps,
        "sampling_mode": sampling_mode,
        "seek_used": start > 0,
        "requested_clip_start_sec": clip_start_sec,
        "sample_start_sec": start,
        "sample_end_sec": start + window,
        "resize_short_side": image_size,
        "crop": "center",
    }


def write_video(path: Path, video: torch.Tensor, *, fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    array = video.detach().float().clamp(-1, 1).add(1).mul(127.5).byte()
    array = np.ascontiguousarray(array.permute(1, 2, 3, 0).cpu().numpy())
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"video must be [3,T,H,W], got {tuple(video.shape)}")
    height, width = int(array.shape[1]), int(array.shape[2])
    writer = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )
    try:
        writer.send(None)
        for frame in array:
            writer.send(frame.tobytes())
    finally:
        writer.close()


def probe_video(path: Path) -> dict[str, Any]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    frame_count = 0
    try:
        metadata = dict(next(reader))
        for _ in reader:
            frame_count += 1
    finally:
        reader.close()
    size = tuple(int(value) for value in metadata.get("size") or metadata.get("source_size") or ())
    if len(size) != 2:
        raise RuntimeError(f"ffmpeg did not report output dimensions for {path}")
    return {
        "fps": float(metadata["fps"]) if metadata.get("fps") else None,
        "frame_count": frame_count,
        "width": size[0],
        "height": size[1],
        "backend": "imageio-ffmpeg",
        "ffmpeg_version": metadata.get("ffmpeg_version"),
    }
