#!/usr/bin/env python3
"""Expand extracted video manifests into fixed 3-second clip manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, ensure_dir, read_jsonl, sample_id, utc_now, write_json, write_jsonl


def row_seconds(row: dict[str, Any]) -> float | None:
    for key in ("seconds", "duration"):
        try:
            value = float(row.get(key))
            if math.isfinite(value) and value > 0:
                return value
        except (TypeError, ValueError):
            pass
    probe = row.get("ffprobe") or {}
    try:
        value = float(probe.get("duration"))
        if math.isfinite(value) and value > 0:
            return value
    except (TypeError, ValueError):
        pass
    try:
        frames = float(row.get("frame"))
        fps = float(row.get("fps"))
        if math.isfinite(frames) and math.isfinite(fps) and fps > 0:
            return frames / fps
    except (TypeError, ValueError):
        pass
    return None


def segment_count(duration: float, *, clip_duration: float, max_segments: int, safety_margin: float) -> int:
    usable = duration - clip_duration - safety_margin
    if usable < -1e-9:
        return 0
    return max(0, min(max_segments, int(math.floor(usable / clip_duration)) + 1))


def expand_rows(
    rows: list[dict[str, Any]],
    *,
    clip_duration: float,
    max_segments: int,
    safety_margin: float,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for row in rows:
        duration = row_seconds(row)
        sid = sample_id(row)
        if duration is None:
            skipped.append({"sample_id": sid, "video": row.get("video"), "reason": "missing_duration"})
            continue
        count = segment_count(duration, clip_duration=clip_duration, max_segments=max_segments, safety_margin=safety_margin)
        if count <= 0:
            skipped.append({"sample_id": sid, "video": row.get("video"), "duration": duration, "reason": "too_short"})
            continue
        counts[count] = counts.get(count, 0) + 1
        for idx in range(count):
            out = dict(row)
            out["segment_source_sample_id"] = sid
            out["clip_index"] = idx
            out["clip_start_sec"] = round(idx * clip_duration, 6)
            out["clip_duration_sec"] = clip_duration
            out["clip_source_duration_sec"] = duration
            out["split"] = split
            out["sample_id"] = f"{sid}_seg{idx:03d}"
            segments.append(out)
    summary = {
        "source_rows": len(rows),
        "segment_rows": len(segments),
        "skipped_rows": len(skipped),
        "clip_duration_sec": clip_duration,
        "max_segments_per_video": max_segments,
        "safety_margin_sec": safety_margin,
        "segments_per_video": {str(k): v for k, v in sorted(counts.items())},
    }
    return segments, skipped, summary


def run(args: argparse.Namespace) -> None:
    rows = [row for manifest in args.manifest for row in read_jsonl(manifest)]
    if args.limit is not None:
        rows = rows[: args.limit]
    segments, skipped, summary = expand_rows(
        rows,
        clip_duration=args.clip_duration,
        max_segments=args.max_segments,
        safety_margin=args.safety_margin,
        split=args.split,
    )
    write_jsonl(args.output, segments)
    write_jsonl(args.skipped, skipped)
    report = {
        "created_at": utc_now(),
        "manifests": [str(path) for path in args.manifest],
        "output": str(args.output),
        "skipped": str(args.skipped),
        **summary,
    }
    write_json(args.report, report)
    print(json.dumps({"source_rows": len(rows), "segment_rows": len(segments), "skipped_rows": len(skipped)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=root / "data/openvid/meta/stage1_train_segments_3s_cap8.jsonl")
    parser.add_argument("--skipped", type=Path, default=root / "data/openvid/meta/stage1_train_segments_3s_cap8_skipped.jsonl")
    parser.add_argument("--report", type=Path, default=root / "reports/stage1_segments/segments_3s_cap8.json")
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--safety-margin", type=float, default=1.0 / 16.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", default="train_stage1_segments")
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
