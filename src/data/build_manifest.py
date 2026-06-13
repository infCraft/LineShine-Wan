#!/usr/bin/env python3
"""Build OpenVid W1 manifests and part plan with bounded memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


URL_OR_HTML_RE = re.compile(r"(https?://|www\.|<[^>]+>|&lt;|&gt;|\.com\b|\.net\b|\.org\b)", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
PART_FILE_RE = re.compile(r"^OpenVid_part(?P<num>\d+)(?P<suffix>\.zip|_part[a-z]+)$")
PART_CSV_RE = re.compile(r"OpenVid_part(?P<num>\d+)\.csv$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = CONTROL_RE.sub(" ", value.strip())
    return re.sub(r"\s+", " ", text)


def caption_feature_tuple(text: str) -> tuple[int, float, bool, bool, bool, int]:
    words = WORD_RE.findall(text)
    ascii_letters = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    english_ratio = ascii_letters / alpha_chars if alpha_chars else 0.0
    filename_like = bool(
        re.fullmatch(r"[\w.\- /]+", text)
        and (" " not in text or text.count(" ") <= 2)
        and (
            text.lower().endswith((".mp4", ".mov", ".avi", ".webm"))
            or re.search(r"\.(mp4|mov|avi|webm)\b", text.lower())
        )
    )
    return (
        len(words),
        english_ratio,
        bool(URL_OR_HTML_RE.search(text)),
        filename_like,
        bool(CONTROL_RE.search(text)),
        len(text),
    )


def source_id_from_video(video: str) -> str:
    stem = re.sub(r"\.[^.]+$", "", str(video))
    for pattern in (r"_\d+_\d+to\d+$", r"_\d+to\d+$", r"_\d{1,5}$"):
        stripped = re.sub(pattern, "", stem)
        if stripped != stem:
            return stripped
    return stem


def scan_shared_parts(shared_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    temp_files: list[dict[str, Any]] = []

    if shared_dir.exists():
        for entry in shared_dir.iterdir():
            if not entry.is_file():
                continue
            match = PART_FILE_RE.match(entry.name)
            if not match:
                continue
            part_num = int(match.group("num"))
            item = groups.setdefault(part_num, {"part_num": part_num, "files": [], "bytes": 0})
            stat = entry.stat()
            item["files"].append(
                {
                    "name": entry.name,
                    "path": str(entry),
                    "bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                }
            )
            item["bytes"] += stat.st_size

        temp_dir = shared_dir / "._____temp"
        if temp_dir.exists():
            for entry in temp_dir.iterdir():
                if entry.is_file() and PART_FILE_RE.match(entry.name):
                    stat = entry.stat()
                    temp_files.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "bytes": stat.st_size,
                            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                        }
                    )

    completed: dict[int, dict[str, Any]] = {}
    incomplete: dict[int, dict[str, Any]] = {}
    for part_num, item in groups.items():
        names = {f["name"] for f in item["files"]}
        has_zip = f"OpenVid_part{part_num}.zip" in names
        split_parts = sorted(name for name in names if f"OpenVid_part{part_num}_part" in name)
        has_complete_split = f"OpenVid_part{part_num}_partaa" in names and any(
            name != f"OpenVid_part{part_num}_partaa" for name in split_parts
        )
        item["complete"] = bool(has_zip or has_complete_split)
        item["format"] = "zip" if has_zip else ("split" if has_complete_split else "incomplete_split")
        item["files"] = sorted(item["files"], key=lambda x: x["name"])
        if item["complete"]:
            completed[part_num] = item
        else:
            incomplete[part_num] = item

    summary = {
        "shared_dir": str(shared_dir),
        "completed_part_groups": len(completed),
        "completed_file_count": sum(len(item["files"]) for item in completed.values()),
        "completed_bytes": sum(item["bytes"] for item in completed.values()),
        "incomplete_top_level_groups": len(incomplete),
        "excluded_temp_file_count": len(temp_files),
        "excluded_temp_bytes": sum(item["bytes"] for item in temp_files),
        "excluded_temp_files": sorted(temp_files, key=lambda x: x["name"]),
    }
    return completed, summary


def load_mapping(mapping_dir: Path) -> tuple[pd.DataFrame, pd.Series, int]:
    files = sorted(
        (mapping_dir / "video_mappings").glob("OpenVid_part*.csv"),
        key=lambda p: int(PART_CSV_RE.search(p.name).group("num")),
    )
    if not files:
        raise FileNotFoundError(f"No mapping CSV files under {mapping_dir / 'video_mappings'}")
    frames = []
    for path in files:
        df = pd.read_csv(path, usecols=["video", "zip_file", "video_path"])
        part_num = int(PART_CSV_RE.search(path.name).group("num"))
        df["part_num"] = np.int16(part_num)
        df["part"] = f"OpenVid_part{part_num}.zip"
        frames.append(df)
    mapping = pd.concat(frames, ignore_index=True)
    mapping = mapping.drop_duplicates(subset=["video"], keep="first")
    mapping_counts = mapping.groupby("part_num").size()
    return mapping, mapping_counts, len(files)


def numeric_quantiles(series: pd.Series) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {key: None for key in ["min", "p05", "p20", "p50", "p80", "p95", "max", "mean"]}
    qs = clean.quantile([0.05, 0.20, 0.50, 0.80, 0.95])
    return {
        "min": float(clean.min()),
        "p05": float(qs.loc[0.05]),
        "p20": float(qs.loc[0.20]),
        "p50": float(qs.loc[0.50]),
        "p80": float(qs.loc[0.80]),
        "p95": float(qs.loc[0.95]),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def update_dist_stats(prefix: str, stats: dict[str, list[float]], frame: pd.DataFrame) -> None:
    for col in ["seconds", "fps", "frame", "caption_word_count", "aesthetic score", "motion score", "temporal consistency score"]:
        key = f"{prefix}:{col}"
        stats.setdefault(key, []).extend(pd.to_numeric(frame[col], errors="coerce").dropna().astype(float).tolist())


def dist_from_stats(stats: dict[str, list[float]], prefix: str, camera_counts: Counter[str]) -> dict[str, Any]:
    name_map = {
        "seconds": "seconds",
        "fps": "fps",
        "frame": "frame",
        "caption_word_count": "caption_word_count",
        "aesthetic score": "aesthetic_score",
        "motion score": "motion_score",
        "temporal consistency score": "temporal_consistency_score",
    }
    out = {name_map[col]: numeric_quantiles(pd.Series(stats.get(f"{prefix}:{col}", []))) for col in name_map}
    out["camera_motion"] = dict(camera_counts)
    return out


def append_jsonl(frame: pd.DataFrame, path: Path, columns: list[str], first_write: bool) -> None:
    mode = "w" if first_write else "a"
    frame.loc[:, columns].to_json(path, orient="records", lines=True, force_ascii=False, mode=mode)


def camera_balance(df: pd.DataFrame, max_fraction: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return df, {"max_fraction": max_fraction, "before": {}, "after": {}}
    before = df["camera motion"].fillna("unknown").value_counts().to_dict()
    cap = max(1, math.ceil(len(df) * max_fraction))
    pieces = []
    for _, group in df.groupby(df["camera motion"].fillna("unknown"), sort=False):
        pieces.append(group.sort_values("quality_score", ascending=False).head(cap))
    balanced = pd.concat(pieces, ignore_index=True).sort_values("quality_score", ascending=False).reset_index(drop=True)
    after = balanced["camera motion"].fillna("unknown").value_counts().to_dict()
    return balanced, {"max_fraction": max_fraction, "cap_per_camera": cap, "before": before, "after": after}


def choose_parts(candidates: pd.DataFrame, mapping_counts: pd.Series, args: argparse.Namespace) -> list[int]:
    part_counts = candidates.groupby("part_num").size().rename("candidate_count")
    part_stats = part_counts.to_frame().join(mapping_counts.rename("mapping_count"), how="left")
    part_stats["mapping_count"] = part_stats["mapping_count"].fillna(part_stats["candidate_count"])
    part_stats["density"] = part_stats["candidate_count"] / part_stats["mapping_count"].clip(lower=1)
    part_stats["bucket"] = part_stats.index.astype(int) % args.num_part_buckets
    buckets = {}
    for bucket, group in part_stats.groupby("bucket"):
        buckets[int(bucket)] = [
            int(x) for x in group.sort_values(["density", "candidate_count"], ascending=[False, False]).index.tolist()
        ]

    selected: list[int] = []
    selected_set: set[int] = set()
    cumulative = 0
    clip_limit = min(args.target_clips, args.byte_budget // args.avg_clip_bytes)
    while cumulative < clip_limit and any(buckets.values()):
        progressed = False
        for bucket in range(args.num_part_buckets):
            queue = buckets.get(bucket, [])
            while queue and queue[0] in selected_set:
                queue.pop(0)
            if not queue:
                continue
            part_num = queue.pop(0)
            selected.append(part_num)
            selected_set.add(part_num)
            cumulative += int(part_stats.loc[part_num, "candidate_count"])
            progressed = True
            if cumulative >= clip_limit:
                break
        if not progressed:
            break
    return selected


def split_train_val(selected: pd.DataFrame, target_val: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = selected.groupby("source_id", sort=False).size().reset_index(name="count")
    groups["hash"] = groups["source_id"].map(lambda x: hashlib.sha1(f"{seed}:{x}".encode("utf-8")).hexdigest())
    groups = groups.sort_values("hash").reset_index(drop=True)

    val_sources: set[str] = set()
    val_count = 0
    for _, row in groups.iterrows():
        count = int(row["count"])
        if val_count + count <= target_val:
            val_sources.add(str(row["source_id"]))
            val_count += count
            if val_count == target_val:
                break
    if val_count < target_val:
        for _, row in groups[(groups["count"] == 1) & (~groups["source_id"].isin(val_sources))].iterrows():
            val_sources.add(str(row["source_id"]))
            val_count += 1
            if val_count == target_val:
                break
    if val_count < target_val:
        for _, row in groups[~groups["source_id"].isin(val_sources)].iterrows():
            val_sources.add(str(row["source_id"]))
            val_count += int(row["count"])
            if val_count >= target_val:
                break

    is_val = selected["source_id"].isin(val_sources)
    return selected[~is_val].copy(), selected[is_val].copy()


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# W1 OpenVid Manifest Summary",
        "",
        f"- created_at_utc: `{summary['created_at_utc']}`",
        f"- csv_rows: `{summary['csv_rows']}`",
        f"- mapping_rows: `{summary['mapping_rows']}`",
        f"- mapping_files: `{summary['mapping_files']}`",
        f"- mapping_join_coverage: `{summary['mapping_join_coverage']:.6f}`",
        f"- hard_filter_pass: `{summary['filter_counts']['hard_filter_pass']}`",
        f"- quality_filter_pass: `{summary['filter_counts']['quality_filter_pass']}`",
        f"- camera_balanced_filtered: `{summary['filter_counts']['camera_balanced_filtered']}`",
        f"- selected_total: `{summary['selection']['selected_total']}`",
        f"- train_count: `{summary['selection']['train_count']}`",
        f"- val_count: `{summary['selection']['val_count']}`",
        f"- selected_parts: `{summary['selection']['selected_parts']}`",
        f"- estimated_selected_gib: `{summary['selection']['estimated_selected_gib']:.2f}`",
        "",
        "## Gate Checks",
    ]
    for key, value in summary["gate_checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    report_root = Path(args.report_root)
    csv_path = Path(args.csv)
    mapping_dir = Path(args.mapping_dir)
    shared_dir = Path(args.shared_openvid_dir)

    for subdir in ["W1.1_metadata_mapping", "W1.2_manifest", "W1.3_subset_plan", "W1.4_manual_review"]:
        ensure_dir(report_root / subdir)
    ensure_dir(output_dir)

    shared_parts, shared_summary = scan_shared_parts(shared_dir)
    write_json(output_dir / "shared_openvid_part_inventory.json", {"summary": shared_summary, "parts": shared_parts})
    with (report_root / "W1.1_metadata_mapping" / "shared_openvid_part_inventory.tsv").open("w", encoding="utf-8") as f:
        f.write("part_num\tformat\tbytes\tfiles\n")
        for part_num, item in sorted(shared_parts.items()):
            f.write(f"{part_num}\t{item['format']}\t{item['bytes']}\t{','.join(file['name'] for file in item['files'])}\n")

    mapping, mapping_counts, mapping_files = load_mapping(mapping_dir)
    mapping_rows = len(mapping)

    raw_path = output_dir / "openvid_manifest_raw.jsonl"
    filtered_path = output_dir / "openvid_manifest_filtered.jsonl"
    for path in [raw_path, filtered_path]:
        if path.exists():
            path.unlink()

    raw_cols = [
        "video",
        "source_id",
        "part",
        "part_num",
        "video_path",
        "mapping_joined",
        "caption_clean",
        "raw_caption",
        "caption_word_count",
        "caption_english_ratio",
        "seconds",
        "fps",
        "frame",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
    ]

    filter_counts = Counter()
    dist_stats: dict[str, list[float]] = {}
    raw_camera_counts: Counter[str] = Counter()
    hard_camera_counts: Counter[str] = Counter()
    hard_frames: list[pd.DataFrame] = []
    raw_first = True

    usecols = [
        "video",
        "caption",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
        "frame",
        "fps",
        "seconds",
    ]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=args.chunksize):
        filter_counts["csv_rows"] += len(chunk)
        chunk["raw_caption"] = chunk["caption"].fillna("")
        chunk["caption_clean"] = chunk["raw_caption"].map(clean_caption)
        features = chunk["caption_clean"].map(caption_feature_tuple)
        chunk["caption_word_count"] = features.map(lambda x: x[0]).astype(np.int16)
        chunk["caption_english_ratio"] = features.map(lambda x: x[1]).astype(np.float32)
        chunk["caption_has_url_or_html"] = features.map(lambda x: x[2])
        chunk["caption_filename_like"] = features.map(lambda x: x[3])
        chunk["caption_has_control"] = features.map(lambda x: x[4])
        chunk["caption_len_chars"] = features.map(lambda x: x[5]).astype(np.int16)
        chunk["source_id"] = chunk["video"].map(source_id_from_video)
        for col in ["seconds", "fps", "frame", "aesthetic score", "motion score", "temporal consistency score"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        merged = chunk.merge(mapping, on="video", how="left", validate="many_to_one")
        merged["mapping_joined"] = merged["part_num"].notna()
        merged["part_num"] = merged["part_num"].astype("Int64")
        mapping_joined_count = int(merged["mapping_joined"].sum())
        mapping_missing_count = int((~merged["mapping_joined"]).sum())
        filter_counts["mapping_joined"] += mapping_joined_count
        filter_counts["mapping_missing"] += mapping_missing_count
        raw_camera_counts.update(merged["camera motion"].fillna("unknown").astype(str).tolist())
        update_dist_stats("raw", dist_stats, merged)
        append_jsonl(merged, raw_path, raw_cols, raw_first)
        raw_first = False

        masks = {
            "duration_ge_4": merged["seconds"] >= 4.0,
            "fps_ge_16": merged["fps"] >= 16.0,
            "frames_ge_49": merged["frame"] >= 49,
            "caption_nonempty": merged["caption_clean"].str.len() > 0,
            "caption_english_main": merged["caption_english_ratio"] >= args.min_english_ratio,
            "caption_min_words": merged["caption_word_count"] >= args.min_caption_words,
            "caption_max_words": merged["caption_word_count"] <= args.max_caption_words,
            "caption_no_url_html": ~merged["caption_has_url_or_html"],
            "caption_not_filename_like": ~merged["caption_filename_like"],
            "caption_no_control_chars": ~merged["caption_has_control"],
            "mapping_joined_filter": merged["mapping_joined"],
        }
        hard_mask = pd.Series(True, index=merged.index)
        for key, mask in masks.items():
            count = int(mask.fillna(False).sum())
            filter_counts[key] += count
            hard_mask &= mask.fillna(False)
        hard = merged.loc[hard_mask].copy()
        filter_counts["hard_filter_pass"] += len(hard)
        if len(hard):
            hard["part_num"] = hard["part_num"].astype(int)
            hard_frames.append(hard)
            hard_camera_counts.update(hard["camera motion"].fillna("unknown").astype(str).tolist())
            update_dist_stats("hard", dist_stats, hard)

    hard_all = pd.concat(hard_frames, ignore_index=True) if hard_frames else pd.DataFrame()
    if hard_all.empty:
        raise RuntimeError("No samples passed hard filters")

    aesthetic_q = hard_all["aesthetic score"].quantile(1.0 - args.aesthetic_top_fraction)
    temporal_q = hard_all["temporal consistency score"].quantile(1.0 - args.temporal_top_fraction)
    motion_low = hard_all["motion score"].quantile(args.motion_low_quantile)
    motion_high = hard_all["motion score"].quantile(args.motion_high_quantile)
    hard_all["aesthetic_rank"] = hard_all["aesthetic score"].rank(method="average", pct=True).astype(np.float32)
    hard_all["temporal_rank"] = hard_all["temporal consistency score"].rank(method="average", pct=True).astype(np.float32)
    denom = max(float(motion_high - motion_low), 1e-12)
    hard_all["motion_rank_clipped"] = ((hard_all["motion score"] - motion_low) / denom).clip(0.0, 1.0).astype(np.float32)
    hard_all["quality_score"] = (
        0.4 * hard_all["aesthetic_rank"] + 0.4 * hard_all["temporal_rank"] + 0.2 * hard_all["motion_rank_clipped"]
    ).astype(np.float32)

    quality = hard_all[
        (hard_all["motion score"] >= motion_low)
        & (hard_all["motion score"] <= motion_high)
    ].copy()
    filter_counts["quality_filter_pass"] = len(quality)
    filtered, camera_balance_summary = camera_balance(quality, args.camera_max_fraction)
    filter_counts["camera_balanced_filtered"] = len(filtered)

    filtered_cols = [
        "video",
        "source_id",
        "part",
        "part_num",
        "video_path",
        "caption_clean",
        "raw_caption",
        "caption_word_count",
        "seconds",
        "fps",
        "frame",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
        "quality_score",
        "aesthetic_rank",
        "temporal_rank",
        "motion_rank_clipped",
    ]
    append_jsonl(filtered, filtered_path, filtered_cols, True)

    selected_parts = choose_parts(filtered, mapping_counts, args)
    part_order = {part: order for order, part in enumerate(selected_parts, start=1)}
    selected_pool = filtered[filtered["part_num"].isin(selected_parts)].copy()
    selected_pool["part_order"] = selected_pool["part_num"].map(part_order)
    selected_pool = selected_pool.sort_values(["part_order", "quality_score"], ascending=[True, False])
    clip_limit = min(args.target_clips, int(args.byte_budget // args.avg_clip_bytes))
    selected = selected_pool.head(clip_limit).copy().reset_index(drop=True)
    selected["sample_id"] = [f"openvid_{i:06d}" for i in range(len(selected))]

    train, val = split_train_val(selected, args.val_clips, args.seed)

    common_cols = [
        "sample_id",
        "video",
        "source_id",
        "part",
        "part_num",
        "video_path",
        "caption_clean",
        "raw_caption",
        "caption_word_count",
        "seconds",
        "fps",
        "frame",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
        "quality_score",
        "aesthetic_rank",
        "temporal_rank",
        "motion_rank_clipped",
    ]
    train.loc[:, common_cols].to_json(output_dir / "openvid_train_98k.jsonl", orient="records", lines=True, force_ascii=False)
    val.loc[:, common_cols].to_json(output_dir / "openvid_val_2k.jsonl", orient="records", lines=True, force_ascii=False)

    selected_counts = selected.groupby("part_num").size().to_dict()
    candidate_counts = filtered.groupby("part_num").size().to_dict()
    plan_parts = []
    for part_num in sorted(selected_counts, key=lambda p: part_order.get(int(p), 10**9)):
        part_num = int(part_num)
        shared = shared_parts.get(part_num)
        mapping_count = int(mapping_counts.get(part_num, 0))
        candidate_count = int(candidate_counts.get(part_num, 0))
        selected_count = int(selected_counts[part_num])
        plan_parts.append(
            {
                "order": int(part_order[part_num]),
                "part": f"OpenVid_part{part_num}.zip",
                "part_num": part_num,
                "source": "shared_readonly" if shared else "local_download",
                "shared_files": shared["files"] if shared else [],
                "shared_bytes": shared["bytes"] if shared else 0,
                "mapping_count": mapping_count,
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "estimated_selected_bytes": selected_count * args.avg_clip_bytes,
                "density": candidate_count / max(mapping_count, 1),
            }
        )

    plan = {
        "created_at_utc": utc_now(),
        "target_clips": args.target_clips,
        "val_clips": args.val_clips,
        "avg_clip_bytes_estimate": args.avg_clip_bytes,
        "byte_budget": args.byte_budget,
        "selected_total": int(len(selected)),
        "train_count": int(len(train)),
        "val_count": int(len(val)),
        "estimated_selected_bytes": int(len(selected) * args.avg_clip_bytes),
        "estimated_selected_gib": float(len(selected) * args.avg_clip_bytes / 1024**3),
        "shared_readonly_parts": sum(1 for item in plan_parts if item["source"] == "shared_readonly"),
        "local_download_parts": sum(1 for item in plan_parts if item["source"] == "local_download"),
        "parts": plan_parts,
    }
    write_json(output_dir / "part_download_plan.json", plan)
    write_json(report_root / "W1.3_subset_plan" / "part_download_plan.summary.json", plan)

    filtered_camera_counts = Counter(filtered["camera motion"].fillna("unknown").astype(str).tolist())
    selected_camera_counts = Counter(selected["camera motion"].fillna("unknown").astype(str).tolist())
    distributions = {
        "raw": dist_from_stats(dist_stats, "raw", raw_camera_counts),
        "hard": dist_from_stats(dist_stats, "hard", hard_camera_counts),
        "filtered": {
            "seconds": numeric_quantiles(filtered["seconds"]),
            "fps": numeric_quantiles(filtered["fps"]),
            "frame": numeric_quantiles(filtered["frame"]),
            "caption_word_count": numeric_quantiles(filtered["caption_word_count"]),
            "aesthetic_score": numeric_quantiles(filtered["aesthetic score"]),
            "motion_score": numeric_quantiles(filtered["motion score"]),
            "temporal_consistency_score": numeric_quantiles(filtered["temporal consistency score"]),
            "camera_motion": dict(filtered_camera_counts),
        },
        "selected": {
            "seconds": numeric_quantiles(selected["seconds"]),
            "fps": numeric_quantiles(selected["fps"]),
            "frame": numeric_quantiles(selected["frame"]),
            "caption_word_count": numeric_quantiles(selected["caption_word_count"]),
            "aesthetic_score": numeric_quantiles(selected["aesthetic score"]),
            "motion_score": numeric_quantiles(selected["motion score"]),
            "temporal_consistency_score": numeric_quantiles(selected["temporal consistency score"]),
            "camera_motion": dict(selected_camera_counts),
        },
    }

    train_sources = set(train["source_id"])
    val_sources = set(val["source_id"])
    gate_checks = {
        "mapping_join_coverage_ge_0.95": bool(filter_counts["mapping_joined"] / filter_counts["csv_rows"] >= 0.95),
        "selected_clips_target_or_budget_limited": bool(len(selected) >= min(args.target_clips, args.byte_budget // args.avg_clip_bytes)),
        "estimated_selected_bytes_le_budget": bool(len(selected) * args.avg_clip_bytes <= args.byte_budget),
        "train_val_source_disjoint": bool(train_sources.isdisjoint(val_sources)),
        "train_val_sample_id_disjoint": bool(set(train["sample_id"]).isdisjoint(set(val["sample_id"]))),
        "train_val_no_duplicate_sample_id": bool(selected["sample_id"].is_unique),
    }
    summary = {
        "created_at_utc": utc_now(),
        "csv_path": str(csv_path),
        "csv_rows": int(filter_counts["csv_rows"]),
        "mapping_dir": str(mapping_dir),
        "mapping_rows": int(mapping_rows),
        "mapping_files": int(mapping_files),
        "mapping_join_coverage": float(filter_counts["mapping_joined"] / filter_counts["csv_rows"]),
        "filter_thresholds": {
            "aesthetic_top_40pct_reference": float(aesthetic_q),
            "temporal_top_50pct_reference": float(temporal_q),
            "motion_low_p20": float(motion_low),
            "motion_high_p95": float(motion_high),
            "min_english_ratio": args.min_english_ratio,
            "min_caption_words": args.min_caption_words,
            "max_caption_words": args.max_caption_words,
            "camera_max_fraction": args.camera_max_fraction,
        },
        "filter_counts": {k: int(v) for k, v in filter_counts.items()},
        "camera_balance": camera_balance_summary,
        "distributions": distributions,
        "selection": {
            "selected_total": int(len(selected)),
            "train_count": int(len(train)),
            "val_count": int(len(val)),
            "selected_parts": int(len(plan_parts)),
            "shared_readonly_parts": int(plan["shared_readonly_parts"]),
            "local_download_parts": int(plan["local_download_parts"]),
            "estimated_selected_bytes": int(plan["estimated_selected_bytes"]),
            "estimated_selected_gib": float(plan["estimated_selected_gib"]),
        },
        "shared_inventory": shared_summary,
        "gate_checks": gate_checks,
    }
    write_json(report_root / "W1.2_manifest" / "filter_stats.json", summary)
    write_summary_md(report_root / "W1.2_manifest" / "summary.md", summary)

    sample = selected.sample(n=min(args.review_samples, len(selected)), random_state=args.seed)
    review_cols = [
        "sample_id",
        "video",
        "source_id",
        "part",
        "caption_clean",
        "seconds",
        "fps",
        "frame",
        "aesthetic score",
        "motion score",
        "temporal consistency score",
        "camera motion",
        "quality_score",
    ]
    sample.loc[:, review_cols].to_csv(report_root / "W1.4_manual_review" / "caption_review_100.tsv", sep="\t", index=False)
    write_json(report_root / "W1.4_manual_review" / "review_distribution_summary.json", distributions["selected"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mapping-dir", required=True)
    parser.add_argument("--meta-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--shared-openvid-dir", default="/mnt/beegfs/home/yezy/openvid")
    parser.add_argument("--target-clips", type=int, default=100_000)
    parser.add_argument("--val-clips", type=int, default=2_000)
    parser.add_argument("--avg-clip-bytes", type=int, default=8_500_000)
    parser.add_argument("--byte-budget", type=int, default=1_200_000_000_000)
    parser.add_argument("--num-part-buckets", type=int, default=10)
    parser.add_argument("--aesthetic-top-fraction", type=float, default=0.40)
    parser.add_argument("--temporal-top-fraction", type=float, default=0.50)
    parser.add_argument("--motion-low-quantile", type=float, default=0.20)
    parser.add_argument("--motion-high-quantile", type=float, default=0.95)
    parser.add_argument("--camera-max-fraction", type=float, default=0.50)
    parser.add_argument("--min-caption-words", type=int, default=5)
    parser.add_argument("--max-caption-words", type=int, default=256)
    parser.add_argument("--min-english-ratio", type=float, default=0.75)
    parser.add_argument("--review-samples", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
