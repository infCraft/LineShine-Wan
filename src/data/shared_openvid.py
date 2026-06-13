#!/usr/bin/env python3
"""Shared-only OpenVid scan, split dry-run, smoke manifest, and extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, ensure_dir, read_jsonl, sample_id, utc_now, write_json, write_jsonl


PART_ZIP_RE = re.compile(r"^(OpenVid_part(?P<num>\d+))\.zip$")
PART_SEG_RE = re.compile(r"^(OpenVid_part(?P<num>\d+))_part(?P<suffix>[a-z]+)$")
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


def part_key_from_name(name: str) -> str | None:
    match = PART_ZIP_RE.match(name) or PART_SEG_RE.match(name)
    if not match:
        return None
    return f"OpenVid_part{int(match.group('num'))}.zip"


def part_num(part: str) -> int:
    match = re.search(r"OpenVid_part(\d+)", part)
    if not match:
        raise ValueError(f"Cannot parse part number from {part}")
    return int(match.group(1))


def scan_shared_parts(shared_dir: Path) -> dict[str, dict[str, Any]]:
    """Return completed top-level shared parts. Hidden/temp files are ignored."""
    groups: dict[str, dict[str, Any]] = {}
    if not shared_dir.exists():
        return groups

    for entry in shared_dir.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        key = part_key_from_name(entry.name)
        if key is None:
            continue
        stat = entry.stat()
        item = groups.setdefault(key, {"part": key, "part_num": part_num(key), "files": [], "bytes": 0})
        item["files"].append(
            {
                "name": entry.name,
                "path": str(entry),
                "bytes": stat.st_size,
                "mtime": int(stat.st_mtime),
            }
        )
        item["bytes"] += stat.st_size

    completed: dict[str, dict[str, Any]] = {}
    for key, item in groups.items():
        names = {f["name"] for f in item["files"]}
        stem = key[:-4]
        has_zip = key in names
        split_names = sorted(name for name in names if name.startswith(f"{stem}_part"))
        has_split = f"{stem}_partaa" in names and len(split_names) >= 2
        if not has_zip and not has_split:
            continue
        item["format"] = "zip" if has_zip else "split"
        item["files"] = sorted(item["files"], key=lambda x: x["name"])
        completed[key] = item
    return dict(sorted(completed.items(), key=lambda kv: part_num(kv[0])))


def manifest_candidates(manifest_path: Path, shared_parts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    shared = set(shared_parts)
    rows = [row for row in read_jsonl(manifest_path) if str(row.get("part", "")) in shared]
    rows.sort(key=lambda r: float(r.get("quality_score", 0.0)), reverse=True)
    return rows


def summarize_candidates(rows: Iterable[dict[str, Any]], shared_parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_part: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    sample_ids: set[str] = set()
    for row in rows:
        part = str(row["part"])
        info = by_part.setdefault(
            part,
            {
                "part": part,
                "part_num": part_num(part),
                "candidate_count": 0,
                "shared_bytes": shared_parts[part]["bytes"],
                "format": shared_parts[part]["format"],
            },
        )
        info["candidate_count"] += 1
        source_ids.add(str(row.get("source_id", "")))
        sample_ids.add(sample_id(row))
    ranked = sorted(by_part.values(), key=lambda x: (-x["candidate_count"], x["part_num"]))
    return {
        "candidate_count": sum(x["candidate_count"] for x in ranked),
        "unique_source_count": len(source_ids),
        "unique_sample_count": len(sample_ids),
        "by_part": ranked,
    }


def write_part_tsv(path: Path, by_part: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("part\tpart_num\tcandidate_count\tshared_bytes\tformat\n")
        for item in by_part:
            f.write(
                f"{item['part']}\t{item['part_num']}\t{item['candidate_count']}\t{item['shared_bytes']}\t{item['format']}\n"
            )


def select_smoke_rows(
    candidates: list[dict[str, Any]],
    *,
    part: str | None,
    limit: int,
    require_unique_source: bool,
) -> list[dict[str, Any]]:
    rows = [row for row in candidates if part is None or row.get("part") == part]
    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    for row in rows:
        source = str(row.get("source_id", ""))
        if require_unique_source and source in used_sources:
            continue
        out = dict(row)
        out["split"] = "smoke"
        selected.append(out)
        used_sources.add(source)
        if len(selected) >= limit:
            break
    return selected


def freeze_shared_split(
    candidates: list[dict[str, Any]],
    *,
    train_count: int,
    val_count: int,
    min_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(candidates) < min_candidates:
        raise RuntimeError(f"shared-only candidates {len(candidates)} < required {min_candidates}; refusing to freeze")

    target = train_count + val_count
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[str(row.get("source_id", sample_id(row)))].append(row)
    ordered_groups = sorted(
        groups.values(),
        key=lambda rows: max(float(row.get("quality_score", 0.0)) for row in rows),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    for rows in ordered_groups:
        rows = sorted(rows, key=lambda row: float(row.get("quality_score", 0.0)), reverse=True)
        if len(selected) + len(rows) > target:
            continue
        selected.extend(rows)
        if len(selected) == target:
            break
    if len(selected) < target:
        raise RuntimeError(f"could only select {len(selected)} rows without source overlap; need {target}")

    val_sources: set[str] = set()
    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for row in reversed(selected):
        source = str(row.get("source_id", sample_id(row)))
        if len(val) < val_count and source not in val_sources:
            val.append(dict(row, split="val"))
            val_sources.add(source)
        else:
            train.append(dict(row, split="train"))
    if len(val) != val_count or len(train) != train_count:
        raise RuntimeError(f"bad split sizes: train={len(train)} val={len(val)}")
    if {sample_id(r) for r in train} & {sample_id(r) for r in val}:
        raise RuntimeError("sample_id overlap between train and val")
    if {str(r.get("source_id", "")) for r in train} & {str(r.get("source_id", "")) for r in val}:
        raise RuntimeError("source_id overlap between train and val")
    return train, val


def open_part_zip(part_info: dict[str, Any], parts_tmp: Path):
    if part_info["format"] == "zip":
        return zipfile.ZipFile(part_info["files"][0]["path"])

    ensure_dir(parts_tmp)
    tmp_path = parts_tmp / part_info["part"]
    with tmp_path.open("wb") as out:
        for file_info in part_info["files"]:
            with open(file_info["path"], "rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    return zipfile.ZipFile(tmp_path)


def member_index(zf: zipfile.ZipFile) -> dict[str, str]:
    index: dict[str, str] = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        base = Path(name).name
        if Path(base).suffix.lower() in VIDEO_EXTS:
            index.setdefault(base, name)
            index.setdefault(name.lstrip("/"), name)
    return index


def row_member_name(row: dict[str, Any], index: dict[str, str]) -> str | None:
    keys = [str(row.get("video", "")), str(row.get("video_path", ""))]
    for value in keys:
        if value in index:
            return index[value]
        base = Path(value).name
        if base in index:
            return index[base]
    return None


def ffprobe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()[:1000]}
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    return {"ok": True, **stream}


def extract_smoke(
    manifest_path: Path,
    shared_parts: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    parts_tmp: Path,
    output_manifest: Path,
    keep_existing: bool,
    state_path: Path | None = None,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    ensure_dir(parts_tmp)
    rows_by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(manifest_path):
        rows_by_part[str(row["part"])].append(row)

    extracted_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for part, rows in sorted(rows_by_part.items(), key=lambda kv: part_num(kv[0])):
        part_started_at = utc_now()
        before_success = len(extracted_rows)
        before_failures = len(failures)
        if part not in shared_parts:
            failures.extend({"video": row.get("video"), "part": part, "error": "part_not_available"} for row in rows)
            continue
        part_out = ensure_dir(output_dir / Path(part).stem)
        zf = open_part_zip(shared_parts[part], parts_tmp)
        try:
            index = member_index(zf)
            for row in rows:
                member = row_member_name(row, index)
                if member is None:
                    failures.append({"video": row.get("video"), "part": part, "error": "member_not_found"})
                    continue
                out_path = part_out / Path(member).name
                if not out_path.exists() or not keep_existing:
                    with zf.open(member, "r") as src, out_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                probe = ffprobe_video(out_path)
                if not probe.get("ok"):
                    failures.append({"video": row.get("video"), "part": part, "path": str(out_path), "error": probe.get("error")})
                    continue
                enriched = dict(row)
                enriched["local_path"] = str(out_path)
                enriched["ffprobe"] = probe
                extracted_rows.append(enriched)
        finally:
            zf.close()
            tmp_zip = parts_tmp / part
            if shared_parts[part]["format"] == "split" and tmp_zip.exists():
                tmp_zip.unlink()
        if state_path is not None:
            state = {
                "updated_at": utc_now(),
                "last_part": part,
                "parts_done": {
                    part: {
                        "started_at": part_started_at,
                        "finished_at": utc_now(),
                        "requested": len(rows),
                        "extracted": len(extracted_rows) - before_success,
                        "failed": len(failures) - before_failures,
                    }
                },
                "total_extracted_so_far": len(extracted_rows),
                "total_failed_so_far": len(failures),
            }
            if state_path.exists():
                try:
                    prev = json.loads(state_path.read_text(encoding="utf-8"))
                    prev.setdefault("parts_done", {}).update(state["parts_done"])
                    prev.update({k: v for k, v in state.items() if k != "parts_done"})
                    state = prev
                except json.JSONDecodeError:
                    pass
            write_json(state_path, state)

    write_jsonl(output_manifest, extracted_rows)
    failure_path = output_manifest.with_suffix(".failed.jsonl")
    write_jsonl(failure_path, failures)
    return {
        "manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "failure_manifest": str(failure_path),
        "requested": sum(len(v) for v in rows_by_part.values()),
        "extracted": len(extracted_rows),
        "failed": len(failures),
        "parts_tmp_empty": not any(parts_tmp.iterdir()) if parts_tmp.exists() else True,
    }


def dry_run(args: argparse.Namespace) -> None:
    shared = scan_shared_parts(args.shared_dir)
    candidates = manifest_candidates(args.manifest, shared)
    summary = summarize_candidates(candidates, shared)
    report = {
        "created_at": utc_now(),
        "mode": "dry_run",
        "manifest": str(args.manifest),
        "shared_dir": str(args.shared_dir),
        "completed_part_count": len(shared),
        "completed_file_count": sum(len(item["files"]) for item in shared.values()),
        "completed_bytes": sum(item["bytes"] for item in shared.values()),
        "min_freeze_candidates": args.min_freeze_candidates,
        "can_freeze": len(candidates) >= args.min_freeze_candidates,
        **summary,
        "parts": list(shared.values()),
    }
    ensure_dir(args.report_dir)
    write_json(args.report_dir / "shared_only_dry_run.json", report)
    write_part_tsv(args.report_dir / "shared_only_by_part.tsv", summary["by_part"])
    print(json.dumps({k: report[k] for k in ["completed_part_count", "candidate_count", "can_freeze"]}, sort_keys=True))


def make_smoke_manifest(args: argparse.Namespace) -> None:
    shared = scan_shared_parts(args.shared_dir)
    candidates = manifest_candidates(args.manifest, shared)
    rows = select_smoke_rows(candidates, part=args.part, limit=args.limit, require_unique_source=args.unique_source)
    if not rows:
        raise RuntimeError("no smoke rows selected")
    count = write_jsonl(args.output, rows)
    counts = Counter(str(row["part"]) for row in rows)
    report = {
        "created_at": utc_now(),
        "mode": "make_smoke_manifest",
        "manifest": str(args.manifest),
        "output": str(args.output),
        "limit": args.limit,
        "part": args.part,
        "selected": count,
        "by_part": dict(counts),
    }
    write_json(args.output.with_suffix(".summary.json"), report)
    print(json.dumps({"selected": count, "by_part": dict(counts)}, sort_keys=True))


def freeze(args: argparse.Namespace) -> None:
    shared = scan_shared_parts(args.shared_dir)
    candidates = manifest_candidates(args.manifest, shared)
    train, val = freeze_shared_split(
        candidates,
        train_count=args.train_count,
        val_count=args.val_count,
        min_candidates=args.min_freeze_candidates,
    )
    train_count = write_jsonl(args.train_output, train)
    val_count = write_jsonl(args.val_output, val)
    plan = summarize_candidates(train + val, shared)
    write_json(args.plan_output, {"created_at": utc_now(), "train": train_count, "val": val_count, **plan})
    print(json.dumps({"train": train_count, "val": val_count}, sort_keys=True))


def extract(args: argparse.Namespace) -> None:
    shared = scan_shared_parts(args.shared_dir)
    summary = extract_smoke(
        args.manifest,
        shared,
        output_dir=args.output_dir,
        parts_tmp=args.parts_tmp,
        output_manifest=args.output_manifest,
        keep_existing=args.keep_existing,
        state_path=args.state,
    )
    write_json(args.report, {"created_at": utc_now(), **summary})
    print(json.dumps(summary, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, default=root / "data/openvid/meta/openvid_manifest_filtered.jsonl")
    common.add_argument("--shared-dir", type=Path, default=Path("/mnt/beegfs/home/yezy/openvid"))
    common.add_argument("--min-freeze-candidates", type=int, default=105_000)

    p = sub.add_parser("dry-run", parents=[common])
    p.add_argument("--report-dir", type=Path, default=root / "reports/W2.1_shared_only_dry_run")
    p.set_defaults(func=dry_run)

    p = sub.add_parser("make-smoke-manifest", parents=[common])
    p.add_argument("--part", default="OpenVid_part115.zip")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--unique-source", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output", type=Path, default=root / "data/openvid/meta/openvid_smoke_64.jsonl")
    p.set_defaults(func=make_smoke_manifest)

    p = sub.add_parser("freeze", parents=[common])
    p.add_argument("--train-count", type=int, default=98_000)
    p.add_argument("--val-count", type=int, default=2_000)
    p.add_argument("--train-output", type=Path, default=root / "data/openvid/meta/openvid_shared_train_98k.jsonl")
    p.add_argument("--val-output", type=Path, default=root / "data/openvid/meta/openvid_shared_val_2k.jsonl")
    p.add_argument("--plan-output", type=Path, default=root / "data/openvid/meta/shared_part_plan.json")
    p.set_defaults(func=freeze)

    p = sub.add_parser("extract", parents=[common])
    p.add_argument("--output-dir", type=Path, default=root / "data/openvid/videos/smoke")
    p.add_argument("--parts-tmp", type=Path, default=root / "data/openvid/parts_tmp")
    p.add_argument("--output-manifest", type=Path, default=root / "data/openvid/meta/openvid_smoke_extracted.jsonl")
    p.add_argument("--report", type=Path, default=root / "reports/W2.2_smoke_extract/extract_summary.json")
    p.add_argument("--state", type=Path, default=root / "data/openvid/meta/shared_extract_state.json")
    p.add_argument("--keep-existing", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=extract)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
