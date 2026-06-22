#!/usr/bin/env python3
"""Build extra Stage-1 OpenVid manifests from currently completed shared parts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, DEFAULT_SHARED_OPENVID_DIR, read_jsonl, sample_id, utc_now, write_json, write_jsonl
from src.data.shared_openvid import manifest_candidates, row_source_id, scan_shared_parts, summarize_candidates


def load_sample_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            ids.add(sample_id(row))
    return ids


def load_source_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            ids.add(row_source_id(row))
    return ids


def select_extra_rows(
    candidates: list[dict[str, Any]],
    *,
    exclude_sample_ids: set[str],
    exclude_source_ids: set[str],
    split: str,
    max_count: int | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for row in candidates:
        sid = sample_id(row)
        if sid in seen:
            skipped["duplicate_candidate_sample_id"] += 1
            continue
        seen.add(sid)
        if sid in exclude_sample_ids:
            skipped["excluded_sample_id"] += 1
            continue
        if row_source_id(row) in exclude_source_ids:
            skipped["excluded_source_id"] += 1
            continue
        out = dict(row)
        out["split"] = split
        rows.append(out)
        if max_count is not None and len(rows) >= max_count:
            break
    return rows, skipped


def build_extra_manifest(args: argparse.Namespace) -> None:
    shared = scan_shared_parts(args.shared_dir)
    candidates = manifest_candidates(args.manifest, shared)
    exclude_sample_ids = load_sample_ids(args.exclude_manifest)
    exclude_source_ids = load_source_ids(args.exclude_source_manifest)
    rows, skipped = select_extra_rows(
        candidates,
        exclude_sample_ids=exclude_sample_ids,
        exclude_source_ids=exclude_source_ids,
        split=args.split,
        max_count=args.max_count,
    )
    count = write_jsonl(args.output, rows)
    by_part = Counter(str(row["part"]) for row in rows)
    candidate_summary = summarize_candidates(candidates, shared)
    selected_summary = summarize_candidates(rows, shared) if rows else {
        "candidate_count": 0,
        "unique_source_count": 0,
        "unique_sample_count": 0,
        "by_part": [],
    }
    report = {
        "created_at": utc_now(),
        "manifest": str(args.manifest),
        "shared_dir": str(args.shared_dir),
        "completed_part_count": len(shared),
        "completed_file_count": sum(len(item["files"]) for item in shared.values()),
        "completed_bytes": sum(item["bytes"] for item in shared.values()),
        "available_candidate_count": len(candidates),
        "available_unique_source_count": candidate_summary["unique_source_count"],
        "available_unique_sample_count": candidate_summary["unique_sample_count"],
        "excluded_sample_count": len(exclude_sample_ids),
        "excluded_source_count": len(exclude_source_ids),
        "selected_count": count,
        "selected_unique_source_count": selected_summary["unique_source_count"],
        "selected_unique_sample_count": selected_summary["unique_sample_count"],
        "selected_by_part": selected_summary["by_part"],
        "selected_by_part_count": dict(sorted(by_part.items())),
        "skipped": dict(skipped),
        "output": str(args.output),
    }
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "available_candidate_count": len(candidates),
                "excluded_sample_count": len(exclude_sample_ids),
                "excluded_source_count": len(exclude_source_ids),
                "selected_count": count,
            },
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=root / "data/openvid/meta/openvid_manifest_filtered.jsonl")
    parser.add_argument("--shared-dir", type=Path, default=DEFAULT_SHARED_OPENVID_DIR)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-source-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=root / "data/openvid/meta/stage1_extra_current_downloads.jsonl")
    parser.add_argument("--report", type=Path, default=root / "reports/stage1_expand/stage1_extra_manifest.json")
    parser.add_argument("--split", default="train_stage1_extra")
    parser.add_argument("--max-count", type=int)
    parser.set_defaults(func=build_extra_manifest)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
