"""Shared helpers for LineShine scripts."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_ROOT = Path(os.environ.get("LINESHINE_ROOT", "/mnt/beegfs/home/huang_z/lineshine"))
DEFAULT_CODE_ROOT = Path(os.environ.get("LINESHINE_CODE_ROOT", DEFAULT_ROOT / "code"))
WAN_ROOT = DEFAULT_CODE_ROOT / "third_party" / "Wan2.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        f.write("\n")


def sample_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id") or row.get("video")
    if not value:
        raise KeyError("manifest row has neither sample_id nor video")
    return str(value)


def stable_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out = list(rows)
    rng.shuffle(out)
    return out
