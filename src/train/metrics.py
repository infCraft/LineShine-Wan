"""Small JSONL metric writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common import ensure_dir, utc_now


class JsonlMetrics:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)

    def write(self, row: dict[str, Any]) -> None:
        out = {"time": utc_now(), **row}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, sort_keys=True) + "\n")


class TbMetrics:
    def __init__(self, log_dir: Path, enabled: bool = True):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(str(log_dir))
            except Exception:
                self.writer = None

    def scalar(self, name: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(name, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
