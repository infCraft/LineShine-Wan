#!/usr/bin/env python3
"""Append-only image index reader and synchronized bucket batch sampler."""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler


SCHEMA_VERSION = 1


def _canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _bucket_dimensions(bucket: str) -> tuple[int, int]:
    try:
        width, height = (int(value) for value in bucket.split("x", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid image bucket {bucket!r}") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image bucket {bucket!r}")
    return width, height


def _stable_seed(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


class _JsonlPartCache:
    def __init__(self, root: Path, max_entries: int):
        if max_entries <= 0:
            raise ValueError("index cache size must be positive")
        self.root = root
        self.max_entries = max_entries
        self._rows: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def get(self, relative_path: str, expected_count: int) -> list[dict[str, Any]]:
        rows = self._rows.pop(relative_path, None)
        if rows is None:
            path = self.root / relative_path
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: row must be a JSON object")
                    rows.append(row)
            if len(rows) != expected_count:
                raise ValueError(
                    f"index part count mismatch for {path}: "
                    f"expected {expected_count}, got {len(rows)}"
                )
        self._rows[relative_path] = rows
        while len(self._rows) > self.max_entries:
            self._rows.popitem(last=False)
        return rows


class ImageBucketView:
    """One homogeneous image shape aggregated across append-only segments."""

    def __init__(
        self,
        root: Path,
        *,
        resolution: int,
        bucket: str,
        parts: list[dict[str, Any]],
        index_cache: _JsonlPartCache,
    ):
        self.root = root
        self.resolution = int(resolution)
        self.bucket = bucket
        self.target_width, self.target_height = _bucket_dimensions(bucket)
        self.parts = parts
        self._index_cache = index_cache
        self.part_starts: list[int] = []
        self._ends: list[int] = []
        total = 0
        for part in self.parts:
            count = int(part["sample_count"])
            if count <= 0:
                raise ValueError(f"non-positive sample count in index part: {part}")
            self.part_starts.append(total)
            total += count
            self._ends.append(total)
        self.sample_count = total

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        part_index = bisect.bisect_right(self._ends, index)
        part_start = 0 if part_index == 0 else self._ends[part_index - 1]
        part = self.parts[part_index]
        rows = self._index_cache.get(str(part["path"]), int(part["sample_count"]))
        row = dict(rows[index - part_start])
        row["source_dataset"] = str(part["dataset_name"])
        row["index_segment"] = str(part["segment_id"])
        row["preprocess_version"] = str(part["preprocess_version"])
        row["resolution"] = self.resolution
        row["bucket"] = self.bucket
        row["target_width"] = self.target_width
        row["target_height"] = self.target_height
        return row


class ImageTrainingDataset(Dataset):
    """Return raw-image descriptors and one deterministically selected caption."""

    def __init__(
        self,
        index_root: Path,
        *,
        resolutions: tuple[int, ...] | None = None,
        datasets: set[str] | None = None,
        caption_seed: int = 1234,
        index_cache_size: int = 32,
        include_all_captions: bool = False,
    ):
        self.root = index_root.resolve()
        self.caption_seed = int(caption_seed)
        self.include_all_captions = bool(include_all_captions)
        catalog_path = self.root / "catalog.json"
        catalog = _read_json(catalog_path)
        if int(catalog.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported image index schema in {catalog_path}")
        if catalog.get("kind") != "image_training_index":
            raise ValueError(f"not an image training index: {catalog_path}")
        self.catalog_fingerprint = str(catalog.get("fingerprint") or "")
        if len(self.catalog_fingerprint) != 64:
            raise ValueError(f"missing catalog fingerprint in {catalog_path}")

        selected_resolutions = None if resolutions is None else {int(value) for value in resolutions}
        selected_datasets = None if datasets is None else {str(value) for value in datasets}
        grouped_parts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for segment in catalog.get("segments", []):
            if segment.get("status") != "ready":
                continue
            dataset_name = str(segment["dataset_name"])
            if selected_datasets is not None and dataset_name not in selected_datasets:
                continue
            segment_id = str(segment["segment_id"])
            manifest = _read_json(self.root / str(segment["manifest"]))
            if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"unsupported image index segment {segment_id}")
            if manifest.get("kind") != "image_training_index_segment":
                raise ValueError(f"invalid image index segment {segment_id}")
            if manifest.get("status") != "ready":
                continue
            stored_fingerprint = str(manifest.get("fingerprint") or "")
            manifest_core = {
                key: value
                for key, value in manifest.items()
                if key not in {"fingerprint", "created_at"}
            }
            if stored_fingerprint != _canonical_fingerprint(manifest_core):
                raise ValueError(f"invalid manifest fingerprint for segment {segment_id}")
            if stored_fingerprint != str(segment.get("fingerprint") or ""):
                raise ValueError(f"catalog fingerprint mismatch for segment {segment_id}")
            preprocess_version = str(manifest.get("preprocess_version") or "unknown")
            for resolution_value, buckets in manifest.get("buckets", {}).items():
                resolution = int(resolution_value)
                if selected_resolutions is not None and resolution not in selected_resolutions:
                    continue
                for bucket, record in buckets.items():
                    key = f"{resolution}/{bucket}"
                    part_count = sum(
                        int(raw_part["sample_count"])
                        for raw_part in record.get("parts", [])
                    )
                    if part_count != int(record.get("sample_count", -1)):
                        raise ValueError(
                            f"bucket count mismatch in segment {segment_id}: {key}"
                        )
                    for raw_part in record.get("parts", []):
                        part = dict(raw_part)
                        part["dataset_name"] = dataset_name
                        part["segment_id"] = segment_id
                        part["preprocess_version"] = preprocess_version
                        grouped_parts[key].append(part)

        if not grouped_parts:
            raise FileNotFoundError(f"no matching buckets in {catalog_path}")
        cache = _JsonlPartCache(self.root, index_cache_size)
        self.buckets: dict[str, ImageBucketView] = {}
        for key, parts in grouped_parts.items():
            resolution_value, bucket = key.split("/", 1)
            self.buckets[key] = ImageBucketView(
                self.root,
                resolution=int(resolution_value),
                bucket=bucket,
                parts=parts,
                index_cache=cache,
            )
        self.bucket_keys = sorted(
            self.buckets,
            key=lambda key: (int(key.split("/", 1)[0]), key.split("/", 1)[1]),
        )

    def __len__(self) -> int:
        return sum(len(view) for view in self.buckets.values())

    @property
    def bucket_sizes(self) -> dict[str, int]:
        return {key: len(self.buckets[key]) for key in self.bucket_keys}

    def _caption_choice(self, sample_id: str) -> int:
        return _stable_seed(self.caption_seed, sample_id) & 1

    def __getitem__(self, request: tuple[str, int] | tuple[str, int, int] | tuple[str, int, int, int]) -> dict[str, Any]:
        if not isinstance(request, tuple) or len(request) not in {2, 3, 4}:
            raise TypeError("image training samples require (bucket, index[, caption_choice[, draw_id]])")
        bucket_key = str(request[0])
        sample_index = int(request[1])
        row = self.buckets[bucket_key][sample_index]
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"sample in {bucket_key} has no sample_id")
        caption_vlm = str(row.get("caption_vlm") or "").strip()
        caption_original = str(row.get("caption_original") or "").strip()
        if not caption_vlm and not caption_original:
            raise ValueError(f"sample {sample_id} has no non-empty caption")
        choice = int(request[2]) & 1 if len(request) >= 3 else self._caption_choice(sample_id)
        if choice == 1 and caption_original:
            caption = caption_original
            caption_source = "original"
        elif caption_vlm:
            caption = caption_vlm
            caption_source = "vlm"
        else:
            caption = caption_original
            caption_source = "original"
        if not self.include_all_captions:
            row.pop("caption_vlm", None)
            row.pop("caption_original", None)
        row["key"] = sample_id
        row["caption"] = caption
        row["caption_source"] = caption_source
        row["caption_choice"] = choice
        row["logical_index"] = sample_index
        row["draw_id"] = int(request[3]) if len(request) == 4 else -1
        return row


class SynchronizedBucketBatchSampler(Sampler[list[tuple[str, int, int, int]]]):
    """Yield shape-homogeneous, disjoint global batches on every DDP rank.

    Every rank builds the same bucket schedule and sample permutations. A global
    batch is then sliced into contiguous per-rank batches, so ranks agree on the
    bucket while never receiving the same logical sample in an epoch.
    """

    def __init__(
        self,
        dataset: ImageTrainingDataset,
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 1234,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is outside num_replicas {num_replicas}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.batch_cursor = 0
        self.global_batch_size = self.batch_size * self.num_replicas
        self._bucket_batches = {
            key: size // self.global_batch_size
            for key, size in self.dataset.bucket_sizes.items()
            if size >= self.global_batch_size
        }
        if not self._bucket_batches:
            raise ValueError(
                "no bucket contains one complete global batch; reduce world size or batch size"
            )

    def _schedule(self) -> list[str]:
        schedule = [
            key
            for key in sorted(self._bucket_batches)
            for _ in range(self._bucket_batches[key])
        ]
        if self.shuffle and len(schedule) > 1:
            generator = torch.Generator().manual_seed(
                _stable_seed(self.seed, self.epoch, "bucket-schedule")
            )
            order = torch.randperm(len(schedule), generator=generator).tolist()
            schedule = [schedule[index] for index in order]
        return schedule

    def _permutation(self, bucket_key: str) -> torch.Tensor:
        view = self.dataset.buckets[bucket_key]
        size = len(view)
        if not self.shuffle:
            return torch.arange(size, dtype=torch.int64)
        part_generator = torch.Generator().manual_seed(
            _stable_seed(self.seed, self.epoch, bucket_key, "parts")
        )
        part_order = torch.randperm(len(view.parts), generator=part_generator).tolist()
        chunks = []
        for part_index in part_order:
            part = view.parts[part_index]
            part_start = view.part_starts[part_index]
            part_count = int(part["sample_count"])
            block_size = int(part.get("shuffle_block_size", 256))
            if block_size <= 0:
                raise ValueError(f"invalid shuffle block size in index part: {part}")
            blocks = [
                (start, min(start + block_size, part_count))
                for start in range(0, part_count, block_size)
            ]
            block_generator = torch.Generator().manual_seed(
                _stable_seed(self.seed, self.epoch, bucket_key, part_index, "blocks")
            )
            block_order = torch.randperm(len(blocks), generator=block_generator).tolist()
            for block_index in block_order:
                start, end = blocks[block_index]
                values = torch.arange(
                    part_start + start,
                    part_start + end,
                    dtype=torch.int64,
                )
                if len(values) > 1:
                    rotation = _stable_seed(
                        self.seed,
                        self.epoch,
                        bucket_key,
                        part_index,
                        block_index,
                        "rotation",
                    ) % len(values)
                    if rotation:
                        values = torch.roll(values, shifts=-rotation)
                chunks.append(values)
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int64)

    def __iter__(self) -> Iterator[list[tuple[str, int, int, int]]]:
        schedule = self._schedule()
        permutations = {key: self._permutation(key) for key in self._bucket_batches}
        bucket_cursors: dict[str, int] = defaultdict(int)
        rank_offset = self.rank * self.batch_size
        for batch_position, bucket_key in enumerate(schedule):
            bucket_batch = bucket_cursors[bucket_key]
            bucket_cursors[bucket_key] += 1
            if batch_position < self.batch_cursor:
                continue
            start = bucket_batch * self.global_batch_size + rank_offset
            indices = permutations[bucket_key][start : start + self.batch_size].tolist()
            draw_base = batch_position * self.global_batch_size + rank_offset
            batch = []
            for offset, sample_index in enumerate(indices):
                draw_id = draw_base + offset
                caption_choice = _stable_seed(
                    self.seed,
                    self.epoch,
                    bucket_key,
                    sample_index,
                    draw_id,
                    "caption",
                ) & 1
                batch.append((bucket_key, int(sample_index), caption_choice, draw_id))
            yield batch

    def __len__(self) -> int:
        return max(0, sum(self._bucket_batches.values()) - self.batch_cursor)

    @property
    def epoch_batches(self) -> int:
        return sum(self._bucket_batches.values())

    def advance(self, batches: int = 1) -> None:
        batches = int(batches)
        if batches < 0:
            raise ValueError("batches must be non-negative")
        self.batch_cursor = min(self.epoch_batches, self.batch_cursor + batches)

    def next_epoch(self) -> None:
        self.epoch += 1
        self.batch_cursor = 0

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch != self.epoch:
            self.epoch = epoch
            self.batch_cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "catalog_fingerprint": self.dataset.catalog_fingerprint,
            "bucket_sizes": self.dataset.bucket_sizes,
            "bucket_batches": dict(self._bucket_batches),
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "epoch": self.epoch,
            "batch_cursor": self.batch_cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "catalog_fingerprint": self.dataset.catalog_fingerprint,
            "bucket_sizes": self.dataset.bucket_sizes,
            "bucket_batches": self._bucket_batches,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise RuntimeError(f"bucket sampler checkpoint has mismatched {name}")
        self.epoch = int(state["epoch"])
        self.batch_cursor = int(state["batch_cursor"])
        if not 0 <= self.batch_cursor <= self.epoch_batches:
            raise RuntimeError("bucket sampler checkpoint cursor is out of range")

    def summary(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "remaining_batches": len(self),
            "epoch_batches": self.epoch_batches,
            "global_batch_size": self.global_batch_size,
            "shuffle_mode": "part_block" if self.shuffle else "sequential",
            "bucket_batches": dict(self._bucket_batches),
            "dropped_samples": {
                key: self.dataset.bucket_sizes[key]
                - self._bucket_batches.get(key, 0) * self.global_batch_size
                for key in self.dataset.bucket_keys
            },
        }
