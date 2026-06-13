"""Dataset readers for cached WebDataset shards."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from torch.utils.data import Dataset


def iter_tar_samples(path: Path):
    grouped: dict[str, dict[str, str]] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name)
            grouped.setdefault(name.stem, {})[name.suffix.lstrip(".")] = member.name
    for key, parts in grouped.items():
        yield key, parts


class CacheDataset(Dataset):
    def __init__(self, shards: list[Path]):
        self.samples: list[tuple[Path, str, dict[str, str]]] = []
        for shard in shards:
            for key, parts in iter_tar_samples(shard):
                if "safetensors" in parts and "json" in parts:
                    self.samples.append((shard, key, parts))
        if not self.samples:
            raise FileNotFoundError("no cache samples found")

    @classmethod
    def from_dir(cls, cache_dir: Path, pattern: str = "*.tar") -> "CacheDataset":
        return cls(sorted(cache_dir.glob(pattern)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard, key, parts = self.samples[idx]
        with tarfile.open(shard, "r") as tar:
            tensor_file = tar.extractfile(parts["safetensors"])
            meta_file = tar.extractfile(parts["json"])
            if tensor_file is None or meta_file is None:
                raise KeyError(f"missing sample parts for {key} in {shard}")
            tensors = load_safetensors(tensor_file.read())
            meta = json.loads(meta_file.read().decode("utf-8"))
        return {
            "key": key,
            "shard": str(shard),
            "latent": tensors["latent"],
            "context": tensors["context"],
            "text_len": int(tensors["text_len"].item()),
            "meta": meta,
        }


class SyntheticDataset(Dataset):
    def __init__(self, *, count: int, latent_shape: tuple[int, int, int, int], text_len: int, text_dim: int, seed: int = 123):
        self.count = count
        self.latent_shape = latent_shape
        self.text_len = text_len
        self.text_dim = text_dim
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> dict[str, Any]:
        generator = torch.Generator().manual_seed(self.seed + idx)
        return {
            "key": f"synthetic_{idx:06d}",
            "latent": torch.randn(self.latent_shape, dtype=torch.bfloat16, generator=generator),
            "context": torch.randn((self.text_len, self.text_dim), dtype=torch.bfloat16, generator=generator),
            "text_len": self.text_len,
            "meta": {"sample_id": f"synthetic_{idx:06d}"},
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "keys": [s["key"] for s in samples],
        "latents": torch.stack([s["latent"] for s in samples], dim=0),
        "contexts": [s["context"] for s in samples],
        "text_lens": torch.tensor([s["text_len"] for s in samples], dtype=torch.long),
        "meta": [s["meta"] for s in samples],
    }
