#!/usr/bin/env python3
"""Cache fixed validation prompt embeddings and the empty prompt."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import DEFAULT_ROOT, WAN_ROOT, ensure_dir, utc_now, write_json
from src.data.preprocess_cache import PROMPTS, add_wan_to_path


def slug(text: str, idx: int) -> str:
    if text == "":
        return "empty"
    value = re.sub(r"[^A-Za-z0-9]+", "_", text.lower()).strip("_")[:48]
    return f"{idx:02d}_{value or 'prompt'}"


@torch.no_grad()
def cache(args: argparse.Namespace) -> None:
    add_wan_to_path()
    from wan.configs.wan_t2v_1_3B import t2v_1_3B
    from wan.modules.t5 import T5EncoderModel

    ensure_dir(args.output_dir)
    device = torch.device(args.device)
    text_encoder = T5EncoderModel(
        text_len=t2v_1_3B.text_len,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=str(args.weights_dir / t2v_1_3B.t5_checkpoint),
        tokenizer_path=str(args.weights_dir / t2v_1_3B.t5_tokenizer),
        shard_fn=None,
    )
    prompts = list(PROMPTS)
    if args.include_empty:
        prompts.append("")
    index = []
    for idx, prompt in enumerate(prompts):
        context = text_encoder([prompt], device)[0].to(torch.bfloat16).cpu()
        name = slug(prompt, idx)
        path = args.output_dir / f"{name}.safetensors"
        save_file({"context": context, "text_len": torch.tensor([context.shape[0]], dtype=torch.int32)}, path)
        index.append({"name": name, "prompt": prompt, "path": str(path), "text_len": int(context.shape[0])})
    write_json(args.output_dir / "prompt_index.json", {"created_at": utc_now(), "prompts": index})
    print(json.dumps({"prompt_count": len(index), "output_dir": str(args.output_dir)}, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    root = DEFAULT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=root / "weights/wan2.1_t2v_1.3b")
    parser.add_argument("--output-dir", type=Path, default=root / "cache/prompts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-empty", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(func=cache)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

