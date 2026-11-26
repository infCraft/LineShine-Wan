"""WanModel construction shared by training, sampling, and evaluation."""

from __future__ import annotations

import sys

from src.common import WAN_ROOT


def add_wan_to_path() -> None:
    wan = str(WAN_ROOT)
    if wan not in sys.path:
        sys.path.insert(0, wan)


def create_model(args):
    add_wan_to_path()
    from wan.configs.wan_t2v_1_3B import t2v_1_3B
    from wan.modules.model import WanModel

    if args.tiny_model:
        return WanModel(
            model_type="t2v",
            patch_size=(1, 2, 2),
            text_len=args.text_len,
            in_dim=args.latent_channels,
            dim=args.tiny_dim,
            ffn_dim=args.tiny_ffn_dim,
            freq_dim=64,
            text_dim=args.text_dim,
            out_dim=args.latent_channels,
            num_heads=args.tiny_heads,
            num_layers=args.tiny_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
        )
    return WanModel(
        model_type="t2v",
        patch_size=t2v_1_3B.patch_size,
        text_len=t2v_1_3B.text_len,
        in_dim=16,
        dim=t2v_1_3B.dim,
        ffn_dim=t2v_1_3B.ffn_dim,
        freq_dim=t2v_1_3B.freq_dim,
        text_dim=4096,
        out_dim=16,
        num_heads=t2v_1_3B.num_heads,
        num_layers=t2v_1_3B.num_layers,
        window_size=t2v_1_3B.window_size,
        qk_norm=t2v_1_3B.qk_norm,
        cross_attn_norm=t2v_1_3B.cross_attn_norm,
        eps=t2v_1_3B.eps,
    )
