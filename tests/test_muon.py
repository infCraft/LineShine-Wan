from __future__ import annotations

from dataclasses import dataclass

import torch

from src.train.muon import MuonWithAuxAdam, zeropower_via_newtonschulz5


@dataclass
class _NamedParam:
    name: str
    ndim: int
    requires_grad: bool = True


def _partition(named_params: list[_NamedParam]) -> tuple[set[str], set[str], set[str]]:
    EXCLUDE = ("embedding", "head", "modulation", "projection")
    muon_params = set()
    adam_decay = set()
    adam_no_decay = set()
    for param in named_params:
        if not param.requires_grad:
            continue
        name_lower = param.name.lower()
        is_aux_1d = param.ndim < 2 or param.name.endswith(".bias") or "norm" in name_lower or "modulation" in param.name
        if param.ndim == 2 and not is_aux_1d and not any(key in name_lower for key in EXCLUDE):
            muon_params.add(param.name)
        elif is_aux_1d:
            adam_no_decay.add(param.name)
        else:
            adam_decay.add(param.name)
    return muon_params, adam_decay, adam_no_decay


def test_partition_is_complete_and_disjoint():
    named_params = [
        _NamedParam("blocks.0.attn.q.weight", 2),
        _NamedParam("blocks.0.ffn.0.weight", 2),
        _NamedParam("blocks.0.attn.q.bias", 1),
        _NamedParam("blocks.0.norm1.weight", 1),
        _NamedParam("embedding.weight", 2),
        _NamedParam("head.weight", 2),
        _NamedParam("blocks.0.modulation", 2),
        _NamedParam("text_projection.weight", 2),
        _NamedParam("patch_embedding.proj.weight", 5),
        _NamedParam("blocks.0.conv.weight", 5),
        _NamedParam("blocks.0.linear.weight", 2, requires_grad=False),
    ]
    muon_params, adam_decay, adam_no_decay = _partition(named_params)
    all_trainable = {param.name for param in named_params if param.requires_grad}

    assert muon_params | adam_decay | adam_no_decay == all_trainable
    assert not (muon_params & adam_decay)
    assert not (muon_params & adam_no_decay)
    assert not (adam_decay & adam_no_decay)
    assert muon_params == {"blocks.0.attn.q.weight", "blocks.0.ffn.0.weight"}
    for name in muon_params:
        param = next(param for param in named_params if param.name == name)
        name_lower = name.lower()
        assert param.ndim == 2
        assert not name.endswith(".bias")
        assert "norm" not in name_lower
        assert not any(key in name_lower for key in ("embedding", "head", "modulation", "projection"))


def _assert_newtonschulz_is_bounded(G: torch.Tensor) -> None:
    X = zeropower_via_newtonschulz5(G, 5).float()
    singular_values = torch.linalg.svdvals(X)
    assert torch.all(torch.isfinite(singular_values))
    assert torch.all(singular_values > 0.5)
    assert torch.all(singular_values < 1.5)
    gram = X @ X.mT if X.size(0) <= X.size(1) else X.mT @ X
    eye = torch.eye(min(X.shape), dtype=gram.dtype, device=gram.device)
    assert torch.linalg.matrix_norm(gram - eye, ord=2) < 1.3


def test_newtonschulz_orthogonal():
    gen = torch.Generator().manual_seed(1234)
    _assert_newtonschulz_is_bounded(torch.randn(16, 32, generator=gen))
    _assert_newtonschulz_is_bounded(torch.randn(32, 16, generator=gen))


def test_step_runs():
    muon_param = torch.nn.Parameter(torch.randn(4, 8))
    adam_param = torch.nn.Parameter(torch.randn(8))
    optimizer = MuonWithAuxAdam(
        [
            dict(params=[muon_param], use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.0),
            dict(params=[adam_param], use_muon=False, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01),
        ]
    )
    muon_start = muon_param.detach().clone()
    adam_start = adam_param.detach().clone()

    for _ in range(3):
        optimizer.zero_grad()
        loss = (muon_param.square().mean() + adam_param.square().mean())
        loss.backward()
        optimizer.step()

    assert not torch.equal(muon_param, muon_start)
    assert not torch.equal(adam_param, adam_start)
    assert torch.isfinite(muon_param).all()
    assert torch.isfinite(adam_param).all()


def test_muon_weight_decay_shrinks():
    param_no_decay = torch.nn.Parameter(torch.ones(8, 8))
    param_decay = torch.nn.Parameter(torch.ones(8, 8))
    optimizer_no_decay = MuonWithAuxAdam(
        [dict(params=[param_no_decay], use_muon=True, lr=0.1, momentum=0.95, weight_decay=0.0)]
    )
    optimizer_decay = MuonWithAuxAdam(
        [dict(params=[param_decay], use_muon=True, lr=0.1, momentum=0.95, weight_decay=0.5)]
    )
    start = param_no_decay.detach().clone()

    param_no_decay.grad = torch.zeros_like(param_no_decay)
    param_decay.grad = torch.zeros_like(param_decay)
    optimizer_no_decay.step()
    optimizer_decay.step()

    assert torch.equal(param_no_decay, start)
    assert torch.linalg.vector_norm(param_decay) < torch.linalg.vector_norm(param_no_decay)


def test_muon_requires_2d():
    param = torch.nn.Parameter(torch.randn(8))
    try:
        MuonWithAuxAdam([dict(params=[param], use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.0)])
    except ValueError:
        return
    raise AssertionError("MuonWithAuxAdam accepted a 1D Muon parameter")


if __name__ == "__main__":
    test_partition_is_complete_and_disjoint()
    test_newtonschulz_orthogonal()
    test_step_runs()
    test_muon_weight_decay_shrinks()
    test_muon_requires_2d()
    print("OK")
