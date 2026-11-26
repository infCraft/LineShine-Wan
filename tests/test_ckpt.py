from types import SimpleNamespace

import pytest
import torch

from src.train import ckpt


def test_save_checkpoint_skips_existing_step_and_keeps_latest(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(run_dir=tmp_path, note="unit")
    original_save = ckpt.torch.save
    save_calls = []

    def counted_save(obj, path):
        save_calls.append(path)
        return original_save(obj, path)

    monkeypatch.setattr(ckpt.torch, "save", counted_save)

    first = ckpt.save_checkpoint(tmp_path, step=1, model=model, optimizer=optimizer, scaler=None, args=args)
    second = ckpt.save_checkpoint(tmp_path, step=1, model=model, optimizer=optimizer, scaler=None, args=args)

    assert first == second
    assert len(save_calls) == 1
    assert ckpt.latest_checkpoint(tmp_path) == first
    assert (tmp_path / "checkpoints/latest").resolve() == first.resolve()
    assert (first / "model.safetensors").is_file()
    assert (first / "trainer.pt").is_file()
    assert (first / "manifest.json").is_file()
    assert (first / "COMPLETED").is_file()
    assert not list((tmp_path / "checkpoints").glob(".*.tmp.*"))
    assert ckpt.load_checkpoint(first, model=model, optimizer=optimizer) == 1


def test_load_checkpoint_can_skip_rng_restore(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(run_dir=tmp_path, note="unit")
    path = ckpt.save_checkpoint(tmp_path, step=3, model=model, optimizer=optimizer, scaler=None, args=args)

    calls = []
    monkeypatch.setattr(ckpt.torch, "set_rng_state", lambda state: calls.append(state))

    assert ckpt.load_checkpoint(path, model=model, optimizer=optimizer, restore_rng=False) == 3
    assert calls == []


def test_save_checkpoint_refuses_incomplete_existing_step(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = ckpt.checkpoint_path(tmp_path, 2)
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="could not reuse checkpoint"):
        ckpt.save_checkpoint(
            tmp_path,
            step=2,
            model=model,
            optimizer=optimizer,
            scaler=None,
            args=SimpleNamespace(run_dir=tmp_path),
        )

    assert ckpt.latest_checkpoint(tmp_path) is None


def test_legacy_checkpoint_still_loads(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "step_00000007.pt"
    torch.save(
        {
            "step": 7,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
        },
        path,
    )

    assert ckpt.load_checkpoint(path, model=model, optimizer=optimizer) == 7


def test_atomic_torch_save_does_not_leave_target_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "state.pt"

    def failing_save(obj, path):
        path.write_bytes(b"partial")
        raise RuntimeError("boom")

    monkeypatch.setattr(ckpt.torch, "save", failing_save)

    with pytest.raises(RuntimeError, match="boom"):
        ckpt.atomic_torch_save({"x": torch.tensor([1])}, target)

    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp.*"))
