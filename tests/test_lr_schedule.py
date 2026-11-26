from argparse import Namespace

import pytest

from src.train.train import lr_lambda


def _args(*, max_steps: int, lr_total_steps: int | None):
    return Namespace(
        max_steps=max_steps,
        lr_total_steps=lr_total_steps,
        warmup_steps=1000,
        lr=1e-4,
        min_lr=1e-5,
    )


def test_lr_total_steps_preserves_full_schedule_for_segments():
    full_schedule = lr_lambda(_args(max_steps=30000, lr_total_steps=None))
    segmented_schedule = lr_lambda(_args(max_steps=5000, lr_total_steps=30000))

    for step in (1000, 5000, 10000, 30000):
        assert segmented_schedule(step) == pytest.approx(full_schedule(step))


def test_lr_total_steps_defaults_to_max_steps():
    default_schedule = lr_lambda(_args(max_steps=5000, lr_total_steps=None))
    explicit_schedule = lr_lambda(_args(max_steps=5000, lr_total_steps=5000))

    for step in (1000, 2500, 5000):
        assert default_schedule(step) == pytest.approx(explicit_schedule(step))
