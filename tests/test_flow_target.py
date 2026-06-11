import importlib.util
from pathlib import Path

import torch


def _load_unipc_scheduler():
    scheduler_path = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "Wan2.1"
        / "wan"
        / "utils"
        / "fm_solvers_unipc.py"
    )
    spec = importlib.util.spec_from_file_location("wan_flow_unipc", scheduler_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


def _sample_logit_normal_sigmas(shape, generator, mean=0.0, std=1.0):
    logits = torch.randn(shape, dtype=torch.float64, generator=generator) * std + mean
    return torch.sigmoid(logits)


def test_flow_target_algebra_identity_fp64():
    generator = torch.Generator().manual_seed(11)
    z = torch.randn(4, 16, 13, 32, 32, dtype=torch.float64, generator=generator)
    eps = torch.randn(z.shape, dtype=z.dtype, generator=generator)
    sigma = torch.rand(4, 1, 1, 1, 1, dtype=torch.float64, generator=generator)

    x = (1.0 - sigma) * z + sigma * eps
    target = eps - z
    recovered = x - sigma * target

    torch.testing.assert_close(recovered, z, rtol=0.0, atol=1e-12)


def test_oracle_velocity_converges_with_wan_unipc_scheduler():
    Scheduler = _load_unipc_scheduler()
    scheduler = Scheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
        solver_order=2,
    )
    scheduler.set_timesteps(50, device="cpu", shift=1)

    generator = torch.Generator().manual_seed(17)
    z_star = torch.randn(1, 2, 3, 4, dtype=torch.float32, generator=generator)
    sample = torch.randn(z_star.shape, dtype=z_star.dtype, generator=generator)

    for timestep in scheduler.timesteps:
        sigma = (timestep.to(dtype=sample.dtype) / scheduler.config.num_train_timesteps).clamp_min(
            torch.finfo(sample.dtype).eps
        )
        oracle_velocity = (sample - z_star) / sigma
        sample = scheduler.step(
            oracle_velocity,
            timestep,
            sample,
            return_dict=False,
        )[0]

    rel_err = (sample - z_star).norm() / z_star.norm()
    assert rel_err.item() < 0.01


def test_logit_normal_sigma_statistics_and_timestep_mapping():
    generator = torch.Generator().manual_seed(23)
    sigmas = _sample_logit_normal_sigmas((200_000,), generator=generator)

    mean = sigmas.mean().item()
    q10, q50, q90 = torch.quantile(
        sigmas,
        torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64),
    ).tolist()

    assert 0.495 < mean < 0.505
    assert 0.215 < q10 < 0.225
    assert 0.495 < q50 < 0.505
    assert 0.775 < q90 < 0.785

    timesteps = sigmas * 1000.0
    torch.testing.assert_close(timesteps / 1000.0, sigmas, rtol=0.0, atol=1e-15)
    assert timesteps.min().item() > 0.0
    assert timesteps.max().item() < 1000.0
