import torch

from src.train.flow import add_flow_noise, apply_cfg_dropout, sample_logit_normal_sigmas
from src.train.model_adapter import latent_seq_len


def test_latent_seq_len_for_wan_49f_256():
    assert latent_seq_len((2, 16, 13, 32, 32), (1, 2, 2)) == 3328


def test_flow_noise_shapes_and_recovery():
    generator = torch.Generator().manual_seed(5)
    clean = torch.randn((3, 16, 13, 32, 32), dtype=torch.float32, generator=generator)
    sigmas = sample_logit_normal_sigmas(3, device=torch.device("cpu"), generator=generator)
    noisy, target, timesteps = add_flow_noise(clean, sigmas=sigmas, generator=generator)
    recovered = noisy - sigmas.reshape(3, 1, 1, 1, 1) * target

    assert noisy.shape == clean.shape
    assert target.shape == clean.shape
    assert timesteps.shape == (3,)
    torch.testing.assert_close(recovered, clean, rtol=1e-5, atol=1e-5)


def test_cfg_dropout_replaces_contexts():
    contexts = [torch.ones((2, 4)), torch.ones((3, 4)) * 2]
    empty = torch.zeros((1, 4))
    out = apply_cfg_dropout(contexts, empty, dropout_prob=1.0)

    assert len(out) == 2
    assert out[0].shape == empty.shape
    assert out[1].shape == empty.shape
    assert torch.count_nonzero(out[0]) == 0

