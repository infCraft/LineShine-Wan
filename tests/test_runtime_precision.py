import pytest

torch = pytest.importorskip('torch')

from src.runtime import autocast_context, resolve_dtype


def test_cpu_auto_stays_fp32_and_bf16_is_opt_in():
    device = torch.device('cpu')
    assert resolve_dtype('auto', device) == torch.float32
    with pytest.raises(RuntimeError, match='CPU BF16 is disabled'):
        resolve_dtype('bf16', device)
    assert resolve_dtype('bf16', device, allow_cpu_bf16=True) == torch.bfloat16


def test_cpu_autocast_keeps_parameters_and_optimizer_state_fp32():
    model = torch.nn.Linear(8, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 8)
    with autocast_context(torch.device('cpu'), torch.bfloat16):
        loss = model(inputs).float().square().mean()
    loss.backward()
    optimizer.step()
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.grad.dtype == torch.float32 for parameter in model.parameters())
    assert all(state['exp_avg'].dtype == torch.float32 for state in optimizer.state.values())
    assert all(state['exp_avg_sq'].dtype == torch.float32 for state in optimizer.state.values())