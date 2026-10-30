import torch

from src.rl import config as C
from src.rl.train_ppo_parallel import (
    ATTENTION_HEADS,
    RECURRENT_ARCHS,
    ProjectedResidualSelfAttention,
    build_net,
    warmstart_net,
)


def test_projected_attention_shapes_and_probability_rows():
    layer = ProjectedResidualSelfAttention(
        input_dim=33, attention_dim=128, num_heads=4, n_nodes=10)
    encoded, weights = layer(torch.randn(2, 10, 33), return_weights=True)
    assert encoded.shape == (2, 10, 128)
    assert weights.shape == (2, 4, 10, 10)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones(2, 4, 10), atol=1e-6)


def test_attention_spatial_only_interface_and_warmstart():
    net = build_net("attention", 128)
    warmstart_net(net, "attention", log_std_init=-2.5)
    assert "attention" not in RECURRENT_ARCHS

    state = torch.randn(3, C.STATE_DIM)
    mean, std, value = net(state)

    assert mean.shape == (3, C.ACTION_DIM)
    assert std.shape == mean.shape
    assert value.shape == (3,)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.isfinite(std).all()
    assert torch.isfinite(value).all()
    logp, entropy, evaluated_value = net.evaluate(
        state, torch.zeros_like(mean))
    loss = -(logp.mean() + 0.01 * entropy.mean()) + evaluated_value.square().mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in net.parameters())


def test_attention_gru_recurrent_interface_and_warmstart():
    net = build_net("attention_gru", 128)
    warmstart_net(net, "attention_gru", log_std_init=-2.5)
    assert "attention_gru" in RECURRENT_ARCHS

    batch, steps = 2, 4
    obs = torch.randn(batch, steps, C.STATE_DIM)
    previous_action = torch.zeros(batch, steps, C.ACTION_DIM)
    mean, std, value, hidden = net(obs, previous_action)

    assert mean.shape == (batch, steps, C.ACTION_DIM)
    assert std.shape == mean.shape
    assert value.shape == (batch, steps)
    assert hidden.shape == (1, batch * C.N_AGENTS, 128)
    assert torch.equal(mean, torch.zeros_like(mean))
    assert torch.isfinite(std).all()
    assert torch.isfinite(value).all()
