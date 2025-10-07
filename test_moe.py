import sys
import torch
import torch.nn as nn
import pytest
import importlib.util
from unittest.mock import MagicMock

sys.path.insert(0, '/home/jovyan/vasiliev/notebooks/Show-o')
sys.path.insert(0, '/home/jovyan/vasiliev/notebooks/Show-o/models/moe_gates')

class MockPhiConfig:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

class MockPhiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, x):
        return self.linear(x)

from gshard_gate import GShardGate
from naive_gate import NaiveGate
from noisy_gate import NoisyGate

sys.modules['models.phi'] = MagicMock()
sys.modules['models.phi'].PhiConfig = MockPhiConfig
sys.modules['models.phi'].PhiModel = MockPhiModel
sys.modules['models.moe_gates'] = MagicMock()
sys.modules['models.moe_gates'].GShardGate = GShardGate
spec = importlib.util.spec_from_file_location("models.moe", "models/moe.py")
moe_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_module)
MoE = moe_module.MoE


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(42)


def test_moe_initialization():
    NUM_EXPERTS = 4
    HIDDEN_SIZE = 64
    TOP_K = 2
    
    moe = MoE(
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        top_k=TOP_K
    )
    
    assert moe.num_experts == NUM_EXPERTS
    assert moe.hidden_size == HIDDEN_SIZE
    assert moe.top_k == TOP_K
    assert len(moe.experts) == NUM_EXPERTS
    assert moe.alpha.shape[0] == NUM_EXPERTS
    assert hasattr(moe, 'gate')


def test_moe_forward_2d():
    BATCH = 16
    HIDDEN_SIZE = 64
    NUM_EXPERTS = 4
    TOP_K = 2
    
    moe = MoE(
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        top_k=TOP_K
    )
    
    x = torch.randn(BATCH, HIDDEN_SIZE)
    output = moe(x)
 
    assert output.shape[0] == BATCH
    assert output.dtype == torch.float32


def test_moe_forward_3d():
    BATCH = 8
    SEQ_LEN = 16
    HIDDEN_SIZE = 64
    NUM_EXPERTS = 4
    TOP_K = 2
    
    moe = MoE(
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        top_k=TOP_K
    )
    
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)
    batch_size, seq_len, hidden_size = x.shape
    x_flat = x.reshape(-1, hidden_size)
    output = moe(x_flat)
    output = output.reshape(batch_size, seq_len, -1)
    
    assert output.shape[0] == BATCH
    assert output.shape[1] == SEQ_LEN


@pytest.mark.parametrize("gate_class,top_k", [
    (NaiveGate, 1),
    (NaiveGate, 2),
    (NoisyGate, 1),
    (NoisyGate, 2),
    (GShardGate, 2),
])
def test_moe_different_gates(gate_class, top_k):
    moe = MoE(num_experts=4, hidden_size=32, top_k=top_k)
    
    moe.gate = gate_class(
        d_model=32,
        num_expert=4,
        world_size=1,
        top_k=top_k
    )
    
    x = torch.randn(8, 32)
    output = moe(x)
    
    assert output.shape[0] == 8
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_moe_parameters():
    moe = MoE(num_experts=4, hidden_size=64, top_k=2)
    
    assert hasattr(moe, 'alpha')
    assert isinstance(moe.alpha, nn.Parameter)
    assert moe.alpha.requires_grad
    assert moe.alpha.shape == (4,)
    
    assert len(moe.experts) == 4
    assert hasattr(moe, 'gate')
    
    total_params = sum(p.numel() for p in moe.parameters())
    trainable_params = sum(p.numel() for p in moe.parameters() if p.requires_grad)
    
    assert total_params > 0
    assert trainable_params == total_params


def test_moe_gradient_flow():
    moe = MoE(num_experts=4, hidden_size=32, top_k=2)
    x = torch.randn(8, 32, requires_grad=True)
    
    output = moe(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.norm() > 0
    
    assert moe.alpha.grad is not None
    assert moe.alpha.grad.norm() > 0
    
    gate_has_grad = any(
        param.grad is not None 
        for param in moe.gate.parameters()
    )
    assert gate_has_grad


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_moe_batch_consistency(batch_size):
    moe = MoE(num_experts=4, hidden_size=32, top_k=2)
    moe.eval()
    
    x = torch.randn(batch_size, 32)
    output = moe(x)
    
    assert output.shape[0] == batch_size
