import sys
import torch
import pytest
sys.path.insert(0, '/home/jovyan/vasiliev/notebooks/Show-o/models/moe_gates')


from base_gate import BaseGate
from naive_gate import NaiveGate
from noisy_gate import NoisyGate
from switch_gate import SwitchGate
from gshard_gate import GShardGate
from dc_gate import DCGate
from faster_gate import FasterGate


def _basic_checks(gate_type, indices, scores, batch, topk, num_expert):
    print(f"Checking {gate_type.__name__} gate")
    assert indices.shape[0] == batch, f"Indices shape {indices.shape} does not match batch {batch}"
    assert scores.shape[0] == batch

    if indices.dim() == 2:
        assert indices.shape[1] == topk
        assert scores.dim() == 2
        assert scores.shape[1] == topk
    else:
        assert indices.dim() == 1
        assert scores.dim() == 1
        assert scores.shape[0] == topk

    print(f"Indices shape: {indices.shape}")
    print(f'indices: {indices}')
    if indices.dim() == 2:
        assert torch.all(indices >= 0)
        assert torch.all(indices < num_expert)
    else:
        assert torch.all(indices >= 0)
        assert torch.all(indices < num_expert)


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(42)

def test_gshard_gate_shapes_and_loss():
    BATCH = 16
    D_MODEL = 64
    NUM_EXPERT = 4
    for TOP_K in [1, 2, 3]:
        GATE_TYPES = [
            NaiveGate,
            NoisyGate,
        ]

        if TOP_K == 1:
            GATE_TYPES.append(SwitchGate)
        if TOP_K == 2:
            GATE_TYPES.append(GShardGate)
            GATE_TYPES.append(DCGate)
            GATE_TYPES.append(FasterGate)

        x = torch.randn(BATCH, D_MODEL)
        scores_list = []
        for gate_type in GATE_TYPES:
            gate = gate_type(d_model=D_MODEL, num_expert=NUM_EXPERT, world_size=1, top_k=TOP_K)
            indices, scores = gate(x)
            _basic_checks(gate_type, indices, scores, BATCH, TOP_K, NUM_EXPERT)
            scores_list.append(scores)

        for scores in scores_list:
            assert scores_list[0].shape == scores.shape, f"Scores shapes are not equal"