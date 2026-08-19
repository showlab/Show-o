import unittest

import torch

from evaluation.c5_boxdiff_adapter.staged_adapter_loss import (
    StagedAdapterLossConfig,
    compose_staged_loss,
)


class StagedAdapterLossTest(unittest.TestCase):
    def test_native_anchor_disables_spatial_terms(self):
        config = StagedAdapterLossConfig(native_steps=2, ramp_steps=2)
        weights = config.weights_at(1)
        self.assertEqual(weights.stage, "native_anchor")
        self.assertEqual(weights.box_u, 0.0)
        self.assertEqual(weights.box_g, 0.0)
        self.assertEqual(weights.guided_flow, 0.0)
        self.assertEqual(weights.retention, 0.0)
        losses = {
            "native_flow": torch.tensor(2.0, requires_grad=True),
            "mmu": torch.tensor(3.0, requires_grad=True),
            "box_u": torch.tensor(100.0, requires_grad=True),
        }
        total, components = compose_staged_loss(losses, weights)
        self.assertAlmostEqual(float(total), 2.03, places=6)
        total.backward()
        self.assertIsNotNone(losses["native_flow"].grad)
        self.assertIsNotNone(losses["mmu"].grad)
        self.assertEqual(float(losses["box_u"].grad), 0.0)

    def test_spatial_weight_ramps_and_joint_is_constant(self):
        config = StagedAdapterLossConfig(native_steps=2, ramp_steps=4)
        ramp = config.weights_at(4)
        self.assertEqual(ramp.stage, "spatial_ramp")
        self.assertAlmostEqual(ramp.progress, 0.5, places=6)
        self.assertAlmostEqual(ramp.box_u, 0.025, places=6)
        self.assertAlmostEqual(ramp.box_g, 0.005, places=6)
        joint = config.weights_at(7)
        self.assertEqual(joint.stage, "joint_stabilize")
        self.assertAlmostEqual(joint.progress, 1.0, places=6)
        self.assertAlmostEqual(joint.box_u, config.box_u_weight, places=6)

    def test_missing_terms_are_zero_and_graph_is_preserved(self):
        config = StagedAdapterLossConfig(native_steps=0, ramp_steps=1)
        weights = config.weights_at(2)
        native = torch.tensor(1.5, requires_grad=True)
        box_g = torch.tensor(2.0, requires_grad=True)
        total, components = compose_staged_loss(
            {"native_flow": native, "box_g": box_g},
            weights,
        )
        self.assertGreater(float(total), 0.0)
        total.backward()
        self.assertAlmostEqual(float(native.grad), config.native_flow_weight)
        self.assertAlmostEqual(float(box_g.grad), config.box_g_weight)

    def test_non_scalar_terms_are_rejected(self):
        weights = StagedAdapterLossConfig().weights_at(1)
        with self.assertRaises(ValueError):
            compose_staged_loss({"native_flow": torch.ones(2)}, weights)


if __name__ == "__main__":
    unittest.main()
