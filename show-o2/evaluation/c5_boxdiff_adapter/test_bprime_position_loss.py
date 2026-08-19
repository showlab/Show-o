import unittest

import torch

from evaluation.c5_boxdiff_adapter.bprime_position_loss import (
    LOSS_NAMES,
    compose_position_loss,
    position_only_losses,
)
from guidance.layout import LayoutSpec, ObjectConstraint


class BprimePositionLossTest(unittest.TestCase):
    def layout(self):
        return LayoutSpec(
            prompt="cat left of dog",
            relation="left",
            objects=(
                ObjectConstraint("cat", (0.0, 0.0, 0.5, 1.0), (0, 3), (1,)),
                ObjectConstraint("dog", (0.5, 0.0, 1.0, 1.0), (12, 15), (4,)),
            ),
        )

    def hidden(self, swapped=False):
        values = torch.zeros((1, 4, 4), dtype=torch.float32)
        cat_dim, dog_dim = (0, 1) if not swapped else (1, 0)
        values[0, 0, cat_dim] = 4.0
        values[0, 2, cat_dim] = 4.0
        values[0, 1, dog_dim] = 4.0
        values[0, 3, dog_dim] = 4.0
        return values

    def test_correct_geometry_beats_swapped_geometry(self):
        phrases = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        correct, _ = position_only_losses(
            self.hidden(False), self.hidden(False), self.layout(), phrases,
            height=2, width=2, temperature=8.0,
        )
        swapped, _ = position_only_losses(
            self.hidden(True), self.hidden(False), self.layout(), phrases,
            height=2, width=2, temperature=8.0,
        )
        self.assertLess(float(correct["box_g"]), float(swapped["box_g"]))
        self.assertLess(float(correct["relation_g"]), float(swapped["relation_g"]))

    def test_base_hidden_is_stop_gradient(self):
        phrases = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        )
        spatial = self.hidden(False).requires_grad_(True)
        base = self.hidden(False).requires_grad_(True)
        losses, _ = position_only_losses(
            spatial, base, self.layout(), phrases,
            height=2, width=2, temperature=8.0,
        )
        weights = {name: 1.0 for name in LOSS_NAMES}
        total, _ = compose_position_loss(losses, weights)
        total.backward()
        self.assertIsNotNone(spatial.grad)
        self.assertIsNone(base.grad)

    def test_flow_term_is_rejected(self):
        losses = {name: torch.ones(()) for name in LOSS_NAMES}
        losses["native_flow"] = torch.ones(())
        with self.assertRaises(KeyError):
            compose_position_loss(losses, {name: 1.0 for name in LOSS_NAMES})


if __name__ == "__main__":
    unittest.main()
