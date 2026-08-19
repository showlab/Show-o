import unittest

from evaluation.c6_mask2former_alignment.train_tifa12_self_cycle import (
    lr_scale,
    parse_save_steps,
    step_seed,
)


class Tifa12TrainerTest(unittest.TestCase):
    def test_step_seed_is_deterministic_and_sample_specific(self):
        self.assertEqual(step_seed(1, "a", 2), step_seed(1, "a", 2))
        self.assertNotEqual(step_seed(1, "a", 2), step_seed(1, "b", 2))

    def test_lr_has_warmup_and_nonzero_cosine_floor(self):
        self.assertAlmostEqual(
            lr_scale(1, max_steps=100, warmup_steps=10, minimum_ratio=0.1), 0.1
        )
        self.assertAlmostEqual(
            lr_scale(10, max_steps=100, warmup_steps=10, minimum_ratio=0.1), 1.0
        )
        self.assertAlmostEqual(
            lr_scale(100, max_steps=100, warmup_steps=10, minimum_ratio=0.1), 0.1
        )

    def test_final_checkpoint_is_always_saved(self):
        self.assertEqual(parse_save_steps("1,2", 3), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
