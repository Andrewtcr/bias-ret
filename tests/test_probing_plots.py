"""Check the regenerated probe plots' depth convention without model inference."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_figures", ROOT / "probing/scripts/plot_regenerated_probes.py"
)
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


class ProbeDepthTests(unittest.TestCase):
    def test_default_normal_and_conditional_depths(self):
        for conditional, layers in ((False, [0, 1, 2, 3]), (True, [1, 2, 3])):
            with self.subTest(conditional=conditional):
                actual = FIGURES.relative_depth(pd.Index(layers), conditional=conditional)
                np.testing.assert_array_equal(actual, np.array(layers) / len(layers))

    def test_sparse_scores_are_rejected_with_plotting_guidance(self):
        for conditional, layers in ((False, [0, 4, 8]), (True, [4, 8])):
            with self.subTest(conditional=conditional), self.assertRaisesRegex(
                ValueError, "contiguous layer scores.*plot actual layer IDs"
            ):
                FIGURES.relative_depth(pd.Index(layers), conditional=conditional)

    def test_empty_or_incorrect_starting_layers_are_rejected(self):
        for conditional, layers in ((False, []), (True, []), (False, [1, 2]), (True, [0, 1])):
            with self.subTest(conditional=conditional, layers=layers), self.assertRaisesRegex(
                ValueError, f"starting at {int(conditional)}"
            ):
                FIGURES.relative_depth(pd.Index(layers), conditional=conditional)


if __name__ == "__main__":
    unittest.main()
