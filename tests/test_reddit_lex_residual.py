"""Reproduce Reddit lexical-control tables from the released numeric scores."""
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reddit_lex_residual", ROOT / "reddit_nat/scripts/analyze_lexical.py"
)
LEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEX)
PAPER_TABLE = {
    "bge-large-en-v1.5": (0.024, 0.007),
    "qwen3-embedding-8b": (0.028, 0.015),
    "llama-embed-nemotron-8b": (0.039, 0.022),
    "octen-embedding-8b": (0.048, 0.033),
    "text-embedding-3-large": (0.062, 0.036),
    "bm25": (0.001, 0.002),
}


class RedditLexicalRegressionTests(unittest.TestCase):
    def test_cached_scores_reproduce_reference_without_text_or_input_changes(self):
        results = ROOT / "reddit_nat/results"
        expected = pd.read_csv(results / "lex_residual_summary.csv").set_index("encoder")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            scores = output / "lex_residual_per_query.csv"
            shutil.copy2(results / scores.name, scores)
            tokens = output / "lex_residual_token_logodds.csv"
            tokens.write_text("unchanged token table\n")
            originals = {path: path.read_bytes() for path in (scores, tokens)}

            with mock.patch.object(sys, "argv", ["analyze_lexical.py", "--from-scores"]), \
                    mock.patch.object(LEX.lib, "run_dir", return_value=output), \
                    mock.patch.object(LEX.lib, "load_reddit_queries", side_effect=AssertionError("read post text")), \
                    mock.patch.object(LEX, "fit_regression", wraps=LEX.fit_regression) as fit, \
                    contextlib.redirect_stdout(io.StringIO()):
                LEX.main()

            actual = pd.read_csv(output / "lex_residual_summary.csv").set_index("encoder")
            # Anchor to the paper's tab:political-lexreg-realarm (raw gap, gamma) so
            # the check catches a changed result even after reproduce.py rewrites
            # the committed summary.
            for encoder, (raw, gamma) in PAPER_TABLE.items():
                self.assertAlmostEqual(actual.loc[encoder, "raw_gap_con_minus_lib"], raw, delta=5e-4)
                self.assertAlmostEqual(actual.loc[encoder, "gamma"], gamma, delta=5e-4)
            pd.testing.assert_frame_equal(
                actual.sort_index(), expected.sort_index(), check_exact=False,
                rtol=1e-10, atol=1e-12,
            )
            self.assertEqual(fit.call_count, len(expected))
            report = (output / "lex_residual.md").read_text()
            for encoder in expected.index:
                self.assertIn(encoder, report)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
