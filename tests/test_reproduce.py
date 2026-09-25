"""Exercise producer failure and output validation without scientific workloads."""
from pathlib import Path
import tempfile
import unittest

from scripts.reproduce import ROOT, STEPS, Step, run_step


class ReproductionTests(unittest.TestCase):
    def test_every_step_script_ships_with_the_repository(self):
        # A step whose producer is missing only fails midway through a long run.
        for step in STEPS:
            self.assertTrue((ROOT / step.script).is_file(), step.script)

    def run_fixture(self, source: str, existing: bool = False) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "producer.py").write_text(source)
            if existing:
                (root / "result.csv").write_text("reference\n")
            run_step(Step("fixture", "tables", "producer.py", ("result.csv",)), root)

    def test_uses_repository_cwd_and_refreshes_existing_output(self):
        self.run_fixture("from pathlib import Path\nPath('result.csv').write_text('updated\\n')", existing=True)

    def test_rejects_stale_output_from_successful_process(self):
        with self.assertRaisesRegex(RuntimeError, "not refreshed"):
            self.run_fixture("pass", existing=True)

    def test_rejects_missing_and_empty_output(self):
        for source in ("pass", "from pathlib import Path\nPath('result.csv').touch()"):
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "missing or empty"):
                self.run_fixture(source)

    def test_propagates_failure_with_log_tail(self):
        with self.assertRaisesRegex(RuntimeError, "(?s)exited 7.*producer failed"):
            self.run_fixture("import sys\nprint('producer failed')\nsys.exit(7)")


if __name__ == "__main__":
    unittest.main()
