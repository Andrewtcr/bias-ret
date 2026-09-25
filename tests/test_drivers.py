"""Check pipeline dispatch without loading models or calling external APIs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
EXPERIMENTS = ("hcmagic", "reddit_nat", "probing")
MODELS = ("fixture/new-model", "other/second-model")
# Scripts each driver dispatches, in stage order; the first is the embedding stage.
STAGES = {
    "hcmagic": ("embed_corpus.py", "embed_queries.py", "retrieve.py", "analyze_retrieval.py"),
    "reddit_nat": ("embed_queries.py", "retrieve.py"),
    "probing": ("generate_embeddings.py", "train_probes.py", "plot_regenerated_probes.py"),
}


@unittest.skipUnless(BASH, "Bash is required for pipeline-driver tests")
class DriverTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for experiment in EXPERIMENTS:
            config = "models.yaml" if experiment == "probing" else "retrieval.yaml"
            driver = self.root / experiment / "scripts/run.sh"
            driver.parent.mkdir(parents=True)
            shutil.copy2(ROOT / experiment / "scripts/run.sh", driver)
            shutil.copytree(ROOT / experiment / "configs", self.root / experiment / "configs")
            target = self.root / experiment / "configs" / config
            contents = yaml.safe_load(target.read_text(encoding="utf-8"))
            if experiment == "probing":
                contents["models"] = dict(zip(("First", "Second"), MODELS))
            else:
                contents["retrieval"]["encoders"] = [
                    {"name": MODELS[0], "devices": ["cuda:9"]}, MODELS[1],
                ]
            target.write_text(yaml.safe_dump(contents))

        self.calls = self.root / "calls.jsonl"
        mock = self.root / "mock.py"
        mock.write_text("""import json, os, sys
args = sys.argv[1:]
if args[0] == "-":
    os.execv(sys.executable, [sys.executable] + args)
fd = os.open(os.environ["DRIVER_CALLS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, (json.dumps(args) + "\\n").encode())
os.close(fd)
target = args[1] if args[0] == "-m" else args[0]
script = target.rsplit(".", 1)[-1] + ".py" if args[0] == "-m" else os.path.basename(target)
if os.environ.get("DRIVER_FAIL_EMBED") == "1" and script == os.environ["DRIVER_EMBED_SCRIPT"]:
    sys.exit(7)
""")
        self.python = self.root / "mock-python"
        self.python.write_text(
            f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(mock))} \"$@\"\n"
        )
        self.python.chmod(0o755)

    def run_driver(self, experiment, *args, fail_embed=False):
        self.calls.unlink(missing_ok=True)
        env = dict(os.environ, PYTHON=str(self.python), DRIVER_CALLS=str(self.calls),
                   DRIVER_FAIL_EMBED="1" if fail_embed else "0",
                   DRIVER_EMBED_SCRIPT=STAGES[experiment][0])
        result = subprocess.run(
            [BASH, str(self.root / experiment / "scripts/run.sh"), *args],
            cwd=self.root.parent, env=env, capture_output=True, text=True, timeout=30,
        )
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] \
            if self.calls.exists() else []
        return result, calls

    @staticmethod
    def stage(experiment, call):
        target = call[1] if call[0] == "-m" else call[0]
        script = target.rsplit(".", 1)[-1] + ".py" if call[0] == "-m" else os.path.basename(target)
        return STAGES[experiment].index(script) + 1

    def test_modified_yaml_rosters_reach_all_embedders(self):
        for experiment, repetitions in (("hcmagic", 2), ("reddit_nat", 1), ("probing", 2)):
            with self.subTest(experiment=experiment):
                result, calls = self.run_driver(experiment)
                self.assertEqual(result.returncode, 0, result.stderr)
                flag = "--model" if experiment == "probing" else "--only"
                actual = [call[call.index(flag) + 1] for call in calls if flag in call]
                self.assertEqual(sorted(actual), sorted(MODELS * repetitions))

    def test_every_stage_script_and_config_ships_with_the_repository(self):
        # The drivers are the documented way to rerun each pipeline, so a script or
        # --config path the repository does not ship makes that stage fail on a
        # fresh clone.
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                result, calls = self.run_driver(experiment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({self.stage(experiment, call) for call in calls},
                                 set(range(1, len(STAGES[experiment]) + 1)))
                for call in calls:
                    config = call[call.index("--config") + 1]
                    self.assertTrue((self.root / config).is_file(), f"missing {config}")
                    script = call[1].replace(".", "/") + ".py" if call[0] == "-m" else call[0]
                    self.assertTrue((ROOT / script).is_file(), f"missing {script}")

    def test_embedding_failure_stops_downstream_stages(self):
        for experiment in EXPERIMENTS:
            with self.subTest(experiment=experiment):
                result, calls = self.run_driver(experiment, fail_embed=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(calls)
                self.assertTrue(all(self.stage(experiment, call) == 1 for call in calls))

    def test_probe_stage_two_skips_embeddings(self):
        result, calls = self.run_driver("probing", "--start-stage", "2", "--restart-incomplete")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([self.stage("probing", call) for call in calls], [2, 2, 3])
        self.assertFalse(any("--restart-incomplete" in call for call in calls))

    def test_probe_restart_flag_only_reaches_embeddings(self):
        result, calls = self.run_driver("probing", "--restart-incomplete")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual("--restart-incomplete" in call, self.stage("probing", call) == 1)


if __name__ == "__main__":
    unittest.main()
