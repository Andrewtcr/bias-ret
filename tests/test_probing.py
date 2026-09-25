"""CPU checks for resumable extraction and sparse-layer probe training.

Run with: python -m unittest discover -s tests -p 'test_probing*.py'
Requires the base dependencies plus h5py and tqdm; no model downloads.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "probing" / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = load_script("probe_generation", "generate_embeddings.py")
train = load_script("probe_training", "train_probes.py")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.table = self.root / "input.csv"
        pd.DataFrame({"id": [0, 1], "text": ["alpha", "beta"]}).to_csv(self.table, index=False)
        self.config = self.root / "extract.yaml"
        self.config.write_text(yaml.safe_dump({
            "input_data": str(self.table), "out_dir": str(self.root / "embeddings"),
            "run_id": "probes", "input_name": "sample", "in_cols": ["text"],
            "idx_col": "id", "batch_size": 2,
        }))
        self.final = self.root / "embeddings" / "fake" / "probes.sample.h5"
        self.partial = self.final.with_suffix(".h5.partial")
        self.argv = ["generate_embeddings.py", "--config", str(self.config), "--model", "test/fake", "--layers", "0", "4", "8"]
        model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=9, hidden_size=3))
        self.loader = self.use_patch(patch.object(gen.lib, "load_encoder", return_value=(model, None)))
        self.use_patch(patch.object(gen.lib, "set_all_seeds"))
        self.use_patch(patch.object(gen, "get_device", return_value="cpu"))
        self.use_patch(patch.object(gen.lib, "create_dataloader", return_value=None))
        self.generator = self.use_patch(patch.object(gen, "generate", side_effect=self.fill))

    def use_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    @staticmethod
    def fill(model, loader, datasets, col, layers, device):
        for dataset in datasets[col].values():
            dataset.resize((2, 3))
            dataset[:] = np.ones((2, 3), dtype=np.float16)

    def run_generation(self, *extra):
        with patch("sys.argv", self.argv + list(extra)):
            gen.main()

    def test_completed_extraction_is_validated_and_skipped_without_loading_model(self):
        self.run_generation()
        self.assertTrue(self.final.exists())
        self.assertFalse(self.partial.exists())
        with h5py.File(self.final, "r") as f:
            self.assertTrue(f.attrs["complete"])
            metadata = json.loads(f.attrs["metadata"])
            self.assertEqual(metadata["layers"], [0, 4, 8])
            self.assertEqual(metadata["input_sha256"], gen.input_digest(self.table))
        self.loader.reset_mock()
        self.generator.reset_mock()
        self.run_generation()
        self.loader.assert_not_called()
        self.generator.assert_not_called()

    def test_changed_input_or_settings_never_overwrite_final_file(self):
        self.run_generation()
        original = self.final.read_bytes()
        for extra in [("--batch-size", "1"), ("--layers", "0", "4")]:
            with self.assertRaisesRegex(ValueError, "Move it aside"):
                self.run_generation(*extra)
            self.assertEqual(self.final.read_bytes(), original)
        self.table.write_text("id,text\n0,changed\n1,beta\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.run_generation("--restart-incomplete")
        self.assertEqual(self.final.read_bytes(), original)

    def test_interrupted_extraction_requires_explicit_restart(self):
        self.generator.side_effect = RuntimeError("simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            self.run_generation()
        self.assertFalse(self.final.exists())
        with h5py.File(self.partial, "r") as f:
            self.assertFalse(f.attrs["complete"])
        self.loader.reset_mock()
        with self.assertRaisesRegex(FileExistsError, "--restart-incomplete"):
            self.run_generation()
        self.loader.assert_not_called()
        self.generator.side_effect = self.fill
        self.run_generation("--restart-incomplete")
        self.assertTrue(self.final.exists())
        self.assertFalse(self.partial.exists())

    def test_incomplete_shapes_are_not_published_or_reused(self):
        self.generator.side_effect = lambda *args: None
        with self.assertRaisesRegex(ValueError, "shape"):
            self.run_generation()
        self.assertFalse(self.final.exists())
        with h5py.File(self.partial, "r") as f:
            self.assertFalse(f.attrs["complete"])
        self.partial.unlink()
        self.generator.side_effect = self.fill
        self.run_generation()
        with h5py.File(self.final, "r+") as f:
            f["text/layer_4"].resize((1, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.run_generation()

    def test_final_file_without_provenance_is_never_reused(self):
        self.run_generation()
        with h5py.File(self.final, "r+") as f:
            del f.attrs["metadata"]
            del f.attrs["complete"]
        original = self.final.read_bytes()
        self.loader.reset_mock()
        with self.assertRaisesRegex(ValueError, "missing completion marker or metadata"):
            self.run_generation()
        self.loader.assert_not_called()
        self.assertEqual(self.final.read_bytes(), original)

    def test_publish_refuses_an_existing_final_path(self):
        self.run_generation()
        original = self.final.read_bytes()
        self.partial.write_bytes(original)
        with h5py.File(self.partial, "r") as f:
            metadata = json.loads(f.attrs["metadata"])
        with self.assertRaises(FileExistsError):
            gen.finalize_dataset(self.partial, self.final, metadata)
        self.assertEqual(self.final.read_bytes(), original)
        self.assertTrue(self.partial.exists())

    def test_default_range_omits_final_state(self):
        self.assertEqual(gen.requested_layers([-1], 9), list(range(9)))
        for invalid in ([0, 0], [-1, 0], [10]):
            with self.assertRaises(ValueError):
                gen.requested_layers(invalid, 9)


class SparseProbeTests(unittest.TestCase):
    def run_training_case(self, layers, modes):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = root / "labels.csv"
            labels = np.tile([0, 1], 10)
            pd.DataFrame({"stance": np.where(labels, "left", "right")}).to_csv(table, index=False)
            h5_path = root / "fake.h5"
            with h5py.File(h5_path, "w") as f:
                for layer in layers:
                    values = np.column_stack([labels, np.arange(len(labels))]) + layer
                    f.create_dataset(f"text/layer_{layer}", data=values.astype(np.float16))
            config = root / "probes.yaml"
            config.write_text(yaml.safe_dump({
                "dataset": {"name": "sample", "h5_template": str(h5_path), "mode": "aligned",
                            "input_data": str(table), "input_type": "text", "label_col": "stance",
                            "pos_label": "left", "n_splits": 2},
                "modes": modes,
                "output": {"probe_dir": str(root / "weights"), "res_dir": str(root / "scores")},
            }))
            roster = root / "models.yaml"
            roster.write_text(yaml.safe_dump({"models": {"Fake": "test/fake"}}))
            with patch("sys.argv", ["train_probes.py", "--config", str(config), "--models-config", str(roster)]):
                train.main()
            normal = pd.read_csv(root / "scores/fake/sample_res.csv")
            self.assertEqual(sorted(normal.layer.unique()), layers)
            if "conditional" in modes:
                conditional = pd.read_csv(root / "scores/fake/sample_cond_res.csv")
                self.assertEqual(sorted(conditional.layer.unique()), [layer for layer in layers if layer != 0])
            self.assertTrue(np.isfinite(normal[["acc", "f1", "nce"]]).all().all())

    def test_training_preserves_sparse_layer_ids(self):
        self.run_training_case([0, 4, 8], ["normal", "conditional"])

    def test_default_normal_and_conditional_layer_ids(self):
        self.run_training_case(list(range(9)), ["normal", "conditional"])

    def test_normal_probes_allow_sparse_layers_without_layer_zero(self):
        self.run_training_case([4, 8], ["normal"])

    def test_conditional_requires_layer_zero_and_groups_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "layers.h5"
            with h5py.File(path, "w") as f:
                for name in ["aal/layer_4", "aal/layer_8", "wme/layer_4"]:
                    f.create_dataset(name, data=np.ones((4, 3), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "require layer 0"):
                train.lib.probe_layer_ids(path, ["aal"], conditional=True)
            with self.assertRaisesRegex(ValueError, "different layer sets"):
                train.lib.probe_layer_ids(path, ["aal", "wme"])
            with h5py.File(path, "r+") as f:
                f.attrs["complete"] = False
            with self.assertRaisesRegex(ValueError, "finish embedding extraction"):
                train.lib.probe_layer_ids(path, ["aal"])


if __name__ == "__main__":
    unittest.main()
