"""Train and evaluate per-layer logistic-regression probes on embeddings.

Each dataset config selects aligned labels or paired text variants. Aligned
rows use shuffled KFold evaluation; paired variants use a seeded train/test
split without replacement. Normal probes use layer L alone; conditional
probes concatenate layer L with layer 0. Plotting subtracts the layer-0
baseline NCE from conditional NCE.

Outputs:
  results/probes/{model_slug}/{ds_name}[_cond]_layer_{L}_{tag}.pkl
  results/probe_scores/{model_slug}/{ds_name}[_cond]_res.csv

Usage:
  python probing/scripts/train_probes.py \
      --config probing/configs/probes/hcmagic_synth.yaml [--only qwen]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from probing import lib  # noqa: E402
from src.io import load_yaml  # noqa: E402
from src.stats import stable_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # probing/


def run_aligned(ds_cfg: dict, h5_path: Path, layer: int, conditional: bool, seed: int, max_iter: int) -> Iterator[tuple[dict, object]]:
    data = lib.load_table(ROOT / ds_cfg["input_data"])
    X, Y = lib.build_aligned_xy(
        h5_path,
        ds_cfg["input_type"],
        layer,
        data,
        ds_cfg["label_col"],
        ds_cfg["pos_label"],
        exclude_label=ds_cfg.get("exclude_label"),
        layer0_concat=conditional,
    )
    kf = KFold(n_splits=int(ds_cfg.get("n_splits", 5)), shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        pipe, metrics = lib.train_eval_lr_probe(
            X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], max_iter=max_iter
        )
        yield {"tag": f"f{fold}", **metrics}, pipe


def run_paired(ds_cfg: dict, h5_path: Path, layer: int, conditional: bool, seed: int, max_iter: int) -> Iterator[tuple[dict, object]]:
    for pos_col in ds_cfg["pos_cols"]:
        X, Y = lib.build_paired_xy(h5_path, pos_col, ds_cfg["neg_col"], layer, layer0_concat=conditional)
        rng = np.random.default_rng(stable_seed(ds_cfg["name"], pos_col, base=seed))
        perm = rng.permutation(len(Y))
        n_train = int(float(ds_cfg.get("train_frac", 0.8)) * len(perm))
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        pipe, metrics = lib.train_eval_lr_probe(
            X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], max_iter=max_iter
        )
        yield {"tag": pos_col, **metrics}, pipe


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="probing/configs/probes/<dataset>.yaml")
    p.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    p.add_argument("--only", default=None, help="substring filter on model short name")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    roster = load_yaml(args.models_config)
    ds_cfg = cfg["dataset"]
    ds_name = ds_cfg["name"]
    modes = cfg.get("modes", ["normal"])
    max_iter = int(cfg.get("max_iter", 5000))
    seed = int(cfg.get("seed", 42))

    out_cfg = cfg.get("output", {})
    probe_dir = ROOT / out_cfg.get("probe_dir", "results/probes")
    res_dir = ROOT / out_cfg.get("res_dir", "results/probe_scores")

    models = roster["models"]
    if args.only:
        models = {k: v for k, v in models.items() if args.only.lower() in k.lower()}

    for model_name, model_chkpt in models.items():
        slug = lib.model_slug(model_chkpt)
        h5_path = ROOT / ds_cfg["h5_template"].format(model_slug=slug)
        if not h5_path.exists():
            print(f"[probes] {ds_name}/{model_name}: missing {h5_path} — skip (run generate_embeddings.py first)")
            continue

        groups = [ds_cfg["input_type"]] if ds_cfg["mode"] == "aligned" else [*ds_cfg["pos_cols"], ds_cfg["neg_col"]]
        layers = lib.probe_layer_ids(h5_path, groups, conditional="conditional" in modes)
        print(f"[probes] {ds_name}/{model_name}: layers {layers}")

        for mode in modes:
            conditional = mode == "conditional"
            suffix = "_cond" if conditional else ""
            runner = run_aligned if ds_cfg["mode"] == "aligned" else run_paired
            layer_range = [layer for layer in layers if layer != 0] if conditional else layers
            if not layer_range:
                raise ValueError(f"{h5_path}: {mode} probing requires at least one non-baseline layer")

            rows = []
            for layer in tqdm(list(layer_range), desc=f"{ds_name}/{model_name}/{mode}"):
                for row, pipe in runner(ds_cfg, h5_path, layer, conditional, seed, max_iter):
                    row["layer"] = layer
                    rows.append(row)
                    probe_path = probe_dir / slug / f"{ds_name}{suffix}_layer_{layer}_{row['tag']}.pkl"
                    lib.save_probe(pipe, probe_path)

            res_path = res_dir / slug / f"{ds_name}{suffix}_res.csv"
            res_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(res_path, index=False)
            print(f"[probes] wrote {res_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
