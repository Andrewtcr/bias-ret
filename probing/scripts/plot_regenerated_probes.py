"""Render the combined probe figure from regenerated results/probe_scores CSVs.

Normal-probe F1 (top row) and conditional V-information (bottom row) for each
configured dataset. Conditional V-information is conditional NCE minus normal
layer-0 NCE. F1 uncertainty bands are 1.96 times the per-layer standard
error, using that layer's actual fold or variant count.

Writes {output_dir}/probe_combined_subset_regenerated.{pdf,png}. For the
paper figure from released scores, use plot_paper_probes.py.

Usage:
  python probing/scripts/plot_regenerated_probes.py --config probing/configs/figures.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figs" / "scripts"))

from probing import lib  # noqa: E402
from src.io import load_yaml  # noqa: E402
from _style import apply_rc, save_both  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # probing/


def mean_se(res: pd.DataFrame, metrics: tuple[str, ...] = ("acc", "f1", "nce")) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = res.groupby("layer")[list(metrics)]
    mean = grouped.mean()
    n = grouped.count()
    se = 1.96 * grouped.std() / np.sqrt(n)
    return mean, se


def load_res(res_dir: Path, ds_name: str, model_chkpt: str, conditional: bool = False) -> pd.DataFrame:
    slug = lib.model_slug(model_chkpt)
    suffix = "_cond" if conditional else ""
    return pd.read_csv(res_dir / slug / f"{ds_name}{suffix}_res.csv")


def relative_depth(index: pd.Index, *, conditional: bool = False) -> np.ndarray:
    """Keep the default depth convention, rejecting sparse or shifted layers."""
    layers = np.array(index.tolist())
    start = 1 if conditional else 0
    if not len(layers) or not np.array_equal(layers, np.arange(start, start + len(layers))):
        mode = "conditional" if conditional else "normal"
        raise ValueError(
            "Default depth normalization expects contiguous layer scores "
            f"starting at {start} for {mode} probes. For sparse runs, plot actual "
            "layer IDs or use model-depth metadata in a separate plotting script."
        )
    return layers / len(index)


# Conditional probing (Hewitt et al., 2021) scores a layer against the model's
# NON-CONTEXTUAL embedding layer, not against itself:
#
#     V-information(L) = NCE(layer_L (+) layer_0) - NCE(layer_0)
#
# i.e. what layer L adds ON TOP OF the token-level baseline. Subtracting the
# same-layer normal probe instead -- NCE(cond L) - NCE(normal L) -- measures the
# reverse (what layer_0 adds on top of layer L) and understates the effect.
BASELINE_LAYER = 0


def baseline_nce(means_bl: pd.DataFrame, ses_bl: pd.DataFrame) -> tuple[float, float]:
    """Scalar NCE (and SE) of the non-contextual layer-0 normal probe."""
    if BASELINE_LAYER not in means_bl.index:
        raise SystemExit(
            f"normal-probe results have no layer {BASELINE_LAYER}; the conditional "
            "baseline is the non-contextual embedding layer, so train_probes.py "
            "must be run with that layer included."
        )
    return float(means_bl["nce"].loc[BASELINE_LAYER]), float(ses_bl["nce"].loc[BASELINE_LAYER])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "figures.yaml"))
    p.add_argument("--models-config", default=str(ROOT / "configs" / "models.yaml"))
    args = p.parse_args()
    cfg = load_yaml(args.config)
    roster = load_yaml(args.models_config)

    res_dir = ROOT / "results" / "probe_scores"
    out_dir = ROOT / cfg.get("output_dir", "../figs/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = cfg["datasets"]
    models = roster["models"]
    subset = {k: models[k] for k in roster.get("subset", list(models)) if k in models}

    apply_rc()
    import matplotlib.pyplot as plt

    # F1 (top) + conditional NCE delta (bottom)
    fig, axs = plt.subplots(2, len(datasets), figsize=(6, 5))
    for i, ds in enumerate(datasets):
        for j, (model_name, model_chkpt) in enumerate(subset.items()):
            res = load_res(res_dir, ds["ds_name"], model_chkpt)
            means, ses = mean_se(res)
            depth = relative_depth(means.index)
            axs[0][i].plot(depth, means["f1"], label=model_name.split("-")[0] if i == 0 else None, color=f"C{j}", zorder=3)
            axs[0][i].fill_between(depth, means["f1"] - ses["f1"], means["f1"] + ses["f1"], color=f"C{j}", alpha=0.3)
        axs[0][i].set_title(ds["display"], fontsize=20)
        axs[0][i].grid(True, zorder=1)

        for j, (model_name, model_chkpt) in enumerate(subset.items()):
            res = load_res(res_dir, ds["ds_name"], model_chkpt, conditional=True)
            baseline = load_res(res_dir, ds["ds_name"], model_chkpt, conditional=False)
            means, _ = mean_se(res)
            means_bl, ses_bl = mean_se(baseline)
            bl_nce, _ = baseline_nce(means_bl, ses_bl)
            delta = (means["nce"] - bl_nce).to_numpy()
            depth = relative_depth(means.index, conditional=True)
            axs[1][i].plot(depth, delta, color=f"C{j}", zorder=3)
        axs[1][i].axhline(0, color="black", zorder=2)
        axs[1][i].grid(True, zorder=1)
    fig.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.04), fontsize=12)
    axs[0][0].set_ylabel("F1 Score", fontsize=15)
    axs[1][0].set_ylabel(r"$\mathcal{V}$-information", fontsize=15)
    fig.supxlabel("Model Depth", fontsize=15, y=0.05)
    fig.tight_layout()
    out_path = out_dir / "probe_combined_subset_regenerated"
    save_both(fig, str(out_path))
    print(f"[plot] wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
