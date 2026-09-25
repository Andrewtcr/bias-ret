"""Render the paper's F1 and conditional V-information figure.

Reads results/consolidated_scores.csv for the models in configs/models.yaml's
subset. HCMagic-synth is on the left and AllSides-synth on the right. The
baseline is the dedicated layer-0 run for each model and dataset; its mean
NCE is subtracted from every conditional layer. Both curve types normalize
depth by max(layer) + 1.

Writes figs/output/probe_combined_subset.{pdf,png}.

Usage:
  python probing/scripts/plot_paper_probes.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "figs" / "scripts"))

from src.io import load_yaml  # noqa: E402
from _style import apply_rc, save_both  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # probing/
SCORES_CSV = ROOT / "results" / "consolidated_scores.csv"


def model_slug(chkpt: str) -> str:
    # Matches model_slug -- inlined to avoid lib.py's top-level h5py
    # import, which this plotting-only script has no other use for.
    return chkpt.split("/")[-1].lower()

SYNTH_DATASETS = [
    # Paper order: HCMagic left, AllSides right.
    {"ds_name": "hcmagic_synth", "display": "HCMagic$_{\\mathrm{synth}}$"},
    {"ds_name": "allsides_synth", "display": "AllSides$_{\\mathrm{synth}}$"},
]


def load_rows() -> list[dict]:
    with SCORES_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def mean_se_by_layer(vals_by_layer: dict[int, list[float]]) -> tuple[list[int], list[float], list[float]]:
    layers = sorted(vals_by_layer)
    means, ses = [], []
    for layer in layers:
        vals = vals_by_layer[layer]
        means.append(statistics.mean(vals))
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ses.append(1.96 * sd / (len(vals) ** 0.5) if len(vals) > 1 else 0.0)
    return layers, means, ses


def normal_f1_series(rows: list[dict], model: str, ds_name: str) -> tuple[list[float], list[float], list[float]] | None:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["ds_name"] == ds_name and r["source"] == "normal":
            by_layer[int(r["layer"])].append(float(r["f1"]))
    if not by_layer:
        return None
    layers, means, ses = mean_se_by_layer(by_layer)
    n_layers = max(layers) + 1
    depths = [layer / n_layers for layer in layers]
    return depths, means, ses


def vinfo_delta_series(rows: list[dict], model: str, ds_name: str) -> tuple[list[float], list[float]] | None:
    baseline_vals = [float(r["nce"]) for r in rows if r["model"] == model and r["ds_name"] == ds_name and r["source"] == "baseline"]
    if not baseline_vals:
        return None
    baseline_nce = statistics.mean(baseline_vals)

    by_layer: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["ds_name"] == ds_name and r["source"] == "conditional":
            by_layer[int(r["layer"])].append(float(r["nce"]))
    if not by_layer:
        return None
    layers, means, _ = mean_se_by_layer(by_layer)
    n_layers = max(layers) + 1
    depths = [layer / n_layers for layer in layers]
    deltas = [m - baseline_nce for m in means]
    return depths, deltas


def main() -> None:
    rows = load_rows()
    roster = load_yaml(ROOT / "configs" / "models.yaml")
    models = roster["models"]
    subset = {k: models[k] for k in roster.get("subset", list(models)) if k in models}

    apply_rc()
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, len(SYNTH_DATASETS), figsize=(6, 5))
    for i, ds in enumerate(SYNTH_DATASETS):
        ds_name = ds["ds_name"]
        for j, (model_name, model_chkpt) in enumerate(subset.items()):
            slug = model_slug(model_chkpt)
            series = normal_f1_series(rows, slug, ds_name)
            if series is None:
                print(f"[plot] {slug}/{ds_name}: no normal-probe data -- skipping F1 row")
                continue
            depth, means, ses = series
            axs[0][i].plot(depth, means, label=model_name.split("-")[0] if i == 0 else None, color=f"C{j}", zorder=3)
            lo = [m - s for m, s in zip(means, ses)]
            hi = [m + s for m, s in zip(means, ses)]
            axs[0][i].fill_between(depth, lo, hi, color=f"C{j}", alpha=0.3)
        axs[0][i].set_title(ds["display"], fontsize=20)
        axs[0][i].grid(True, zorder=1)

        for j, (model_name, model_chkpt) in enumerate(subset.items()):
            slug = model_slug(model_chkpt)
            series = vinfo_delta_series(rows, slug, ds_name)
            if series is None:
                print(f"[plot] {slug}/{ds_name}: missing baseline or conditional data -- skipping delta row")
                continue
            depth, delta = series
            axs[1][i].plot(depth, delta, color=f"C{j}", zorder=3)
        axs[1][i].axhline(0, color="black", zorder=2)
        axs[1][i].grid(True, zorder=1)

    fig.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, -0.04), fontsize=12)
    axs[0][0].set_ylabel("F1 Score", fontsize=15)
    axs[1][0].set_ylabel(r"$\mathcal{V}$-information", fontsize=15)
    fig.supxlabel("Model Depth", fontsize=15, y=0.05)
    fig.tight_layout()

    out_dir = REPO_ROOT / "figs" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_combined_subset"
    save_both(fig, str(out_path))
    print(f"[plot] wrote {out_path}.png / .pdf")


if __name__ == "__main__":
    main()
