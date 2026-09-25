"""Plot paired-query similarity distributions from committed per-anchor scores.

Writes figs/output/pairing_sim_boxes_grouped.{pdf,png} and
allsides_synth/results/pairing_similarity/pairing_mean_summary.csv (mean
cosines with 95% cluster-bootstrap CIs for the appendix pairing check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "allsides_synth" / "results" / "pairing_similarity"
FIG_DIR = REPO / "figs" / "output"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Shared paper style (scienceplots + Tol bright_extended).
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figs" / "scripts"))
from src.stats import cluster_bootstrap_mean  # noqa: E402
from _style import ENCODER_DISPLAY, ENCODER_ORDER, apply_rc, save_both  # noqa: E402

ENCODERS = [(e, ENCODER_DISPLAY[e]) for e in ENCODER_ORDER]
# (per-anchor column, scheme name in pairing_mean_summary.csv, box color).
# Non-semantic ordered triple (tightest -> loosest pairing). Deliberately
# off the political blue/red and dialect orange/purple bindings so this
# robustness figure does not read as a stance/dialect contrast.
SCHEMES = [
    ("within_cell", "within-cell", "#33BBEE"),
    ("within_article_cross_frame", "within-article × frame", "#CCBB44"),
    ("cross_article", "cross-article", "#BBBBBB"),
]


def load_data() -> dict[str, pd.DataFrame]:
    return {enc_full: pd.read_csv(DATA_DIR / f"per_anchor_sims__{enc_full}.csv") for enc_full, _ in ENCODERS}


def write_mean_summary(data: dict[str, pd.DataFrame]) -> None:
    """Mean cosine per (encoder, scheme) with a 95% cluster-bootstrap CI
    (B = 5,000, cluster = article)."""
    rows = []
    for enc_full, enc_short in ENCODERS:
        for col, scheme, _ in SCHEMES:
            sub = data[enc_full][[col, "article_id"]].dropna()
            mean, lo, hi = cluster_bootstrap_mean(
                sub[col].to_numpy(), sub["article_id"].to_numpy(), n_boot=5000, seed=42,
            )
            rows.append({"encoder_full": enc_full, "encoder_short": enc_short, "scheme": scheme,
                         "mean": mean, "lo95": lo, "hi95": hi, "n": len(sub)})
    pd.DataFrame(rows).to_csv(DATA_DIR / "pairing_mean_summary.csv", index=False)
    print("Wrote pairing_mean_summary.csv")


def common_setup(ax):
    ax.set_ylabel(r"cosine$(L, R)$")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def grouped_boxes(data: dict[str, pd.DataFrame]) -> None:
    short = {
        "within_cell": "same frame",
        "within_article_cross_frame": "cross-frame",
        "cross_article": "diff. article",
    }
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    n_enc = len(ENCODERS)
    n_scheme = len(SCHEMES)
    sub_w = 0.27
    group_centers = np.arange(n_enc) * 1.25
    for j, (col, _, color) in enumerate(SCHEMES):
        offset = (j - (n_scheme - 1) / 2) * sub_w
        positions = group_centers + offset
        vals = [data[enc_full][col].dropna().to_numpy() for enc_full, _ in ENCODERS]
        ax.boxplot(
            vals, positions=positions, widths=sub_w * 0.85,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="0.15", linewidth=1.0),
            whiskerprops=dict(color="0.3", linewidth=0.8),
            capprops=dict(color="0.3", linewidth=0.8),
            boxprops=dict(facecolor=color, edgecolor="0.3", linewidth=0.5),
        )
        ax.scatter([], [], color=color, label=short[col], s=20)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([s for _, s in ENCODERS])
    common_setup(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout()
    save_both(fig, str(FIG_DIR / "pairing_sim_boxes_grouped"))
    print("Wrote pairing_sim_boxes_grouped")


def main() -> None:
    apply_rc()
    data = load_data()
    write_mean_summary(data)
    grouped_boxes(data)


if __name__ == "__main__":
    main()
