"""Reddit real-arm k-sweep figure: per-encoder lean(lib/con) vs. k in a
3x2 panel grid (matches the AllSides synthetic-arm k-sweep layout).
Per-subplot ylim zooms each encoder to its own data range.

Reads (recomputes lean@k from candidate rankings for every k):
  reddit_nat/results/retrieval_candidates/{encoder}.csv (ID-only)
Writes:
  figs/output/real_arm_political_ksweep.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figs" / "scripts"))

from src.stats import cluster_bootstrap_mean  # noqa: E402
from _style import ENCODER_DISPLAY, ENCODER_ORDER, STANCE_COLOR, apply_rc, save_both  # noqa: E402

OUT = REPO / "figs" / "output"
DISPLAY_ORDER = list(ENCODER_ORDER) + ["bm25"]
DISPLAY_NAMES = dict(ENCODER_DISPLAY, **{"bm25": "BM25"})
K_GRID = [5, 10, 20, 50, 100]

LIB_COLOR = STANCE_COLOR["left"]   # bright_extended blue
CON_COLOR = STANCE_COLOR["right"]  # bright_extended dark red


def _per_query_lean_from_candidates(cand: pd.DataFrame, k: int) -> pd.DataFrame:
    """One row per (query_id, ideology, subreddit, encoder) with
    lean@k = (#R - #L) / k_actual, from the top-k retrieved_stance values."""
    topk = cand[cand["rank"] <= k]
    grp = topk.groupby(["query_id", "ideology", "subreddit", "encoder"])
    n_left = grp["retrieved_stance"].apply(lambda s: int((s == "left").sum()))
    n_right = grp["retrieved_stance"].apply(lambda s: int((s == "right").sum()))
    n_tot = grp["retrieved_stance"].size()
    lean = (n_right - n_left) / n_tot
    return lean.rename(f"lean@{k}").reset_index()


def _load_all_candidates() -> pd.DataFrame:
    cols = ["query_id", "ideology", "subreddit", "encoder",
            "rank", "retrieved_stance"]
    rd = REPO / "reddit_nat" / "results"
    return pd.concat([pd.read_csv(p, usecols=cols)
                      for p in sorted((rd / "retrieval_candidates").glob("*.csv"))],
                     ignore_index=True)


def compute_means() -> pd.DataFrame:
    cand = _load_all_candidates()
    cand = cand[cand["ideology"].isin(["liberal", "conservative"])]
    rows = []
    for k in K_GRID:
        per_q = _per_query_lean_from_candidates(cand, k)
        col = f"lean@{k}"
        for (enc, ideology), g in per_q.groupby(["encoder", "ideology"]):
            m, lo, hi = cluster_bootstrap_mean(
                g[col].to_numpy(), g["query_id"].to_numpy(),
                n_boot=5000, seed=42,
            )
            rows.append({"encoder": enc, "ideology": ideology, "k": k,
                         "mean_lean": m, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def main() -> None:
    apply_rc()
    long = compute_means()

    fig, axes = plt.subplots(3, 2, figsize=(3.4, 4.4),
                             sharey=False, sharex=True)
    axes = axes.flatten()
    for ax, enc in zip(axes, DISPLAY_ORDER):
        sub = long[long["encoder"] == enc]
        y_los, y_his = [], []
        for ideology, color in [("liberal", LIB_COLOR),
                                ("conservative", CON_COLOR)]:
            s = sub[sub["ideology"] == ideology].sort_values("k")
            if s.empty:
                continue
            ax.errorbar(s["k"], s["mean_lean"],
                        yerr=[s["mean_lean"] - s["ci_lo"],
                              s["ci_hi"] - s["mean_lean"]],
                        fmt="o-", color=color, markersize=3.5,
                        elinewidth=0.7, capsize=1.5, capthick=0.5)
            y_los.extend(s["ci_lo"].tolist())
            y_his.extend(s["ci_hi"].tolist())
        ax.axhline(0, color="#333", linewidth=0.5, linestyle=":", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(K_GRID)
        ax.set_xticklabels([str(k) for k in K_GRID], fontsize=6.5)
        ax.set_xlim(4.0, 120)
        ax.set_title(DISPLAY_NAMES[enc], fontsize=7.5, pad=2)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.grid(axis="y", alpha=0.25, lw=0.3)
        if y_los and y_his:
            lo = min(y_los + [0.0])
            hi = max(y_his + [0.0])
            pad = max((hi - lo) * 0.18, 0.005)
            ax.set_ylim(lo - pad, hi + pad)

    fig.text(0.52, 0.005, "top-$k$ cutoff", ha="center", fontsize=8)
    fig.text(0.005, 0.5, r"mean retrieval lean ($\%R-\%L$)",
             va="center", rotation="vertical", fontsize=8)
    fig.tight_layout(rect=[0.035, 0.025, 1.0, 1.0], w_pad=0.8, h_pad=0.6)
    save_both(fig, str(OUT / "real_arm_political_ksweep"))
    print("wrote real_arm_political_ksweep")


if __name__ == "__main__":
    main()
