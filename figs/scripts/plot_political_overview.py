"""Combined political forest figure: (a) AllSides synth gold-filtered paired
lean@10 + (b) Reddit real-arm relevance-filtered lean@10.

Both subplots
use the shared STANCE_COLOR (blue=left/liberal, red=right/conservative).
Asterisks on each encoder y-label encode the per-encoder p-value for that
subplot's headline test:
  - LEFT: gold-filtered sign-flip permutation p (`perm_p`) at k=10 from
          figs/results/gold_filtered_paired_test_k10.csv
          (figs/scripts/analyze_political_gold.py).
  - RIGHT: two-sided Mann-Whitney U p, recomputed from queries with at least
           one document graded relevant by both judges.

Outputs:
  figs/output/political_combined.{pdf,png}
  reddit_nat/results/gold_filtered_lean_summary.csv  (per-encoder numbers
      behind panel (b): per-ideology mean lean with bootstrap CI, the
      conservative-minus-liberal gap with a two-sample bootstrap CI, and
      the Mann-Whitney U p)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figs" / "scripts"))

from _style import (  # noqa: E402
    ENCODER_DISPLAY, ENCODER_ORDER, STANCE_COLOR, apply_rc, save_both,
)
from plot_political_retrieval import (  # noqa: E402
    _gold_filtered_bias, _per_stance_lean_filtered,
    DENSE_RUN, BM25_RUN,
)
from reddit_nat.lib import filtered_per_query_lean  # noqa: E402

DISPLAY_ORDER = list(ENCODER_ORDER) + ["bm25"]
DISPLAY_NAMES = dict(ENCODER_DISPLAY, **{"bm25": "BM25"})
LEFT_COLOR = STANCE_COLOR["left"]
RIGHT_COLOR = STANCE_COLOR["right"]

REDDIT_RUN = REPO / "reddit_nat" / "results"
REDDIT_SUMMARY = REDDIT_RUN / "gold_filtered_lean_summary.csv"
SYNTH_GOLD_TEST = REPO / "figs" / "results" / "gold_filtered_paired_test_k10.csv"

K = 10
B_BOOT = 5000
RNG = np.random.default_rng(42)


def _stars(p: float) -> str:
    if p is None or pd.isna(p):
        return ""
    if p < 1e-3:
        return " ***"
    if p < 1e-2:
        return " **"
    if p < 5e-2:
        return " *"
    return ""


def _synth_pvals() -> dict[str, float]:
    """Per-encoder gold-filtered sign-flip permutation p at k=10."""
    test = pd.read_csv(SYNTH_GOLD_TEST)
    return dict(zip(test["encoder"], test["perm_p"].astype(float)))


def _bootstrap_ci(x: np.ndarray, b: int = B_BOOT, alpha: float = 0.05):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    n = len(x)
    means = np.empty(b)
    for i in range(b):
        idx = RNG.integers(0, n, n)
        means[i] = x[idx].mean()
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _reddit_summary() -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """Per-(encoder, ideology) AGREE-both gold-filtered mean + bootstrap CI.
    Also returns per-encoder MWU p (con vs lib) and a per-encoder table with
    the con-minus-lib gap and its two-sample bootstrap CI."""
    from scipy import stats as scipy_stats
    per_q = filtered_per_query_lean(REDDIT_RUN, k=K, judge="both")
    rows, pvals, gaps = [], {}, []
    # One seed-42 stream for the gap bootstrap, consumed in DISPLAY_ORDER; the
    # paper's tab:reddit-goldtest CIs use this stream and order.
    gap_rng = np.random.default_rng(42)
    for enc in DISPLAY_ORDER:
        sub = per_q[per_q["encoder"] == enc]
        lib_v = sub.loc[sub["ideology"] == "liberal", "lean"].to_numpy()
        con_v = sub.loc[sub["ideology"] == "conservative", "lean"].to_numpy()
        stats = {"encoder": enc, "n_lib": len(lib_v), "n_con": len(con_v)}
        for ide, vals in [("liberal", lib_v), ("conservative", con_v)]:
            m, lo, hi = _bootstrap_ci(vals)
            rows.append({"encoder": enc, "ideology": ide, "n": len(vals),
                         "mean_lean": m, "ci_low": lo, "ci_high": hi})
            key = "lib" if ide == "liberal" else "con"
            stats.update({f"mean_{key}": m, f"ci_low_{key}": lo, f"ci_high_{key}": hi})
        if len(lib_v) >= 10 and len(con_v) >= 10:
            _, p = scipy_stats.mannwhitneyu(con_v, lib_v, alternative="two-sided")
            pvals[enc] = float(p)
            draws = np.empty(B_BOOT)
            for i in range(B_BOOT):
                draws[i] = (con_v[gap_rng.integers(0, len(con_v), len(con_v))].mean()
                            - lib_v[gap_rng.integers(0, len(lib_v), len(lib_v))].mean())
            stats.update({
                "gap_con_minus_lib": float(con_v.mean() - lib_v.mean()),
                "gap_ci_low": float(np.quantile(draws, 0.025)),
                "gap_ci_high": float(np.quantile(draws, 0.975)),
                "mwu_p": float(p),
            })
        gaps.append(stats)
    return pd.DataFrame(rows), pvals, pd.DataFrame(gaps)


def _draw_synth(ax) -> None:
    bm = pd.concat(
        [_gold_filtered_bias(DENSE_RUN), _gold_filtered_bias(BM25_RUN)],
        ignore_index=True,
    )
    means = _per_stance_lean_filtered(bm, k=K).set_index(["encoder", "stance"])
    pvals = _synth_pvals()

    offset = 0.18
    yticks, yticklabels = [], []
    for i, enc in enumerate(DISPLAY_ORDER):
        ypos = len(DISPLAY_ORDER) - 1 - i
        yticks.append(ypos)
        yticklabels.append(DISPLAY_NAMES[enc] + _stars(pvals.get(enc)))
        for stance, color, dy in [
            ("left", LEFT_COLOR, +offset),
            ("right", RIGHT_COLOR, -offset),
        ]:
            if (enc, stance) not in means.index:
                continue
            r = means.loc[(enc, stance)]
            m, lo, hi = r["mean"], r["ci_low"], r["ci_high"]
            ax.errorbar(m, ypos + dy, xerr=[[m - lo], [hi - m]],
                        fmt="o", color=color, markersize=4.5,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.5, elinewidth=1.1, capsize=2.0,
                        capthick=0.8, zorder=3)
    ax.axvline(0, color="#333", linewidth=0.7, linestyle="--", zorder=0)
    for ypos in yticks:
        ax.axhline(ypos, color="#EEE", linewidth=0.4, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7.5)
    ax.set_xlabel(r"Mean Retrieval Lean ($\%R-\%L$)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(-0.6, len(DISPLAY_ORDER) - 0.3)
    ax.set_title(r"(a) AllSides$_{\mathrm{synth}}$", fontsize=9, pad=4)

    # Per-row n column (single paired-pair count per encoder).
    n_pairs = (means.reset_index()
                    .pivot_table(index="encoder", columns="stance",
                                 values="n_cells", aggfunc="first"))
    XN = 1.09
    ax.text(XN, len(DISPLAY_ORDER) - 0.3, "n",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="#333", ha="center", va="top")
    for i, enc in enumerate(DISPLAY_ORDER):
        ypos = len(DISPLAY_ORDER) - 1 - i
        if enc in n_pairs.index:
            n_e = int(n_pairs.loc[enc, "left"]) if "left" in n_pairs.columns else 0
            ax.text(XN, ypos, f"{n_e:,}",
                    transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color="#666",
                    ha="center", va="center")


def _draw_real(ax) -> pd.DataFrame:
    summary, pvals, gaps = _reddit_summary()

    offset = 0.18
    yticks, yticklabels = [], []
    for i, enc in enumerate(DISPLAY_ORDER):
        ypos = len(DISPLAY_ORDER) - 1 - i
        yticks.append(ypos)
        yticklabels.append(DISPLAY_NAMES[enc] + _stars(pvals.get(enc)))
        for ide, color, dy in [
            ("liberal", LEFT_COLOR, +offset),
            ("conservative", RIGHT_COLOR, -offset),
        ]:
            row = summary[(summary["encoder"] == enc) & (summary["ideology"] == ide)].iloc[0]
            m, lo, hi = row["mean_lean"], row["ci_low"], row["ci_high"]
            ax.errorbar(m, ypos + dy, xerr=[[m - lo], [hi - m]],
                        fmt="o", color=color, markersize=4.5,
                        markerfacecolor=color, markeredgecolor="white",
                        markeredgewidth=0.5, elinewidth=1.1, capsize=2.0,
                        capthick=0.8, zorder=3)
    for ypos in yticks:
        ax.axhline(ypos, color="#EEE", linewidth=0.4, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7.5)
    ax.set_xlabel(r"Mean Retrieval Lean ($\%R-\%L$)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(-0.6, len(DISPLAY_ORDER) - 0.3)
    # Allow some right padding (BM25's conservative marker is near 0).
    xs = pd.concat([summary["ci_low"], summary["ci_high"]])
    ax.set_xlim(xs.min() - 0.01, max(0.02, xs.max() + 0.01))
    ax.axvline(0, color="#333", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_title(r"(b) Reddit$_{\mathrm{nat}}$", fontsize=9, pad=4)

    # Per-row n column: colored lib_n / con_n with a gray slash.
    # XL,XR positioned so the visual left margin from the right spine
    # matches panel (a)'s XN (center of single number ~1.09); since
    # lib_n is right-anchored at XL, we bump XL outward to compensate
    # for the 3-digit text width that extends leftward from XL.
    XL, XSLASH, XR = 1.100, 1.120, 1.140
    ax.text((XL + XR) / 2, len(DISPLAY_ORDER) - 0.3, "n",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="#333", ha="center", va="top")
    for i, enc in enumerate(DISPLAY_ORDER):
        ypos = len(DISPLAY_ORDER) - 1 - i
        lib_n = int(summary[(summary["encoder"] == enc) & (summary["ideology"] == "liberal")].iloc[0]["n"])
        con_n = int(summary[(summary["encoder"] == enc) & (summary["ideology"] == "conservative")].iloc[0]["n"])
        ax.text(XL, ypos, f"{lib_n}",
                transform=ax.get_yaxis_transform(),
                fontsize=6.5, color=LEFT_COLOR, ha="right", va="center")
        ax.text(XSLASH, ypos, "/",
                transform=ax.get_yaxis_transform(),
                fontsize=6.5, color="#666", ha="center", va="center")
        ax.text(XR, ypos, f"{con_n}",
                transform=ax.get_yaxis_transform(),
                fontsize=6.5, color=RIGHT_COLOR, ha="left", va="center")
    return gaps


def main() -> None:
    apply_rc()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(7.2, 2.4))
    _draw_synth(ax_left)
    gaps = _draw_real(ax_right)
    gaps.to_csv(REDDIT_SUMMARY, index=False)
    print(f"wrote {REDDIT_SUMMARY}")

    # Legend is dropped; left/right colors are defined in the LaTeX
    # preamble (\leftq, \rightq macros) and used in the caption itself.
    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0], w_pad=1.4)
    out_base = REPO / "figs" / "output" / "political_combined"
    save_both(fig, str(out_base))
    print(f"wrote {out_base}.pdf and .png")


if __name__ == "__main__":
    main()
