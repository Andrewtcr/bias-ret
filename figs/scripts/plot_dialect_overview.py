"""Combined HCMagic AAL-vs-WME figure: synth-paired + real-unpaired forests.

Builds a single two-subplot figure:
the HCMagic synth paired MRR forest and the HCMagic_nat real-arm unpaired
MRR forest.

LEFT subplot:  mean MRR for WME-real vs avg(AAL_synth1/2/3) on the
               paired synth arm (n=4,999 paired rows per encoder).
RIGHT subplot: mean MRR for natural WME (n=659) vs natural AAL
               (n=647) on the unpaired real arm.

Both reuse cluster_bootstrap_mean for per-encoder CIs and pull per-encoder
significance from `hcmagic_synth_paired.csv` / `hcmagic_real_paired.csv`
(filter=unfiltered, metric=mrr) to drive the asterisks on the y-tick labels.

Output: figs/output/aalwme_combined.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "figs" / "scripts"))

from src.stats import cluster_bootstrap_mean  # noqa: E402
from _style import ENCODER_DISPLAY, ENCODER_ORDER, apply_rc, save_both  # noqa: E402

# Dialect colors: WME = orange, AAL = purple (matches _style.DIALECT_COLOR).
WME_COLOR = "#EE7733"   # orange
AAL_COLOR = "#AA3377"   # purple

RES = REPO / "hcmagic" / "results"
ENCS = ENCODER_ORDER + ["bm25"]


def _display(enc: str) -> str:
    return ENCODER_DISPLAY.get(enc, "BM25" if enc == "bm25" else enc)


# ---------------------------------------------------------------------------
# Significance helpers
# ---------------------------------------------------------------------------
def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _p_lookup_synth() -> dict[str, float]:
    """Per-encoder paired p-value on the synth arm (mrr, unfiltered).
    Uses wilcoxon_p when finite; otherwise falls back to perm_p."""
    df = pd.read_csv(RES / "hcmagic_synth_paired.csv")
    sub = df[(df["metric"] == "mrr")
             & (df["filter"] == "unfiltered")
             & (df["comparison"] == "wme_vs_aal_synth_mean")]
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        p = r["wilcoxon_p"]
        if not np.isfinite(p):
            p = r["perm_p"]
        out[r["encoder"]] = float(p)
    return out


def _p_lookup_real() -> dict[str, float]:
    """Per-encoder unpaired p-value on the real arm (mrr, unfiltered).
    The real-arm CSV stores the unpaired MWU result under
    comparison=='wme_real_vs_aal_real_unpaired' in perm_p (wilcoxon_p NaN
    because the groups are different sizes — 659 vs 647)."""
    df = pd.read_csv(RES / "hcmagic_real_paired.csv")
    sub = df[(df["metric"] == "mrr")
             & (df["filter"] == "unfiltered")
             & (df["comparison"] == "wme_real_vs_aal_real_unpaired")]
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        p = r["perm_p"]
        if not np.isfinite(p):
            p = r["wilcoxon_p"]
        out[r["encoder"]] = float(p)
    return out


# ---------------------------------------------------------------------------
# Per-arm CIs (cluster bootstrap on per-query MRR)
# ---------------------------------------------------------------------------
def _avg_aal_per_query(df: pd.DataFrame, metric: str) -> pd.Series:
    sub = df[df["variant"].str.startswith("aal_synth")]
    pivot = sub.pivot_table(index="query_id", columns="variant",
                            values=metric, aggfunc="mean")
    return pivot.mean(axis=1)


def _ci_synth(enc: str, variant_or_avg: str
              ) -> tuple[float, float, float, int]:
    df = pd.read_csv(RES / enc / "hcmagic_synth_per_query_metrics.csv")
    if variant_or_avg == "aal_synth_avg":
        series = _avg_aal_per_query(df, "mrr")
        vals = series.to_numpy()
        qids = series.index.to_numpy()
    else:
        sub = df[df["variant"] == variant_or_avg]
        vals = sub["mrr"].to_numpy()
        qids = sub["query_id"].to_numpy()
    m, lo, hi = cluster_bootstrap_mean(vals, qids, n_boot=5000, seed=42)
    return m, lo, hi, len(vals)


def _ci_real(enc: str, variant: str) -> tuple[float, float, float, int]:
    df = pd.read_csv(RES / enc / "hcmagic_real_per_query_metrics.csv")
    sub = df[df["variant"] == variant]
    vals = sub["mrr"].to_numpy()
    qids = sub["query_id"].to_numpy()
    m, lo, hi = cluster_bootstrap_mean(vals, qids, n_boot=5000, seed=42)
    return m, lo, hi, len(vals)


# ---------------------------------------------------------------------------
# Subplot renderers
# ---------------------------------------------------------------------------
def _draw_forest(ax, rows_by_enc: dict[str, dict],
                 p_by_enc: dict[str, float],
                 right_n_format,  # callable(ax, ypos, row) -> None
                 title: str, xlabel: str) -> None:
    """Shared forest renderer.

    rows_by_enc[enc] = {
        "wme": (m, lo, hi, n),
        "aal": (m, lo, hi, n),
    }
    right_n_format(ax, ypos, row) draws the n column for one encoder row:
    LEFT draws a single gray paired n (e.g. "4,999"); RIGHT draws an
    orange/gray/purple "n_w / n_a" triplet.
    """
    offset = 0.18
    yticks, yticklabels = [], []
    for i, enc in enumerate(ENCS):
        ypos = len(ENCS) - 1 - i
        yticks.append(ypos)
        star = _stars(p_by_enc.get(enc, float("nan")))
        yticklabels.append(_display(enc) + star)
        wme = rows_by_enc[enc]["wme"]
        aal = rows_by_enc[enc]["aal"]
        for (m, lo, hi, _n), color, dy in [
            (wme, WME_COLOR, +offset),
            (aal, AAL_COLOR, -offset),
        ]:
            ax.errorbar(
                m, ypos + dy, xerr=[[m - lo], [hi - m]],
                fmt="o", color=color, markersize=5.0,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, elinewidth=1.1, capsize=2.0,
                capthick=0.8, zorder=3,
            )
        right_n_format(ax, ypos, rows_by_enc[enc])
    for ypos in yticks:
        ax.axhline(ypos, color="#EEE", linewidth=0.4, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_ylim(-0.6, len(ENCS) - 0.3)
    ax.set_title(title, fontsize=9)


_XN_LEFT = 1.09
# XN_W positioned so the visual left margin from the right spine
# matches panel (a)'s _XN_LEFT (center of single number); since wme_n
# is right-anchored at _XN_W, we bump it outward to compensate for the
# 3-digit text width that extends leftward from _XN_W.
_XN_W, _XN_SLASH, _XN_A = 1.100, 1.115, 1.130


def _n_left(ax, ypos: float, row: dict) -> None:
    n = int(row["wme"][3])  # paired -> single n
    ax.text(_XN_LEFT, ypos, f"{n:,}",
            transform=ax.get_yaxis_transform(),
            fontsize=6.5, color="#666", ha="center", va="center")


def _n_right(ax, ypos: float, row: dict) -> None:
    """Orange n_w / gray slash / purple n_a, stacked horizontally to the
    right of the axes. Uses ha='right' / 'center' / 'left' alignment so
    digit widths line up without overlap."""
    n_w = int(row["wme"][3])
    n_a = int(row["aal"][3])
    trans = ax.get_yaxis_transform()
    # WME n: right-aligned to the slash; AAL n: left-aligned past the slash.
    ax.text(_XN_W, ypos, f"{n_w}", transform=trans,
            fontsize=6.5, color=WME_COLOR, ha="right", va="center")
    ax.text(_XN_SLASH, ypos, "/", transform=trans,
            fontsize=6.5, color="#888", ha="center", va="center")
    ax.text(_XN_A, ypos, f"{n_a}", transform=trans,
            fontsize=6.5, color=AAL_COLOR, ha="left", va="center")


def _draw_n_header(ax, x: float) -> None:
    """Centered 'n' header just inside the top spine of the subplot
    (top of text touches the spine) — matches Fig 2."""
    ax.text(x, len(ENCS) - 0.3, "n",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="#333", ha="center", va="top")


def _build_left_rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for enc in ENCS:
        out[enc] = {
            "wme": _ci_synth(enc, "wme_real"),
            "aal": _ci_synth(enc, "aal_synth_avg"),
        }
    return out


def _build_right_rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for enc in ENCS:
        out[enc] = {
            "wme": _ci_real(enc, "wme_real"),
            "aal": _ci_real(enc, "aal_real"),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    apply_rc()

    p_synth = _p_lookup_synth()
    p_real = _p_lookup_real()
    print("synth p-values:", {e: f"{p:.2e}" for e, p in p_synth.items()})
    print("real  p-values:", {e: f"{p:.2e}" for e, p in p_real.items()})

    print("computing left (synth) CIs...")
    left_rows = _build_left_rows()
    print("computing right (real) CIs...")
    right_rows = _build_right_rows()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax_l, ax_r = axes

    xlabel = r"Mean MRR"
    _draw_forest(
        ax_l, left_rows, p_synth, _n_left,
        title=r"(a) HCMagic$_{\mathrm{synth}}$",
        xlabel=xlabel,
    )
    _draw_n_header(ax_l, _XN_LEFT)
    _draw_forest(
        ax_r, right_rows, p_real, _n_right,
        title=r"(b) HCMagic$_{\mathrm{nat}}$",
        xlabel=xlabel,
    )
    _draw_n_header(ax_r, _XN_SLASH)

    # Legend is dropped; WME/AAL colors are defined in the LaTeX
    # preamble (\wmeq, \aalq macros) and used in the caption itself.
    fig.tight_layout(rect=[0.0, 0.0, 0.97, 1.0], w_pad=1.4)
    # Extra right padding for n column on each subplot (axes-relative).
    # Right subplot needs slightly more room because the n column has
    # three text segments (n_w / n_a) rather than one ("4,999").
    for ax, shrink in ((ax_l, 0.93), (ax_r, 0.88)):
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0,
                         pos.width * shrink, pos.height])

    out_base = REPO / "figs" / "output" / "aalwme_combined"
    save_both(fig, str(out_base))
    print(f"wrote {out_base}.{{pdf,png}}")


if __name__ == "__main__":
    main()
