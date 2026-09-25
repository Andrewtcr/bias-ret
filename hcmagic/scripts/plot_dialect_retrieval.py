"""Build HealthCareMagic retrieval figures from committed per-query results.

Inputs: hcmagic/results/{encoder}/hcmagic_{synth,real}_per_query_metrics.csv
and hcmagic/results/hcmagic_real_paired.csv for translation significance.

Outputs under figs/output/ (PDF and PNG):
  aalwme_synth_ksweep               synthetic-pair recall@k
  aalwme_real_unpaired_ksweep        natural AAL/WME recall@k
  aalwme_translation_forest         paired translation MRR
  aalwme_translation_ksweep         paired translation recall@k

Query-level bootstrap confidence intervals use 5,000 replicates with seed 42.
They are computed in worker processes, cached in memory, and plotted
sequentially with matplotlib.
"""
from __future__ import annotations

import multiprocessing as mp
import os
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
from _style import DIALECT_COLOR, ENCODER_DISPLAY, ENCODER_ORDER, apply_rc, save_both  # noqa: E402

RES = REPO / "hcmagic" / "results"
OUT = REPO / "figs" / "output"; OUT.mkdir(parents=True, exist_ok=True)

# Use the shared dialect palette across all figures.
WME_COLOR = DIALECT_COLOR["wme"]    # orange
AAL_COLOR = DIALECT_COLOR["aal"]    # purple

K_GRID = [1, 5, 10, 20, 50, 100]   # k grid for the recall@k sweeps

# CSV kinds used as part of the cache key. Maps short label -> filename.
_CSV_KIND = {
    "synth": "hcmagic_synth_per_query_metrics.csv",
    "real":  "hcmagic_real_per_query_metrics.csv",
}

# Cache populated by `_precompute_all_cis`. Key is
#   (enc, csv_kind, variant, metric) -> (m, lo, hi, n)
_CI_CACHE: dict[tuple[str, str, str, str], tuple[float, float, float, int]] = {}


def _per_query_csv(enc: str, kind: str = "synth") -> Path:
    return RES / enc / _CSV_KIND[kind]


# BM25 is appended to the encoder list when its results exist on disk.
def _enc_with_results(kind: str = "synth") -> list[str]:
    out = [e for e in ENCODER_ORDER if _per_query_csv(e, kind).exists()]
    if _per_query_csv("bm25", kind).exists():
        out.append("bm25")
    return out


# Display names: ENCODER_DISPLAY for the 5 dense, "BM25" for the lex baseline.
def _display(enc: str) -> str:
    return ENCODER_DISPLAY.get(enc, "BM25" if enc == "bm25" else enc)


def _avg_aal_per_query(df: pd.DataFrame, metric: str) -> pd.Series:
    """Average a metric across aal_synth1/2/3 within query_id."""
    sub = df[df["variant"].str.startswith("aal_synth")]
    pivot = sub.pivot_table(index="query_id", columns="variant",
                            values=metric, aggfunc="mean")
    return pivot.mean(axis=1)


def _series_for(df: pd.DataFrame, variant: str,
                metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, qids) for a (variant, metric) slice.

    `variant` can be a real variant name (e.g. "wme_real", "aal_real",
    "aal_synth1", "trans_aal") OR the synthetic "aal_synth_avg" which is
    averaged across aal_synth1/2/3.
    """
    if variant == "aal_synth_avg":
        series = _avg_aal_per_query(df, metric)
        return series.to_numpy(), series.index.to_numpy()
    sub = df[df["variant"] == variant]
    return sub[metric].to_numpy(), sub["query_id"].to_numpy()


def _ci_for(df: pd.DataFrame, variant_or_avg: str, metric: str,
            enc: str | None = None,
            csv_kind: str = "synth") -> tuple[float, float, float, int]:
    """Returns (mean, ci_lo, ci_hi, n) for the variant.

    Consults `_CI_CACHE` first (populated by `_precompute_all_cis`). Falls back
    to live computation when the key is missing.
    """
    if enc is not None:
        cached = _CI_CACHE.get((enc, csv_kind, variant_or_avg, metric))
        if cached is not None:
            return cached
    vals, qids = _series_for(df, variant_or_avg, metric)
    m, lo, hi = cluster_bootstrap_mean(vals, qids, n_boot=5000, seed=42)
    return m, lo, hi, len(vals)


# ---------------------------------------------------------------------------
# Parallel bootstrap precompute
# ---------------------------------------------------------------------------
def _bootstrap_one(args: tuple) -> tuple[tuple[str, str, str, str],
                                         float, float, float, int]:
    """Worker: load CSV, slice, run cluster_bootstrap_mean.

    Args is the cache key: (enc, csv_kind, variant, metric).
    Returns (key, m, lo, hi, n).
    """
    enc, csv_kind, variant, metric = args
    csv = RES / enc / _CSV_KIND[csv_kind]
    df = pd.read_csv(csv)
    vals, qids = _series_for(df, variant, metric)
    m, lo, hi = cluster_bootstrap_mean(vals, qids, n_boot=5000, seed=42)
    return (enc, csv_kind, variant, metric), m, lo, hi, len(vals)


def _enumerate_tuples(encs_synth: list[str],
                       encs_real: list[str]) -> list[tuple]:
    """Enumerate every (enc, csv_kind, variant, metric) needed by any figure
    builder."""
    tuples: list[tuple] = []

    # make_paired_ksweep: synth CSV, recall@k, both variants.
    for enc in encs_synth:
        for k in K_GRID:
            for v in ("wme_real", "aal_synth_avg"):
                tuples.append((enc, "synth", v, f"recall@{k}"))

    # Translation forest: original AAL and translated WME MRR.
    for enc in encs_real:
        for v in ("aal_real", "trans_aal"):
            tuples.append((enc, "real", v, "mrr"))

    # make_natural_ksweep + make_translation_ksweep: real CSV,
    # recall@k, variants {wme_real, aal_real, trans_aal}.
    for enc in encs_real:
        for k in K_GRID:
            for v in ("wme_real", "aal_real", "trans_aal"):
                tuples.append((enc, "real", v, f"recall@{k}"))
    return tuples


def _precompute_all_cis(encs_synth: list[str], encs_real: list[str]) -> None:
    """Fan out every needed cluster_bootstrap_mean to a multiprocessing.Pool
    and populate the global `_CI_CACHE`."""
    tasks = _enumerate_tuples(encs_synth, encs_real)
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    n_proc = min(n_proc, len(tasks))
    print(f"precompute: {len(tasks)} bootstraps across {n_proc} workers")
    # spawn keeps workers clean (no inherited matplotlib state).
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc) as pool:
        for key, m, lo, hi, n in pool.imap_unordered(_bootstrap_one, tasks,
                                                     chunksize=4):
            _CI_CACHE[key] = (m, lo, hi, n)
    print(f"precompute: cached {len(_CI_CACHE)} CIs")


def make_paired_ksweep() -> None:
    """K-sweep: unfiltered recall@k vs k per encoder (full n=4999)."""
    encs = _enc_with_results()

    rows = []
    for enc in encs:
        df = pd.read_csv(_per_query_csv(enc))
        for k in K_GRID:
            metric_col = f"recall@{k}"
            for variant_label in ("wme_real", "aal_synth_avg"):
                m, lo, hi, _ = _ci_for(df, variant_label, metric_col,
                                       enc=enc, csv_kind="synth")
                rows.append({"encoder": enc, "variant": variant_label,
                             "k": k, "mean": m, "ci_lo": lo, "ci_hi": hi})
    long = pd.DataFrame(rows)

    # Per-encoder ylim = tight zoom around the data range, so CIs are visible.
    # sharey=False; sharex=True (k values are common).
    # 3x2 to fit up to 6 encoders (5 dense + BM25).
    fig, axes = plt.subplots(3, 2, figsize=(3.4, 5.0),
                             sharey=False, sharex=True)
    axes = axes.flatten()
    for ax, enc in zip(axes, encs):
        _ksweep_panel(ax, long[long["encoder"] == enc], WME_COLOR, AAL_COLOR,
                      "wme_real", "aal_synth_avg", _display(enc))

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=WME_COLOR,
                   markersize=4, label="WME"),
        plt.Line2D([0], [0], marker="o", linestyle="-", color=AAL_COLOR,
                   markersize=4, label=r"avg AAL$_\mathrm{synth}$"),
    ]
    fig.tight_layout(rect=[0.04, 0.14, 1.0, 1.0])
    fig.text(0.52, 0.095, "top-$k$ cutoff", ha="center", fontsize=8)
    fig.text(0.005, 0.55, "mean recall@$k$",
             va="center", rotation="vertical", fontsize=8)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=7)
    save_both(fig, str(OUT / "aalwme_synth_ksweep"))
    print("wrote aalwme_synth_ksweep")


def _per_variant_ci_real(encoder: str, variant: str,
                          metric: str = "mrr") -> tuple[float, float, float, int]:
    """Bootstrap CI for mean of `metric` on the real-arm per-query CSV
    (variant in {aal_real, wme_real, trans_aal}).

    Consults `_CI_CACHE` first; falls back to live compute when missing.
    """
    cached = _CI_CACHE.get((encoder, "real", variant, metric))
    if cached is not None:
        return cached
    per_q = RES / encoder / _CSV_KIND["real"]
    df = pd.read_csv(per_q)
    sub = df[df["variant"] == variant]
    vals = sub[metric].to_numpy()
    qids = sub["query_id"].to_numpy()
    m, lo, hi = cluster_bootstrap_mean(vals, qids, n_boot=5000, seed=42)
    return m, lo, hi, len(vals)


def _stars_trans(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return " ***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return " *"
    return ""


def _p_lookup_trans() -> dict[str, float]:
    """Per-encoder paired-Wilcoxon p for trans_wme vs aal_real on MRR."""
    df = pd.read_csv(RES / "hcmagic_real_paired.csv")
    sub = df[(df["metric"] == "mrr")
             & (df["filter"] == "unfiltered")
             & (df["comparison"] == "trans_wme_vs_aal_real_paired")]
    out: dict[str, float] = {}
    for _, r in sub.iterrows():
        p = r["wilcoxon_p"]
        if not np.isfinite(p):
            p = r["perm_p"]
        out[r["encoder"]] = float(p)
    return out


def make_translation_forest() -> None:
    """Translation-mitigation paired forest: aal_real vs trans_aal on MRR.
    Same query content, dialect-translated via GPT.
    Tested via paired Wilcoxon. Answers: does back-translation help?

    Colors are identified in the figure caption. A single-value 'n' header is
    top-aligned with the top spine, values centered without 'n=' prefix,
    significance asterisks on y-tick labels.
    """
    apply_rc()
    encs = [e for e in ENCODER_ORDER if (RES / e / "hcmagic_real_per_query_metrics.csv").exists()]
    if (RES / "bm25" / "hcmagic_real_per_query_metrics.csv").exists():
        encs.append("bm25")

    p_trans = _p_lookup_trans()

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    offset = 0.18
    yticks, yticklabels = [], []
    XN = 1.09
    for i, enc in enumerate(encs):
        ypos = len(encs) - 1 - i
        yticks.append(ypos)
        yticklabels.append(_display(enc) + _stars_trans(p_trans.get(enc, float("nan"))))
        for variant, color, dy in [
            ("trans_aal", WME_COLOR, +offset),
            ("aal_real",  AAL_COLOR, -offset),
        ]:
            m, lo, hi, n = _per_variant_ci_real(enc, variant)
            ax.errorbar(
                m, ypos + dy, xerr=[[m - lo], [hi - m]],
                fmt="o", color=color, markersize=5.5,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, elinewidth=1.3, capsize=2.4,
                capthick=0.9, zorder=3,
            )
        n_e = _per_variant_ci_real(enc, "aal_real")[3]
        ax.text(XN, ypos, f"{n_e:,}",
                transform=ax.get_yaxis_transform(),
                fontsize=7.0, color="#666", ha="center", va="center")
    for ypos in yticks:
        ax.axhline(ypos, color="#EEE", linewidth=0.4, zorder=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_xlabel(r"Mean MRR", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_ylim(-0.6, len(encs) - 0.3)

    # Centered 'n' header, top of text aligned with bottom of top spine.
    ax.text(XN, len(encs) - 0.3, "n",
            transform=ax.get_yaxis_transform(),
            fontsize=7, color="#333", ha="center", va="top")

    fig.tight_layout()
    save_both(fig, str(OUT / "aalwme_translation_forest"))
    print("wrote aalwme_translation_forest")


def _ksweep_panel(ax, sub: pd.DataFrame, color_a: str, color_b: str,
                  variant_a: str, variant_b: str, title: str) -> list[float]:
    """Shared k-sweep panel renderer. Returns list of plotted values for ylim."""
    vals_all: list[float] = []
    for variant, color in [(variant_a, color_a), (variant_b, color_b)]:
        s = sub[sub["variant"] == variant].sort_values("k")
        if s.empty:
            continue
        ax.errorbar(s["k"], s["mean"],
                    yerr=[s["mean"] - s["ci_lo"], s["ci_hi"] - s["mean"]],
                    fmt="o-", color=color, markersize=3.5,
                    elinewidth=0.7, capsize=1.5, capthick=0.5)
        vals_all.extend(s["ci_lo"].tolist() + s["ci_hi"].tolist())
    if vals_all:
        lo, hi = min(vals_all), max(vals_all)
        pad = max(0.005, (hi - lo) * 0.10)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xscale("log")
    ax.set_xticks(K_GRID)
    ax.set_xticklabels([str(k) for k in K_GRID], fontsize=6.5)
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    return vals_all


def _generic_recall_ksweep(per_query_csv_name: str, variant_a: str, variant_b: str,
                            out_path: Path, label_a: str, label_b: str,
                            ) -> None:
    """Build a 3x2 recall@k vs k panel grid for two variants per encoder."""
    encs = [e for e in ENCODER_ORDER if (RES / e / per_query_csv_name).exists()]
    if (RES / "bm25" / per_query_csv_name).exists():
        encs.append("bm25")

    # Determine csv_kind from filename so we can hit the cache.
    csv_kind = next((k for k, v in _CSV_KIND.items()
                     if v == per_query_csv_name), None)

    rows = []
    for enc in encs:
        df = None  # lazy-load if cache misses
        for k in K_GRID:
            metric_col = f"recall@{k}"
            for v in (variant_a, variant_b):
                cached = (_CI_CACHE.get((enc, csv_kind, v, metric_col))
                          if csv_kind is not None else None)
                if cached is not None:
                    m, lo, hi, _n = cached
                else:
                    if df is None:
                        df = pd.read_csv(RES / enc / per_query_csv_name)
                    if metric_col not in df.columns:
                        continue
                    sub_v = df[df["variant"] == v]
                    if sub_v.empty:
                        continue
                    vals = sub_v[metric_col].to_numpy()
                    qids = sub_v["query_id"].to_numpy()
                    m, lo, hi = cluster_bootstrap_mean(vals, qids,
                                                       n_boot=5000, seed=42)
                rows.append({"encoder": enc, "variant": v, "k": k,
                             "mean": m, "ci_lo": lo, "ci_hi": hi})
    long = pd.DataFrame(rows)

    fig, axes = plt.subplots(3, 2, figsize=(3.4, 5.0),
                             sharey=False, sharex=True)
    axes = axes.flatten()
    for ax, enc in zip(axes, encs):
        sub = long[long["encoder"] == enc]
        _ksweep_panel(ax, sub, WME_COLOR, AAL_COLOR, variant_a, variant_b,
                      _display(enc))

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="-", color=WME_COLOR,
                   markersize=4, label=label_a),
        plt.Line2D([0], [0], marker="o", linestyle="-", color=AAL_COLOR,
                   markersize=4, label=label_b),
    ]
    fig.tight_layout(rect=[0.04, 0.14, 1.0, 1.0])
    fig.text(0.52, 0.095, "top-$k$ cutoff", ha="center", fontsize=8)
    fig.text(0.005, 0.55, "mean recall@$k$",
             va="center", rotation="vertical", fontsize=8)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=7)
    save_both(fig, str(out_path))
    print(f"wrote {out_path.name}")


def make_natural_ksweep() -> None:
    _generic_recall_ksweep(
        "hcmagic_real_per_query_metrics.csv",
        variant_a="wme_real", variant_b="aal_real",
        out_path=OUT / "aalwme_real_unpaired_ksweep",
        label_a="natural WME (n=659)", label_b="natural AAL (n=647)",
    )


def make_translation_ksweep() -> None:
    _generic_recall_ksweep(
        "hcmagic_real_per_query_metrics.csv",
        variant_a="trans_aal", variant_b="aal_real",
        out_path=OUT / "aalwme_translation_ksweep",
        label_a="GPT-translated WME", label_b="original AAL",
    )


def main() -> None:
    apply_rc()
    encs_synth = _enc_with_results("synth")
    encs_real = _enc_with_results("real")
    _precompute_all_cis(encs_synth, encs_real)
    make_paired_ksweep()
    make_natural_ksweep()
    make_translation_forest()
    make_translation_ksweep()


if __name__ == "__main__":
    main()
