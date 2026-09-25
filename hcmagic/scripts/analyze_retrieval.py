"""Aggregate HealthCareMagic retrieval metrics and statistical comparisons.

HCMagic-synth: paired WME-minus-mean-AAL MRR differences over all queries, with
Wilcoxon tests, sign-flip permutation tests, and query-level bootstrap
confidence intervals.

HCMagic-nat: paired AAL-to-WME translation comparisons and unpaired natural
AAL/WME comparisons. Unpaired comparisons use Mann–Whitney U tests and a
two-group percentile bootstrap confidence interval that resamples each group
independently.

Outputs under hcmagic/results/:
  hcmagic_synth_paired.csv
  hcmagic_real_paired.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.io import ensure_dir, load_yaml  # noqa: E402
from src.stats import cluster_bootstrap_mean, signflip_permutation_p  # noqa: E402
from hcmagic.lib import RES, model_slug  # noqa: E402


def _signflip_p(diff: np.ndarray, n_perm: int = 50000, seed: int = 42) -> float:
    """Wrapper: return two-sided sign-flip permutation p (or NaN for empty)."""
    if len(diff) == 0:
        return float("nan")
    perm = signflip_permutation_p(diff, n_perm=n_perm, seed=seed)
    return float(perm["p_two_sided"])


HCSYNTH_METRICS = ["mrr"]


def hcmagic_synth_paired(df: pd.DataFrame, encoder: str) -> list[dict]:
    """Per-encoder paired Δ(WME_real - mean(AAL_synth1/2/3)) per metric.

    Variants in df: wme_real, aal_synth1, aal_synth2, aal_synth3. All share
    query_id per row, so pairing is index-based. Uses all 4999 paired rows.
    """
    rows = []
    wme = df[df["variant"] == "wme_real"].set_index("query_id")
    aal_sub = df[df["variant"].str.startswith("aal_synth")]
    if wme.empty or aal_sub.empty:
        return rows
    aal_mean = (aal_sub.pivot_table(index="query_id", columns="variant",
                                    values=HCSYNTH_METRICS, aggfunc="mean")
                         .T.groupby(level=0).mean().T)   # avg across the 3 trials (pandas>=3 safe)
    common = wme.index.intersection(aal_mean.index)
    if len(common) == 0:
        return rows

    qids_arr = np.asarray(list(common))
    for metric in HCSYNTH_METRICS:
        w = wme.loc[common, metric].to_numpy()
        a = aal_mean.loc[common, metric].to_numpy()
        diff = w - a
        try:
            wilc = stats.wilcoxon(w, a, zero_method="wilcox",
                                  alternative="two-sided")
            wilc_p = float(wilc.pvalue)
        except ValueError:
            wilc_p = 1.0
        boot_m, boot_lo, boot_hi = cluster_bootstrap_mean(
            diff, qids_arr, n_boot=5000, seed=42,
        )
        rows.append({
            "encoder": encoder, "metric": metric, "filter": "unfiltered",
            "comparison": "wme_vs_aal_synth_mean",
            "wme_mean": float(w.mean()), "variant_mean": float(a.mean()),
            "delta": boot_m, "ci_low": boot_lo, "ci_high": boot_hi,
            "rel_drop_pct": 100.0 * boot_m / float(w.mean()) if w.mean() else float("nan"),
            "wilcoxon_p": wilc_p,
            "perm_p": _signflip_p(diff),
            "n_pairs": int(len(common)),
        })
    return rows


HCREAL_METRICS = ["mrr"]


def hcmagic_real_paired(df: pd.DataFrame, encoder: str,
                        unpaired_rng: np.random.Generator) -> list[dict]:
    """Per-encoder analysis of the real-arm (qrels-based, unified 100k corpus).

    Variants in df: aal_real, wme_real, trans_aal. Two comparisons:
      - PAIRED  aal_real vs trans_aal (same gold doc, dialect-only contrast)
      - UNPAIRED aal_real vs wme_real (different patients/conditions)

    Pairing for aal vs trans is via the numeric suffix of query_id:
      hcm_aal_K   <-> hcm_trans_K

    `unpaired_rng` draws the unpaired bootstrap; main() shares one stream
    across encoders.
    """
    rows = []
    aal   = df[df["variant"] == "aal_real"].copy()
    wme   = df[df["variant"] == "wme_real"].copy()
    trans = df[df["variant"] == "trans_aal"].copy()
    aal["aal_idx"]   = aal["query_id"].str.rsplit("_", n=1).str[-1].astype(int)
    trans["aal_idx"] = trans["query_id"].str.rsplit("_", n=1).str[-1].astype(int)

    # Paired aal vs trans (same gold; dialect-only).
    paired = aal.merge(trans, on="aal_idx", suffixes=("_aal", "_trans"))
    qids_arr = paired["aal_idx"].astype(str).to_numpy()
    for metric in HCREAL_METRICS:
        a = paired[f"{metric}_aal"].to_numpy()
        t = paired[f"{metric}_trans"].to_numpy()
        # Δ = trans - aal ("WME minus AAL"): positive = the WME translation scores higher.
        diff = t - a
        try:
            wilc = stats.wilcoxon(t, a, zero_method="wilcox", alternative="two-sided")
            wilc_p = float(wilc.pvalue)
        except ValueError:
            wilc_p = 1.0
        boot_m, boot_lo, boot_hi = cluster_bootstrap_mean(
            diff, qids_arr, n_boot=5000, seed=42,
        )
        rows.append({
            "encoder": encoder, "metric": metric, "filter": "unfiltered",
            "comparison": "trans_wme_vs_aal_real_paired",
            "wme_mean": float(t.mean()), "variant_mean": float(a.mean()),
            "delta": boot_m, "ci_low": boot_lo, "ci_high": boot_hi,
            "rel_drop_pct": 100.0 * boot_m / float(t.mean()) if t.mean() else float("nan"),
            "wilcoxon_p": wilc_p,
            "perm_p": _signflip_p(diff),
            "n_pairs": int(len(paired)),
        })

    # Unpaired aal_real vs wme_real (different content, between-group).
    for metric in HCREAL_METRICS:
        a_arr = aal[metric].to_numpy()
        w_arr = wme[metric].to_numpy()
        if len(a_arr) == 0 or len(w_arr) == 0:
            continue
        mwu = stats.mannwhitneyu(w_arr, a_arr, alternative="two-sided")
        # Two-group percentile bootstrap: resample each group independently.
        boots = np.empty(5000)
        for b in range(len(boots)):
            boots[b] = (w_arr[unpaired_rng.integers(0, len(w_arr), len(w_arr))].mean()
                        - a_arr[unpaired_rng.integers(0, len(a_arr), len(a_arr))].mean())
        rows.append({
            "encoder": encoder, "metric": metric, "filter": "unfiltered",
            "comparison": "wme_real_vs_aal_real_unpaired",
            "wme_mean": float(w_arr.mean()), "variant_mean": float(a_arr.mean()),
            "delta": float(w_arr.mean() - a_arr.mean()),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "rel_drop_pct": float("nan"),
            "wilcoxon_p": float("nan"),
            "perm_p": float(mwu.pvalue),   # MWU p in the perm_p slot
            "n_pairs": int(min(len(a_arr), len(w_arr))),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(RES)

    hcsynth_rows: list[dict] = []
    hcreal_rows: list[dict] = []
    # Include the configured dense encoders and any BM25 results.
    enc_slugs: list[str] = []
    for enc in cfg["retrieval"]["encoders"]:
        name = enc["name"] if isinstance(enc, dict) else enc
        enc_slugs.append(model_slug(name))
    for extra_slug in ("bm25",):
        if (RES / extra_slug).is_dir() and extra_slug not in enc_slugs:
            enc_slugs.append(extra_slug)
    # One seed-42 stream for the unpaired HCMagic-nat bootstrap, consumed in
    # encoder order (config order, then BM25); the paper's tab:hcmagic-nat-gap
    # CIs use this stream and order.
    unpaired_rng = np.random.default_rng(42)
    for slug in enc_slugs:
        result_dir = RES / slug

        # Natural-query relevance metrics.
        hcreal_csv = result_dir / "hcmagic_real_per_query_metrics.csv"
        if hcreal_csv.exists():
            df = pd.read_csv(hcreal_csv)
            hcreal_rows.extend(hcmagic_real_paired(df, slug, unpaired_rng))

        hcsynth_csv = result_dir / "hcmagic_synth_per_query_metrics.csv"
        if hcsynth_csv.exists():
            df = pd.read_csv(hcsynth_csv)
            hcsynth_rows.extend(hcmagic_synth_paired(df, slug))

    if hcsynth_rows:
        df = pd.DataFrame(hcsynth_rows)
        df.to_csv(out_dir / "hcmagic_synth_paired.csv", index=False)
        print(f"\n[analyze] wrote {out_dir / 'hcmagic_synth_paired.csv'}")
        print(df.round(4).to_string(index=False))

    if hcreal_rows:
        df = pd.DataFrame(hcreal_rows)
        df.to_csv(out_dir / "hcmagic_real_paired.csv", index=False)
        print(f"\n[analyze] wrote {out_dir / 'hcmagic_real_paired.csv'}")
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
