"""Per-row lex-residual decomposition for the dialect (AAL/WME) arm.

Replicates the per-cell Monroe-Colaresi-Quinn log-odds + OLS-residual
decomposition that the political arm uses
(allsides_synth/scripts/analyze_lexical_scores.py +
allsides_synth/scripts/analyze_lexical_regression.py), but applied to the
hcmagic_100k paired synth rows.

Model
-----
For each of the 4,999 paired questions r:
  - wme_real: the natural WME query (analogue of L)
  - aal_synth1..3: three AAL paraphrases (analogue of R, averaged)
We compute per-token MCQ log-odds asymmetry ζ_w with
  L-corpus = bag of tokens across all wme_real queries
  R-corpus = bag of tokens across all aal_synth{1,2,3} paraphrases
            (so each row contributes one WME doc and three AAL docs)
matched to the political setup: ALPHA=0.01 with
background-proportional smoothing, log-odds and ζ as in
Monroe-Colaresi-Quinn (2008).

Per-row asymmetry
-----------------
  lex_WME(r) = Σ_t∈tok(wme_real)  ζ_t
  lex_AAL(r) = mean_{i=1..3} Σ_t∈tok(aal_synth_i) ζ_t
  asym(r)    = lex_WME(r) − lex_AAL(r)
By construction, asym(r) is expected to be positive when the row's
AAL paraphrases use AAL-distinctive tokens (since those tokens have
ζ < 0, pulling lex_AAL down).

Per-row Δ
---------
  Δ(r) = MRR(wme_real) − mean(MRR(aal_synth1..3))
(matches `hcmagic/scripts/analyze_retrieval.py::hcmagic_synth_paired`.)

Per-encoder OLS
---------------
  Δ(r) = α + β · asym(r) + ε(r)
with HC1 robust SE — each row is its own unit, so HC1 is the natural
analogue of the political-arm cluster-robust SE on article_id (where
each cell maps to a unique article-frame). α is the residual Δ at zero
lex asymmetry; β · mean(asym) is the implied per-row lex contribution.

Outputs
-------
  hcmagic/results/lex_residual_summary.csv
  hcmagic/results/lex_residual_token_logodds.csv
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "hcmagic" / "results"
PAIRED_JSONL = ROOT / "hcmagic" / "data" / "hcmagic_100k" / "synth_paired.jsonl"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ALPHA = 0.01  # MCQ Dirichlet prior weight (matches political arm)

ENCODERS = (
    "bge-large-en-v1.5",
    "qwen3-embedding-8b",
    "llama-embed-nemotron-8b",
    "octen-embedding-8b",
    "text-embedding-3-large",
    "bm25",
)
AAL_KEYS = ("aal_synth1", "aal_synth2", "aal_synth3")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(str(t).lower())


def load_paired_rows() -> pd.DataFrame:
    rows = []
    with PAIRED_JSONL.open() as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    needed = {"qid", "wme_real", *AAL_KEYS}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"synth_paired.jsonl missing columns: {missing}")
    return df


def fit_mcq_logodds(wme_docs: list[list[str]], aal_docs: list[list[str]]) -> dict[str, float]:
    """MCQ informative-Dirichlet log-odds; signed positive = WME-distinctive."""
    wme_counts: Counter[str] = Counter()
    aal_counts: Counter[str] = Counter()
    for toks in wme_docs:
        wme_counts.update(toks)
    for toks in aal_docs:
        aal_counts.update(toks)

    all_terms = set(wme_counts) | set(aal_counts)
    nW = sum(wme_counts.values())
    nA = sum(aal_counts.values())
    bg = Counter()
    bg.update(wme_counts)
    bg.update(aal_counts)
    n_bg = nW + nA

    zeta_map: dict[str, float] = {}
    diag_rows = []
    for w in all_terms:
        yW = wme_counts.get(w, 0)
        yA = aal_counts.get(w, 0)
        alpha_w = ALPHA * (bg[w] / n_bg)  # background-proportional smoothing
        nW_w = yW + alpha_w
        nA_w = yA + alpha_w
        nW_not = (nW + ALPHA) - nW_w
        nA_not = (nA + ALPHA) - nA_w
        log_odds = np.log(nW_w / nW_not) - np.log(nA_w / nA_not)
        var = 1.0 / nW_w + 1.0 / nA_w
        zeta = log_odds / np.sqrt(var)
        zeta_map[w] = float(zeta)
        diag_rows.append({
            "token": w,
            "count_WME": yW,
            "count_AAL": yA,
            "log_odds": float(log_odds),
            "zeta": float(zeta),
        })
    diag = pd.DataFrame(diag_rows).sort_values(["zeta", "token"], ascending=[False, True])
    diag.to_csv(RES_DIR / "lex_residual_token_logodds.csv", index=False)
    return zeta_map


def per_row_lex(paired: pd.DataFrame, zeta_map: dict[str, float]) -> pd.DataFrame:
    """One row per qid with WME/AAL lexical scores, their difference, and token counts."""

    def score(toks: list[str]) -> float:
        return float(sum(zeta_map.get(t, 0.0) for t in toks))

    out = []
    for r in paired.itertuples(index=False):
        wme_toks = tokenize(getattr(r, "wme_real"))
        aal_toks_list = [tokenize(getattr(r, k)) for k in AAL_KEYS]
        lex_W = score(wme_toks)
        lex_A_each = [score(t) for t in aal_toks_list]
        lex_A = float(np.mean(lex_A_each))
        out.append({
            "qid": r.qid,
            "lex_WME": lex_W,
            "lex_AAL": lex_A,
            "lex_diff_wme_aal": lex_W - lex_A,
            "n_tok_wme": len(wme_toks),
            "n_tok_aal_mean": float(np.mean([len(t) for t in aal_toks_list])),
        })
    return pd.DataFrame(out)


def per_row_delta(encoder: str) -> pd.DataFrame:
    """One row per qid with Δ_mrr = MRR(wme_real) − mean MRR over 3 aal_synth."""
    csv = RES_DIR / encoder / "hcmagic_synth_per_query_metrics.csv"
    df = pd.read_csv(csv, usecols=["query_id", "variant", "mrr"])
    wme = df[df.variant == "wme_real"].set_index("query_id")["mrr"]
    aal = (df[df.variant.isin(AAL_KEYS)]
           .groupby("query_id")["mrr"].mean())
    common = wme.index.intersection(aal.index)
    out = pd.DataFrame({
        "qid": common,
        "mrr_wme": wme.loc[common].to_numpy(),
        "mrr_aal_mean": aal.loc[common].to_numpy(),
    })
    out["delta_mrr"] = out["mrr_wme"] - out["mrr_aal_mean"]
    return out


def fit_per_encoder(per_row: pd.DataFrame) -> dict:
    """OLS Δ = α + β·asym with HC1 robust SE."""
    y = per_row["delta_mrr"].to_numpy()
    x = per_row["lex_diff_wme_aal"].to_numpy()
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit(cov_type="HC1")
    alpha = float(model.params[0])
    beta = float(model.params[1])
    asym_mean = float(np.mean(x))
    return {
        "n_rows": int(len(per_row)),
        "raw_delta": float(np.mean(y)),
        "asym_mean": asym_mean,
        "asym_sd": float(np.std(x, ddof=1)),
        "alpha": alpha,
        "alpha_se": float(model.bse[0]),
        "alpha_ci_low": float(model.conf_int()[0, 0]),
        "alpha_ci_high": float(model.conf_int()[0, 1]),
        "alpha_p": float(model.pvalues[0]),
        "beta": beta,
        "beta_se": float(model.bse[1]),
        "beta_ci_low": float(model.conf_int()[1, 0]),
        "beta_ci_high": float(model.conf_int()[1, 1]),
        "beta_p": float(model.pvalues[1]),
        "beta_times_asym_mean": beta * asym_mean,
        "r2": float(model.rsquared),
        "residual_pct_of_raw": (100.0 * alpha / float(np.mean(y))
                                if np.mean(y) != 0 else float("nan")),
    }


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[lex_residual] loading paired rows from {PAIRED_JSONL}")
    paired = load_paired_rows()
    print(f"[lex_residual]   n rows = {len(paired)}")

    # Tokenize once for the corpus-level MCQ fit
    wme_docs = [tokenize(t) for t in paired["wme_real"]]
    aal_docs = []
    for k in AAL_KEYS:
        aal_docs.extend(tokenize(t) for t in paired[k])
    print(f"[lex_residual] WME corpus: {sum(len(d) for d in wme_docs)} tokens; "
          f"AAL corpus: {sum(len(d) for d in aal_docs)} tokens")

    print("[lex_residual] fitting MCQ log-odds...")
    zeta_map = fit_mcq_logodds(wme_docs, aal_docs)
    print(f"[lex_residual]   vocab size = {len(zeta_map)}; "
          f"wrote token table to {RES_DIR / 'lex_residual_token_logodds.csv'}")

    print("[lex_residual] computing per-row lex_WME / lex_AAL / asym ...")
    lex_per_row = per_row_lex(paired, zeta_map)
    print(f"[lex_residual]   asym summary:\n{lex_per_row['lex_diff_wme_aal'].describe().round(3)}")

    summary_rows = []
    for enc in ENCODERS:
        try:
            d = per_row_delta(enc)
        except FileNotFoundError:
            print(f"[lex_residual] WARN: missing per-query CSV for {enc}; skipping")
            continue
        merged = d.merge(lex_per_row, on="qid", how="inner")
        if len(merged) == 0:
            print(f"[lex_residual] WARN: no overlap for {enc}; skipping")
            continue
        s = fit_per_encoder(merged)
        s["encoder"] = enc
        summary_rows.append(s)
        print(f"[lex_residual] {enc:30s}  n={s['n_rows']:5d}  "
              f"Δ̄={s['raw_delta']:+.4f}  α={s['alpha']:+.4f}  β={s['beta']:+.2e}  "
              f"(p={s['beta_p']:.3g})  β·μ(asym)={s['beta_times_asym_mean']:+.4f}")

    cols = [
        "encoder", "n_rows", "raw_delta",
        "asym_mean", "asym_sd",
        "alpha", "alpha_se", "alpha_ci_low", "alpha_ci_high", "alpha_p",
        "beta", "beta_se", "beta_ci_low", "beta_ci_high", "beta_p",
        "beta_times_asym_mean",
        "r2", "residual_pct_of_raw",
    ]
    summary = pd.DataFrame(summary_rows)[cols]
    summary.to_csv(RES_DIR / "lex_residual_summary.csv", index=False)
    print(f"[lex_residual] wrote {RES_DIR / 'lex_residual_summary.csv'}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
