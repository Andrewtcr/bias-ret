"""Between-group lex-residual decomposition for the dialect REAL arm.

Unpaired analogue of `analyze_lexical_synthetic.py` (synth arm). Whereas the synth-arm
script regresses a per-row paired Δ_MRR = MRR(wme_real) − mean MRR(aal_synth_i)
on a per-row lex asymmetry, this script regresses the per-query MRR on a
per-query lex score plus a binary group indicator, across the union of
aal_real + wme_real queries — because the real arm has different patient
pools (no within-row pairing).

Model
-----
MCQ informative-Dirichlet log-odds ζ_t with
  L-corpus = bag of tokens across all aal_real queries
  R-corpus = bag of tokens across all wme_real queries
ALPHA = 0.01 with background-proportional smoothing.
Sign convention: positive ζ = WME-distinctive; negative = AAL-distinctive.

Per query:
  lex_q = Σ_{t ∈ q} ζ_t

Per encoder we then fit, across the n_AAL + n_WME queries:
  MRR_q = α + β · lex_q + γ · 1[q ∈ wme_real] + ε_q
with HC1 robust SE. γ is the WME-vs-AAL gap after partialling out lex;
β is the within-group lex slope. We compare γ to the raw unpaired delta
from `hcmagic_real_paired.csv` (`wme_real_vs_aal_real_unpaired`).

Outputs
-------
  hcmagic/results/lex_residual_real_summary.csv
  hcmagic/results/lex_residual_real_token_logodds.csv
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
RES_DIR = ROOT / "hcmagic" / "results"
SRC_CSV = ROOT / "hcmagic" / "data" / "healthq_aal" / "hcmagic_aal_wme.csv"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ALPHA = 0.01  # MCQ Dirichlet prior weight (matches synth-arm script)

ENCODERS = (
    "bge-large-en-v1.5",
    "qwen3-embedding-8b",
    "llama-embed-nemotron-8b",
    "octen-embedding-8b",
    "text-embedding-3-large",
    "bm25",
)


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(str(t).lower())


def load_source() -> pd.DataFrame:
    """Load hcmagic_aal_wme.csv. Build qid via dial + idx."""
    df = pd.read_csv(SRC_CSV)
    needed = {"input", "dial", "idx"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"source CSV missing columns: {missing}")
    df = df[df["dial"].isin(["aal", "wme"])].copy()
    df["qid"] = df.apply(
        lambda r: f"hcm_{r['dial']}_{int(r['idx'])}", axis=1
    )
    return df[["qid", "dial", "input"]]


def fit_mcq_logodds(
    aal_docs: list[list[str]], wme_docs: list[list[str]]
) -> dict[str, float]:
    """MCQ log-odds; positive ζ = WME-distinctive."""
    aal_counts: Counter[str] = Counter()
    wme_counts: Counter[str] = Counter()
    for toks in aal_docs:
        aal_counts.update(toks)
    for toks in wme_docs:
        wme_counts.update(toks)

    all_terms = set(aal_counts) | set(wme_counts)
    nA = sum(aal_counts.values())
    nW = sum(wme_counts.values())
    bg = Counter()
    bg.update(aal_counts)
    bg.update(wme_counts)
    n_bg = nA + nW

    zeta_map: dict[str, float] = {}
    diag_rows = []
    for w in all_terms:
        yA = aal_counts.get(w, 0)
        yW = wme_counts.get(w, 0)
        alpha_w = ALPHA * (bg[w] / n_bg)
        nA_w = yA + alpha_w
        nW_w = yW + alpha_w
        nA_not = (nA + ALPHA) - nA_w
        nW_not = (nW + ALPHA) - nW_w
        # Signed positive when WME has higher relative odds than AAL.
        log_odds = np.log(nW_w / nW_not) - np.log(nA_w / nA_not)
        var = 1.0 / nW_w + 1.0 / nA_w
        zeta = log_odds / np.sqrt(var)
        zeta_map[w] = float(zeta)
        diag_rows.append({
            "token": w,
            "count_AAL": yA,
            "count_WME": yW,
            "log_odds": float(log_odds),
            "zeta": float(zeta),
        })
    diag = pd.DataFrame(diag_rows).sort_values(["zeta", "token"], ascending=[False, True])
    diag.to_csv(RES_DIR / "lex_residual_real_token_logodds.csv", index=False)
    return zeta_map


def per_query_lex(src: pd.DataFrame, zeta_map: dict[str, float]) -> pd.DataFrame:
    """One row per qid (across both groups) with lex_q, n_tok, group."""

    def score(toks: list[str]) -> float:
        return float(sum(zeta_map.get(t, 0.0) for t in toks))

    out = []
    for r in src.itertuples(index=False):
        toks = tokenize(r.input)
        out.append({
            "qid": r.qid,
            "group": r.dial,
            "lex": score(toks),
            "n_tok": len(toks),
        })
    return pd.DataFrame(out)


def per_query_mrr(encoder: str) -> pd.DataFrame:
    """Per-query MRR for variants in {aal_real, wme_real}. Returns qid + mrr."""
    csv = RES_DIR / encoder / "hcmagic_real_per_query_metrics.csv"
    df = pd.read_csv(csv, usecols=["query_id", "variant", "mrr"])
    df = df[df["variant"].isin(["aal_real", "wme_real"])].copy()
    df = df.rename(columns={"query_id": "qid"})
    return df[["qid", "variant", "mrr"]]


def fit_per_encoder(df: pd.DataFrame) -> dict:
    """OLS MRR = α + β·lex + γ·1[wme] with HC1 robust SE.

    Also includes raw_delta = mean MRR(wme) − mean MRR(aal),
    and gamma / raw_delta.
    """
    is_wme = (df["group"] == "wme").astype(float).to_numpy()
    lex = df["lex"].to_numpy()
    y = df["mrr"].to_numpy()
    X = np.column_stack([np.ones_like(y), lex, is_wme])
    model = sm.OLS(y, X).fit(cov_type="HC1")
    alpha = float(model.params[0])
    beta = float(model.params[1])
    gamma = float(model.params[2])

    mrr_aal = float(df.loc[df["group"] == "aal", "mrr"].mean())
    mrr_wme = float(df.loc[df["group"] == "wme", "mrr"].mean())
    raw_delta = mrr_wme - mrr_aal
    lex_mean_wme = float(df.loc[df["group"] == "wme", "lex"].mean())
    lex_mean_aal = float(df.loc[df["group"] == "aal", "lex"].mean())

    ci = model.conf_int()
    return {
        "n_aal": int((df["group"] == "aal").sum()),
        "n_wme": int((df["group"] == "wme").sum()),
        "mrr_aal_mean": mrr_aal,
        "mrr_wme_mean": mrr_wme,
        "raw_delta": raw_delta,
        "lex_aal_mean": lex_mean_aal,
        "lex_wme_mean": lex_mean_wme,
        "alpha": alpha,
        "alpha_se": float(model.bse[0]),
        "alpha_ci_low": float(ci[0, 0]),
        "alpha_ci_high": float(ci[0, 1]),
        "alpha_p": float(model.pvalues[0]),
        "beta": beta,
        "beta_se": float(model.bse[1]),
        "beta_ci_low": float(ci[1, 0]),
        "beta_ci_high": float(ci[1, 1]),
        "beta_p": float(model.pvalues[1]),
        "beta_times_lex_wme_mean": beta * lex_mean_wme,
        "gamma": gamma,
        "gamma_se": float(model.bse[2]),
        "gamma_ci_low": float(ci[2, 0]),
        "gamma_ci_high": float(ci[2, 1]),
        "gamma_p": float(model.pvalues[2]),
        "gamma_over_raw": (gamma / raw_delta) if raw_delta != 0 else float("nan"),
        "r2": float(model.rsquared),
    }


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[lex_residual_real] loading source from {SRC_CSV}")
    src = load_source()
    n_aal_src = int((src["dial"] == "aal").sum())
    n_wme_src = int((src["dial"] == "wme").sum())
    print(f"[lex_residual_real]   source rows: aal={n_aal_src}, wme={n_wme_src}")

    aal_docs = [tokenize(t) for t in src.loc[src["dial"] == "aal", "input"]]
    wme_docs = [tokenize(t) for t in src.loc[src["dial"] == "wme", "input"]]
    print(
        f"[lex_residual_real] AAL corpus: {sum(len(d) for d in aal_docs)} tokens; "
        f"WME corpus: {sum(len(d) for d in wme_docs)} tokens"
    )

    print("[lex_residual_real] fitting MCQ log-odds...")
    zeta_map = fit_mcq_logodds(aal_docs, wme_docs)
    print(
        f"[lex_residual_real]   vocab size = {len(zeta_map)}; "
        f"wrote token table to {RES_DIR / 'lex_residual_real_token_logodds.csv'}"
    )

    print("[lex_residual_real] computing per-query lex scores ...")
    lex_per_q = per_query_lex(src, zeta_map)
    print(
        f"[lex_residual_real]   AAL lex: mean={lex_per_q.loc[lex_per_q.group=='aal','lex'].mean():.2f} "
        f"WME lex: mean={lex_per_q.loc[lex_per_q.group=='wme','lex'].mean():.2f}"
    )

    summary_rows = []
    for enc in ENCODERS:
        try:
            mrr_df = per_query_mrr(enc)
        except FileNotFoundError:
            print(f"[lex_residual_real] WARN: missing per-query CSV for {enc}; skipping")
            continue
        # Map variant → group label for merge clarity (variant is aal_real/wme_real).
        mrr_df["group"] = mrr_df["variant"].map({"aal_real": "aal", "wme_real": "wme"})
        merged = mrr_df.merge(lex_per_q[["qid", "lex", "n_tok"]], on="qid", how="inner")
        if len(merged) == 0:
            print(f"[lex_residual_real] WARN: no overlap for {enc}; skipping")
            continue
        s = fit_per_encoder(merged)
        s["encoder"] = enc
        summary_rows.append(s)
        print(
            f"[lex_residual_real] {enc:30s}  "
            f"nA={s['n_aal']:4d} nW={s['n_wme']:4d}  "
            f"rawΔ={s['raw_delta']:+.4f}  "
            f"γ={s['gamma']:+.4f} [p={s['gamma_p']:.3g}]  "
            f"β={s['beta']:+.2e} [p={s['beta_p']:.3g}]  "
            f"γ/raw={s['gamma_over_raw']:+.2f}"
        )

    cols = [
        "encoder", "n_aal", "n_wme",
        "mrr_aal_mean", "mrr_wme_mean", "raw_delta",
        "lex_aal_mean", "lex_wme_mean",
        "alpha", "alpha_se", "alpha_ci_low", "alpha_ci_high", "alpha_p",
        "beta", "beta_se", "beta_ci_low", "beta_ci_high", "beta_p",
        "beta_times_lex_wme_mean",
        "gamma", "gamma_se", "gamma_ci_low", "gamma_ci_high", "gamma_p",
        "gamma_over_raw", "r2",
    ]
    summary = pd.DataFrame(summary_rows)[cols]
    summary.to_csv(RES_DIR / "lex_residual_real_summary.csv", index=False)
    print(f"[lex_residual_real] wrote {RES_DIR / 'lex_residual_real_summary.csv'}")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
