"""Between-group MCQ lex-residual decomposition for the Reddit real arm.

Reddit-arm analog of the paired-cell synth-arm decomposition in
``allsides_synth/scripts/analyze_lexical_regression.py``: for each retriever, we ask
how much of the con-asker minus lib-asker lean@20 gap survives after we
partial out per-query lexical partisanship.

Setup
-----
We treat liberal-asker post titles as the L corpus and conservative-asker
titles as the R corpus (centrist queries are dropped for the headline
between-group contrast). Tokenizer matches ``retrieve_bm25.py``
(``[a-z0-9]+(?:'[a-z]+)?`` over the lowercased title). MCQ informative-
Dirichlet log-odds with background-proportional smoothing
``ALPHA = 0.01`` gives one token-level ζ per term; sign convention is
**positive ζ = conservative-distinctive**, since L = lib and R = con.

Per-query lex score is
  lex_q = Σ_{t ∈ tok(title_q)} ζ_t                                        (1)
so positive lex_q means the query uses conservative-distinctive
vocabulary on net. Per encoder, at k = 20 (the appendix analysis),
we fit between-group OLS with HC1 robust SE:
  lean@20_q = α + β · lex_q + γ · 1[ideology_q = conservative] + ε        (2)

γ̂ is the **partialled** con−lib gap (the part of the raw gap that
survives stripping out per-query lexical partisanship). β̂ is the
within-group lex slope. Comparing γ̂ with the raw con−lib gap quantifies
how much group difference remains after this linear lexical control.
The residual does not isolate a causal or purely semantic effect.

Use --from-scores to refit the regression from committed per-query lexical
scores and retrieval lean without post text. This writes only the summary
CSV and Markdown report, preserving the per-query table.

Use --gold-filtered for the appendix robustness fit: the same regression
with the gold-filtered lean@10 (mean stance over the articles both judges
grade relevant, queries with at least one such article) as the outcome. It
reads the committed per-query lexical scores and writes only the gold summary.

Outputs
-------
  reddit_nat/results/lex_residual_token_logodds.csv  (not released)
  reddit_nat/results/lex_residual_per_query.csv
  reddit_nat/results/lex_residual_summary.csv
  reddit_nat/results/lex_residual.md
  reddit_nat/results/lex_residual_gold_summary.csv   (--gold-filtered)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reddit_nat import lib  # noqa: E402
from src.io import load_yaml  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
ALPHA = 0.01  # MCQ Dirichlet prior weight (matches political-synth fit)
K = 20  # unfiltered lexical-control analysis in the appendix
GOLD_K = 10  # gold-filtered robustness fit (outcome of the main Reddit figure)
DENSE_ENCODERS = (
    "bge-large-en-v1.5",
    "qwen3-embedding-8b",
    "llama-embed-nemotron-8b",
    "octen-embedding-8b",
    "text-embedding-3-large",
)


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(str(t).lower())


def fit_mcq_logodds(
    lib_docs: list[list[str]],
    con_docs: list[list[str]],
    token_out_csv: Path,
) -> dict[str, float]:
    """MCQ informative-Dirichlet log-odds; positive ζ = conservative-distinctive."""
    lib_counts: Counter[str] = Counter()
    con_counts: Counter[str] = Counter()
    for toks in lib_docs:
        lib_counts.update(toks)
    for toks in con_docs:
        con_counts.update(toks)

    all_terms = set(lib_counts) | set(con_counts)
    nL = sum(lib_counts.values())
    nR = sum(con_counts.values())
    bg: Counter[str] = Counter()
    bg.update(lib_counts)
    bg.update(con_counts)
    n_bg = nL + nR

    zeta_map: dict[str, float] = {}
    diag_rows = []
    for w in all_terms:
        yL = lib_counts.get(w, 0)
        yR = con_counts.get(w, 0)
        alpha_w = ALPHA * (bg[w] / n_bg)  # background-proportional smoothing
        # Match hcmagic/scripts/analyze_lexical_synthetic.py: smooth each
        # cell by alpha_w and use (nL + ALPHA), (nR + ALPHA) as totals.
        nR_w = yR + alpha_w
        nL_w = yL + alpha_w
        nR_not = (nR + ALPHA) - nR_w
        nL_not = (nL + ALPHA) - nL_w
        log_odds = np.log(nR_w / nR_not) - np.log(nL_w / nL_not)
        var = 1.0 / nR_w + 1.0 / nL_w
        zeta = log_odds / np.sqrt(var)
        zeta_map[w] = float(zeta)
        diag_rows.append({
            "token": w,
            "count_lib": yL,
            "count_con": yR,
            "log_odds": float(log_odds),
            "zeta": float(zeta),
        })
    diag = pd.DataFrame(diag_rows).sort_values(["zeta", "token"], ascending=[False, True])
    diag.to_csv(token_out_csv, index=False)
    return zeta_map


def per_query_lex(
    queries: pd.DataFrame, zeta_map: dict[str, float]
) -> pd.DataFrame:
    rows = []
    for r in queries.itertuples(index=False):
        toks = tokenize(getattr(r, "text"))
        lex = float(sum(zeta_map.get(t, 0.0) for t in toks))
        rows.append({
            "query_id": getattr(r, "query_id"),
            "ideology": getattr(r, "ideology"),
            "subreddit": getattr(r, "subreddit"),
            "n_tok": len(toks),
            "lex": lex,
        })
    return pd.DataFrame(rows)


def load_lean(rdir: Path, encoder: str) -> pd.DataFrame:
    """Mean retrieved stance over the first K candidates, including center."""
    path = lib.candidates_dir(rdir) / f"{encoder}.csv"
    candidates = pd.read_csv(path, usecols=[
        "query_id", "ideology", "rank", "retrieved_stance",
    ])
    top = candidates[candidates["rank"] <= K].copy()
    top["lean"] = top["retrieved_stance"].map({"left": -1, "right": 1}).fillna(0)
    return top.groupby(["query_id", "ideology"], sort=False)["lean"].mean().reset_index()


def fit_regression(merged: pd.DataFrame) -> dict[str, float]:
    """OLS lean = α + β·lex + γ·1[con] with HC1 robust SE."""
    is_con = (merged["ideology"] == "conservative").astype(float).to_numpy()
    lex = merged["lex"].to_numpy()
    y = merged["lean"].to_numpy()
    X = np.column_stack([np.ones_like(y), lex, is_con])
    model = sm.OLS(y, X).fit(cov_type="HC1")
    raw_gap = float(
        merged.loc[is_con == 1, "lean"].mean()
        - merged.loc[is_con == 0, "lean"].mean()
    )
    lex_mean_con = float(merged.loc[is_con == 1, "lex"].mean())
    lex_mean_lib = float(merged.loc[is_con == 0, "lex"].mean())
    alpha = float(model.params[0])
    beta = float(model.params[1])
    gamma = float(model.params[2])
    ci = model.conf_int()
    return {
        "n_lib": int((is_con == 0).sum()),
        "n_con": int((is_con == 1).sum()),
        "raw_gap_con_minus_lib": raw_gap,
        "lex_mean_lib": lex_mean_lib,
        "lex_mean_con": lex_mean_con,
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
        "beta_times_lex_mean_con": beta * lex_mean_con,
        "gamma": gamma,
        "gamma_se": float(model.bse[2]),
        "gamma_ci_low": float(ci[2, 0]),
        "gamma_ci_high": float(ci[2, 1]),
        "gamma_p": float(model.pvalues[2]),
        "gamma_over_raw": gamma / raw_gap if raw_gap != 0 else float("nan"),
        "r2": float(model.rsquared),
    }


def write_report(rdir: Path, summary: pd.DataFrame) -> None:
    """Write the fitted coefficients and their interpretation without post text."""
    columns = [
        "encoder", "n_lib", "n_con", "raw_gap_con_minus_lib",
        "gamma", "gamma_ci_low", "gamma_ci_high", "gamma_p",
        "gamma_over_raw", "beta", "beta_p",
    ]
    lines = [
        "# Reddit lexical-control analysis", "",
        f"At k={K}, fit lean = alpha + beta * lex + gamma * I[conservative] "
        "with HC1 robust standard errors. The lexical score sums token-level "
        f"MCQ log-odds with prior weight {ALPHA}.", "",
        "The raw gap is conservative minus liberal mean retrieval lean. "
        "Gamma is the residual group difference after the linear lexical "
        "control. It does not identify a causal effect or exclude nonlinear "
        "lexical and topic differences. Ratios are unstable near a zero raw gap.", "",
        "```", summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "```", "",
    ]
    (rdir / "lex_residual.md").write_text("\n".join(lines))
    print(f"[lex_residual] wrote {rdir / 'lex_residual.md'}")


def write_summary(rdir: Path, summary_rows: list[dict],
                  name: str = "lex_residual_summary.csv") -> pd.DataFrame:
    """Write the regression summary shared by full and cached-score runs."""
    summary = pd.DataFrame(summary_rows)
    summary = summary[[
        "encoder", "n_lib", "n_con",
        "raw_gap_con_minus_lib",
        "alpha", "alpha_se", "alpha_ci_low", "alpha_ci_high", "alpha_p",
        "beta", "beta_se", "beta_ci_low", "beta_ci_high", "beta_p",
        "lex_mean_lib", "lex_mean_con", "beta_times_lex_mean_con",
        "gamma", "gamma_se", "gamma_ci_low", "gamma_ci_high", "gamma_p",
        "gamma_over_raw", "r2",
    ]]
    summary_out = rdir / name
    summary.to_csv(summary_out, index=False)
    print(f"\n[lex_residual] wrote {summary_out}")
    print(summary.round(4).to_string(index=False))

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "reddit_nat" / "configs"
                    / "retrieval.yaml"),
    )
    p.add_argument(
        "--from-scores", action="store_true",
        help="refit summary/report from committed per-query scores without post text",
    )
    p.add_argument(
        "--gold-filtered", action="store_true",
        help="fit the gold-filtered lean@10 outcome from committed per-query scores",
    )
    args = p.parse_args()

    cfg = load_yaml(args.config)
    rdir = lib.run_dir(cfg)
    print(f"[lex_residual] run dir = {rdir}")

    if args.gold_filtered:
        scores_path = rdir / "lex_residual_per_query.csv"
        lex_scores = (pd.read_csv(scores_path, usecols=["query_id", "lex"])
                      .drop_duplicates(subset=["query_id"]))
        gold = lib.filtered_per_query_lean(rdir, k=GOLD_K, judge="both")
        summary_rows = []
        for enc in list(DENSE_ENCODERS) + ["bm25"]:
            merged = gold[gold["encoder"] == enc].merge(lex_scores, on="query_id", how="inner")
            if merged.empty:
                print(f"[lex_residual] {enc}: no gold-filtered queries; skip")
                continue
            fitted = fit_regression(merged)
            fitted["encoder"] = enc
            summary_rows.append(fitted)
        write_summary(rdir, summary_rows, name="lex_residual_gold_summary.csv")
        return

    if args.from_scores:
        scores_path = rdir / "lex_residual_per_query.csv"
        scores = pd.read_csv(scores_path)
        required = {"encoder", "ideology", "lean", "lex"}
        missing = required - set(scores.columns)
        if missing:
            raise ValueError(f"{scores_path}: missing columns {sorted(missing)}")
        scores = scores[scores["ideology"].isin(["liberal", "conservative"])]
        scores = scores.dropna(subset=["encoder", "lean", "lex"])
        if scores.empty:
            raise ValueError(f"{scores_path}: no liberal/conservative rows to fit")
        summary_rows = []
        for encoder, group in scores.groupby("encoder", sort=False):
            fitted = fit_regression(group)
            fitted["encoder"] = encoder
            summary_rows.append(fitted)
        summary = write_summary(rdir, summary_rows)
        write_report(rdir, summary)
        return

    queries = lib.load_reddit_queries(cfg)
    # The MCQ corpus split uses only lib + con. Centrist queries are kept in
    # `lex_q` (and get a lex score from the same ζ map) but excluded from
    # corpus counts and from the regression sample.
    lib_q = queries[queries["ideology"] == "liberal"].copy()
    con_q = queries[queries["ideology"] == "conservative"].copy()
    print(f"[lex_residual] L corpus (lib): {len(lib_q)} queries; "
          f"R corpus (con): {len(con_q)} queries; "
          f"centrist dropped from MCQ fit: "
          f"{(queries['ideology']=='centrist').sum()}")

    lib_docs = [tokenize(t) for t in lib_q["text"]]
    con_docs = [tokenize(t) for t in con_q["text"]]
    print(f"[lex_residual] L tokens: {sum(len(d) for d in lib_docs):,}; "
          f"R tokens: {sum(len(d) for d in con_docs):,}")

    token_out = rdir / "lex_residual_token_logodds.csv"
    zeta_map = fit_mcq_logodds(lib_docs, con_docs, token_out)
    print(f"[lex_residual] vocab = {len(zeta_map):,}; "
          f"wrote {token_out}")

    # Per-query lex score (kept for ALL queries incl. centrist for the per-
    # query CSV; regression filters to lib+con only).
    lex_q = per_query_lex(queries, zeta_map)
    print(f"[lex_residual] per-query lex score distribution:\n"
          f"{lex_q['lex'].describe().round(2)}")

    # Spot-check tokens.
    diag = pd.read_csv(token_out)
    big = diag[(diag["count_lib"] + diag["count_con"]) >= 20]
    print("[lex_residual] top conservative-distinctive (count>=20):")
    print(big.sort_values("zeta", ascending=False).head(10).to_string(index=False))
    print("[lex_residual] top liberal-distinctive (count>=20):")
    print(big.sort_values("zeta").head(10).to_string(index=False))

    # Per-encoder regression.
    summary_rows = []
    per_query_long = []
    encoders = list(DENSE_ENCODERS) + ["bm25"]
    for enc in encoders:
        try:
            lean = load_lean(rdir, enc)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[lex_residual] {enc}: skip ({e})")
            continue
        merged = lex_q.merge(lean[["query_id", "lean"]], on="query_id",
                              how="inner")
        # Filter to lib + con for the regression.
        merged = merged[merged["ideology"].isin(["liberal", "conservative"])]
        merged = merged.dropna(subset=["lean", "lex"])
        if len(merged) == 0:
            print(f"[lex_residual] {enc}: no overlap; skip")
            continue
        s = fit_regression(merged)
        s["encoder"] = enc
        summary_rows.append(s)
        # Keep per-query (encoder, query, lean, lex, ideology).
        out_pq = merged[["query_id", "ideology", "subreddit",
                          "n_tok", "lex", "lean"]].copy()
        out_pq["encoder"] = enc
        per_query_long.append(out_pq)
        print(f"[lex_residual] {enc:25s}  n={s['n_lib']+s['n_con']:5d}  "
              f"raw={s['raw_gap_con_minus_lib']:+.4f}  "
              f"γ̂={s['gamma']:+.4f} (p={s['gamma_p']:.3g})  "
              f"β̂={s['beta']:+.2e} (p={s['beta_p']:.3g})  "
              f"γ̂/raw={s['gamma_over_raw']:+.2%}")

    summary = write_summary(rdir, summary_rows)

    per_query_long_df = pd.concat(per_query_long, ignore_index=True)[
        ["encoder", "query_id", "ideology", "subreddit", "n_tok", "lex", "lean"]
    ]
    pq_out = rdir / "lex_residual_per_query.csv"
    per_query_long_df.to_csv(pq_out, index=False)
    print(f"[lex_residual] wrote {pq_out} ({len(per_query_long_df)} rows)")

    write_report(rdir, summary)


if __name__ == "__main__":
    main()
