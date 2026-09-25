r"""Monroe-Colaresi-Quinn log-odds with informative Dirichlet prior, for L vs R
Stage-B queries.

Method (Monroe, Colaresi, Quinn 2008):
Across the combined L/R vocabulary, compute a log-odds ratio with an informative
Dirichlet prior using the union vocab counts. The MCQ z-score
$\zeta_w = (\delta_w) / \sqrt{1/(y_w^L + \alpha_w) + 1/(y_w^R + \alpha_w)}$
where $\delta_w$ is the log odds ratio of (left counts + prior) vs (right counts
+ prior). Tokens with high positive ζ are L-distinctive; high negative ζ are
R-distinctive.

We then compute a per-query lexical-asymmetry score:
  lex_score(q) = sum_{w in q's tokens} ζ_w     (informational mass of partisan words)
For a query q with stance L the score is *expected* to be positive (it uses
L-distinctive words), and vice versa for R. The per-cell lexical asymmetry
$\mathrm{lex}_L - \mathrm{lex}_R$ is the right-hand variable we partial out
of $\Delta_c$ in `analyze_lexical_regression.py`.

Outputs:
  allsides_synth/results/lexical_logodds.csv          (one row per token: zeta, count_L, count_R)
  allsides_synth/results/cell_lex_asymmetry.csv       (one row per (article_id, frame): lex_L, lex_N, lex_R, lex_diff_lr)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "allsides_synth" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SRC = ROOT / "allsides_synth" / "data" / "generated_queries.jsonl"

ALPHA = 0.01  # MCQ Dirichlet prior weight


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def main() -> None:
    rows = []
    with SRC.open() as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["tokens"] = df["text"].map(tokenize)

    # MCQ uses only L and R; we keep N separate.
    left_tokens = Counter()
    right_tokens = Counter()
    for _, r in df.iterrows():
        if r["stance"] == "left":
            left_tokens.update(r["tokens"])
        elif r["stance"] == "right":
            right_tokens.update(r["tokens"])

    all_terms = set(left_tokens) | set(right_tokens)
    nL = sum(left_tokens.values())
    nR = sum(right_tokens.values())
    # Background prior alpha_w = ALPHA * (total count of w / total tokens)
    bg = Counter()
    bg.update(left_tokens); bg.update(right_tokens)
    n_bg = nL + nR

    out = []
    for w in all_terms:
        yL = left_tokens.get(w, 0)
        yR = right_tokens.get(w, 0)
        alpha_w = ALPHA * (bg[w] / n_bg)  # background-proportional smoothing
        nL_w = yL + alpha_w
        nR_w = yR + alpha_w
        nL_not = (nL + ALPHA) - nL_w
        nR_not = (nR + ALPHA) - nR_w
        # log odds ratio
        log_odds = np.log(nL_w / nL_not) - np.log(nR_w / nR_not)
        # variance of log odds ratio (delta-method approx)
        var = 1.0 / nL_w + 1.0 / nR_w
        zeta = log_odds / np.sqrt(var)
        out.append({
            "token": w,
            "count_L": yL,
            "count_R": yR,
            "log_odds": float(log_odds),
            "zeta": float(zeta),
        })
    lo_df = pd.DataFrame(out).sort_values(["zeta", "token"], ascending=[False, True])
    lo_df.to_csv(OUT_DIR / "lexical_logodds.csv", index=False)

    # Per-query lex_score (full vocab, no stopword filter)
    zeta_map = dict(zip(lo_df["token"], lo_df["zeta"]))
    df["lex_score"] = df["tokens"].map(lambda toks: float(sum(zeta_map.get(t, 0.0) for t in toks)))

    # Per-cell lex asymmetry: lex_L − lex_R
    pivot = df.pivot_table(index=["article_id", "frame"], columns="stance",
                           values="lex_score").reset_index()
    if "neutral" not in pivot.columns:
        pivot["neutral"] = np.nan
    pivot["lex_diff_lr"] = pivot["left"] - pivot["right"]
    pivot = pivot.rename(columns={"left": "lex_L", "right": "lex_R", "neutral": "lex_N"})
    pivot[["article_id", "frame", "lex_L", "lex_N", "lex_R", "lex_diff_lr"]].to_csv(
        OUT_DIR / "cell_lex_asymmetry.csv", index=False,
    )
    print("\nCell-level lex_diff_lr summary:")
    print(pivot["lex_diff_lr"].describe().round(3))


if __name__ == "__main__":
    main()
