"""Compare gpt-5-mini and Qwen3.5-35B-A3B relevance judgments.

Computes overall, per-encoder, and per-ideology Cohen's kappa, relevance rates,
and the conservative-minus-liberal lean@10 gap under each relevance filter
(raw, gpt-only, Qwen-only, both judges). Reads ID-keyed judgments and
retrieval candidates; no post text is required.

Writes judge_relevance/{agreement_overall,kappa_per_encoder,lean_gap_by_filter}.csv
and agreement_summary.md. agreement_overall.csv holds
the overall statistics the paper's judge appendix reports: kappa, raw
agreement, relevance rates, each judge's recall against the other as gold,
the per-side kappas, and the gold (both judges relevant) pair count and share.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reddit_nat import lib  # noqa: E402

RUN_DIR = REPO / "reddit_nat" / "results"
JUDGE_DIR = RUN_DIR / "judge_relevance"
OVERALL_CSV = JUDGE_DIR / "agreement_overall.csv"
ENC_CSV = JUDGE_DIR / "kappa_per_encoder.csv"
OUT_MD = JUDGE_DIR / "agreement_summary.md"

BM25_CSV = lib.bm25_candidate_file(RUN_DIR)
K = 10
ENCODERS = ["bge-large-en-v1.5", "qwen3-embedding-8b", "llama-embed-nemotron-8b",
            "octen-embedding-8b", "text-embedding-3-large", "bm25"]


def load_judgments(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({"query_id": d["query_id"],
                         "article_id": d["article_id"],
                         "grade": d["grade"]})
    return pd.DataFrame(rows)


def main() -> None:
    print("loading judgments...")
    gpt = load_judgments(JUDGE_DIR / "judgments.jsonl")
    qwen = load_judgments(JUDGE_DIR / "judgments_qwen.jsonl")
    print(f"  gpt-5-mini: {len(gpt):,} judgments")
    print(f"  Qwen3.5:    {len(qwen):,} judgments")

    # Strip Qwen parse-failures (grade=None) before merge
    qwen_clean = qwen[qwen["grade"].notna()].copy()
    qwen_clean["grade"] = qwen_clean["grade"].astype(int)
    print(f"  Qwen valid: {len(qwen_clean):,}")

    # Inner-join
    m = gpt.merge(qwen_clean, on=["query_id", "article_id"], suffixes=("_gpt", "_qwen"))
    print(f"  intersected: {len(m):,} pairs")

    # === 1. Overall κ ===
    kappa = cohen_kappa_score(m["grade_gpt"], m["grade_qwen"])
    pct_agree = (m["grade_gpt"] == m["grade_qwen"]).mean()
    # Cross-tab
    ct = pd.crosstab(m["grade_gpt"], m["grade_qwen"],
                     rownames=["gpt-5-mini"], colnames=["qwen"])
    print()
    print(f"=== Overall (n={len(m):,}) ===")
    print(f"  Cohen's κ: {kappa:.4f}")
    print(f"  raw agreement: {pct_agree*100:.2f}%")
    print()
    print(ct.to_string())
    print()
    print(f"  gpt-5-mini grade=1 rate: {m['grade_gpt'].mean()*100:.2f}%")
    print(f"  Qwen       grade=1 rate: {m['grade_qwen'].mean()*100:.2f}%")
    recall_qwen = recall_gpt = float("nan")
    if (m["grade_gpt"] == 1).sum() > 0:
        recall_qwen = ((m["grade_qwen"] == 1) & (m["grade_gpt"] == 1)).sum() / (m["grade_gpt"] == 1).sum()
        print(f"  Qwen recall vs gpt-5-mini-as-truth: {recall_qwen*100:.2f}%")
    if (m["grade_qwen"] == 1).sum() > 0:
        recall_gpt = ((m["grade_qwen"] == 1) & (m["grade_gpt"] == 1)).sum() / (m["grade_qwen"] == 1).sum()
        print(f"  gpt-5-mini recall vs Qwen-as-truth: {recall_gpt*100:.2f}%")
    n_gold = int(((m["grade_gpt"] == 1) & (m["grade_qwen"] == 1)).sum())
    print(f"  gold (both judges grade=1): {n_gold:,} pairs ({n_gold/len(m)*100:.2f}%)")

    # === 2. Per-encoder κ ===
    print()
    print("=== Per-encoder κ ===")
    cand_cols = ["query_id", "encoder", "rank", "retrieved_article_id"]
    dense = lib.load_dense_candidates(RUN_DIR, usecols=cand_cols)
    bm = pd.read_csv(BM25_CSV, usecols=cand_cols)
    cands = pd.concat([dense, bm], ignore_index=True)
    cands = cands[cands["rank"] <= K].drop_duplicates(subset=["query_id", "encoder", "retrieved_article_id"])
    cands = cands.rename(columns={"retrieved_article_id": "article_id"})
    cands_m = cands.merge(m, on=["query_id", "article_id"], how="inner")

    per_enc_rows = []
    for enc in ENCODERS:
        sub = cands_m[cands_m["encoder"] == enc]
        if len(sub) < 10:
            continue
        k_enc = cohen_kappa_score(sub["grade_gpt"], sub["grade_qwen"])
        agree = (sub["grade_gpt"] == sub["grade_qwen"]).mean()
        both = (sub["grade_gpt"] == 1) & (sub["grade_qwen"] == 1)
        per_enc_rows.append({
            "encoder": enc, "n_pairs": len(sub),
            "kappa": k_enc, "raw_agreement": agree,
            "rel_rate_gpt": sub["grade_gpt"].mean(),
            "rel_rate_qwen": sub["grade_qwen"].mean(),
            "rel_rate_both": both.mean(),
        })
        print(f"  {enc:<25s} n={len(sub):>6,d} κ={k_enc:.3f} agree={agree*100:5.2f}% "
              f"gpt={sub['grade_gpt'].mean()*100:5.2f}% qwen={sub['grade_qwen'].mean()*100:5.2f}% "
              f"both={both.mean()*100:5.2f}%")
    enc_df = pd.DataFrame(per_enc_rows)

    # === 3. Per-ideology κ ===
    print()
    print("=== Per-ideology κ ===")
    dense_meta = lib.load_dense_candidates(
        RUN_DIR, usecols=["query_id", "ideology"]).drop_duplicates()
    m_with_id = m.merge(dense_meta, on="query_id", how="left")
    ide_rows = []
    for ide in ["liberal", "conservative"]:
        sub = m_with_id[m_with_id["ideology"] == ide]
        if len(sub) < 10:
            continue
        k_ide = cohen_kappa_score(sub["grade_gpt"], sub["grade_qwen"])
        agree = (sub["grade_gpt"] == sub["grade_qwen"]).mean()
        ide_rows.append({
            "ideology": ide, "n_pairs": len(sub),
            "kappa": k_ide, "raw_agreement": agree,
            "rel_rate_gpt": sub["grade_gpt"].mean(),
            "rel_rate_qwen": sub["grade_qwen"].mean(),
        })
        print(f"  {ide:<15s} n={len(sub):>6,d} κ={k_ide:.3f} agree={agree*100:5.2f}% "
              f"gpt={sub['grade_gpt'].mean()*100:5.2f}% qwen={sub['grade_qwen'].mean()*100:5.2f}%")
    ide_df = pd.DataFrame(ide_rows)

    # === 4. Conservative-minus-liberal lean@10 gap under each relevance filter ===
    print()
    print("=== con-lib lean@10 gap: raw / gpt-only / Qwen-only / gold (both judges) ===")
    tags = {"none": "raw", "gpt": "gpt", "qwen": "qwen", "both": "gold"}
    per_filter = {judge: lib.filtered_per_query_lean(RUN_DIR, k=K, judge=judge) for judge in tags}
    gap_rows = []
    for enc in ENCODERS:
        row: dict[str, float] = {"encoder": enc}
        for judge, tag in tags.items():
            sub = per_filter[judge][per_filter[judge]["encoder"] == enc]
            lib_v = sub.loc[sub["ideology"] == "liberal", "lean"].to_numpy()
            con_v = sub.loc[sub["ideology"] == "conservative", "lean"].to_numpy()
            row[f"n_lib_{tag}"], row[f"n_con_{tag}"] = len(lib_v), len(con_v)
            if len(lib_v) >= 10 and len(con_v) >= 10:
                _, p = scipy_stats.mannwhitneyu(con_v, lib_v, alternative="two-sided")
                row[f"gap_{tag}"], row[f"mwu_p_{tag}"] = float(con_v.mean() - lib_v.mean()), float(p)
            else:
                row[f"gap_{tag}"], row[f"mwu_p_{tag}"] = np.nan, np.nan
        gap_rows.append(row)
        print(f"  {enc:<25s} " + "  ".join(f"{tag}={row[f'gap_{tag}']:+.3f} (p={row[f'mwu_p_{tag}']:.3g})"
                                          for tag in tags.values()))
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(JUDGE_DIR / "lean_gap_by_filter.csv", index=False)

    # === Write outputs ===
    print()
    overall = {"n_queries": m["query_id"].nunique(), "n_pairs": len(m),
               "kappa": kappa, "raw_agreement": pct_agree,
               "rel_rate_gpt": m["grade_gpt"].mean(), "rel_rate_qwen": m["grade_qwen"].mean(),
               "recall_gpt_vs_qwen_gold": recall_gpt, "recall_qwen_vs_gpt_gold": recall_qwen}
    overall.update({f"kappa_{r['ideology']}": r["kappa"] for r in ide_rows})
    overall.update({"n_gold": n_gold, "frac_gold": n_gold / len(m)})
    pd.DataFrame([overall]).to_csv(OVERALL_CSV, index=False)
    enc_df.to_csv(ENC_CSV, index=False)

    md = []
    md.append("# Judge agreement: gpt-5-mini vs Qwen3.5-35B-A3B\n")
    md.append(f"- n = **{len(m):,}** pairs judged by both")
    md.append(f"- gpt-5-mini relevance rate: **{m['grade_gpt'].mean()*100:.2f}%**")
    md.append(f"- Qwen relevance rate: **{m['grade_qwen'].mean()*100:.2f}%**")
    md.append(f"- Cohen's κ: **{kappa:.4f}** (raw agreement {pct_agree*100:.2f}%)\n")
    md.append("## Cross-tab\n```")
    md.append(ct.to_string())
    md.append("```\n## Per-encoder κ\n")
    md.append(enc_df.to_markdown(index=False, floatfmt=".4f") if len(enc_df) else "(empty)")
    md.append("\n## Per-ideology κ\n")
    md.append(ide_df.to_markdown(index=False, floatfmt=".4f") if len(ide_df) else "(empty)")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")
    print(f"      {OVERALL_CSV}")
    print(f"      {ENC_CSV}")


if __name__ == "__main__":
    main()
