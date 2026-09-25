"""Gold-filtering robustness appendix figures.

Generates, into figs/output/:
  gold_filter_unfiltered.{pdf,png}   2-panel forest: synth + reddit, lean over FULL top-10 (no relevance filter)
  gold_filter_synth_llm.{pdf,png}    single forest: AllSides lean over AGREE-both LLM-judged-relevant docs
  gold_filter_reddit_full10.{pdf,png} single forest: Reddit lean over FULL top-10, retention-gated (synth-style)
  gold_filter_relevance_dist.{pdf,png} #relevant docs in top-10 per (query,retriever), synth vs reddit (retained; raw reported)
  gold_filter_confusion.{pdf,png}    synth LLM-judge (AGREE-both) vs editorial same-roundup gold, 2x2

Numerical output: figs/results/gold_filter_conditions.csv
  Per-retriever gap under every condition (supports the gold-filter appendix text).

All forests reuse the Fig 2 style (figs/scripts/plot_political_overview.py).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from figs.scripts._style import (  # noqa: E402
    ENCODER_DISPLAY, ENCODER_ORDER, STANCE_COLOR, apply_rc, save_both,
)

AS = REPO / "allsides_synth" / "results"
ASJ = AS / "judge_relevance"
ASDATA = REPO / "allsides_synth" / "data"
RD = REPO / "reddit_nat" / "results"
RDJ = RD / "judge_relevance"
FIGS = REPO / "figs" / "output"
RESULTS = REPO / "figs" / "results"

DISPLAY_ORDER = list(ENCODER_ORDER) + ["bm25"]
DISPLAY_NAMES = dict(ENCODER_DISPLAY, **{"bm25": "BM25"})
LEFT, RIGHT = STANCE_COLOR["left"], STANCE_COLOR["right"]
RNG = np.random.default_rng(0)
B = 5000


def _stars(p):
    if p is None or np.isnan(p):
        return ""
    return " ***" if p < .001 else " **" if p < .01 else " *" if p < .05 else ""


def _agree_gold(jd):
    with open(jd / "judgments.jsonl") as f:
        g = pd.DataFrame([json.loads(l) for l in f])
    with open(jd / "judgments_qwen.jsonl") as f:
        q = pd.DataFrame([json.loads(l) for l in f])
    q = q[q["grade"].notna()].copy(); q["grade"] = q["grade"].astype(int)
    m = g.merge(q, on=["query_id", "article_id"], suffixes=("_g", "_q"))
    m["gold"] = ((m["grade_g"] == 1) & (m["grade_q"] == 1)).astype(int)
    return m


def _ci(x):
    x = np.asarray(x, float)
    means = np.array([RNG.choice(x, len(x)).mean() for _ in range(B)])
    return x.mean(), np.quantile(means, .025), np.quantile(means, .975)


def _lean(docs):
    return ((docs["retrieved_stance"] == "right").sum() - (docs["retrieved_stance"] == "left").sum()) / max(len(docs), 1)


def per_query_lean(sub, mode):
    out = {}
    for qid, g in sub.groupby("query_id"):
        if mode == "relevant":
            docs = g[g["gold"] == 1]
            if len(docs) == 0:
                continue
        elif mode == "full_retained":
            if g["gold"].sum() < 1:
                continue
            docs = g
        else:  # full
            docs = g
        if len(docs) == 0:
            continue
        out[qid] = _lean(docs)
    return out


# ---------- load data ----------
def load_synth():
    m = _agree_gold(ASJ)
    with open(ASDATA / "generated_queries.jsonl") as f:
        qmeta = {d["query_id"]: (d["article_id"], d["frame"], d["stance"])
                 for d in (json.loads(l) for l in f)
                 if d["stance"] in ("left", "right")}
    cols = ["query_id", "encoder", "rank", "retrieved_article_id", "retrieved_stance"]
    c = pd.concat([pd.read_csv(p, usecols=cols)
                   for p in sorted((AS / "retrieval_candidates").glob("*.csv"))], ignore_index=True)
    c = c[(c["rank"] <= 10) & c["query_id"].isin(qmeta)].rename(columns={"retrieved_article_id": "article_id"})
    c = c.merge(m[["query_id", "article_id", "gold"]], on=["query_id", "article_id"], how="left")
    c["gold"] = c["gold"].fillna(0).astype(int)
    return c, qmeta, m


def load_reddit():
    m = _agree_gold(RDJ)
    cols = ["query_id", "ideology", "encoder", "rank", "retrieved_article_id", "retrieved_stance"]
    c = pd.concat([pd.read_csv(p, usecols=cols)
                   for p in sorted((RD / "retrieval_candidates").glob("*.csv"))], ignore_index=True)
    c = c[(c["rank"] <= 10) & c["ideology"].isin(["liberal", "conservative"])].rename(columns={"retrieved_article_id": "article_id"})
    c = c.merge(m[["query_id", "article_id", "gold"]], on=["query_id", "article_id"], how="left")
    c["gold"] = c["gold"].fillna(0).astype(int)
    return c


# ---------- summaries ----------
def synth_summary(cand, qmeta, mode):
    """Return (rows DataFrame: encoder,group,mean,ci_low,ci_high,n), pvals dict (paired Wilcoxon on R-L)
    and gaps dict (encoder -> (mean paired R-L, n pairs))."""
    rows, pvals, gaps = [], {}, {}
    for enc in DISPLAY_ORDER:
        leans = per_query_lean(cand[cand["encoder"] == enc], mode)
        byg = {"left": [], "right": []}
        byaf = {}
        for qid, l in leans.items():
            a, f, s = qmeta[qid]
            byg[s].append(l)
            byaf.setdefault((a, f), {})[s] = l
        for s in ("left", "right"):
            m, lo, hi = _ci(byg[s])
            rows.append({"encoder": enc, "group": s, "mean": m, "ci_low": lo, "ci_high": hi, "n": len(byg[s])})
        d = np.array([v["right"] - v["left"] for v in byaf.values() if "left" in v and "right" in v])
        pvals[enc] = stats.wilcoxon(d)[1] if len(d) > 5 and d.any() else np.nan
        gaps[enc] = (d.mean(), len(d))
    return pd.DataFrame(rows), pvals, gaps


def reddit_summary(cand, mode):
    rows, pvals, gaps = [], {}, {}
    for enc in DISPLAY_ORDER:
        sub = cand[cand["encoder"] == enc]
        leans = per_query_lean(sub, mode)
        ide = sub.drop_duplicates("query_id").set_index("query_id")["ideology"].to_dict()
        byg = {"liberal": [], "conservative": []}
        for qid, l in leans.items():
            byg[ide[qid]].append(l)
        for g in ("liberal", "conservative"):
            m, lo, hi = _ci(byg[g])
            rows.append({"encoder": enc, "group": g, "mean": m, "ci_low": lo, "ci_high": hi, "n": len(byg[g])})
        lib, con = np.array(byg["liberal"]), np.array(byg["conservative"])
        pvals[enc] = stats.mannwhitneyu(con, lib, alternative="two-sided")[1] if len(lib) > 5 and len(con) > 5 else np.nan
        gaps[enc] = (con.mean() - lib.mean(), len(lib) + len(con))
    return pd.DataFrame(rows), pvals, gaps


# ---------- forest drawing (Fig 2 style) ----------
def draw_forest(ax, summary, pvals, title, gleft, gright):
    off = 0.18
    yt = []
    for i, enc in enumerate(DISPLAY_ORDER):
        y = len(DISPLAY_ORDER) - 1 - i
        yt.append(y)
        for grp, color, dy in [(gleft, LEFT, +off), (gright, RIGHT, -off)]:
            r = summary[(summary.encoder == enc) & (summary.group == grp)]
            if r.empty:
                continue
            m, lo, hi = r.iloc[0][["mean", "ci_low", "ci_high"]]
            ax.errorbar(m, y + dy, xerr=[[m - lo], [hi - m]], fmt="o", color=color, markersize=4.5,
                        markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
                        elinewidth=1.1, capsize=2.0, capthick=0.8, zorder=3)
    for y in yt:
        ax.axhline(y, color="#EEE", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#333", linewidth=0.7, linestyle="--", zorder=0)
    ax.set_yticks(yt)
    ax.set_yticklabels([DISPLAY_NAMES[e] + _stars(pvals.get(e)) for e in DISPLAY_ORDER], fontsize=7.5)
    ax.set_xlabel(r"Mean Retrieval Lean ($\%R-\%L$)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(-0.6, len(DISPLAY_ORDER) - 0.3)
    ax.set_title(title, fontsize=9, pad=4)
    # n_L / n_R margin
    XL, XS, XR = 1.16, 1.20, 1.24
    ax.text((XL + XR) / 2, len(DISPLAY_ORDER) - 0.3, "n", transform=ax.get_yaxis_transform(),
            fontsize=7, color="#333", ha="center", va="top")
    for i, enc in enumerate(DISPLAY_ORDER):
        y = len(DISPLAY_ORDER) - 1 - i
        nl = int(summary[(summary.encoder == enc) & (summary.group == gleft)].iloc[0]["n"])
        nr = int(summary[(summary.encoder == enc) & (summary.group == gright)].iloc[0]["n"])
        ax.text(XL, y, f"{nl}", transform=ax.get_yaxis_transform(), fontsize=6.0, color=LEFT, ha="right", va="center")
        ax.text(XS, y, "/", transform=ax.get_yaxis_transform(), fontsize=6.0, color="#666", ha="center", va="center")
        ax.text(XR, y, f"{nr}", transform=ax.get_yaxis_transform(), fontsize=6.0, color=RIGHT, ha="left", va="center")


def main():
    apply_rc()
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)

    scand, qmeta, sm = load_synth()
    rcand = load_reddit()

    s_unf, s_unf_p, s_unf_g = synth_summary(scand, qmeta, "full")
    s_llm, s_llm_p, s_llm_g = synth_summary(scand, qmeta, "relevant")
    r_unf, r_unf_p, r_unf_g = reddit_summary(rcand, "full")
    r_f10, r_f10_p, r_f10_g = reddit_summary(rcand, "full_retained")

    # ---- Fig: unfiltered 2-panel ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.4))
    draw_forest(a1, s_unf, s_unf_p, r"(a) AllSides$_{\mathrm{synth}}$, unfiltered", "left", "right")
    draw_forest(a2, r_unf, r_unf_p, r"(b) Reddit$_{\mathrm{nat}}$, unfiltered", "liberal", "conservative")
    fig.tight_layout(w_pad=5.0)
    save_both(fig, str(FIGS / "gold_filter_unfiltered"))

    # ---- Fig: synth LLM-judge (single) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw_forest(ax, s_llm, s_llm_p, r"AllSides$_{\mathrm{synth}}$, LLM-judged relevant", "left", "right")
    fig.tight_layout()
    save_both(fig, str(FIGS / "gold_filter_synth_llm"))

    # ---- Fig: reddit full-top-10 (single) ----
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw_forest(ax, r_f10, r_f10_p, r"Reddit$_{\mathrm{nat}}$, full top-10 (retention-gated)", "liberal", "conservative")
    fig.tight_layout()
    save_both(fig, str(FIGS / "gold_filter_reddit_full10"))

    # ---- Fig: relevance distribution (retained; raw reported) ----
    syn_ng = scand.groupby(["encoder", "query_id"])["gold"].sum()
    red_ng = rcand.groupby(["encoder", "query_id"])["gold"].sum()
    syn_ret = syn_ng[syn_ng >= 1]
    red_ret = red_ng[red_ng >= 1]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bins = np.arange(0.5, 11.5, 1)
    ax.hist(syn_ret.values, bins=bins, density=True, histtype="step", linewidth=1.6,
            color="#009988", label=f"AllSides$_{{\\mathrm{{synth}}}}$ (med {int(syn_ret.median())})")
    ax.hist(red_ret.values, bins=bins, density=True, histtype="step", linewidth=1.6,
            color="#332288", label=f"Reddit$_{{\\mathrm{{nat}}}}$ (med {int(red_ret.median())})")
    ax.set_xlabel("# relevant docs in top-10 (retained queries)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save_both(fig, str(FIGS / "gold_filter_relevance_dist"))
    raw = {"synth_pct_ge1": 100 * (syn_ng >= 1).mean(), "reddit_pct_ge1": 100 * (red_ng >= 1).mean(),
           "synth_med_retained": float(syn_ret.median()), "reddit_med_retained": float(red_ret.median()),
           "synth_mean_all": float(syn_ng.mean()), "reddit_mean_all": float(red_ng.mean())}

    # ---- Fig: confusion (synth LLM vs editorial same-roundup) ----
    with open(ASDATA / "corpus.jsonl") as f:
        roundup = {d["article_id"]: d.get("roundup_title", "")
                   for d in (json.loads(l) for l in f)}
    src = {qid: qmeta[qid][0] for qid in qmeta}
    sm2 = sm[sm["query_id"].isin(qmeta)].copy()
    sm2["edit"] = sm2.apply(lambda r: int(roundup.get(r["article_id"], "_x") == roundup.get(src.get(r["query_id"]), "_y")), axis=1)
    ct = pd.crosstab(sm2["edit"], sm2["gold"]).reindex(index=[1, 0], columns=[1, 0])
    po = (ct.loc[1, 1] + ct.loc[0, 0]) / ct.values.sum()
    tot = ct.values.sum()
    pe = (((ct.loc[1].sum()) * (ct[1].sum()) + (ct.loc[0].sum()) * (ct[0].sum())) / tot**2)
    kappa = (po - pe) / (1 - pe)
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    im = ax.imshow(ct.values, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["relevant", "not"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["same-\nroundup", "not"], fontsize=8)
    ax.set_xlabel("LLM-judge (both agree)", fontsize=8)
    ax.set_ylabel("editorial gold", fontsize=8)
    ax.set_title(rf"AllSides$_{{\mathrm{{synth}}}}$  ($\kappa={kappa:.2f}$)", fontsize=9, pad=4)
    for i in range(2):
        for j in range(2):
            v = ct.values[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > ct.values.max() / 2 else "#222")
    fig.tight_layout()
    save_both(fig, str(FIGS / "gold_filter_confusion"))

    # ---- summary CSV ----
    rows = []
    for label, gaps, pv in [("synth_unfiltered", s_unf_g, s_unf_p), ("synth_llm", s_llm_g, s_llm_p),
                            ("reddit_unfiltered", r_unf_g, r_unf_p), ("reddit_full10", r_f10_g, r_f10_p)]:
        for enc in DISPLAY_ORDER:
            gap, n = gaps[enc]
            rows.append({"condition": label, "retriever": enc, "gap": round(gap, 4), "n": n, "p": pv[enc]})
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "gold_filter_conditions.csv", index=False)

    print("=== relevance (raw) ===")
    for k, v in raw.items():
        print(f"  {k}: {v:.2f}")
    print(f"=== confusion kappa={kappa:.3f}, raw agreement {100*po:.0f}% ===")
    print(ct)
    print("wrote figures to", FIGS)


if __name__ == "__main__":
    main()
