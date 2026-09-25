"""
Within-cell pairing similarity check for the AllSides political-bias arm.

Tests, per encoder, whether L vs R query cosine similarity differs across three
pairing schemes:

  1. within-cell:                cos(L_c, R_c) for c=(article, frame)
  2. within-article-cross-frame: cos(L_c, R_c') with c' a same-article-other-frame cell
  3. cross-article:              cos(L_c, R_c') with c' from a different article

The paired Wilcoxon test compares within-cell and within-article cross-frame
similarities to assess the contribution of frame-level pairing.

Run:
    python allsides_synth/scripts/analyze_pairing_similarity.py \
        --config allsides_synth/configs/pairing_similarity.yaml

Rebuilds per-anchor scores and summary statistics from the embeddings bundle.
Render the paper figure separately with
allsides_synth/scripts/plot_pairing_similarity.py.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# ----------------------------- IO ---------------------------------------------

def load_queries(emb_dir: Path) -> list[dict]:
    with open(emb_dir / "queries.jsonl") as f:
        rows = [json.loads(l) for l in f]
    with open(emb_dir / "query_ids.json") as f:
        qids = json.load(f)
    assert len(rows) == len(qids), f"row/id count mismatch in {emb_dir}"
    for r, qid in zip(rows, qids):
        assert r["query_id"] == qid, f"order mismatch in {emb_dir}: {r['query_id']} vs {qid}"
    return rows


def load_embeddings(emb_dir: Path) -> np.ndarray:
    arr = np.load(emb_dir / "query_embeddings.npy")
    # L2-normalize once so cosine = dot.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# ----------------------------- core -------------------------------------------

def build_cell_index(rows: list[dict], left_label: str, right_label: str):
    """Return:
       cells: list of dicts {article_id, frame, left_idx, right_idx, cell_key}
              for cells with BOTH a left and a right query present.
       by_article: article_id -> list of cell indices into `cells`.
    """
    # Map (article, frame, stance) -> row idx
    triple_to_idx: dict[tuple[str, str, str], int] = {}
    for i, r in enumerate(rows):
        key = (r["article_id"], r["frame"], r["stance"])
        if key in triple_to_idx:
            raise ValueError(f"duplicate (article, frame, stance): {key}")
        triple_to_idx[key] = i

    # Enumerate cells with both L and R.
    cells = []
    by_article: dict[str, list[int]] = defaultdict(list)
    pairs = set()
    for (art, frame, stance), idx in triple_to_idx.items():
        if (art, frame) in pairs:
            continue
        pairs.add((art, frame))
        l = triple_to_idx.get((art, frame, left_label))
        r = triple_to_idx.get((art, frame, right_label))
        if l is None or r is None:
            continue
        cells.append({
            "article_id": art,
            "frame": frame,
            "left_idx": l,
            "right_idx": r,
            "cell_key": f"{art}__{frame}",
        })
        by_article[art].append(len(cells) - 1)

    return cells, by_article


def compute_per_anchor_sims(
    embeddings: np.ndarray,
    cells: list[dict],
    by_article: dict[str, list[int]],
    n_draws: int,
    rng: np.random.Generator,
):
    """For each anchor cell c, return three scalars:
       (within_cell, within_article_cross_frame, cross_article).
       within_article_cross_frame is NaN for anchors with no sibling frame in the same article.
       Baseline values are averages over n_draws random partner cells.
    """
    n = len(cells)
    left_idx = np.array([c["left_idx"] for c in cells])
    right_idx = np.array([c["right_idx"] for c in cells])
    article_of_cell = np.array([c["article_id"] for c in cells])
    all_cell_idx = np.arange(n)

    L = embeddings[left_idx]   # (n, d) already normalized
    R = embeddings[right_idx]  # (n, d)

    # within-cell: row-wise dot
    within_cell = np.einsum("nd,nd->n", L, R)

    # within-article-cross-frame: per anchor, sample other cells from same article
    within_art = np.full(n, np.nan)
    has_sibling = np.zeros(n, dtype=bool)
    for ci in range(n):
        sibs = [s for s in by_article[article_of_cell[ci]] if s != ci]
        if not sibs:
            continue
        has_sibling[ci] = True
        draws = rng.choice(sibs, size=min(n_draws, len(sibs)), replace=(len(sibs) < n_draws))
        sims = L[ci] @ R[draws].T   # (k,)
        within_art[ci] = float(sims.mean())

    # cross-article: per anchor, sample cells from different articles
    cross_art = np.full(n, np.nan)
    for ci in range(n):
        # Sample a candidate pool, exclude the anchor article, and keep n_draws.
        cand = rng.choice(all_cell_idx, size=n_draws * 4, replace=False)
        cand = cand[article_of_cell[cand] != article_of_cell[ci]]
        if len(cand) < n_draws:
            # Fall back to the full eligible pool if rejection leaves too few draws.
            cand = np.array([j for j in all_cell_idx if article_of_cell[j] != article_of_cell[ci]])
            cand = rng.choice(cand, size=n_draws, replace=False)
        else:
            cand = cand[:n_draws]
        sims = L[ci] @ R[cand].T
        cross_art[ci] = float(sims.mean())

    return within_cell, within_art, cross_art, has_sibling


# ----------------------------- stats ------------------------------------------

def summarize_dist(x: np.ndarray) -> dict:
    x = x[~np.isnan(x)]
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q1": float(np.quantile(x, 0.25)),
        "q3": float(np.quantile(x, 0.75)),
    }


def paired_wilcoxon(a: np.ndarray, b: np.ndarray):
    """Paired one-sided Wilcoxon: H1 a > b. Drops NaNs pairwise."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return float("nan"), float("nan"), int(len(a))
    res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue), int(len(a))


def cluster_bootstrap_median_delta(
    delta: np.ndarray,
    article_of_anchor: np.ndarray,
    B: int,
    rng: np.random.Generator,
):
    """Cluster (article) bootstrap of median(delta). Returns (point, lo95, hi95)."""
    mask = ~np.isnan(delta)
    delta = delta[mask]
    arts = article_of_anchor[mask]

    # Group anchors by article.
    art_to_pos: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(arts):
        art_to_pos[a].append(i)
    unique_arts = np.array(list(art_to_pos.keys()))

    point = float(np.median(delta))
    medians = np.empty(B)
    for b in range(B):
        sampled = rng.choice(unique_arts, size=len(unique_arts), replace=True)
        idx_chunks = [art_to_pos[a] for a in sampled]
        idx = np.fromiter((i for chunk in idx_chunks for i in chunk), dtype=np.int64)
        medians[b] = np.median(delta[idx])
    lo, hi = np.quantile(medians, [0.025, 0.975])
    return point, float(lo), float(hi)


# ----------------------------- main -------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    emb_root = Path(cfg["embeddings_root"])
    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["random_seed"])
    n_draws = int(cfg["n_baseline_draws_per_anchor"])
    B = int(cfg["bootstrap_B"])

    # Build cell index from the first encoder (schema identical across encoders;
    # we re-verify per encoder below).
    first = cfg["encoders"][0]
    rows0 = load_queries(emb_root / first)
    cells, by_article = build_cell_index(rows0, cfg["left_label"], cfg["right_label"])
    n_cells = len(cells)
    n_articles = len(by_article)
    n_articles_multi = sum(1 for v in by_article.values() if len(v) > 1)
    n_cells_with_sibling = sum(1 for c in cells if len(by_article[c["article_id"]]) > 1)
    print(f"[index] cells={n_cells}, articles={n_articles}, "
          f"articles-with-multi-frame={n_articles_multi}, "
          f"cells-with-sibling={n_cells_with_sibling}")

    article_of_anchor = np.array([c["article_id"] for c in cells])

    summary_rows = []

    for enc in cfg["encoders"]:
        emb_dir = emb_root / enc
        rows_e = load_queries(emb_dir)
        # Sanity: schema identical to rows0.
        assert len(rows_e) == len(rows0), f"{enc} row count differs"
        for i in (0, len(rows_e) // 2, len(rows_e) - 1):
            assert rows_e[i]["query_id"] == rows0[i]["query_id"], f"{enc} order mismatch"

        emb = load_embeddings(emb_dir)

        rng = np.random.default_rng(seed)
        wc, wa, ca, has_sib = compute_per_anchor_sims(emb, cells, by_article, n_draws, rng)

        # Stats
        s_wc = summarize_dist(wc)
        s_wa = summarize_dist(wa)
        s_ca = summarize_dist(ca)

        W_wa, p_wa, n_wa = paired_wilcoxon(wc, wa)
        W_ca, p_ca, n_ca = paired_wilcoxon(wc, ca)

        # Effect size: median(wc - wa) with cluster-bootstrap 95% CI.
        delta_wa = wc - wa
        med_d, lo_d, hi_d = cluster_bootstrap_median_delta(
            delta_wa, article_of_anchor, B, np.random.default_rng(seed + 1)
        )
        # Also median(wc - ca) for completeness (no CI; sanity baseline).
        delta_ca = wc - ca
        med_d_ca = float(np.nanmedian(delta_ca))

        summary_rows.append({
            "encoder": enc,
            "n_cells_within_cell": s_wc["n"],
            "wc_mean": s_wc["mean"], "wc_median": s_wc["median"],
            "wc_q1": s_wc["q1"], "wc_q3": s_wc["q3"],
            "n_cells_within_art": s_wa["n"],
            "wa_mean": s_wa["mean"], "wa_median": s_wa["median"],
            "wa_q1": s_wa["q1"], "wa_q3": s_wa["q3"],
            "n_cells_cross_art": s_ca["n"],
            "ca_mean": s_ca["mean"], "ca_median": s_ca["median"],
            "ca_q1": s_ca["q1"], "ca_q3": s_ca["q3"],
            "wilcoxon_W_wc_vs_wa": W_wa, "wilcoxon_p_wc_vs_wa": p_wa, "n_paired_wa": n_wa,
            "wilcoxon_W_wc_vs_ca": W_ca, "wilcoxon_p_wc_vs_ca": p_ca, "n_paired_ca": n_ca,
            "median_delta_wc_minus_wa": med_d,
            "median_delta_wc_minus_wa_lo95": lo_d,
            "median_delta_wc_minus_wa_hi95": hi_d,
            "median_delta_wc_minus_ca": med_d_ca,
        })

        # Persist per-anchor sims for traceability.
        df_anchor = pd.DataFrame({
            "article_id": [c["article_id"] for c in cells],
            "frame": [c["frame"] for c in cells],
            "cell_key": [c["cell_key"] for c in cells],
            "within_cell": wc,
            "within_article_cross_frame": wa,
            "cross_article": ca,
            "has_sibling": has_sib,
        })
        df_anchor.to_csv(out_root / f"per_anchor_sims__{enc}.csv", index=False)
        print(f"[{enc}] wc med={s_wc['median']:.4f}  wa med={s_wa['median']:.4f}  "
              f"ca med={s_ca['median']:.4f}  Δmed(wc-wa)={med_d:.4f} "
              f"[{lo_d:.4f}, {hi_d:.4f}]  p(wc>wa)={p_wa:.2e}")

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_root / "summary.csv", index=False)

    # Write index metadata once.
    with open(out_root / "index_meta.json", "w") as f:
        json.dump({
            "n_cells": n_cells,
            "n_articles": n_articles,
            "n_articles_with_multi_frame": n_articles_multi,
            "n_cells_with_sibling": n_cells_with_sibling,
            "n_articles_dropped_single_frame": n_articles - n_articles_multi,
            "n_cells_no_sibling": n_cells - n_cells_with_sibling,
            "left_label": cfg["left_label"],
            "right_label": cfg["right_label"],
            "n_baseline_draws_per_anchor": n_draws,
            "bootstrap_B": B,
            "random_seed": seed,
        }, f, indent=2)



if __name__ == "__main__":
    main()
