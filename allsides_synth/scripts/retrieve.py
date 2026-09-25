"""Retrieve ranked articles from cached embeddings and compute bias metrics.

Reads allsides_synth/embeddings/{slug}/ (produced by embed.py), masks each
query's source article, and writes rankings and metrics to allsides_synth/results/.
Directory overrides are set in the config's run section.

Usage:
  python allsides_synth/scripts/retrieve.py \
      --config allsides_synth/configs/retrieval.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from allsides_synth import lib  # noqa: E402
from src.embedding import model_slug  # noqa: E402
from src.io import ensure_dir, load_yaml, read_jsonl  # noqa: E402
from src.retrieval import dense_retrieve_with_mask  # noqa: E402


def _resolve_encoder_items(cfg: dict) -> list[dict]:
    return [{"name": e} if isinstance(e, str) else dict(e) for e in cfg["retrieval"]["encoders"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    rdir = lib.run_dir(cfg)
    ddir = lib.data_dir(cfg)
    edir = lib.embeddings_dir(cfg)

    corpus = read_jsonl(ddir / "corpus.jsonl")
    stance_by_id = {r["article_id"]: r["stance"] for r in corpus}
    print(f"[retrieve] corpus={len(corpus)}")

    items = _resolve_encoder_items(cfg)
    candidate_rows = []
    for item in items:
        enc = item["name"]
        slug = model_slug(enc)
        emb_dir = edir / slug
        cpath = emb_dir / "corpus_embeddings.npy"
        qpath = emb_dir / "query_embeddings.npy"
        queries_path = emb_dir / "queries.jsonl"
        if not (cpath.exists() and qpath.exists() and queries_path.exists()):
            raise FileNotFoundError(
                f"Missing embeddings or queries.jsonl for {slug}; "
                f"run embed.py first."
            )
        corpus_embs = np.load(cpath)
        query_embs = np.load(qpath)
        queries = read_jsonl(queries_path)
        cids = json.loads((emb_dir / "corpus_ids.json").read_text())
        print(f"[retrieve] {slug}: corpus={corpus_embs.shape} query={query_embs.shape} "
              f"queries={len(queries)}")

        query_srcs = [q["article_id"] for q in queries]
        top_k = int(cfg["retrieval"]["top_k"])
        ranked = dense_retrieve_with_mask(query_embs, corpus_embs, cids, query_srcs, top_k=top_k)
        for q, hits in zip(queries, ranked):
            for rank, (doc_id, score) in enumerate(hits, start=1):
                candidate_rows.append({
                    "query_id": q.get("query_id"),
                    "article_id": q["article_id"],
                    "frame": q["frame"],
                    "stance": q["stance"],
                    "encoder": slug,
                    "rank": rank,
                    "retrieved_article_id": doc_id,
                    "retrieved_stance": stance_by_id.get(doc_id, "unknown"),
                    "score": score,
                })

    cand_df = pd.DataFrame(candidate_rows)
    # The retrieval output is an ID-only ranking, written one file per encoder.
    # Query-side metadata (article_id, frame, stance) joins back from the query
    # JSONL on `query_id`; retrieved document text joins from the corpus on
    # `retrieved_article_id`. Downstream code does those joins on demand, so we
    # never persist the (large) denormalized text. The files keep the top 10
    # ranks that the relevance judge and figures read; the full top_k cand_df
    # is retained below for the story ranks and bias metrics.
    cand_cols = ["query_id", "encoder", "rank",
                 "retrieved_article_id", "retrieved_stance"]
    cand_dir = ensure_dir(rdir / "retrieval_candidates")
    for enc, g in cand_df.groupby("encoder"):
        top10 = g[g["rank"] <= 10]
        top10[cand_cols].to_csv(cand_dir / f"{enc}.csv", index=False)
        print(f"[retrieve] wrote retrieval_candidates/{enc}.csv ({len(top10)} rows)")

    print("[story] building article -> story_id lookup...")
    story_lookup = lib.build_story_lookup(cfg)

    # Gold-filter cache: smallest rank at which a same-story (same AllSides
    # roundup) document is retrieved, per (query_id, encoder). The appendix
    # gold-filter figures/tables read this committed parquet directly.
    aid_to_story = {a: info["story_id"] for a, info in story_lookup.items()}
    sr = cand_df[["query_id", "encoder", "rank",
                  "article_id", "retrieved_article_id"]].copy()
    sr["src_story"] = sr["article_id"].map(aid_to_story)
    sr["ret_story"] = sr["retrieved_article_id"].map(aid_to_story)
    sr = sr[sr["src_story"].notna() & sr["ret_story"].notna()
            & (sr["src_story"] == sr["ret_story"])]
    (sr.groupby(["query_id", "encoder"])["rank"].min()
       .rename("min_story_rank").reset_index()
       .to_parquet(rdir / "_min_story_rank.parquet"))

    K_VALUES = (5, 10, 20, 50, 100)
    metrics_df = lib.compute_bias_metrics(cand_df, k_values=K_VALUES)
    metrics_df.to_csv(rdir / "bias_metrics.csv", index=False)

    print(f"[retrieve] wrote rankings, story ranks, and bias_metrics -> {rdir}")


if __name__ == "__main__":
    main()
