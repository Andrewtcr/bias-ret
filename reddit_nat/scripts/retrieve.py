"""Retrieve top-k Qbias articles per Reddit query per encoder.

Reads query embeddings and paired metadata written by embed_queries.py
and the corpus embeddings produced by allsides_synth. No Reddit post text
is required. Writes one ID-only ranking per encoder (text is joined from
the corpus on demand by downstream code):

  reddit_nat/results/retrieval_candidates/{encoder}.csv
    columns: query_id, ideology, subreddit, encoder, rank,
             retrieved_article_id, retrieved_stance, score
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reddit_nat import lib  # noqa: E402
from src.embedding import model_slug  # noqa: E402
from src.io import load_yaml, read_jsonl  # noqa: E402


def _resolve_encoder_slugs(cfg: dict) -> list[str]:
    return [model_slug(e["name"] if isinstance(e, dict) else e) for e in cfg["retrieval"]["encoders"]]


def _load_corpus_meta(cfg: dict) -> dict[str, dict]:
    """{article_id: {stance}}"""
    out = {}
    for r in read_jsonl(lib.corpus_jsonl_path(cfg)):
        out[r["article_id"]] = {"stance": r["stance"]}
    return out


def _load_query_meta(edir: Path, query_ids: list[str]) -> dict[str, dict]:
    """Use the metadata paired with the encoded rows, without requiring post text."""
    rows = read_jsonl(edir / "queries.jsonl")
    meta = {
        r["query_id"]: {"ideology": r["ideology"], "subreddit": r["subreddit"]}
        for r in rows
    }
    if len(meta) != len(rows) or len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{edir}: duplicate query IDs in the embedding bundle")
    if set(meta) != set(query_ids):
        raise ValueError(f"{edir}: queries.jsonl and query_ids.json contain different query IDs")
    return meta


def _retrieve_for_encoder(slug: str, cfg: dict, corp_meta: dict, top_k: int) -> list[dict]:
    edir = lib.embeddings_dir(cfg) / slug
    cdir = lib.corpus_embeddings_path(cfg, slug)
    q_emb = np.load(edir / "query_embeddings.npy")
    q_ids = json.loads((edir / "query_ids.json").read_text())
    # The embedding sidecar preserves the filtered/deduplicated metadata used
    # during encoding. Re-reading the source CSV can select different edit states.
    query_meta = _load_query_meta(edir, q_ids)
    if len(q_emb) != len(q_ids):
        raise ValueError(f"{edir}: query embedding row count does not match query_ids.json")
    c_emb = np.load(cdir / "corpus_embeddings.npy")
    c_ids = json.loads((cdir / "corpus_ids.json").read_text())
    print(f"[retrieve] {slug}: q={q_emb.shape} c={c_emb.shape}")

    # Normalize defensively (embeddings should already be unit-norm).
    q_emb = q_emb / np.clip(np.linalg.norm(q_emb, axis=1, keepdims=True), 1e-12, None)
    c_emb = c_emb / np.clip(np.linalg.norm(c_emb, axis=1, keepdims=True), 1e-12, None)

    # Compute cosine similarities for the full query/corpus matrix.
    sim = q_emb @ c_emb.T
    topk = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
    # Sort within top_k by score desc
    row_idx = np.arange(len(q_ids))[:, None]
    topk_scores = sim[row_idx, topk]
    order = np.argsort(-topk_scores, axis=1)
    topk = np.take_along_axis(topk, order, axis=1)
    topk_scores = np.take_along_axis(topk_scores, order, axis=1)

    rows = []
    for i, qid in enumerate(q_ids):
        meta = query_meta[qid]
        for rank in range(top_k):
            aid = c_ids[topk[i, rank]]
            cm = corp_meta.get(aid, {})
            rows.append({
                "query_id": qid,
                "ideology": meta["ideology"],
                "subreddit": meta["subreddit"],
                "encoder": slug,
                "rank": rank + 1,
                "retrieved_article_id": aid,
                "retrieved_stance": cm.get("stance", ""),
                "score": float(topk_scores[i, rank]),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--top_k", type=int, default=None,
                   help="Override retrieval.top_k from cfg.")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    top_k = args.top_k or int(cfg["retrieval"].get("top_k", 100))
    print(f"[retrieve] run.tag={cfg['run']['tag']}  top_k={top_k}")

    corp_meta = _load_corpus_meta(cfg)
    print(f"[retrieve] corpus_articles={len(corp_meta)}")

    slugs = _resolve_encoder_slugs(cfg)
    cand_dir = lib.run_dir(cfg) / "retrieval_candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    for slug in slugs:
        rows = _retrieve_for_encoder(slug, cfg, corp_meta, top_k)
        if not rows:
            print(f"[retrieve] {slug}: no rows — skip")
            continue
        out_path = cand_dir / f"{slug}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"[retrieve] {slug}: wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
