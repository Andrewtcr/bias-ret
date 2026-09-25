"""Retrieve top-k AllSides articles for Reddit queries with BM25.

Existing committed candidates (results/retrieval_candidates/bm25.csv) are
reused, and --top_k is then ignored; delete that file to rebuild them.
Building candidates requires rehydrated post titles and the AllSides article
corpus.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reddit_nat import lib  # noqa: E402
from src.io import load_yaml, read_jsonl  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def build_bm25_candidates(cfg: dict, out_path: Path, top_k: int = 100) -> pd.DataFrame:
    df_q = lib.load_reddit_queries(cfg)
    corpus = read_jsonl(lib.corpus_jsonl_path(cfg))
    corpus_ids = [r["article_id"] for r in corpus]
    stance_by_id = {r["article_id"]: r["stance"] for r in corpus}
    print(f"[bm25] tokenizing {len(corpus)} corpus docs...")
    t0 = time.time()
    tokenized = [tokenize(r["text"]) for r in corpus]
    print(f"[bm25]   tokenized in {time.time()-t0:.1f}s")
    from rank_bm25 import BM25Okapi
    t1 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"[bm25]   index built in {time.time()-t1:.1f}s")

    rows = []
    t2 = time.time()
    for qi, q in enumerate(df_q.to_dict(orient="records")):
        toks = tokenize(q["text"])
        scores = bm25.get_scores(toks)
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        for rank, idx in enumerate(top_idx, start=1):
            doc_id = corpus_ids[idx]
            rows.append({
                "query_id": q["query_id"],
                "ideology": q["ideology"],
                "subreddit": q["subreddit"],
                "encoder": "bm25",
                "rank": rank,
                "retrieved_article_id": doc_id,
                "retrieved_stance": stance_by_id.get(doc_id, "unknown"),
                "score": float(scores[idx]),
            })
        if (qi + 1) % 100 == 0:
            print(f"[bm25]   {qi+1}/{len(df_q)} in {time.time()-t2:.1f}s")
    print(f"[bm25] scored all in {time.time()-t2:.1f}s")
    df_out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[bm25] wrote {out_path} ({len(df_out)} rows)")
    return df_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--top_k", type=int, default=100)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    rdir = lib.run_dir(cfg)
    bm25_path = lib.bm25_candidate_file(rdir)
    if bm25_path.exists():
        print(f"[bm25] reusing existing {bm25_path}; delete it to rebuild")
    else:
        build_bm25_candidates(cfg, bm25_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
