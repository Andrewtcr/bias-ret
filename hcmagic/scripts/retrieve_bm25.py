"""BM25 baseline for the shared HealthCareMagic answer corpus.
Uses `bm25s` for indexing and scoring.

Scores BOTH arms when enabled in the config:
  - hcmagic_synth (synth-paired): 4 variants per row, 19,996 queries
  - hcmagic real:                 aal_real / wme_real / trans_aal queries

Outputs (under hcmagic/results/bm25/):
  hcmagic_synth_per_query_metrics.csv   (if synth arm enabled)
  hcmagic_real_per_query_metrics.csv    (if real arm enabled)

Matches the dense per-encoder per-query CSV schema so analyze_retrieval.py picks
both up via the bm25 slug.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.io import ensure_dir, load_yaml  # noqa: E402
from hcmagic.lib import (  # noqa: E402
    ROOT, RES, load_hcmagic_real, load_hcmagic_synth_paired,
)
from hcmagic.metrics import score_metrics  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(t: str) -> list[str]:
    return TOKEN_RE.findall(t.lower())


def _scores_for(retriever: bm25s.BM25, q_texts: list[str],
                n_corpus: int) -> np.ndarray:
    """bm25s.retrieve returns top-k indices/scores. We need full per-doc
    scores so the existing score_metrics can argsort. Use get_scores per
    query, which returns a (n_corpus,) score vector."""
    out = np.zeros((len(q_texts), n_corpus), dtype=np.float32)
    for i, qtext in enumerate(q_texts):
        toks = tokenize(qtext)
        scores = retriever.get_scores(toks)
        out[i] = scores.astype(np.float32)
        if (i + 1) % 5000 == 0:
            print(f"[bm25]   scored {i+1}/{len(q_texts)}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    corpus_jsonl = ROOT / cfg["corpus"]["corpus_jsonl"]

    # Load corpus
    doc_ids, texts = [], []
    with open(corpus_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["answer"])
    print(f"[bm25] corpus: {len(doc_ids):,} docs")

    # Index
    t0 = time.time()
    corpus_tokens = [tokenize(t) for t in texts]
    print(f"[bm25] tokenized in {time.time()-t0:.1f}s")
    t1 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)
    print(f"[bm25] index built in {time.time()-t1:.1f}s")

    out_dir = ensure_dir(RES / "bm25")

    # --- Synth arm ---
    if cfg["datasets"].get("hcmagic_synth", {}).get("enabled"):
        hs = cfg["datasets"]["hcmagic_synth"]
        with open(ROOT / hs["qrels_path"]) as f:
            qrels = json.load(f)
        variants = load_hcmagic_synth_paired(ROOT / hs["paired_jsonl"])
        synth_rows = []
        for vname in ("wme_real", "aal_synth1", "aal_synth2", "aal_synth3"):
            recs = variants[vname]
            qids = [r["query_id"] for r in recs]
            q_texts = [r["text"] for r in recs]
            t2 = time.time()
            scores_mat = _scores_for(retriever, q_texts, len(doc_ids))
            print(f"[bm25] synth {vname}: scored in {time.time()-t2:.1f}s")
            rows = score_metrics(scores_mat, qids, doc_ids, qrels, vname)
            synth_rows.extend(rows)
        out_path = out_dir / "hcmagic_synth_per_query_metrics.csv"
        pd.DataFrame(synth_rows).to_csv(out_path, index=False)
        print(f"[bm25] wrote {out_path}  ({len(synth_rows)} rows)")

    # --- Real arm ---
    hd = cfg["datasets"].get("hcmagic", {})
    if hd.get("enabled"):
        with open(ROOT / hd["qrels_path"]) as f:
            qrels = json.load(f)
        groups = load_hcmagic_real(
            ROOT / hd["source_csv"],
            ROOT / hd["translated_jsonl"],
        )
        real_rows = []
        for variant in ("aal_real", "wme_real", "trans_aal"):
            recs = groups[variant]
            qids = [r["query_id"] for r in recs]
            q_texts = [r["text"] for r in recs]
            t2 = time.time()
            scores_mat = _scores_for(retriever, q_texts, len(doc_ids))
            print(f"[bm25] real {variant}: scored in {time.time()-t2:.1f}s")
            rows = score_metrics(scores_mat, qids, doc_ids, qrels, variant)
            real_rows.extend(rows)
        out_path = out_dir / "hcmagic_real_per_query_metrics.csv"
        pd.DataFrame(real_rows).to_csv(out_path, index=False)
        print(f"[bm25] wrote {out_path}  ({len(real_rows)} rows)")


if __name__ == "__main__":
    main()
