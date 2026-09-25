"""Relevance metrics shared by dense and BM25 HealthCareMagic retrieval."""
from __future__ import annotations

import numpy as np

K_VALUES = [1, 5, 10, 20, 50, 100]


# Each query has ONE gold answer; the qrels entry lists every doc_id whose
# answer text is byte-identical to it (an interchangeable equivalence class,
# see prepare_corpus.py). All metrics therefore reduce to the rank of the
# best-ranked member of that class.
def first_relevant_rank(ranked_ids, grade_map) -> int | None:
    for rank, did in enumerate(ranked_ids, start=1):
        if grade_map.get(did, 0) >= 1:
            return rank
    return None


def reciprocal_rank(ranked_ids, grade_map) -> float:
    rank = first_relevant_rank(ranked_ids, grade_map)
    return 1.0 / rank if rank is not None else 0.0


def recall_at_k(ranked_ids, grade_map, k) -> float:
    rank = first_relevant_rank(ranked_ids, grade_map)
    return 1.0 if rank is not None and rank <= k else 0.0


def score_metrics(scores, qids, corpus_ids, qrels, variant):
    """Return list of per-query rows."""
    rows = []
    for qi, qid in enumerate(qids):
        grade_map = qrels.get(qid)
        if not grade_map:
            continue
        ranked = np.argsort(-scores[qi])
        ranked_ids = [corpus_ids[i] for i in ranked]
        row = {
            "query_id": qid,
            "variant": variant,
            "mrr": reciprocal_rank(ranked_ids, grade_map),
        }
        for k in K_VALUES:
            row[f"recall@{k}"] = recall_at_k(ranked_ids, grade_map, k)
        rows.append(row)
    return rows

