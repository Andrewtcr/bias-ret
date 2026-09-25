"""Dense retrieval with per-query source-article masking."""
from __future__ import annotations

import numpy as np


def dense_retrieve_with_mask(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    corpus_ids: list[str],
    query_source_article_ids: list[str],
    top_k: int,
) -> list[list[tuple[str, float]]]:
    """Cosine top-k retrieval with per-query masking of a single source-article id."""
    scores = query_embs @ corpus_embs.T
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    for qi, src in enumerate(query_source_article_ids):
        idx = id_to_idx.get(src)
        if idx is not None:
            scores[qi, idx] = -np.inf
    out: list[list[tuple[str, float]]] = []
    for qi in range(scores.shape[0]):
        order = np.argsort(-scores[qi])[:top_k]
        out.append([(corpus_ids[j], float(scores[qi, j])) for j in order])
    return out


__all__ = [
    "dense_retrieve_with_mask",
]
