"""Retrieval metrics for the AllSides synthetic-frame experiment.

Computes per-query political lean scores from ranked candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STANCE_NUM = {"left": -1, "center": 0, "right": +1}


def compute_bias_metrics(
    candidates_df: pd.DataFrame,
    k_values: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Per (query_id, encoder, k): lean_score, the mean stance value
    (left = -1, center = 0, right = +1) of the top-k retrieved articles.

    Each row also carries the query's article_id, frame, and stance, so
    downstream analyses can pair the left and right query of a cell.
    """
    rows = []
    grouped = candidates_df.groupby(["query_id", "encoder"])
    for (query_id, encoder), g in grouped:
        g_sorted = g.sort_values("rank")
        sample = g_sorted.iloc[0]
        article_id = sample["article_id"]
        frame = sample["frame"]
        stance = sample["stance"]
        for k in k_values:
            top = g_sorted.head(k)
            stances = top["retrieved_stance"].tolist()
            lean = float(np.mean([STANCE_NUM[s] for s in stances])) if stances else 0.0
            rows.append(
                {
                    "query_id": query_id,
                    "article_id": article_id,
                    "frame": frame,
                    "stance": stance,
                    "encoder": encoder,
                    "k": k,
                    "lean_score": lean,
                }
            )
    return pd.DataFrame(rows)
