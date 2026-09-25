"""Reddit real-query political-bias arm.

Loads the Reddit political-questions CSV, normalizes ideology labels, and
exposes path helpers. The retrieval corpus is shared with allsides_synth —
we never re-embed it here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.io import ensure_dir

REDDIT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REDDIT_ROOT.parent

EMBEDDINGS_ROOT = REDDIT_ROOT / "embeddings"
RESULTS_BASE = REDDIT_ROOT / "results"

IDEOLOGY_CANONICAL = {"liberal", "conservative", "centrist"}


def run_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(RESULTS_BASE)


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    return ensure_dir(EMBEDDINGS_ROOT)


def _resolve_csv(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["dataset"]["csv_path"])
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Reddit CSV missing: {p}")
    return p


def load_reddit_queries(cfg: dict[str, Any]) -> pd.DataFrame:
    """Return DataFrame with columns: query_id, text, ideology, subreddit.

    NOTE: the public release ships post IDs and labels only -- the Reddit post
    text was deliberately withheld (see reddit_nat/README.md, "Released data"). This loader therefore raises unless the caller has
    rehydrated the text locally by re-fetching it from the Reddit API by post
    id; `title_sha256` in the shipped CSV lets a rehydrator verify per post
    that it recovered the exact string used in the paper.
    """
    ds = cfg["dataset"]
    df = pd.read_csv(_resolve_csv(cfg))
    qfield = ds.get("query_field", "title")
    ifield = ds.get("ideology_field", "ideology")
    idfield = ds.get("id_field", "id")
    min_chars = int(ds.get("min_chars", 30))

    if qfield not in df.columns:
        raise SystemExit(
            f"'{qfield}' is not in {_resolve_csv(cfg).name}. The release ships post IDs and\n"
            "labels only; Reddit post text is withheld so that content a user later deletes is\n"
            "not redistributed (paper, Ethical considerations). Every committed result, figure\n"
            "and table is reproducible without it -- see reddit_nat/README.md. To re-run the\n"
            "text-dependent stages (embed / BM25 / judge / lexical), re-fetch the posts by id\n"
            f"from the Reddit API into a '{qfield}' column (the judges also need 'selftext')\n"
            "and verify each title against title_sha256."
        )

    df = df[[idfield, "subreddit", qfield, ifield]].rename(
        columns={idfield: "query_id", qfield: "text", ifield: "ideology"}
    )
    n0 = len(df)
    df = df.dropna(subset=["text", "ideology"])
    df = df[df["text"].str.len() >= min_chars]
    df = df[df["ideology"].isin(IDEOLOGY_CANONICAL)]
    # Drop duplicate post ids (ids appear more than once in the source pull as
    # edit states). Keep first occurrence -- deterministic.
    n_pre_dedupe = len(df)
    df = df.drop_duplicates(subset=["query_id"], keep="first").reset_index(drop=True)
    n_dropped = n_pre_dedupe - len(df)
    print(f"[reddit] loaded {n0} rows; after filter: {len(df)} (dedupe dropped {n_dropped})")
    print(f"[reddit] ideology counts: {dict(df['ideology'].value_counts())}")
    print(f"[reddit] subreddit counts: {dict(df['subreddit'].value_counts())}")
    return df


def candidates_dir(results_dir: Path | None = None) -> Path:
    """Directory holding the per-encoder ID-only retrieval candidates."""
    base = Path(results_dir) if results_dir is not None else RESULTS_BASE
    return base / "retrieval_candidates"


def dense_candidate_files(results_dir: Path | None = None) -> list[Path]:
    """Per-encoder dense candidate CSVs (the BM25 file is excluded)."""
    return sorted(p for p in candidates_dir(results_dir).glob("*.csv")
                  if p.stem != "bm25")


def bm25_candidate_file(results_dir: Path | None = None) -> Path:
    return candidates_dir(results_dir) / "bm25.csv"


def load_dense_candidates(results_dir: Path | None = None,
                          usecols: list[str] | None = None) -> pd.DataFrame:
    """Concatenate the dense retrieval candidates across encoders (no BM25)."""
    files = dense_candidate_files(results_dir)
    if not files:
        raise FileNotFoundError(
            f"no dense candidate files under {candidates_dir(results_dir)}")
    return pd.concat([pd.read_csv(p, usecols=usecols) for p in files],
                     ignore_index=True)


RELEVANCE_FILTERS = ("none", "gpt", "qwen", "both")


def filtered_per_query_lean(results_dir: Path | None = None, k: int = 10,
                            judge: str = "both") -> pd.DataFrame:
    """Per-query lean@k over the top-k retrievals that pass a relevance filter.

    `judge` selects the filter: "none" keeps every top-k article (raw lean,
    centre articles count in the denominator); "gpt" and "qwen" keep the
    articles that gpt-5-mini or Qwen3.5-35B-A3B grade relevant; "both" keeps
    the gold set both judges grade relevant, the outcome behind the paper's
    Reddit gold-filtered figure and tables. Queries with no retained article
    are dropped. Returns one row per (encoder, query_id) with columns encoder,
    query_id, ideology, n_relevant, lean.
    """
    if judge not in RELEVANCE_FILTERS:
        raise ValueError(f"judge must be one of {RELEVANCE_FILTERS}, got {judge!r}")
    base = Path(results_dir) if results_dir is not None else RESULTS_BASE
    judge_dir = base / "judge_relevance"
    gpt = pd.DataFrame([json.loads(l) for l in open(judge_dir / "judgments.jsonl")])
    qwen = pd.DataFrame([json.loads(l) for l in open(judge_dir / "judgments_qwen.jsonl")])
    qwen = qwen[qwen["grade"].notna()].copy()
    qwen["grade"] = qwen["grade"].astype(int)
    merged = gpt.merge(qwen, on=["query_id", "article_id"], suffixes=("_gpt", "_qwen"), how="outer")
    g_gpt = merged["grade_gpt"].fillna(0).astype(int) == 1
    g_qwen = merged["grade_qwen"].fillna(0).astype(int) == 1
    merged["g_agree"] = {"none": pd.Series(True, index=merged.index), "gpt": g_gpt,
                         "qwen": g_qwen, "both": g_gpt & g_qwen}[judge].astype(int)

    cols = ["query_id", "ideology", "encoder", "rank",
            "retrieved_article_id", "retrieved_stance"]
    cands = pd.concat(
        [pd.read_csv(p, usecols=cols) for p in sorted(candidates_dir(base).glob("*.csv"))],
        ignore_index=True,
    )
    cands = cands[cands["rank"] <= k].rename(columns={"retrieved_article_id": "article_id"})
    cands = cands[cands["ideology"].isin(["liberal", "conservative"])]
    cands = cands.merge(merged[["query_id", "article_id", "g_agree"]],
                        on=["query_id", "article_id"], how="left")
    cands["g_agree"] = cands["g_agree"].fillna(1 if judge == "none" else 0).astype(int)

    rows = []
    for (enc, qid, ide), g in cands.groupby(["encoder", "query_id", "ideology"], sort=False):
        rel = g[g["g_agree"] == 1]
        if len(rel) == 0:
            continue
        n_right = int((rel["retrieved_stance"] == "right").sum())
        n_left = int((rel["retrieved_stance"] == "left").sum())
        rows.append({"encoder": enc, "query_id": qid, "ideology": ide,
                     "n_relevant": len(rel), "lean": (n_right - n_left) / len(rel)})
    return pd.DataFrame(rows)


def corpus_embeddings_path(cfg: dict[str, Any], slug: str) -> Path:
    root = Path(cfg["corpus"]["embeddings_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    p = root / slug
    if not p.exists():
        raise FileNotFoundError(f"Corpus embedding dir missing: {p}")
    return p


def corpus_jsonl_path(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["corpus"]["corpus_jsonl"])
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Corpus jsonl missing: {p}")
    return p
