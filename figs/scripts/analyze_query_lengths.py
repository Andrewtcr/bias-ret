"""Query counts and character-length statistics for the four query datasets.

Reproduces the appendix table of per-group query counts and query lengths
from the committed query files: AllSides_synth left/right queries, Reddit_nat
liberal/conservative queries (post text is withheld, so lengths come from the
released per-post title character counts), HCMagic_synth WME sources and their
three AAL paraphrases, and the filtered HCMagic_nat WME/AAL pools. Also
lists the HCMagic_nat source rows that the natural-query loader's input filter
drops (the appendix table of dropped rows).

Writes figs/results/query_length_stats.csv and
figs/results/hcmagic_dropped_queries.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hcmagic.lib import load_hcmagic_real, load_hcmagic_synth_paired  # noqa: E402

OUT = REPO / "figs" / "results" / "query_length_stats.csv"
DROPPED_OUT = REPO / "figs" / "results" / "hcmagic_dropped_queries.csv"


def stats(dataset: str, group: str, lengths: pd.Series) -> dict:
    lengths = pd.Series(lengths, dtype=float)
    return {
        "dataset": dataset, "group": group, "n": int(lengths.size),
        "median_chars": float(lengths.median()), "mean_chars": round(float(lengths.mean()), 1),
        "min_chars": int(lengths.min()), "max_chars": int(lengths.max()),
    }


def main() -> None:
    rows = []

    allsides = pd.read_json(REPO / "allsides_synth" / "data" / "generated_queries.jsonl", lines=True)
    for stance in ("left", "right"):
        rows.append(stats("AllSides_synth", stance, allsides.loc[allsides["stance"] == stance, "text"].str.len()))

    reddit = pd.read_csv(REPO / "reddit_nat" / "data" / "reddit_nat_raw.csv")
    evaluated = pd.read_csv(REPO / "reddit_nat" / "results" / "retrieval_candidates" / "bm25.csv",
                            usecols=["query_id"])["query_id"].unique()
    reddit = reddit[reddit["id"].isin(evaluated)].drop_duplicates(subset=["id"])
    for ideology in ("liberal", "conservative"):
        rows.append(stats("Reddit_nat", ideology, reddit.loc[reddit["ideology"] == ideology, "title_n_chars"]))

    synth = load_hcmagic_synth_paired(REPO / "hcmagic" / "data" / "hcmagic_100k" / "synth_paired.jsonl")
    rows.append(stats("HCMagic_synth", "WME", pd.Series([len(q["text"]) for q in synth["wme_real"]])))
    aal = [len(q["text"]) for v in ("aal_synth1", "aal_synth2", "aal_synth3") for q in synth[v]]
    rows.append(stats("HCMagic_synth", "AAL", pd.Series(aal)))

    natural_csv = REPO / "hcmagic" / "data" / "healthq_aal" / "hcmagic_aal_wme.csv"
    natural = load_hcmagic_real(natural_csv, None)
    rows.append(stats("HCMagic_nat", "WME", pd.Series([len(q["text"]) for q in natural["wme_real"]])))
    rows.append(stats("HCMagic_nat", "AAL", pd.Series([len(q["text"]) for q in natural["aal_real"]])))

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"wrote {OUT}")

    # Source rows that the loader's input filter drops.
    kept = {q["query_id"] for group in ("aal_real", "wme_real") for q in natural[group]}
    source = pd.read_csv(natural_csv)
    source = source[[f"hcm_{d}_{int(i)}" not in kept for d, i in zip(source["dial"], source["idx"])]]
    text = source["input"].fillna("").astype(str).str.strip()
    dropped = pd.DataFrame({"dial": source["dial"], "idx": source["idx"],
                            "char_len": text.str.len(), "query_first_30_chars": text.str[:30]})
    dropped.to_csv(DROPPED_OUT, index=False)
    print(dropped.to_string(index=False))
    print(f"wrote {DROPPED_OUT}")


if __name__ == "__main__":
    main()
