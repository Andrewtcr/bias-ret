"""Shared loaders and paths for HealthCareMagic dialect retrieval.

HCMagic-nat contains natural AAL/WME queries and WME translations paired with the
AAL queries. HCMagic-synth contains 4,999 WME questions with three AAL paraphrases
per question. Both use the same 112,165-answer HealthCareMagic corpus.

Corpus embeddings: embeddings/{slug}-{corpus_tag}/corpus_embeddings.npy
Query embeddings:  embeddings/{slug}/{variant}_embeddings.npy
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent   # hcmagic/
EMB = ROOT / "embeddings"
RES = ROOT / "results"


def model_slug(name: str) -> str:
    return name.split("/")[-1].lower()


def corpus_emb_dir(slug: str, corpus_tag: str = "full") -> Path:
    return EMB / f"{slug}-{corpus_tag}"


def queries_emb_dir(slug: str) -> Path:
    return EMB / slug


def load_corpus(corpus_jsonl: Path) -> tuple[list[str], list[str]]:
    """Load HealthCareMagic answer records → (doc_ids, answer texts)."""
    doc_ids, texts = [], []
    with open(corpus_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            texts.append(rec["answer"])
    return doc_ids, texts


# ---------- hcmagic real-arm ----------
def load_hcmagic_real(csv_path: Path, translated_jsonl: Path | None) -> dict[str, list[dict]]:
    """Return three named lists of {query_id, text, dial}.

    Keys:
      - "aal_real": each retained AAL row (n=647). query_id = f"hcm_aal_{idx}".
      - "wme_real": each retained WME row (n=659). query_id = f"hcm_wme_{idx}".
      - "trans_aal": WME translations paired with aal_real by the numeric
        query-ID suffix (n=647). Loaded from translated_jsonl if present.
    """
    df = pd.read_csv(csv_path)
    # Exclude missing or blank queries before converting text to strings.
    df = df[df["input"].notna() & df["input"].astype(str).str.strip().ne("")]
    # Require at least 20 characters after stripping surrounding whitespace.
    df = df[df["input"].astype(str).str.strip().str.len() >= 20]
    aal = df[df["dial"] == "aal"].copy()
    wme = df[df["dial"] == "wme"].copy()
    aal_real = [
        {"query_id": f"hcm_aal_{int(r.idx)}", "text": str(r.input), "dial": "aal"}
        for r in aal.itertuples()
    ]
    wme_real = [
        {"query_id": f"hcm_wme_{int(r.idx)}", "text": str(r.input), "dial": "wme"}
        for r in wme.itertuples()
    ]

    trans_aal: list[dict] = []
    if translated_jsonl and translated_jsonl.exists():
        # Translation jsonl uses 1-indexed row_id matching the original AAL file's
        # row position: aal_idx k (0-indexed in CSV) → row_id k+1.
        by_row_id: dict[int, str] = {}
        with open(translated_jsonl) as f:
            for line in f:
                rec = json.loads(line)
                t = rec.get("trans_aal") or ""
                by_row_id[int(rec["row_id"])] = str(t)
        for r in aal_real:
            idx_int = int(r["query_id"].rsplit("_", 1)[-1])
            t = by_row_id.get(idx_int + 1, "")
            if t:
                trans_aal.append({
                    "query_id": f"hcm_trans_{idx_int}",
                    "text": t,
                    "dial": "trans_wme",
                    "paired_aal_qid": r["query_id"],
                })
    return {"aal_real": aal_real, "wme_real": wme_real, "trans_aal": trans_aal}


# ---------- hcmagic_synth paired arm ----------
def load_hcmagic_synth_paired(paired_jsonl: Path) -> dict[str, list[dict]]:
    """Return four named lists of {query_id, text}, one per variant.

    All four variants share `query_id = f"hcsynth_q_{row_idx}"`; only the text
    differs. The qrels from prepare_corpus.py map each query to its
    gold answer and any identical-answer documents in the shared corpus.
    """
    variants = {"wme_real": [], "aal_synth1": [], "aal_synth2": [], "aal_synth3": []}
    with open(paired_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            qid = rec["qid"]
            for vname in variants:
                variants[vname].append({"query_id": qid, "text": rec[vname]})
    return variants
