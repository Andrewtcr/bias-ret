"""Prepare the unified 112k HealthCareMagic corpus + combined qrels covering
BOTH the synth-paired arm and the real-arm.

Source corpus: hcmagic/data/healthq_aal/HealthCareMagic-100k.json
  (112,165 records; despite the "100k" name)

Source query sets:
  - hcmagic_wme_synth.csv (synth-paired arm; 4,999 surviving rows after filter)
  - hcmagic_aal_wme.csv   (real arm: aal_real, wme_real)
  - hcmagic_aal_translated.jsonl (real arm: trans_aal, paired to aal_real)

Output:
  data/hcmagic_100k/corpus.jsonl         — 112,165 doctor-answer docs
  data/hcmagic_100k/synth_paired.jsonl   — paired WME/AAL query records
  data/hcmagic_100k/qrels.json           — combined: hcsynth_q_* + hcm_aal_* +
                                            hcm_wme_* + hcm_trans_* qids

Each qrels entry maps qid → {doc_id: 1}. Doc IDs are `hcm100k_{row_idx}` from
the 100k file's row order. (input, output) pairs in each source are resolved
against the 100k file via dict lookup. Qrels also credit every document
with answer text identical to the gold answer.

Filters (applied per query variant):
  - non-null, non-empty stripped
  - len.strip() >= 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]   # hcmagic/
SRC_100K = ROOT / "data" / "healthq_aal" / "HealthCareMagic-100k.json"
SRC_SYNTH = ROOT / "data" / "healthq_aal" / "hcmagic_wme_synth.csv"
SRC_REAL  = ROOT / "data" / "healthq_aal" / "hcmagic_aal_wme.csv"
SRC_TRANS = ROOT / "data" / "healthq_aal" / "hcmagic_aal_translated.jsonl"
OUT_DIR   = ROOT / "data" / "hcmagic_100k"

MIN_CHARS = 20   # require stripped length >= 20


def _ok(s) -> bool:
    if not isinstance(s, str):
        return False
    return len(s.strip()) >= MIN_CHARS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Load 100k corpus ===
    print(f"[prep] loading {SRC_100K}")
    with open(SRC_100K) as f:
        full = json.load(f)
    print(f"[prep]   {len(full):,} records")

    # === Build corpus.jsonl ===
    # The corpus contains byte-identical duplicate answer texts under distinct
    # doc_ids. Retrieval scores depend only on the answer text, so any doc with
    # text identical to the gold is an equally correct retrieval: qrels credit
    # the whole identical-text equivalence class, not one arbitrary doc_id.
    corpus_path = out_dir / "corpus.jsonl"
    in_out_to_idx: dict[tuple[str, str], int] = {}
    out_to_indices: dict[str, list[int]] = {}
    with open(corpus_path, "w") as f:
        for i, rec in enumerate(full):
            inp = (rec.get("input") or "").strip()
            out = (rec.get("output") or "").strip()
            doc_id = f"hcm100k_{i}"
            f.write(json.dumps({"doc_id": doc_id, "answer": out}) + "\n")
            in_out_to_idx[(inp, out)] = i
            out_to_indices.setdefault(out, []).append(i)
    print(f"[prep] wrote {corpus_path}  ({len(full):,} docs)")
    print(f"[prep]   unique (input,output) keys: {len(in_out_to_idx):,}")
    print(f"[prep]   unique answer texts: {len(out_to_indices):,}")

    def gold_grades(out_text: str) -> dict[str, int]:
        return {f"hcm100k_{j}": 1 for j in out_to_indices[out_text]}

    qrels: dict[str, dict[str, int]] = {}

    # === Synth arm ===
    print("\n[prep] === synth arm ===")
    synth = pd.read_csv(SRC_SYNTH)
    n_in = len(synth)
    query_fields = ["input", "aal_message1", "aal_message2", "aal_message3"]
    mask = pd.Series(True, index=synth.index)
    for col in query_fields + ["output"]:
        col_ok = synth[col].map(_ok)
        n_drop = int((~col_ok & mask).sum())
        print(f"[prep]   filter {col!r:18s} -> drop {n_drop}")
        mask &= col_ok
    synth = synth[mask].reset_index(drop=True)
    print(f"[prep]   kept {len(synth):,} / {n_in:,} rows after filters")

    # Resolve gold doc per synth row
    paired_path = out_dir / "synth_paired.jsonl"
    n_unresolved = 0
    with open(paired_path, "w") as f:
        for i, row in synth.iterrows():
            inp = str(row["input"]).strip()
            out = str(row["output"]).strip()
            idx100k = in_out_to_idx.get((inp, out))
            if idx100k is None:
                n_unresolved += 1
                continue
            qid = f"hcsynth_q_{i}"
            doc_id = f"hcm100k_{idx100k}"
            f.write(json.dumps({
                "qid": qid, "row_idx": int(i), "gold_doc_id": doc_id,
                "wme_real":   inp,
                "aal_synth1": str(row["aal_message1"]).strip(),
                "aal_synth2": str(row["aal_message2"]).strip(),
                "aal_synth3": str(row["aal_message3"]).strip(),
            }) + "\n")
            qrels[qid] = gold_grades(out)
    print(f"[prep]   wrote {paired_path} ({len(synth) - n_unresolved:,} paired rows)")
    if n_unresolved:
        print(f"[prep]   WARNING: {n_unresolved} synth rows unresolvable to 100k corpus")

    # === Real arm ===
    print("\n[prep] === real arm ===")
    real = pd.read_csv(SRC_REAL)
    print(f"[prep]   raw real-arm rows: {len(real):,} ({(real['dial']=='aal').sum()} aal + {(real['dial']=='wme').sum()} wme)")
    real = real[real["input"].notna() & real["input"].astype(str).str.strip().ne("")]
    real = real[real["input"].astype(str).str.strip().str.len() >= MIN_CHARS]
    real = real[real["output"].notna() & real["output"].astype(str).str.strip().ne("")]
    real = real[real["output"].astype(str).str.strip().str.len() >= MIN_CHARS]
    aal = real[real["dial"] == "aal"].copy()
    wme = real[real["dial"] == "wme"].copy()
    print(f"[prep]   after filters: {len(aal):,} aal + {len(wme):,} wme")

    n_real_unresolved = 0
    for df, dial_prefix in [(aal, "aal"), (wme, "wme")]:
        for r in df.itertuples():
            inp = str(r.input).strip()
            out = str(r.output).strip()
            idx100k = in_out_to_idx.get((inp, out))
            if idx100k is None:
                n_real_unresolved += 1
                continue
            qid = f"hcm_{dial_prefix}_{int(r.idx)}"
            qrels[qid] = gold_grades(out)
    print(f"[prep]   real-arm aal+wme qrels added: {sum(1 for k in qrels if k.startswith('hcm_aal_') or k.startswith('hcm_wme_'))}")
    if n_real_unresolved:
        print(f"[prep]   WARNING: {n_real_unresolved} real-arm rows unresolvable")

    # === trans_aal (paired to aal_real on same gold) ===
    print("\n[prep] === trans_aal (real-arm back-translation) ===")
    aal_idx_to_out: dict[int, str] = {}
    for r in aal.itertuples():
        inp = str(r.input).strip()
        out = str(r.output).strip()
        idx100k = in_out_to_idx.get((inp, out))
        if idx100k is not None:
            aal_idx_to_out[int(r.idx)] = out

    n_trans = 0
    with open(SRC_TRANS) as f:
        for line in f:
            rec = json.loads(line)
            # row_id in translation file is 1-indexed against the aal rows' position
            row_id = int(rec["row_id"])
            aal_idx = row_id - 1
            gold_out = aal_idx_to_out.get(aal_idx)
            if gold_out is None:
                continue
            trans_text = (rec.get("trans_aal") or "").strip()
            if not _ok(trans_text):
                continue
            qid = f"hcm_trans_{aal_idx}"
            qrels[qid] = gold_grades(gold_out)
            n_trans += 1
    print(f"[prep]   trans_aal qrels added: {n_trans}")

    # === Write combined qrels ===
    qrels_path = out_dir / "qrels.json"
    with open(qrels_path, "w") as f:
        json.dump(qrels, f)
    print(f"\n[prep] wrote {qrels_path}  ({len(qrels):,} total qids)")
    print(f"[prep]   synth qids:  {sum(1 for k in qrels if k.startswith('hcsynth_'))}")
    print(f"[prep]   real aal:    {sum(1 for k in qrels if k.startswith('hcm_aal_'))}")
    print(f"[prep]   real wme:    {sum(1 for k in qrels if k.startswith('hcm_wme_'))}")
    print(f"[prep]   real trans:  {sum(1 for k in qrels if k.startswith('hcm_trans_'))}")


if __name__ == "__main__":
    main()
