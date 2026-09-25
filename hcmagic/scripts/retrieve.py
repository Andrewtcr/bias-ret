"""Retrieve and score across all encoders + variants for the HCMagic dialect run.

For each encoder, hcmagic_synth and hcmagic (natural) variants are scored
against their qrels: MRR/Recall@{1,5,10,20,50,100} per
(query, variant).

Outputs under hcmagic/results/{slug}/:
  hcmagic_synth_per_query_metrics.csv
  hcmagic_real_per_query_metrics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.io import ensure_dir, load_yaml  # noqa: E402
from hcmagic.lib import (  # noqa: E402
    ROOT, RES, corpus_emb_dir, model_slug, queries_emb_dir,
)


from hcmagic.metrics import score_metrics  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    corpus_tag = cfg["corpus"].get("tag", "full")
    qrels_hcsynth: dict = {}
    qrels_hcm: dict = {}
    if cfg["datasets"].get("hcmagic_synth", {}).get("enabled"):
        with open(ROOT / cfg["datasets"]["hcmagic_synth"]["qrels_path"]) as f:
            qrels_hcsynth = json.load(f)
        print(f"[score] loaded hcmagic_synth qrels: {len(qrels_hcsynth)} qids")
    hd = cfg["datasets"].get("hcmagic", {})
    if hd.get("enabled"):
        with open(ROOT / hd["qrels_path"]) as f:
            qrels_hcm = json.load(f)
        print(f"[score] loaded hcmagic (real) qrels: {len(qrels_hcm)} qids")

    encs = cfg["retrieval"]["encoders"]
    if args.only:
        encs = [e for e in encs if args.only.lower() in (e["name"] if isinstance(e, dict) else e).lower()]

    for enc in encs:
        name = enc["name"] if isinstance(enc, dict) else enc
        slug = model_slug(name)
        edir_q = queries_emb_dir(slug)
        edir_c = corpus_emb_dir(slug, corpus_tag)
        result_dir = ensure_dir(RES / slug)

        if not (edir_c / "corpus_embeddings.npy").exists():
            print(f"[score] {slug}: no corpus embeddings — skip")
            continue

        corpus_emb = np.load(edir_c / "corpus_embeddings.npy")
        with open(edir_c / "corpus_ids.json") as f:
            corpus_ids = json.load(f)
        print(f"\n[score] === {slug} ===  corpus shape={corpus_emb.shape}")

        # discover variants from query emb dir
        variant_files = sorted(p.stem.replace("_embeddings", "")
                                for p in edir_q.glob("*_embeddings.npy"))
        hcm_metric_rows: list[dict] = []
        hcsynth_rows: list[dict] = []
        for variant in variant_files:
            q_emb = np.load(edir_q / f"{variant}_embeddings.npy")
            with open(edir_q / f"{variant}_ids.json") as f:
                qids = json.load(f)
            scores = q_emb @ corpus_emb.T   # cosine since L2-normalized
            if variant.startswith("hcsynth_"):
                v_label = variant.replace("hcsynth_", "")
                # Score against the gold-answer equivalence class.
                metric_rows = score_metrics(scores, qids, corpus_ids, qrels_hcsynth, v_label)
                hcsynth_rows.extend(metric_rows)
                print(f"[score]   {variant:24s}  metric_rows={len(metric_rows)}")
            elif variant.startswith("hcm_"):
                v_label = variant.replace("hcm_", "")
                metric_rows = score_metrics(scores, qids, corpus_ids, qrels_hcm, v_label)
                hcm_metric_rows.extend(metric_rows)
                print(f"[score]   {variant:24s}  metric_rows={len(metric_rows)} (qrels)")

        if hcsynth_rows:
            df = pd.DataFrame(hcsynth_rows)
            df.to_csv(result_dir / "hcmagic_synth_per_query_metrics.csv", index=False)
            print(f"[score] {slug}: wrote hcmagic_synth_per_query_metrics.csv ({len(df)} rows)")
        if hcm_metric_rows:
            df = pd.DataFrame(hcm_metric_rows)
            df.to_csv(result_dir / "hcmagic_real_per_query_metrics.csv", index=False)
            print(f"[score] {slug}: wrote hcmagic_real_per_query_metrics.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
