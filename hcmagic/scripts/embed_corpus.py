"""Embed the HealthCareMagic answer corpus with each encoder listed in the config.

Reuses existing embeddings at `hcmagic/embeddings/{slug}-{tag}/corpus_*.npy`
if they exist. Multi-GPU via `src.embedding.encode_texts_multi`.

Usage:
  python hcmagic/scripts/embed_corpus.py \
         --config hcmagic/configs/retrieval.yaml \
         [--only octen]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.embedding import encode_texts_multi  # noqa: E402
from src.io import ensure_dir, load_yaml  # noqa: E402
from hcmagic.lib import (  # noqa: E402
    ROOT, corpus_emb_dir, load_corpus, model_slug,
)


def _enc_items(cfg: dict) -> list[dict]:
    items = cfg["retrieval"]["encoders"]
    return [{"name": e} if isinstance(e, str) else dict(e) for e in items]


def _merge_enc_cfg(cfg: dict, item: dict) -> dict:
    out = dict(cfg)
    rcfg = dict(cfg["retrieval"])
    for k, v in item.items():
        if k == "name":
            continue
        rcfg[k] = v
    out["retrieval"] = rcfg
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", default=None,
                   help="substring filter; only matching encoders run")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    corpus_jsonl = ROOT / cfg["corpus"]["corpus_jsonl"]
    corpus_tag = cfg["corpus"].get("tag", "full")

    print(f"[corpus] loading {corpus_jsonl}")
    doc_ids, texts = load_corpus(corpus_jsonl)
    print(f"[corpus]   {len(doc_ids):,} docs")

    items = _enc_items(cfg)
    if args.only:
        items = [it for it in items if args.only.lower() in it["name"].lower()]

    for item in items:
        name = item["name"]
        slug = model_slug(name)
        edir = ensure_dir(corpus_emb_dir(slug, corpus_tag))
        emb_path = edir / "corpus_embeddings.npy"
        ids_path = edir / "corpus_ids.json"
        if emb_path.exists() and ids_path.exists():
            print(f"[corpus] {slug}: exists — skip")
            continue

        enc_cfg = _merge_enc_cfg(cfg, item)
        devices = enc_cfg["retrieval"].get("devices") or ["cuda:0"]
        print(f"\n[corpus] === {slug} === devices={devices}")

        t0 = time.time()
        embs = encode_texts_multi(name, [(texts, "document")], enc_cfg)[0]
        dt = time.time() - t0
        np.save(emb_path, embs)
        ids_path.write_text(json.dumps(doc_ids))
        (edir / "encode_meta.json").write_text(json.dumps({
            "encoder": name,
            "slug": slug,
            "corpus_jsonl": str(corpus_jsonl.relative_to(REPO_ROOT)),
            "shape": list(embs.shape),
            "elapsed_s": round(dt, 1),
            "devices": devices,
        }, indent=2))
        print(f"[corpus] {slug}: shape={embs.shape} in {dt:.1f}s")


if __name__ == "__main__":
    main()
