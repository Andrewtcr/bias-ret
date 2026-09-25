"""Encode the corpus and generated queries with each configured encoder.

Reads allsides_synth/data/{corpus,generated_queries}.jsonl and writes per-encoder
arrays and metadata under allsides_synth/embeddings/{slug}/. The directories
can be overridden with run.data_dir and run.embeddings_dir in the config.

Outputs: corpus_embeddings.npy, corpus_ids.json, query_embeddings.npy,
query_ids.json, queries.jsonl, and encode_meta.json. Encoders with both existing
embedding arrays are skipped.

Usage:
  python allsides_synth/scripts/embed.py \
      --config allsides_synth/configs/retrieval.yaml [--only ENCODER_SUBSTRING]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from allsides_synth import lib  # noqa: E402
from src.embedding import encode_texts_multi, model_slug  # noqa: E402
from src.io import ensure_dir, load_yaml, read_jsonl, write_jsonl  # noqa: E402


def _resolve_encoder_items(cfg: dict) -> list[dict]:
    return [{"name": e} if isinstance(e, str) else dict(e) for e in cfg["retrieval"]["encoders"]]


def _merge_encoder_cfg(cfg: dict, item: dict) -> dict:
    import copy as _copy

    cfg_copy = _copy.copy(cfg)
    rcfg = dict(cfg["retrieval"])
    for k, v in item.items():
        if k == "name":
            continue
        rcfg[k] = v
    cfg_copy["retrieval"] = rcfg
    return cfg_copy


def _load_data(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load corpus.jsonl and generated_queries.jsonl from the data directory."""
    ddir = lib.data_dir(cfg)
    corpus_path = ddir / "corpus.jsonl"
    queries_path = ddir / "generated_queries.jsonl"
    if not (corpus_path.exists() and queries_path.exists()):
        raise FileNotFoundError(
            f"Missing corpus or query data: {corpus_path}, {queries_path}."
        )
    return read_jsonl(corpus_path), read_jsonl(queries_path)


def _encode_one(
    item: dict,
    cfg: dict,
    corpus_ids: list[str],
    corpus_texts: list[str],
    queries: list[dict],
    edir: Path,
) -> None:
    enc = item["name"]
    slug = model_slug(enc)
    emb_dir = ensure_dir(edir / slug)
    cpath = emb_dir / "corpus_embeddings.npy"
    qpath = emb_dir / "query_embeddings.npy"
    cids_path = emb_dir / "corpus_ids.json"
    qids_path = emb_dir / "query_ids.json"
    queries_path = emb_dir / "queries.jsonl"
    meta_path = emb_dir / "encode_meta.json"

    if cpath.exists() and qpath.exists():
        print(f"[encode] {slug}: already done (corpus + query .npy present). Skipping.")
        return

    enc_cfg = _merge_encoder_cfg(cfg, item)
    devices = enc_cfg["retrieval"].get("devices") or []
    if not devices:
        devices = [enc_cfg["retrieval"].get("device", "cuda:0")]
    print(f"\n[encode] === {slug} ({enc}) ===")
    print(
        f"[encode] devices={devices} batch_size={enc_cfg['retrieval'].get('batch_size')}"
        f" dtype={enc_cfg['retrieval'].get('torch_dtype')}"
    )

    query_texts = [q["text"] for q in queries]
    query_ids = [q.get("query_id") or f"{q['article_id']}__{q['frame']}__{q['stance']}" for q in queries]

    t0 = time.time()
    # tqdm bars are driven inside encode_texts_multi via the DP queue protocol
    # (multi-GPU) or via sentence-transformers' built-in progress (single GPU).
    corpus_embs, query_embs = encode_texts_multi(
        enc,
        [(corpus_texts, "document"), (query_texts, "query")],
        enc_cfg,
    )
    dt = time.time() - t0

    np.save(cpath, corpus_embs)
    np.save(qpath, query_embs)
    cids_path.write_text(json.dumps(corpus_ids))
    qids_path.write_text(json.dumps(query_ids))
    write_jsonl(queries_path, queries)
    meta = {
        "encoder": enc,
        "slug": slug,
        "corpus_shape": list(corpus_embs.shape),
        "query_shape": list(query_embs.shape),
        "dtype": str(corpus_embs.dtype),
        "elapsed_s": round(dt, 1),
        "devices": devices,
        "batch_size": enc_cfg["retrieval"].get("batch_size"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[encode] {slug}: corpus={corpus_embs.shape} query={query_embs.shape} in {dt:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", default=None,
                        help="Substring match on encoder name; runs only matching encoders.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    print(f"[encode] run.tag={cfg['run']['tag']}  config={args.config}")
    corpus, queries = _load_data(cfg)
    print(f"[encode] corpus={len(corpus)} queries={len(queries)}")

    corpus_ids = [r["article_id"] for r in corpus]
    corpus_texts = [r["text"] for r in corpus]

    items = _resolve_encoder_items(cfg)
    if args.only:
        items = [it for it in items if args.only.lower() in it["name"].lower()]
        print(f"[encode] --only filter: {[it['name'] for it in items]}")
    if not items:
        raise SystemExit("No encoders to run.")

    edir = lib.embeddings_dir(cfg)
    for item in tqdm(items, desc="encoders", unit="enc"):
        _encode_one(item, cfg, corpus_ids, corpus_texts, queries, edir)
    print("[encode] all encoders done.")


if __name__ == "__main__":
    main()
