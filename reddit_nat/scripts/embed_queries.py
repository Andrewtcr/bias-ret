"""Encode Reddit queries with each encoder listed in the config.

Writes per encoder:
  reddit_nat/embeddings/{slug}/query_embeddings.npy
  reddit_nat/embeddings/{slug}/query_ids.json
  reddit_nat/embeddings/{slug}/queries.jsonl
  reddit_nat/embeddings/{slug}/encode_meta.json

Corpus embeddings are NOT touched — we reuse them in-place from
`allsides_synth/embeddings/{slug}/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from reddit_nat import lib  # noqa: E402
from src.embedding import encode_texts_multi, model_slug  # noqa: E402
from src.io import ensure_dir, load_yaml, write_jsonl  # noqa: E402


def _resolve_encoder_items(cfg: dict) -> list[dict]:
    enc_list = cfg["retrieval"].get("encoders") or []
    return [{"name": e} if isinstance(e, str) else dict(e) for e in enc_list]


def _merge_encoder_cfg(cfg: dict, item: dict) -> dict:
    import copy as _copy

    out = _copy.copy(cfg)
    rcfg = dict(cfg["retrieval"])
    for k, v in item.items():
        if k == "name":
            continue
        rcfg[k] = v
    out["retrieval"] = rcfg
    return out


def _encode_one(item: dict, cfg: dict, queries: list[dict], edir: Path) -> None:
    enc = item["name"]
    slug = model_slug(enc)
    emb_dir = ensure_dir(edir / slug)
    qpath = emb_dir / "query_embeddings.npy"
    qids_path = emb_dir / "query_ids.json"
    queries_path = emb_dir / "queries.jsonl"
    meta_path = emb_dir / "encode_meta.json"

    # Sanity: the corpus embedding for this slug must exist.
    corp_dir = lib.corpus_embeddings_path(cfg, slug)
    if not (corp_dir / "corpus_embeddings.npy").exists():
        raise FileNotFoundError(
            f"[encode] {slug}: corpus embeddings absent at {corp_dir}. "
            f"Run allsides_synth encoding for {cfg['corpus']['source_run']} first."
        )

    if qpath.exists():
        print(f"[encode] {slug}: queries already encoded. Skipping.")
        return

    enc_cfg = _merge_encoder_cfg(cfg, item)
    devices = enc_cfg["retrieval"].get("devices") or [enc_cfg["retrieval"].get("device", "cuda:0")]
    print(f"\n[encode] === {slug} ({enc}) === devices={devices}")

    query_texts = [q["text"] for q in queries]
    query_ids = [q["query_id"] for q in queries]
    t0 = time.time()
    embs = encode_texts_multi(enc, [(query_texts, "query")], enc_cfg)[0]
    dt = time.time() - t0

    np.save(qpath, embs)
    qids_path.write_text(json.dumps(query_ids))
    write_jsonl(queries_path, queries)
    meta_path.write_text(json.dumps({
        "encoder": enc,
        "slug": slug,
        "query_shape": list(embs.shape),
        "dtype": str(embs.dtype),
        "elapsed_s": round(dt, 1),
        "devices": devices,
        "batch_size": enc_cfg["retrieval"].get("batch_size"),
        "source_run": cfg["corpus"]["source_run"],
    }, indent=2))
    print(f"[encode] {slug}: queries={embs.shape} in {dt:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    print(f"[encode] run.tag={cfg['run']['tag']}  config={args.config}")
    df = lib.load_reddit_queries(cfg)
    queries = df.to_dict(orient="records")
    print(f"[encode] queries: {len(queries)}")

    items = _resolve_encoder_items(cfg)
    if args.only:
        items = [it for it in items if args.only.lower() in it["name"].lower()]
        print(f"[encode] --only -> {[it['name'] for it in items]}")
    if not items:
        raise SystemExit("No encoders to run.")

    edir = lib.embeddings_dir(cfg)
    for item in items:
        _encode_one(item, cfg, queries, edir)
    print("[encode] all encoders done.")


if __name__ == "__main__":
    main()
