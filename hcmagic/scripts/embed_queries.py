"""Embed the natural, translated, and synthetic HealthCareMagic query variants.

HCMagic-nat: 647 natural AAL queries, 659 natural WME queries, and 647 WME
translations paired with the AAL queries. HCMagic-synth: 4,999 WME questions,
each with three synthetic AAL variants.

Writes under hcmagic/embeddings/{slug}/:
  {variant}_embeddings.npy
  {variant}_ids.json
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
    ROOT, load_hcmagic_real, load_hcmagic_synth_paired,
    model_slug, queries_emb_dir,
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


def gather_query_groups(cfg: dict) -> list[tuple[str, list[str], list[str]]]:
    """Return list of (variant_label, ids, texts). One per variant."""
    out: list[tuple[str, list[str], list[str]]] = []

    if cfg["datasets"]["hcmagic"]["enabled"]:
        hd = cfg["datasets"]["hcmagic"]
        groups = load_hcmagic_real(
            ROOT / hd["source_csv"],
            ROOT / hd["translated_jsonl"],
        )
        for variant in ("aal_real", "wme_real", "trans_aal"):
            recs = groups[variant]
            ids = [r["query_id"] for r in recs]
            texts = [r["text"] for r in recs]
            out.append((f"hcm_{variant}", ids, texts))

    if cfg["datasets"].get("hcmagic_synth", {}).get("enabled"):
        hs = cfg["datasets"]["hcmagic_synth"]
        groups = load_hcmagic_synth_paired(ROOT / hs["paired_jsonl"])
        for variant in ("wme_real", "aal_synth1", "aal_synth2", "aal_synth3"):
            recs = groups[variant]
            ids = [r["query_id"] for r in recs]
            texts = [r["text"] for r in recs]
            out.append((f"hcsynth_{variant}", ids, texts))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--only", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    items = _enc_items(cfg)
    if args.only:
        items = [it for it in items if args.only.lower() in it["name"].lower()]

    query_groups = gather_query_groups(cfg)
    print("[queries] variants:")
    for vname, ids, _ in query_groups:
        print(f"  {vname:24s}  n={len(ids):4d}")

    for item in items:
        name = item["name"]
        slug = model_slug(name)
        edir = ensure_dir(queries_emb_dir(slug))
        # which variants already done?
        pending: list[tuple[str, list[str], list[str]]] = []
        for vname, ids, texts in query_groups:
            if (edir / f"{vname}_embeddings.npy").exists():
                continue
            pending.append((vname, ids, texts))
        if not pending:
            print(f"[queries] {slug}: all variants exist — skip")
            continue

        enc_cfg = _merge_enc_cfg(cfg, item)
        devices = enc_cfg["retrieval"].get("devices") or ["cuda:0"]
        print(f"\n[queries] === {slug} === devices={devices}; "
              f"{len(pending)} variants to encode")

        # batch all pending texts into a single encode_texts_multi call to
        # amortize model load
        text_groups = [(texts, "query") for _, _, texts in pending]
        t0 = time.time()
        embs_list = encode_texts_multi(name, text_groups, enc_cfg)
        dt = time.time() - t0

        for (vname, ids, _), embs in zip(pending, embs_list):
            np.save(edir / f"{vname}_embeddings.npy", embs)
            (edir / f"{vname}_ids.json").write_text(json.dumps(ids))
            print(f"[queries]   {vname:24s}  shape={embs.shape}")
        print(f"[queries] {slug}: total elapsed {dt:.1f}s")


if __name__ == "__main__":
    main()
