"""SentenceTransformer wrapper + helpers shared across experiments."""
from __future__ import annotations

from typing import Any

import numpy as np


def model_slug(name: str) -> str:
    return name.split("/")[-1].lower()


_TRUST_REMOTE_CODE_PATTERNS = (
    "nvidia/llama-embed-nemotron",
)

def _is_openai_encoder(name: str) -> bool:
    lname = name.lower()
    return lname.startswith("openai/") or lname.startswith("text-embedding-")


def _openai_model_id(name: str) -> str:
    if name.lower().startswith("openai/"):
        return name.split("/", 1)[1]
    return name


def _encode_texts_openai(
    model_name: str,
    texts: list[str],
    cfg: dict[str, Any],
    kind: str,
) -> np.ndarray:
    """Encode via OpenAI embeddings API. Returns L2-normalized embeddings.

    Reads `retrieval.batch_size` (API batch, default 100) and optional
    `retrieval.dimensions` (Matryoshka truncation; default = model default).
    """
    from openai import OpenAI

    rcfg = cfg["retrieval"]
    bs = int(rcfg.get("batch_size", 100))
    mid = _openai_model_id(model_name)
    dim = rcfg.get("dimensions")
    client = OpenAI()

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(total=len(texts), desc=f"  openai/{mid} {kind}", unit="doc", leave=False)
    except Exception:  # noqa: BLE001
        pbar = None
    out_rows: list[list[float]] = []
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        kwargs: dict[str, Any] = {"model": mid, "input": chunk}
        if dim:
            kwargs["dimensions"] = int(dim)
        resp = client.embeddings.create(**kwargs)
        # OpenAI returns rows in input order
        for d in resp.data:
            out_rows.append(d.embedding)
        if pbar is not None:
            pbar.update(len(chunk))
    if pbar is not None:
        pbar.close()
    arr = np.asarray(out_rows, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _prepare_hf_texts(model_name: str, texts: list[str], kind: str) -> tuple[list[str], dict[str, Any]]:
    """Return (texts, extra encode_kwargs) with model-specific prefixes applied."""
    encode_kwargs: dict[str, Any] = {}
    name_lower = model_name.lower()
    prepared = list(texts)
    if kind == "query":
        if "qwen3" in name_lower:
            encode_kwargs["prompt_name"] = "query"
        if "bge" in name_lower:
            prepared = [
                "Represent this sentence for searching relevant passages: " + t for t in prepared
            ]
        if "nemotron" in name_lower:
            prepared = [
                "Instruct: Given a search query, retrieve relevant passages.\nQuery: " + t
                for t in prepared
            ]
        # Octen-Embedding-8B uses no query prefix per model card.
    elif kind == "document":
        if "qwen3" in name_lower and "octen" not in name_lower:
            encode_kwargs["prompt_name"] = "document"
        if "octen-embedding" in name_lower:
            # Octen recommends a "- " prefix on documents to avoid upstream
            # Qwen3 quirks (see model card).
            prepared = ["- " + t for t in prepared]
    return prepared, encode_kwargs


def _dp_worker(device: str, model_name: str, cfg: dict, in_q, out_q) -> None:
    """Per-GPU worker: load model on `device` once, then loop on in_q.

    Protocol:
      Sends ("ready", device) once after loading, or ("error", traceback).
      Receives (chunk_id, texts, kind) tuples; processes and replies
      ("done", chunk_id, embeddings_ndarray) or ("error", traceback).
      Receives None to exit.
    """
    import sys
    import traceback as _tb
    from pathlib import Path as _Path

    try:
        # Workers reimport modules fresh under spawn; reattach the repo to
        # sys.path so this module's helpers resolve.
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

        import numpy as _np
        import torch
        from sentence_transformers import SentenceTransformer

        from src.embedding import (  # noqa: WPS433
            _prepare_hf_texts,
            _TRUST_REMOTE_CODE_PATTERNS,
        )

        rcfg = cfg["retrieval"]
        msl = int(rcfg.get("max_seq_length", 512))
        bs = int(rcfg.get("batch_size", 8))
        trust_remote_code = bool(rcfg.get("trust_remote_code")) or any(
            p in model_name for p in _TRUST_REMOTE_CODE_PATTERNS
        )
        st_kwargs: dict[str, Any] = {"device": device}
        if trust_remote_code:
            st_kwargs["trust_remote_code"] = True
        dtype = rcfg.get("torch_dtype")
        if dtype:
            st_kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, dtype)}
        model = SentenceTransformer(model_name, **st_kwargs)
        model.max_seq_length = msl
    except Exception:  # noqa: BLE001
        out_q.put(("error", f"load failure on {device}:\n{_tb.format_exc()}"))
        return

    out_q.put(("ready", device))

    while True:
        try:
            msg = in_q.get()
            if msg is None:
                break
            chunk_id, texts, kind = msg
            prepared, extra_kwargs = _prepare_hf_texts(model_name, texts, kind)
            embs = model.encode(
                prepared,
                batch_size=bs,
                show_progress_bar=False,
                convert_to_numpy=True,
                **extra_kwargs,
            )
            out_q.put(("done", chunk_id, _np.asarray(embs)))
        except Exception:  # noqa: BLE001
            out_q.put(("error", f"encode failure on {device}:\n{_tb.format_exc()}"))


def _encode_texts_multi_dp(
    model_name: str,
    text_groups: list[tuple[list[str], str]],
    cfg: dict[str, Any],
    devices: list[str],
) -> list[np.ndarray]:
    """Data-parallel encode across multiple GPUs via manual mp processes.

    One worker per device loads the model directly on its GPU. Each text group
    is round-robin sharded across workers; results reassembled in original order.
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()
    procs: list = []
    print(f"[encode-dp] {model_name}: spawning {len(devices)} workers on {devices}")
    for device in devices:
        p = ctx.Process(
            target=_dp_worker,
            args=(device, model_name, cfg, in_q, out_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    # Wait for every worker to finish model load.
    ready = 0
    while ready < len(devices):
        msg = out_q.get()
        if msg[0] == "error":
            raise RuntimeError(f"worker failed during load:\n{msg[1]}")
        if msg[0] != "ready":
            raise RuntimeError(f"unexpected worker message during load: {msg}")
        ready += 1
        print(f"[encode-dp]   worker on {msg[1]} ready ({ready}/{len(devices)})")

    # Chunk size for the work queue. Smaller chunks = smoother progress bar
    # but more pickle overhead. ~200 fits one tqdm refresh + decent batching.
    chunk_size = int(cfg["retrieval"].get("dp_chunk_size", 200))

    out_groups: list[np.ndarray] = []
    try:
        for texts, kind in text_groups:
            # Split sequentially: chunks[i] = texts[i*cs : (i+1)*cs].
            # Reassembly is just np.concatenate over chunk_id order.
            chunks = []
            for start in range(0, len(texts), chunk_size):
                chunks.append((len(chunks), texts[start : start + chunk_size], kind))
            for chunk in chunks:
                in_q.put(chunk)
            results: list = [None] * len(chunks)
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=len(texts), desc=f"  {kind}", unit="doc", leave=False)
            except Exception:  # noqa: BLE001
                pbar = None
            for _ in range(len(chunks)):
                msg = out_q.get()
                if msg[0] == "error":
                    raise RuntimeError(f"worker encode error:\n{msg[1]}")
                _, chunk_id, embs = msg
                results[chunk_id] = embs
                if pbar is not None:
                    pbar.update(embs.shape[0])
            if pbar is not None:
                pbar.close()
            final = np.concatenate(results, axis=0)
            norms = np.linalg.norm(final, axis=1, keepdims=True)
            out_groups.append(final / np.clip(norms, 1e-12, None))
    finally:
        for _ in devices:
            in_q.put(None)
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
    return out_groups


def encode_texts_multi(
    model_name: str,
    text_groups: list[tuple[list[str], str]],
    cfg: dict[str, Any],
) -> list[np.ndarray]:
    """Encode multiple text groups with a single model load.

    Use this for large models (e.g. 8B+) where loading twice can OOM. Each
    element of `text_groups` is (texts, kind) where kind ∈ {"document","query"}.
    Returns a list of L2-normalized embedding arrays in the same order.

    If ``cfg["retrieval"]["devices"]`` is a list of length >1, dispatches to a
    multi-GPU data-parallel encoder.
    """
    if _is_openai_encoder(model_name):
        return [_encode_texts_openai(model_name, t, cfg, k) for t, k in text_groups]

    devices = cfg["retrieval"].get("devices")
    if isinstance(devices, list) and len(devices) > 1:
        return _encode_texts_multi_dp(model_name, text_groups, cfg, devices)

    import gc as _gc

    import torch
    from sentence_transformers import SentenceTransformer

    rcfg = cfg["retrieval"]
    bs = int(rcfg.get("batch_size", 32))
    msl = int(rcfg.get("max_seq_length", 512))
    device = rcfg.get("device")
    if not device and isinstance(devices, list) and devices:
        device = devices[0]
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    st_kwargs: dict[str, Any] = {"device": device}
    trust_remote_code = bool(rcfg.get("trust_remote_code")) or any(
        p in model_name for p in _TRUST_REMOTE_CODE_PATTERNS
    )
    if trust_remote_code:
        st_kwargs["trust_remote_code"] = True
    dtype = rcfg.get("torch_dtype")
    if dtype:
        st_kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, dtype)}
    model = SentenceTransformer(model_name, **st_kwargs)
    model.max_seq_length = msl

    out: list[np.ndarray] = []
    for texts, kind in text_groups:
        prepared, extra_kwargs = _prepare_hf_texts(model_name, texts, kind)
        encode_kwargs: dict[str, Any] = {
            "batch_size": bs,
            "show_progress_bar": True,
            "convert_to_numpy": True,
            **extra_kwargs,
        }
        embs = model.encode(prepared, **encode_kwargs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.clip(norms, 1e-12, None)
        out.append(embs)

    del model
    _gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return out


__all__ = ["encode_texts_multi", "model_slug"]
