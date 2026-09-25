"""Generate mean-pooled embeddings for one or more text columns of a table.

For every configured `in_cols` column of the input table, batches text
through `--model`'s tokenizer, mean-pools the raw token-embedding layer and
each requested hidden layer, and writes half-precision embeddings to:

  {out_dir}/{model_slug}/{run_id}.{input_name}.h5
    {col}/embed        — pooled raw (layer-0) token embeddings
    {col}/layer_{N}     — pooled hidden state at layer N, for each requested N

One process handles one model; loop over `probing/configs/models.yaml` (see
`run.sh`) to cover the full roster.

Usage:
  python probing/scripts/generate_embeddings.py \
      --config probing/configs/embeddings/hcmagic_synth.yaml \
      --model BAAI/bge-large-en-v1.5 --device 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from probing import lib  # noqa: E402
from src.io import load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # probing/


def get_device(device_arg: int) -> torch.device:
    import torch

    return torch.device("cpu") if device_arg == -1 else torch.device(f"cuda:{device_arg}")


def _append(dataset: h5py.Dataset, values: torch.Tensor) -> None:
    n = values.shape[0]
    cur = dataset.shape[0]
    dataset.resize((cur + n, dataset.shape[1]))
    dataset[-n:] = values.numpy()


def init_dataset(
    partial_path: Path,
    in_cols: list[str],
    layers: list[int],
    hidden_size: int,
    metadata: dict,
) -> tuple[h5py.File, dict]:
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    h5_file = h5py.File(partial_path, "x")
    h5_file.attrs["complete"] = False
    h5_file.attrs["metadata"] = json.dumps(metadata, sort_keys=True)
    layer_dss: dict = {col: {} for col in in_cols}
    for col in in_cols:
        h5_file.create_group(col)
        layer_dss[col]["embed"] = h5_file.create_dataset(
            f"{col}/embed", shape=(0, hidden_size), maxshape=(None, hidden_size), dtype=np.float16
        )
        for layer in layers:
            layer_dss[col][layer] = h5_file.create_dataset(
                f"{col}/layer_{layer}", shape=(0, hidden_size), maxshape=(None, hidden_size), dtype=np.float16
            )
    return h5_file, layer_dss


def requested_layers(selection: list[int], num_layers: int) -> list[int]:
    # Intentional paper convention: layer 0 through N-1; omit the final state N.
    if selection == [-1]:
        return list(range(num_layers))
    if not selection or len(set(selection)) != len(selection) or any(layer < 0 or layer > num_layers for layer in selection):
        raise ValueError(f"--layers must be unique indices between 0 and {num_layers}, or -1")
    return sorted(selection)


def input_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def skip_completed(path: Path, request: dict, selection: list[int]) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            stored = json.loads(f.attrs.get("metadata", "null"))
            if not bool(f.attrs.get("complete", False)) or not isinstance(stored, dict):
                raise ValueError("missing completion marker or metadata")
        num_layers, width = stored["num_hidden_layers"], stored["hidden_size"]
        if not isinstance(num_layers, int) or not isinstance(width, int) or min(num_layers, width) <= 0:
            raise ValueError("invalid model dimensions in metadata")
        layers = requested_layers(selection, num_layers)
        expected = dict(request, num_hidden_layers=num_layers, hidden_size=width, layers=layers)
        lib.validate_embeddings(path, request["in_cols"], request["rows"], layers, width, metadata=expected)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot reuse {path}: {exc}. Move it aside or choose a new run_id; "
            "existing final files are never overwritten."
        ) from exc
    print(f"[gen-embeds] {path}: validated complete; skipping")
    return True


def finalize_dataset(partial_path: Path, final_path: Path, metadata: dict) -> None:
    """Validate before exposing an extraction at its final filename."""
    lib.validate_embeddings(
        partial_path, metadata["in_cols"], metadata["rows"], metadata["layers"],
        metadata["hidden_size"], metadata=metadata, require_complete=False,
    )
    with h5py.File(partial_path, "r+") as f:
        f.attrs["complete"] = True
        f.flush()
    # Linking within the same directory atomically exposes the completed file
    # and fails if another process created the final path in the meantime.
    os.link(partial_path, final_path)
    partial_path.unlink()


def generate(model, dataloader, layer_dss: dict, in_col: str, layers: list[int], device: torch.device) -> None:
    import torch

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=in_col):
            batch = {k: v.to(device) for k, v in batch.items()}

            embed_layer = model.get_input_embeddings()
            tok_embeddings = lib.mean_pooling(embed_layer(batch["input_ids"]), batch["attention_mask"]).cpu()
            _append(layer_dss[in_col]["embed"], tok_embeddings.half())

            outputs = model(**batch)
            for layer in layers:
                pooled = lib.mean_pooling(outputs["hidden_states"][layer], batch["attention_mask"]).cpu()
                _append(layer_dss[in_col][layer], pooled.half())

            del batch, outputs
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=None, help="overrides config's batch_size")
    p.add_argument("--layers", type=int, nargs="+", default=[-1], help="-1 = paper range: hidden states 0 through N-1")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--restart-incomplete", action="store_true", help="discard a .partial extraction and start it again")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    batch_size = args.batch_size or int(cfg.get("batch_size", 4))

    data_path = ROOT / cfg["input_data"]
    data = lib.load_table(data_path)
    if len(data) == 0 or not cfg["in_cols"]:
        raise ValueError("Embedding extraction requires nonempty input rows and text columns")
    if any(col not in data for col in cfg["in_cols"]):
        raise ValueError(f"{data_path}: configured text columns are missing")
    print(f"[gen-embeds] loaded {data_path} ({len(data):,} rows)")

    out_dir = ROOT / cfg["out_dir"]
    slug = lib.model_slug(args.model)
    ds_path = out_dir / slug / f"{cfg['run_id']}.{cfg['input_name']}.h5"
    partial_path = ds_path.with_suffix(ds_path.suffix + ".partial")
    request = {
        "format_version": 1, "model": args.model, "input_sha256": input_digest(data_path),
        "in_cols": cfg["in_cols"], "idx_col": cfg["idx_col"], "rows": len(data),
        "seed": args.seed, "batch_size": batch_size, "max_length": 512,
    }
    if skip_completed(ds_path, request, args.layers):
        return
    if partial_path.exists():
        if not args.restart_incomplete:
            raise FileExistsError(
                f"{partial_path}: interrupted extraction. Rerun with --restart-incomplete "
                "to discard this partial file and regenerate it."
            )
        partial_path.unlink()

    lib.set_all_seeds(args.seed)
    device = get_device(args.device)
    model, tokenizer = lib.load_encoder(args.model, device=device)
    num_layers, hidden_size = lib.model_info(model)
    layers = requested_layers(args.layers, num_layers)
    metadata = dict(request, num_hidden_layers=num_layers, hidden_size=hidden_size, layers=layers)
    h5_file, layer_dss = init_dataset(partial_path, cfg["in_cols"], layers, hidden_size, metadata)
    print(f"[gen-embeds] writing {partial_path}")

    try:
        for in_col in cfg["in_cols"]:
            dataloader = lib.create_dataloader(data, tokenizer, args.model, in_col, cfg["idx_col"], batch_size)
            generate(model, dataloader, layer_dss, in_col, layers, device)
            print(f"[gen-embeds] completed column {in_col}")
    finally:
        h5_file.close()
    finalize_dataset(partial_path, ds_path, metadata)
    print(f"[gen-embeds] done: {ds_path}")


if __name__ == "__main__":
    main()
