"""Shared model loading, pooling, dataset construction, and probe evaluation.

Torch, Transformers, and Datasets are imported only by embedding-generation
helpers. Probe training and plotting do not require those packages.
"""
from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import torch

TASK = "Given a web search query, retrieve relevant passages that answer the query"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    import torch
    import transformers

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------
def model_slug(chkpt: str) -> str:
    return chkpt.split("/")[-1].lower()


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSONL input table, dispatched on file suffix."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def layer_ids(group: h5py.Group) -> list[int]:
    """Return the actual stored hidden-state indices, including sparse sets."""
    layers = []
    for name in group:
        if name.startswith("layer_"):
            if not re.fullmatch(r"layer_(0|[1-9][0-9]*)", name):
                raise ValueError(f"{group.name}: invalid layer name {name!r}")
            layers.append(int(name.removeprefix("layer_")))
    if not layers:
        raise ValueError(f"{group.name}: no layer embeddings found")
    return sorted(layers)


def validate_embeddings(
    path: str | Path,
    in_cols: list[str],
    expected_rows: int,
    expected_layers: list[int],
    hidden_size: int,
    *,
    metadata: dict,
    require_complete: bool = True,
) -> None:
    """Validate an extraction's schema and shape before reuse or publication."""
    with h5py.File(path, "r") as f:
        if require_complete and "complete" in f.attrs and not bool(f.attrs["complete"]):
            raise ValueError(f"{path}: extraction is not marked complete")
        if json.loads(f.attrs.get("metadata", "null")) != metadata:
            raise ValueError(f"{path}: extraction metadata does not match this request")
        if set(f) != set(in_cols):
            raise ValueError(f"{path}: text columns do not match this request")
        names = {"embed", *(f"layer_{layer}" for layer in expected_layers)}
        for col in in_cols:
            if not isinstance(f[col], h5py.Group) or set(f[col]) != names:
                raise ValueError(f"{path}: {col} has a different or incomplete layer set")
            for name in names:
                ds = f[col][name]
                if not isinstance(ds, h5py.Dataset) or ds.shape != (expected_rows, hidden_size):
                    raise ValueError(
                        f"{path}: {col}/{name} must have shape "
                        f"({expected_rows}, {hidden_size})"
                    )
                if ds.dtype != np.dtype("float16"):
                    raise ValueError(f"{path}: {col}/{name} must contain float16 embeddings")


def probe_layer_ids(path: str | Path, groups: list[str], conditional: bool = False) -> list[int]:
    """Check the requested groups and return layer IDs used for probe training."""
    if str(path).endswith(".partial"):
        raise ValueError(f"{path}: finish embedding extraction before training probes")
    with h5py.File(path, "r") as f:
        if "complete" in f.attrs and not bool(f.attrs["complete"]):
            raise ValueError(f"{path}: finish embedding extraction before training probes")
        layers = None
        width = None
        for group in groups:
            current = layer_ids(f[group])
            if layers is not None and current != layers:
                raise ValueError(f"{path}: text groups have different layer sets")
            layers = current
            rows = None
            for layer in layers:
                ds = f[group][f"layer_{layer}"]
                if not isinstance(ds, h5py.Dataset) or ds.ndim != 2 or ds.shape[0] == 0:
                    raise ValueError(f"{path}: {group}/layer_{layer} is empty or malformed")
                if rows is not None and ds.shape[0] != rows:
                    raise ValueError(f"{path}: {group} has inconsistent layer row counts")
                if width is not None and ds.shape[1] != width:
                    raise ValueError(f"{path}: inconsistent hidden-state dimensions")
                rows, width = ds.shape
        if conditional and 0 not in (layers or []):
            raise ValueError(f"{path}: conditional probes require layer 0; regenerate with --layers 0 ...")
        return layers or []


# ---------------------------------------------------------------------------
# Encoder loading + pooling (generate_embeddings.py)
# ---------------------------------------------------------------------------
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def load_encoder(model_chkpt: str, device: "torch.device | None" = None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_chkpt, padding_side="left", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_chkpt, output_hidden_states=True, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


def model_info(model) -> tuple[int, int]:
    return model.config.num_hidden_layers, model.config.hidden_size


def mean_pooling(token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
    """From https://github.com/MilaNLProc/socio-probe/blob/main/embedder.py"""
    import torch

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def make_collate_fn(tokenizer, in_col: str):
    def collate_fn(batch: list[dict]):
        texts = [item[in_col] for item in batch]
        return tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    return collate_fn


def create_dataloader(
    data: pd.DataFrame, tokenizer, model_name: str, in_col: str, idx_col: str, batch_size: int
):
    from datasets import Dataset as HFDataset
    from torch.utils.data import DataLoader

    data = data.copy()
    if idx_col not in data.columns:
        # The shipped hcmagic_wme_synth.csv has no idx column (regenerating it
        # with hcmagic/scripts/synthesize_aal.py adds one). h5<->table alignment
        # is positional (shuffle=False), so row positions are equivalent.
        print(f"[lib] '{idx_col}' not in table; synthesizing from row positions")
        data[idx_col] = range(len(data))
    data[in_col] = data[in_col].fillna("")
    if "qwen" in model_name.lower():
        data[in_col] = data[in_col].apply(lambda q: get_detailed_instruct(TASK, q))

    ds = HFDataset.from_pandas(data[[idx_col, in_col]])
    return DataLoader(
        ds, batch_size=batch_size, collate_fn=make_collate_fn(tokenizer, in_col), shuffle=False
    )


# ---------------------------------------------------------------------------
# Probe dataset construction (train_probes.py)
# ---------------------------------------------------------------------------
def _read_layer(h5_file: h5py.File, group: str, layer: int) -> np.ndarray:
    arr = h5_file[group][f"layer_{layer}"][:, :].astype(np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_aligned_xy(
    h5_path: str | Path,
    input_type: str,
    layer: int,
    data: pd.DataFrame,
    label_col: str,
    pos_label: str,
    exclude_label: str | None = None,
    layer0_concat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One h5 group's embeddings, row-aligned against `data[label_col]`.

    X contains one input type's per-layer embeddings; Y is derived from a
    label column. `exclude_label` drops a third class (such as neutral
    AllSides queries) before the binary split.
    """
    with h5py.File(h5_path, "r") as f:
        X = _read_layer(f, input_type, layer)
        if layer0_concat and layer != 0:
            X = np.concatenate([X, _read_layer(f, input_type, 0)], axis=1)

    labels = data[label_col]
    keep = pd.Series(True, index=data.index)
    if exclude_label is not None:
        keep &= labels != exclude_label
    X = X[keep.to_numpy()]
    Y = (labels[keep] == pos_label).astype(int).to_numpy()
    return X, Y


def build_paired_xy(
    h5_path: str | Path,
    pos_col: str,
    neg_col: str,
    layer: int,
    layer0_concat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One "positive" h5 group (e.g. an `aal_message{k}` dialect-transform
    variant) vs. one "negative" group (e.g. the original WME text), as a
    single binary dataset. The caller trains one probe per variant against
    the shared negative baseline. Labels identify the source h5 group.
    """
    with h5py.File(h5_path, "r") as f:
        pos = _read_layer(f, pos_col, layer)
        neg = _read_layer(f, neg_col, layer)
        X = np.concatenate([pos, neg], axis=0)
        if layer0_concat and layer != 0:
            pos0 = _read_layer(f, pos_col, 0)
            neg0 = _read_layer(f, neg_col, 0)
            X = np.concatenate([X, np.concatenate([pos0, neg0], axis=0)], axis=1)
    Y = np.array([1] * len(pos) + [0] * len(neg))
    return X, Y


# ---------------------------------------------------------------------------
# Probe train/eval
# ---------------------------------------------------------------------------
def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        # Negative cross-entropy; subtract a layer-0 baseline to obtain
        # conditional V-information.
        "nce": float(-log_loss(y_true, y_prob)),
    }


def train_eval_lr_probe(
    X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray, max_iter: int = 5000
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scaler", StandardScaler()), ("probe", LogisticRegression(max_iter=max_iter))])
    pipe.fit(X_train, Y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return pipe, class_metrics(Y_test, y_pred, y_prob)


def save_probe(pipe, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)


__all__ = [
    "TASK",
    "set_all_seeds",
    "model_slug",
    "load_table",
    "layer_ids",
    "validate_embeddings",
    "probe_layer_ids",
    "get_detailed_instruct",
    "load_encoder",
    "model_info",
    "mean_pooling",
    "make_collate_fn",
    "create_dataloader",
    "build_aligned_xy",
    "build_paired_xy",
    "class_metrics",
    "train_eval_lr_probe",
    "save_probe",
]
