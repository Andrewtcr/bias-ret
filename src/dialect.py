"""Dialect-proxy classification for HealthCareMagic data preparation.

Wraps the externally-cloned `twitteraae` repo (slanglab/twitteraae, Blodgett
et al.), which ships a pretrained CVB0 topic model producing a posterior over
four demographic-proxy categories (African-American, Hispanic, Asian, White)
for a token list. Not vendored here: see docs/reproduction.md for
installation instructions.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Posterior order returned by twitteraae's predict(); also the column order
# used throughout hcmagic's `AA, H, A, W` CSVs.
DIALECT_LABELS = ("AA", "H", "A", "W")

_predict_module = None  # cached, lazily-imported `twitteraae/code/predict.py`


def _resolve_twitteraae_dir(twitteraae_dir: str | Path | None = None) -> Path:
    candidate = twitteraae_dir or os.environ.get("TWITTERAAE_HOME") or "../twitteraae"
    path = Path(candidate).resolve()
    if not (path / "code" / "predict.py").exists():
        raise FileNotFoundError(
            f"twitteraae not found at '{path}'. Clone it first:\n"
            f"  git clone https://github.com/slanglab/twitteraae {path}\n"
            "or point at an existing clone via the TWITTERAAE_HOME env var "
            "or a --twitteraae-dir CLI flag. See docs/reproduction.md."
        )
    return path


def _load_predict_module(twitteraae_dir: str | Path | None = None):
    global _predict_module
    if _predict_module is not None:
        return _predict_module
    repo = _resolve_twitteraae_dir(twitteraae_dir)
    code_dir = str(repo / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    import predict as _predict  # type: ignore  # twitteraae/code/predict.py

    # Upstream predict.py resolves its model files relative to the process
    # CWD ("./model/..."); point them at the actual clone so callers don't
    # need to chdir into it.
    _predict.vocabfile = str(repo / "model" / "model_vocab.txt")
    _predict.modelfile = str(repo / "model" / "model_count_table.txt")
    _predict.load_model()
    _predict_module = _predict
    return _predict_module


def ensure_available(twitteraae_dir: str | Path | None = None) -> None:
    """Load the twitteraae model now, so a missing/misconfigured clone fails
    fast and loudly instead of being swallowed by a per-row try/except and
    silently degrading every row to an all-zero posterior. Call this once
    before classifying in bulk."""
    _load_predict_module(twitteraae_dir)


def classify_dialect(
    text: str, twitteraae_dir: str | Path | None = None
) -> np.ndarray | None:
    """Return the (AA, H, A, W) posterior for `text`.

    Returns None if too few of `text`'s tokens are in the model's vocabulary
    (twitteraae's own in-vocab count/fraction thresholds; see its predict()).
    """
    predict = _load_predict_module(twitteraae_dir)
    tokens = text.split()
    posterior = predict.predict(tokens)
    if posterior is None:
        return None
    return np.asarray(posterior, dtype=np.float64)


def dominant_dialect(posterior: np.ndarray | None) -> str | None:
    """Argmax label over DIALECT_LABELS; None if `posterior` is None.

    Ties break toward the earlier label in DIALECT_LABELS (AA > H > A > W).
    """
    if posterior is None:
        return None
    return DIALECT_LABELS[int(np.argmax(posterior))]


__all__ = ["DIALECT_LABELS", "classify_dialect", "dominant_dialect", "ensure_available"]
