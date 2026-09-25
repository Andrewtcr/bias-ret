"""AllSides synthetic-frame retrieval-bias experiment library.

  - corpus comes from the Qbias CSV release (Haak & Schaer, WebSci 2023);
  - story_id (roundup id) is derived from Qbias's `title` column
    (the AllSides headline-roundup title), which is the editorial linkage
    AllSides uses to group L/C/R articles covering one event;
  - stance comes from the Qbias `bias_rating` column.

The path layout and the story lookup are defined here; the shared
retrieval metrics are defined in
`allsides_synth/metrics.py` and re-exported below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.io import ensure_dir, read_jsonl

# Retrieval metrics (re-exported for retrieve.py).
from allsides_synth.metrics import compute_bias_metrics  # noqa: F401

# ---------------------------------------------------------------------------
# Paths relative to allsides_synth/
# ---------------------------------------------------------------------------
QBIAS_ROOT = Path(__file__).resolve().parent

DATA_ROOT = QBIAS_ROOT / "data"
EMBEDDINGS_ROOT = QBIAS_ROOT / "embeddings"
RESULTS_BASE = QBIAS_ROOT / "results"


def _cfg_dir(cfg: dict[str, Any], key: str, default_root: Path) -> Path:
    """Resolve a directory from cfg.run.<key> (relative to allsides_synth/),
    falling back to the default data, embeddings, or results directory."""
    override = (cfg.get("run") or {}).get(key)
    if override:
        p = Path(override)
        return ensure_dir(p if p.is_absolute() else QBIAS_ROOT / p)
    return ensure_dir(default_root)


def run_dir(cfg: dict[str, Any]) -> Path:
    return _cfg_dir(cfg, "results_dir", RESULTS_BASE)


def data_dir(cfg: dict[str, Any]) -> Path:
    return _cfg_dir(cfg, "data_dir", DATA_ROOT)


def embeddings_dir(cfg: dict[str, Any]) -> Path:
    return _cfg_dir(cfg, "embeddings_dir", EMBEDDINGS_ROOT)


def build_story_lookup(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build {article_id -> {story_id, side}} keyed by the AllSides roundup title
    (the editorial story id: every roundup is one event with L/C/R counterparts).

    Reads the processed `corpus.jsonl`, which carries the roundup title and
    stance needed to identify counterparts among corpus articles.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in read_jsonl(data_dir(cfg) / "corpus.jsonl"):
        roundup = (r.get("roundup_title") or "").strip()
        side = r.get("stance")
        if not roundup or side not in ("left", "center", "right"):
            continue
        out[r["article_id"]] = {"story_id": roundup, "side": side}
    return out

