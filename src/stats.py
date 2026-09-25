"""Shared statistical helpers for the retrieval experiments.

Functions are deterministic given a seed and operate on NumPy arrays.

Conventions:
- `clusters` is the unit at which observations are correlated (article_id in
  allsides; the per-query ID in hcmagic and reddit, which makes it an
  ordinary per-query bootstrap). Cluster-bootstrap resamples cluster
  IDs with replacement, then takes ALL observations belonging to the
  resampled clusters.
- All p-values are two-sided unless otherwise stated.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def stable_seed(*parts, base: int = 0) -> int:
    """Deterministic 31-bit seed derived from `parts` (strings/ints/tuples).

    Use instead of Python's hash(), which is salted per process for str inputs
    (PYTHONHASHSEED) and therefore breaks run-to-run reproducibility.
    """
    import zlib

    return base + (zlib.crc32(repr(parts).encode("utf-8")) & 0x7FFFFFFF)


# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap_mean(
    values: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 5000,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Cluster-bootstrap the mean of `values`, resampling unique cluster IDs.

    Returns (point_estimate, ci_low, ci_high) at the 95% level.
    Either pass an existing rng OR pass a seed (the function will create one).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    values = np.asarray(values)
    clusters = np.asarray(clusters)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(clusters)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in uniq}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled])
        boots[b] = values[idx].mean()
    return (
        float(values.mean()),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


# ---------------------------------------------------------------------------
# Sign-flip permutation null
# ---------------------------------------------------------------------------
def signflip_permutation_p(
    deltas: np.ndarray,
    n_perm: int = 10000,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Sign-flip permutation null for a paired Δ array.

    Under H0 the sign of each Δ is symmetric, so randomly flipping each Δ's
    sign yields the null distribution of the mean. Returns the observed mean,
    the [2.5, 97.5] interval of permutation means, and the two-sided p-value
    with the +1 correction (Phipson & Smyth 2010), so p >= 1/(n_perm+1).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas, dtype=float)
    n = len(deltas)
    if n == 0:
        return {
            "n": 0,
            "delta_obs": float("nan"),
            "null_mean": float("nan"),
            "null_ci_low": float("nan"),
            "null_ci_high": float("nan"),
            "p_two_sided": float("nan"),
        }
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * deltas).mean(axis=1)
    obs = float(deltas.mean())
    return {
        "n": n,
        "delta_obs": obs,
        "null_mean": float(null_means.mean()),
        "null_ci_low": float(np.percentile(null_means, 2.5)),
        "null_ci_high": float(np.percentile(null_means, 97.5)),
        "p_two_sided": float(
            (1 + (np.abs(null_means) >= abs(obs)).sum()) / (n_perm + 1)
        ),
    }


__all__ = [
    "stable_seed",
    "cluster_bootstrap_mean",
    "signflip_permutation_p",
]
