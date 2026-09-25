"""Dialect-proxy label the raw HealthCareMagic-100k question set.

Classifies each row's patient-authored `input` text with the twitteraae
demographic-proxy model (`src.dialect.classify_dialect`) and derives an
AAL/WME split for corpus preparation and query synthesis.

Input:
  hcmagic/data/healthq_aal/HealthCareMagic-100k.json
    (raw, gitignored upstream file; 112,165 records — see docs/data.md)

Output (all under hcmagic/data/healthq_aal/):
  hcmagic_aal_labels.csv   — every row + AA,H,A,W posterior + aa_max/w_max
  hcmagic_aal_wme.csv      — aal ∪ wme, `dial` in {aal, wme}, `idx` = row
                              order (this is prepare_corpus.py's
                              SRC_REAL / the hcmagic real arm; the
                              committed file differs, see hcmagic/README.md)
  hcmagic_big_wme_samp.csv — a larger W-argmax sample (config
                              `sampling.wme_big_n`, default 5000): the base
                              population synthesize_aal.py draws
                              synthetic AAL paraphrases from

Rows without a classifier posterior receive an all-zero vector. Argmax
ties resolve to "AA", the first label. Labels are demographic proxies,
not verified speaker identities. Use the committed query sets to reproduce
the reported retrieval results; rerunning labeling and sampling creates
new query sets.

Requires an external `twitteraae` clone — see docs/reproduction.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.dialect import DIALECT_LABELS, classify_dialect, dominant_dialect, ensure_available  # noqa: E402
from src.io import load_yaml  # noqa: E402
from src.stats import stable_seed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # hcmagic/


def label_row(text: object, twitteraae_dir: str | None) -> dict[str, float]:
    try:
        posterior = classify_dialect(str(text), twitteraae_dir=twitteraae_dir)
    except Exception as e:  # noqa: BLE001 — a handful of malformed rows shouldn't kill a 112k-row run
        print(f"[label-dialect] WARNING: classification failed ({e!r}); defaulting to zeros")
        posterior = None
    if posterior is None:
        return {label: 0.0 for label in DIALECT_LABELS}
    return dict(zip(DIALECT_LABELS, posterior.tolist()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--twitteraae-dir", default=None, help="overrides config + $TWITTERAAE_HOME")
    args = p.parse_args()
    cfg = load_yaml(args.config)

    src_json = ROOT / cfg["source_json"]
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    twitteraae_dir = args.twitteraae_dir or cfg.get("dialect", {}).get("twitteraae_dir")
    wme_big_n = int(cfg.get("sampling", {}).get("wme_big_n", 5000))

    print(f"[label-dialect] loading {src_json}")
    data = pd.read_json(src_json)
    print(f"[label-dialect]   {len(data):,} records")

    # Validate the classifier installation before processing individual rows.
    ensure_available(twitteraae_dir)

    print("[label-dialect] classifying dialect posteriors (this walks all rows once)")
    labels_df = pd.DataFrame(
        [label_row(t, twitteraae_dir) for t in data["input"]]
    )
    data = pd.concat([data.reset_index(drop=True), labels_df], axis=1)

    dominant = data[list(DIALECT_LABELS)].apply(lambda r: dominant_dialect(r.to_numpy()), axis=1)
    data["aa_max"] = dominant == "AA"
    data["w_max"] = dominant == "W"
    print(f"[label-dialect]   aa_max: {int(data['aa_max'].sum()):,}  w_max: {int(data['w_max'].sum()):,}")

    labels_path = out_dir / "hcmagic_aal_labels.csv"
    data.to_csv(labels_path, index=False)
    print(f"[label-dialect] wrote {labels_path} ({len(data):,} rows)")

    aal = data[data["aa_max"]].reset_index(drop=True)
    wme_pool = data[data["w_max"]]
    if len(wme_pool) < len(aal):
        raise ValueError(
            f"only {len(wme_pool):,} w_max rows available, need {len(aal):,} to match "
            f"the aa_max count — check twitteraae is classifying correctly on this input "
            f"(e.g. non-English or very short `input` text can push everything into one bucket)."
        )

    rng = np.random.default_rng(stable_seed("hcmagic-wme-matched"))
    wme = wme_pool.sample(n=len(aal), replace=False, random_state=rng).reset_index(drop=True)

    aal_tagged = aal.copy()
    wme_tagged = wme.copy()
    aal_tagged["dial"] = "aal"
    wme_tagged["dial"] = "wme"
    combined = pd.concat([aal_tagged, wme_tagged], ignore_index=True)
    combined = combined.drop(columns=["aa_max", "w_max"])
    # Translation row_id is idx + 1 for rows with dial == "aal".
    combined["idx"] = combined.index
    combined_path = out_dir / "hcmagic_aal_wme.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[label-dialect] wrote {combined_path} ({len(combined):,} rows)")

    rng_big = np.random.default_rng(stable_seed("hcmagic-wme-big"))
    n_big = min(wme_big_n, len(wme_pool))
    big_wme = wme_pool.sample(n=n_big, replace=False, random_state=rng_big).reset_index(drop=True)
    big_wme_path = out_dir / "hcmagic_big_wme_samp.csv"
    big_wme.to_csv(big_wme_path, index=False)
    print(f"[label-dialect] wrote {big_wme_path} ({len(big_wme):,} rows)")


if __name__ == "__main__":
    main()
