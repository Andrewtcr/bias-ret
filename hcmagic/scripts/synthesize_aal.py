"""Synthesize AAL paraphrases of WME HealthCareMagic questions.

Uses `multivalue` (Multi-VALUE, `pip install value-nlp`) for
`Dialects.AfricanAmericanVernacular().transform(...)` plus the externally-
cloned PhonATe (github.com/NickDeas/PhonATe) for phonological augmentation,
to produce `num_variants` (default 3) AAL paraphrase variants per WME
question. See docs/reproduction.md for setup of both.

Before constructing the dialect transformer, `_patch_multivalue_coref`
aligns Stanza coreference mentions to spaCy tokens by character offsets.
Unaligned mentions are skipped so tokenizer disagreements do not stop
query synthesis.

Input:
  hcmagic/data/healthq_aal/hcmagic_big_wme_samp.csv
    (written by label_dialect.py; `input` holds the WME question)

Output:
  hcmagic/data/healthq_aal/hcmagic_wme_synth.csv
    — input rows + aal_message1/2/3 (empty string on rows where the
      transform pipeline raised — malformed sentence splits, spaCy/Stanza
      tokenization mismatches, etc. are expected on a small fraction of
      rows and are skipped rather than failing the run) + `idx` = row order
      (the committed file differs; see hcmagic/README.md)

Sentence splitting keeps `[...]`-bracketed placeholder tokens (occasional
redaction markers in the source data) untouched; each sentence is
dialect-transformed independently, then all `num_variants` phon-augmented
variants for a row are produced in one `AALPhonate.full_phon_aug` call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # hcmagic/

_SENT_SPLIT_RE = re.compile(r"(?<=[,.!?:;])|(\[\w+\])")


def _patch_multivalue_coref() -> None:
    """Replace BaseDialect.create_coref_cluster with a char-offset mapping
    that skips unaligned mentions instead of asserting. See module docstring."""
    from multivalue.BaseDialect import BaseDialect

    def create_coref_cluster(self, string):
        doc = self.coref(string)
        clusters = []
        for chain in doc.coref:
            cluster = []
            for ment in chain.mentions:
                stanza_sent = doc.sentences[ment.sentence]
                start_char = stanza_sent.words[ment.start_word].start_char
                end_char = stanza_sent.words[ment.end_word - 1].end_char
                spacy_span = self.doc.char_span(start_char, end_char)
                if spacy_span is None:
                    continue
                cluster.append(spacy_span)
            if cluster:
                clusters.append(cluster)
        return clusters if clusters else None

    BaseDialect.create_coref_cluster = create_coref_cluster


def _resolve_phonate_dir(cli_value: str | None, cfg_value: str | None) -> Path:
    candidate = cli_value or cfg_value or os.environ.get("PHONATE_HOME") or "../PhonATe"
    path = Path(candidate).resolve()
    if not (path / "phonate" / "default_config.json").exists():
        raise FileNotFoundError(
            f"PhonATe not found at '{path}' (missing phonate/default_config.json). "
            f"Clone it first:\n  git clone https://github.com/NickDeas/PhonATe {path}\n"
            "or set PHONATE_HOME / pass --phonate-dir. "
            "See docs/reproduction.md."
        )
    return path


def _load_phonate(phonate_dir: Path):
    """Construct AALPhonate from PhonATe's default_config.json, resolving its
    `p2g_model` field (a "./byt5-aal-p2g" path, relative to PhonATe's own
    repo root) to an absolute path first — passing config= directly would
    resolve that path against this process's CWD instead and fail."""
    if str(phonate_dir) not in sys.path:
        sys.path.insert(0, str(phonate_dir))
    from phonate import AALPhonate  # type: ignore

    config_path = phonate_dir / "phonate" / "default_config.json"
    with open(config_path) as f:
        config = json.load(f)
    p2g_model = (phonate_dir / config["p2g_model"]).resolve()
    if not p2g_model.exists():
        raise FileNotFoundError(
            f"PhonATe's phoneme-to-grapheme model not found at '{p2g_model}'. "
            f"Untar it first: tar -xzf {phonate_dir / 'byt5-aal-p2g.tar.gz'} -C {phonate_dir}\n"
            "See docs/reproduction.md."
        )
    return AALPhonate(
        p2g_model=str(p2g_model),
        g2p_model=config["g2p_model"],
        tok=config["tok"],
        device=config["device"],
        probs=config["probs"],
        augs=config["augs"],
    )


def load_transform_libs(phonate_dir: Path):
    from multivalue import Dialects  # type: ignore  # pip install value-nlp

    _patch_multivalue_coref()
    aal_transform = Dialects.AfricanAmericanVernacular()
    aal_phonate = _load_phonate(phonate_dir)
    return aal_transform, aal_phonate


def transform_texts(aal_transform, aal_phonate, texts: list[str], num_variants: int) -> list[list[str]]:
    all_aug_texts = []
    for text in texts:
        aug_text_vars = []
        sents = _SENT_SPLIT_RE.split(text)
        sents = [s for s in sents if s is not None and s.strip() != ""]
        for _ in range(num_variants):
            full_aug = ""
            for sent in sents:
                if "[" not in sent and "]" not in sent:
                    ms_aug = aal_transform.transform(sent.strip())
                else:
                    ms_aug = sent
                full_aug += ms_aug + " "
            aug_text_vars.append(full_aug.strip())
        _, _, _, aug_text_vars = aal_phonate.full_phon_aug(aug_text_vars)
        all_aug_texts.append(aug_text_vars)
    return all_aug_texts


def split_variants(variants: list[str], num_variants: int) -> list[str]:
    if len(variants) == num_variants:
        return list(variants)
    return [""] * num_variants


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--phonate-dir", default=None)
    args = p.parse_args()
    cfg = load_yaml(args.config)

    src_csv = ROOT / cfg["source_csv"]
    in_col = cfg.get("input_col", "input")
    out_csv = ROOT / cfg["output_csv"]
    num_variants = int(cfg.get("num_variants", 3))
    ext_cfg = cfg.get("external", {})

    ph_dir = _resolve_phonate_dir(args.phonate_dir, ext_cfg.get("phonate_dir"))
    print(f"[synth-aal] PhonATe: {ph_dir}")
    aal_transform, aal_phonate = load_transform_libs(ph_dir)

    print(f"[synth-aal] loading {src_csv}")
    data = pd.read_csv(src_csv)
    print(f"[synth-aal]   {len(data):,} rows")

    augmented: list[list[str]] = []
    n_errors = 0
    for i, message in enumerate(tqdm(data[in_col], desc="synth-aal")):
        try:
            augmented += transform_texts(aal_transform, aal_phonate, [str(message)], num_variants)
        except Exception as e:  # noqa: BLE001 — expected on a fraction of rows (tokenizer mismatches etc.)
            n_errors += 1
            print(f"[synth-aal] row {i} errored: {e!r}")
            augmented.append([])
    print(f"[synth-aal] {n_errors:,} / {len(data):,} rows errored during transform")

    data["aal_messages"] = augmented
    variant_cols = [f"aal_message{k}" for k in range(1, num_variants + 1)]
    data[variant_cols] = data["aal_messages"].apply(
        lambda v: pd.Series(split_variants(v, num_variants))
    )
    data = data.drop(columns=["aal_messages"])
    data["idx"] = data.index

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_csv, index=False)
    n_complete = int((data[variant_cols[0]] != "").sum())
    print(f"[synth-aal] wrote {out_csv} ({len(data):,} rows, {n_complete:,} with all {num_variants} variants)")


if __name__ == "__main__":
    main()
