# HealthCareMagic dialect queries

The dialect-bias (AAL vs. WME) experiment over the HealthCareMagic-100k corpus.
Two query sets share the same retrievable corpus:

- **HCMagic-synth** (paired): real WME questions + synthetic AAL paraphrases of each,
  with a shared gold answer per pair.
- **HCMagic-nat** (unpaired): separate naturally-occurring AAL and WME questions,
  each with its own gold answer.

`real` in file names (e.g. `hcmagic_real_paired.csv`,
`hcmagic_real_per_query_metrics.csv`, `lex_residual_real_*.csv`) denotes the
natural set (HCMagic-nat).

## Layout

```text
hcmagic/
├── configs/
│   ├── retrieval.yaml          # encoder set + shared corpus/query paths
│   ├── label_dialect.yaml
│   └── synthesize_aal.yaml
├── data/
│   ├── hcmagic_100k/           # 112,165 answers, paired queries, and qrels
│   └── healthq_aal/            # natural queries, AAL paraphrases, translations
├── embeddings/                 # gitignored — embeddings bundle
├── lib.py                      # loaders and paths
├── metrics.py                  # reciprocal rank and recall@k shared by dense + BM25 scoring
├── scripts/
│   ├── label_dialect.py        # dialect-label the raw HealthCareMagic-100k
│   │                           #   set (twitteraae) → healthq_aal/*.csv
│   ├── synthesize_aal.py       # synthetic AAL paraphrases (Multi-VALUE +
│   │                           #   PhonATe) → hcmagic_wme_synth.csv
│   ├── prepare_corpus.py       # unified corpus + qrels for both query sets
│   ├── embed_corpus.py  embed_queries.py  retrieve.py  retrieve_bm25.py
│   ├── analyze_retrieval.py    # per-encoder gaps, confidence intervals, tests
│   ├── analyze_lexical_synthetic.py  analyze_lexical_natural.py   # lex-residual tables
│   ├── plot_dialect_retrieval.py    # aalwme_{synth,real_unpaired,translation}_ksweep
│   │                           #   + aalwme_translation_forest
│   └── run.sh                  # full pipeline driver (encode → retrieve → analyze)
└── results/                    # per-encoder per-query metrics + the above outputs
```

## Dialect labeling / synthesis

`data/healthq_aal/*.csv` is checked in pre-built, so the
encode→retrieve→analyze pipeline below needs no re-run of this step. To
regenerate it from the raw `HealthCareMagic-100k.json` (gitignored — see
[Sources and terms](../docs/data.md#sources-and-terms)), in order:

```bash
python hcmagic/scripts/label_dialect.py --config hcmagic/configs/label_dialect.yaml
python hcmagic/scripts/synthesize_aal.py --config hcmagic/configs/synthesize_aal.yaml
python hcmagic/scripts/prepare_corpus.py
```

The AAL→WME back-translations (`hcmagic_aal_translated.jsonl`) used in the
paper's translation-mitigation appendix were produced with `gpt-5-mini` at its
default temperature via the OpenAI Batch API; the shipped file is the one the
paper used, and that pipeline's script is not part of this release. Each translation's `row_id` is the AAL row's `idx` in
`hcmagic_aal_wme.csv` plus one; regenerating the natural query set requires
corresponding translations before running `prepare_corpus.py`.

The first two steps need external tooling (a `twitteraae` clone, `pip
install value-nlp`, and a `PhonATe` clone) — see
[external dependency setup](../docs/reproduction.md#dialect-query-data).
These steps overwrite the committed files in `data/healthq_aal/` and
`data/hcmagic_100k/` (restore them with `git checkout -- hcmagic/data`); use the
committed queries for the reported results. The committed CSVs differ from what
these scripts write: `hcmagic_aal_wme.csv` has 656 AAL and 659 WME rows, and
the WME rows' `idx` is the raw HealthCareMagic-100k row index;
`hcmagic_wme_synth.csv` has no `idx` column (row position is used) and has
extra `Unnamed: 0` and `aal_messages` columns. Rerunning the scripts yields
equal-size samples with `idx` = row order.

The paired synthetic set contains 4,999 source questions, each with
three AAL variants. After filtering, the natural set contains 647 AAL and
659 WME questions, with WME translations for all 647 AAL questions.
`figs/scripts/analyze_query_lengths.py` writes the nine AAL rows that the
input filter in `lib.py::load_hcmagic_real` drops to
`figs/results/hcmagic_dropped_queries.csv`.

## Reproduce

`python scripts/reproduce.py` (see the [top-level README](../README.md)) runs
every HCMagic step. The individual commands, from the repository root after
`pip install -e .`, read committed per-query metrics:

```bash
python hcmagic/scripts/analyze_retrieval.py --config hcmagic/configs/retrieval.yaml   # gap tables
python figs/scripts/plot_dialect_overview.py        # main dialect figure (aalwme_combined)
python hcmagic/scripts/plot_dialect_retrieval.py    # appendix dialect figures
python hcmagic/scripts/analyze_lexical_synthetic.py # synthetic lexical analysis
python hcmagic/scripts/analyze_lexical_natural.py   # natural lexical analysis
```

The analysis writes `results/hcmagic_synth_paired.csv` and
`results/hcmagic_real_paired.csv`. The natural-query summary includes both
paired translation comparisons and unpaired natural AAL/WME comparisons.
For unpaired comparisons, `perm_p` stores the Mann–Whitney U p-value,
`n_pairs` stores the smaller group size, and `ci_low`/`ci_high` come from a
two-group percentile bootstrap that resamples each group independently
(5,000 draws from one seed-42 stream shared across encoders in table order);
the queries are not paired.

Paired permutation p-values use the +1 correction `(b + 1) / (B + 1)`, where
`b` counts null draws at least as extreme as the observed statistic and
`B = 50,000`, so the smallest reportable value is 0.00002.

Re-running dense retrieval requires the [embeddings bundle](../docs/data.md) or
new embeddings. `bash hcmagic/scripts/run.sh` runs encoding, retrieval, and
analysis using the active Python environment (`PYTHON` can override the
interpreter). Adjust the device assignments in `configs/retrieval.yaml` to
match the available GPUs; the OpenAI encoder also requires an API key.

BM25 can be run separately without embeddings:

```bash
python hcmagic/scripts/retrieve_bm25.py --config hcmagic/configs/retrieval.yaml
```

The committed per-query metrics, and the paper, credit one gold answer per
query. `data/hcmagic_100k/qrels.json` also credits byte-identical copies of that
answer (148 of 6,952 queries have more than one gold document, at most 123).
Re-running retrieval therefore raises MRR for those queries, and ties between
identically scored documents can break differently; together these shift
HCMagic-nat gaps by up to 0.001 MRR and HCMagic-synth gaps by less than 0.0001.
