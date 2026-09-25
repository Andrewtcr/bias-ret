# AllSides synthetic queries

Synthetic political-frame queries over the AllSides article corpus (§2 of the
paper). For each (article, MFC frame), the data contain a left, neutral, and
right query; the neutral query is not used in the paper's left/right contrasts.
Retrieval masks the source article, and the analysis measures the political
lean of the retrieved set for each retriever.

## Files

- `configs/query_generation.yaml`: frame extraction and query-generation prompts
  and response schemas. It records the prompts that produced the committed
  frames and queries; the generation script is not part of this release.
- `configs/retrieval.yaml`: retrieval depth and per-encoder settings.
- `configs/judge_relevance.yaml`: binary relevance-judging prompts and settings.
- `configs/pairing_similarity.yaml`: within-cell query-pair similarity analysis.
- `data/`: processed corpus, extracted frames, and queries.
- `scripts/`: encoding, retrieval, relevance judging, pairing-similarity
  analysis, and lexical-residual regression.
- `results/`: retrieval metrics, parsed relevance judgments, lexical scores and
  regressions, and pairing-similarity scores.
  `retrieval_candidates/` contains top-10 rankings
  as document IDs; `_min_story_rank.parquet` records the first same-story hit.

## Reproduce from committed data

`python scripts/reproduce.py` (see the [top-level README](../README.md)) runs
every AllSides step: the gold-filtered paired test, the lexical-residual
regression, the pairing-similarity figure, and the political figures that read
this directory. The individual commands, from the repository root after
installing the dependencies:

```bash
python figs/scripts/analyze_political_gold.py              # gold-filtered paired test at k=10
python figs/scripts/plot_political_overview.py             # main political figure (also reads reddit_nat)
python figs/scripts/plot_political_retrieval.py            # AllSides k-sweep appendix figure
python allsides_synth/scripts/plot_pairing_similarity.py   # pairing-check figure and mean cosines
```

The lexical-residual commands are listed below.

Re-running retrieval requires the [embeddings bundle](../docs/data.md):

```bash
python allsides_synth/scripts/retrieve.py --config allsides_synth/configs/retrieval.yaml
```

To rebuild the query-pair similarity scores from the embeddings bundle:

```bash
python allsides_synth/scripts/analyze_pairing_similarity.py --config allsides_synth/configs/pairing_similarity.yaml
```

The scripts write to `results/pairing_similarity/`. The per-anchor scores
support the boxplot; `summary.csv` contains the paired median-difference tests
and `index_meta.json` records the sampling settings and counts.
`plot_pairing_similarity.py` computes `pairing_mean_summary.csv`, the mean
cosine confidence intervals reported in the appendix's query-side pairing
check, from the per-anchor scores.

To re-encode the corpus and queries, install the additional dependencies in
[the reproduction guide](../docs/reproduction.md#retriever-embeddings), set the devices in
`configs/retrieval.yaml`, and run `scripts/embed.py` with that config.
`scripts/run.sh` runs the configured encoders concurrently before retrieval;
it uses the active Python environment, with optional `PYTHON` and `LOG_DIR`
overrides. The OpenAI encoder and relevance judge require API credentials and
incur usage charges.

```bash
python allsides_synth/scripts/embed.py --config allsides_synth/configs/retrieval.yaml   # add --only <encoder> for one encoder
bash allsides_synth/scripts/run.sh allsides_synth/configs/retrieval.yaml               # all encoders in parallel, then retrieval
```

## Relevance judging

The relevance-filter appendix figures read the committed judgments in
`results/judge_relevance/`. To judge the left/right top-10 pairs of a new
ranking, from the repository root:

```bash
python allsides_synth/scripts/judge_relevance.py generate   # requests for unique (query, article) pairs
python allsides_synth/scripts/judge_relevance.py submit     # billable OpenAI batches (gpt-5-mini)
python allsides_synth/scripts/judge_relevance.py poll       # writes results/judge_relevance/judgments.jsonl
python allsides_synth/scripts/judge_relevance_qwen.py       # Qwen3.5-35B-A3B via vLLM; skips pairs already in judgments_qwen.jsonl
```

## Balanced corpus

`data/corpus.jsonl` is the balanced corpus the paper uses. It keeps Qbias
articles of at least 100 characters from roundups that have both a left and a
right article. Within each roundup it keeps min(n_L, n_R) left and right
articles, sampled with a per-roundup seed, and every center article: 7,019
left, 7,019 right, and 3,958 center articles in total. The construction script
is not part of this release.

## Lexical-residual analysis

The lexical analysis estimates how much of the paired political-lean gap
remains after accounting for vocabulary differences between left- and
right-coded queries:

```bash
python allsides_synth/scripts/analyze_lexical_scores.py
python allsides_synth/scripts/analyze_lexical_regression.py
```

`analyze_lexical_scores.py` computes token log-odds and, from per-query
lexical scores, the within-cell lexical differences.
`analyze_lexical_regression.py` fits each retriever's regression on cells where
both queries retrieve a same-story article in the top 10. The committed results in `results/` support the paper's lexical-residual
tables. The separate layer-wise analysis is in [probing](../probing/README.md).

Sign convention: the AllSides outputs store Δ = lean(L) − lean(R)
(`lex_regression_per_encoder_gold_filtered.csv` and
`figs/results/gold_filtered_paired_test_k10.csv`) and lex_L − lex_R
(`cell_lex_asymmetry.csv`). The paper reports R − L, so its Δ, α, and
confidence intervals are these values negated; β and the p-values are
unchanged.
