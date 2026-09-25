# Data

What the repo ships, and what the embeddings bundle adds (see [the README](../README.md)).

Run the commands below from the repository root.

## Included in the repository (no download)

- **Source corpora**: `allsides_synth/data/corpus.jsonl` (AllSides articles),
  `hcmagic/data/hcmagic_100k/corpus.jsonl` (the full 112,165-answer
  HealthCareMagic corpus).
- **Queries**: AllSides LLM-extracted frames and LLM-generated queries
  (`allsides_synth/data/`); HealthCareMagic natural questions, synthetic AAL
  rewrites, LLM AAL→WME translations, paired queries, and qrels
  (`hcmagic/data/`). Reddit ships no query text (see below).
- **AllSides and Reddit retrieval candidates**: the ID-only retrieval ranking at
  `<exp>/results/retrieval_candidates/{encoder}.csv` (BM25 included as
  `retrieval_candidates/bm25.csv`). AllSides rows are `query_id, encoder, rank,
  retrieved_article_id, retrieved_stance`; Reddit rows add `ideology,
  subreddit` after `query_id` and a trailing `score`. No row carries retrieved
  text: document text joins from the corpus on `retrieved_article_id`, and
  AllSides query metadata joins from `allsides_synth/data/generated_queries.jsonl`
  on `query_id`.
- **HCMagic retrieval metrics**: per-query scores at
  `hcmagic/results/{encoder}/hcmagic_{real,synth}_per_query_metrics.csv`.
- **AllSides and Reddit judge labels**: `<exp>/results/judge_relevance/judgments*.jsonl`.
- **Probe scores**: `probing/results/consolidated_scores.csv`, including normal,
  conditional, and dedicated layer-0 baseline scores. These render the
  probing figure without per-layer embeddings or trained probe weights.
- **Aggregates** used by figures/tables: `bias_metrics.csv`, `lex_*` CSVs, the
  pairing-similarity CSVs (`allsides_synth/results/pairing_similarity/`), the
  AllSides gold-filter cache (`_min_story_rank.parquet`), etc.
- **Reddit IDs + labels**: `reddit_nat/data/reddit_nat_raw.csv` (point-in-time
  pull, 2026-05-18) — post IDs, subreddit, ideology label, and length/hash
  fields only. The post text is deliberately withheld, per the paper's Ethical
  considerations; see the "Released data" section of [the Reddit README](../reddit_nat/README.md).

## Sources and terms

- **Qbias** (Haak and Schaer, WebSci 2023;
  [doi:10.1145/3578503.3583628](https://doi.org/10.1145/3578503.3583628)) is the
  source of `allsides_synth/data/corpus.jsonl`: AllSides articles with their
  outlets, lean labels, and roundups.
- **ChatDoctor HealthCareMagic-100k** (Li et al., Cureus 2023) is the source of
  the files under `hcmagic/data/`, including the corpus and the
  `healthq_aal/*.csv` queries. To regenerate the dialect CSVs, get
  `HealthCareMagic-100k.json` through the
  [ChatDoctor repository](https://github.com/Kent0n-Li/ChatDoctor) and place it
  at `hcmagic/data/healthq_aal/HealthCareMagic-100k.json`. The corpus and
  queries keep the upstream text, which may contain contact details that the
  upstream release left unmasked.

These derived files are redistributed for research use under the upstream
datasets' terms. Rehydrated Reddit text is subject to Reddit's Data API terms.

## The embeddings bundle

Only the pre-computed dense embeddings (~10 GB). They are large, take
GPU-hours / OpenAI credits to recompute, and are not needed to rebuild any
figure or table — only to re-run retrieval and the AllSides query-pairing
similarity analysis. AllSides and HCMagic embeddings can instead be regenerated
from the committed corpora and queries. Regenerating Reddit query embeddings
requires the withheld post text.

The [embeddings bundle is on Google Drive](https://drive.google.com/drive/folders/1xQmMrxQALkfm4g31L4Ofdwu-QkkqeTHg)
(folder ID `1xQmMrxQALkfm4g31L4Ofdwu-QkkqeTHg`). Downloads use
[rclone](https://rclone.org), which verifies checksums as it copies.

1. Open the folder link and verify that your Google account can view its contents.
   Access is granted on request through Google Drive.
2. Install rclone and run `rclone config`. Create a Google Drive remote named
   `gdrive`, choose read-only access, and complete browser authorization with an
   account that can read the folder. Leave the remote's root-folder setting
   empty: the fetch script selects the bundle by its folder ID.
3. From this repository's root, run:

```bash
bash scripts/fetch_embeddings.sh
python scripts/verify_embeddings.py
```

To use a differently named Drive remote or an alternative bundle copy:

```bash
BUNDLE_DRIVE_REMOTE=mydrive: bash scripts/fetch_embeddings.sh
BUNDLE_REMOTE=<remote>:<path> bash scripts/fetch_embeddings.sh
```

AllSides and Reddit files land under `<exp>/embeddings/{encoder}/`.
HCMagic corpus embeddings land under `hcmagic/embeddings/{encoder}-hcmagic_100k/`,
and its query arrays at `hcmagic/embeddings/{encoder}/{variant}_embeddings.npy`.
These directories are excluded by `.gitignore`. The committed
[`scripts/embeddings_manifest.csv`](../scripts/embeddings_manifest.csv) holds the
size and sha256 of every bundle file; `scripts/verify_embeddings.py` checks the
download against it. Rebuild
specs (checkpoint, instruction prefixes, pooling, 512-token cap, L2 norm,
dtype) are in [the reproduction guide](reproduction.md#retriever-embeddings).

## Deliberately absent

Raw OpenAI batch request/output JSONLs (the parsed `judgments.jsonl` is kept),
the upstream `HealthCareMagic-100k.json` (see [Sources and terms](#sources-and-terms)),
the retrieved document text (joined from the corpus on demand rather than
denormalized into the candidate files), and local job logs and temporary
outputs. The release also omits the scripts that built the balanced AllSides
corpus and generated its frames, queries, and BM25 outputs (candidates,
`allsides_synth/results/bm25/bias_metrics.csv`, and the BM25 gold-filter
cache); the `gpt-5-mini` AAL→WME
back-translation script (its output `hcmagic_aal_translated.jsonl` ships); and
the per-layer probing embeddings and trained probe weights.

Also deliberately absent, for the privacy reason in the paper's Ethical
considerations: **all Reddit post text** (post titles and bodies, the
per-post classifier rationales that quoted them, and the ideology-
classification code that consumed them). The bundle's
`reddit_nat/embeddings/*/queries.jsonl` likewise carry IDs, labels, a
`text_sha256` (the same hex SHA-256 digest as `title_sha256` in
`reddit_nat/data/reddit_nat_raw.csv`) and a length — not the query strings.
The per-post `query_embeddings.npy` files are included to support retrieval reproduction.
These encode the withheld titles; text withholding does not remove all
derivatives of the posts.

Also absent: the `twitteraae` and `PhonATe` clones and the `multivalue` package
(`pip install value-nlp`), needed only to regenerate
`hcmagic/data/healthq_aal/*.csv`; see
[dialect query data](reproduction.md#dialect-query-data).
