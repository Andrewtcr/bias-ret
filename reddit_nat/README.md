# Naturalistic Reddit queries

Political questions from Reddit are labeled by asker ideology and retrieved
against the AllSides article corpus. Analyses compare retrieval lean across
ideology groups, with relevance filtering and lexical controls.

## Released data

`data/reddit_nat_raw.csv` contains post IDs, subreddit names, ideology labels,
character counts (`title_n_chars`, `selftext_n_chars`), and title SHA-256 hashes.
The ideology labels are classifier inferences, not self-reports; use them only
for aggregate analysis, not to characterize individual users. The file has
3,472 rows for 3,340 posts: 129 posts appear more than once in the pull, 24 of
them with different ideology labels across rows. The pipeline keeps the first
row per ID, which gives the paper's 1,327 liberal, 1,225 conservative, and 788
centrist queries.
Post text and classifier responses are withheld so that deleted content is
not redistributed. Character counts support query-length statistics;
`title_sha256` (a hex SHA-256 digest, equal to `text_sha256` in the embeddings
bundle's `queries.jsonl`) lets researchers check rehydrated titles against the
encoded text. The collection date is 2026-05-18.

`results/` contains ID-keyed intermediates and the aggregate tables the paper
reports:

- `retrieval_candidates/{encoder}.csv`: retrieval candidates
- `judge_relevance/judgments*.jsonl`: parsed relevance judgments of both judges
- `lex_residual_per_query.csv`: per-query lexical scores and lean
- `judge_relevance/agreement_overall.csv`, `judge_relevance/kappa_per_encoder.csv`: judge agreement
- `judge_relevance/lean_gap_by_filter.csv`: lean gap under each relevance filter
- `gold_filtered_lean_summary.csv`: gold-filtered own-lean gap
- `lex_residual_summary.csv`, `lex_residual_gold_summary.csv`: lexical-control regressions

These artifacts support figure rendering without post text or model inference.
`embeddings/` is supplied separately; see [docs/data.md](../docs/data.md).

## Ideology labels

The paper describes ideology labeling with **gpt-5.4-mini** and
**Qwen3.5-4B**, using the ideology-classification prompt in the paper's
"Reddit Data Collection" appendix. The released labels are those used in the
reported experiments. Label extraction selected the first matching substring
in the response, in the order conservative, centrist, liberal, rather than
restricting extraction to a final verdict.
The text-consuming classification stage is not included.

## Reproduce figures and analysis

`python scripts/reproduce.py` (see [the top-level README](../README.md)) runs
every Reddit step. The individual commands, from the repository root after
`pip install -e .`:

```bash
python reddit_nat/scripts/analyze_judge_agreement.py            # judge agreement tables
python reddit_nat/scripts/analyze_lexical.py --from-scores      # unfiltered lexical control (k=20)
python reddit_nat/scripts/analyze_lexical.py --gold-filtered    # gold-filtered lexical control (k=10)
python figs/scripts/plot_political_overview.py                  # main political figure + gold-filtered gap table
python figs/scripts/plot_relevance_robustness.py                # relevance-filter appendix figures
python figs/scripts/plot_reddit_k_sweep.py                      # retrieval-depth appendix figure
```

The `--from-scores` mode refits the lexical-control regression from the
committed `results/lex_residual_per_query.csv`. It writes
`lex_residual_summary.csv` and `lex_residual.md` without reading post text or
changing the per-query table. `--gold-filtered` fits the same regression with
the gold-filtered lean (articles both judges grade relevant) as the outcome
and writes `lex_residual_gold_summary.csv`. Fitting the lexical scores
themselves requires post text; run `analyze_lexical.py` without either flag
after rehydrating the queries. That mode also writes the token log-odds table
`lex_residual_token_logodds.csv`, which is not released.

| Scripts | Purpose |
| --- | --- |
| `embed_queries.py`, `retrieve.py` | Encode questions and retrieve articles with the dense retrievers |
| `retrieve_bm25.py` | BM25 retrieval baseline (reuses the committed candidates) |
| `judge_relevance.py`, `judge_relevance_qwen.py` | Generate relevance judgments with gpt-5-mini and Qwen3.5-35B-A3B |
| `analyze_judge_agreement.py` | Compare the two judges and their relevance filters |
| `analyze_lexical.py` | Fit the lexical controls on unfiltered and gold-filtered lean |

## Regenerate retrieval and annotations

To rerun dense retrieval, fetch the embeddings bundle and run:

```bash
python reddit_nat/scripts/retrieve.py --config reddit_nat/configs/retrieval.yaml
```

The distributed bundle pairs each encoder's embeddings with `query_ids.json`
and an ID-only `queries.jsonl`. Retrieval uses these files to preserve the
encoded query set without post text. Re-encoding locally with
`embed_queries.py` writes the rehydrated text to
`embeddings/{encoder}/queries.jsonl` in place of the ID-only file; that path is
gitignored, and the file should not be redistributed.

Re-encoding, generating BM25 candidates, fitting lexical scores, and running
the judges require rehydrating posts by ID. Add a `title` column to
`data/reddit_nat_raw.csv` for the retrieval and lexical stages; the judges also
need a `selftext` column. Verify titles against `title_sha256`. This file is
tracked, so keep the rehydrated copy local and never commit or redistribute it
(for example, run `git update-index --skip-worktree reddit_nat/data/reddit_nat_raw.csv`
before adding the columns). Deleted posts
may no longer be available. `retrieve_bm25.py` reuses the committed BM25
candidates without text; to rebuild them from rehydrated titles, delete
`results/retrieval_candidates/bm25.csv` first. OpenAI stages require an API
key; the local judge stage requires vLLM and suitable GPU resources.

`bash reddit_nat/scripts/run.sh` encodes and retrieves using the active Python
environment (`PYTHON` can override the interpreter). Configure device lists in
`configs/retrieval.yaml` for your hardware before running it.
