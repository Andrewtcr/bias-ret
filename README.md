# Retrieval Sensitivity to Identity Signals in Queries

Code and intermediate data accompanying the paper by Andrew Tang, Nicholas
Deas, Kathleen McKeown, and Vishal Misra
([arXiv:2609.36534](https://arxiv.org/abs/2609.36534); see [Citation](#citation)). Every
analysis figure and table can be regenerated from committed CSV, JSONL, and
Parquet files without a large data download.

## Layout

```text
src/                      shared embedding, retrieval, judging, I/O, and statistics helpers
allsides_synth/           AllSides-synth political-frame queries (+ lexical-residual regression, §4)
hcmagic/                  HCMagic-synth and HCMagic-nat AAL/WME health questions (+ lexical residuals, §4)
reddit_nat/               Reddit-nat political queries (+ lexical-residual regression, §4)
probing/                  per-layer probes and their committed scores (§4)
figs/scripts/             paper figure and table producers (+ _style.py)
figs/results/             numerical tables written by figs/scripts
figs/output/              rendered figures land here (gitignored)
scripts/                  paper reproduction and embeddings-bundle utilities
tests/                    CPU regression tests (unittest)
docs/                     data access and advanced reproduction instructions
```

Each experiment directory holds its configs and pipeline scripts; `data/`
holds inputs and `results/` holds numerical outputs. In file and figure names,
`real` marks a natural query set (HCMagic-nat, Reddit-nat) and `synth` a
synthetic one. AllSides and Reddit
retrieval rankings are stored by encoder under
`results/retrieval_candidates/`; these files contain IDs and ranks, with
article text joined from the corpus when needed. HCMagic stores per-query
retrieval metrics under `results/{encoder}/`.

## Reproducing the paper

You can render committed results, rerun retrieval from downloaded embeddings,
or regenerate embeddings from the released text:

```text
corpus + queries ─[encode]─▶ embeddings ─[retrieve + judge]─▶ results ─[render]─▶ figures + tables
(see docs/data.md)           (embeddings bundle)              (committed)
```

### Environment

```bash
conda create -n bias-ret python=3.11 -y && conda activate bias-ret
pip install -e .          # deps for the figure, table, and retrieval code
```

The tested stack is Python 3.11 with numpy 2.4, pandas 3.0, scipy 1.17,
statsmodels 0.15, matplotlib 3.11, scikit-learn 1.9, and bm25s 0.3.

Figure rendering uses Matplotlib's built-in mathtext and bundled fonts; no
system LaTeX installation is required. Typography can differ from the paper's
TeX-rendered figures, while the underlying data and statistics are unchanged.

Re-encoding the corpus additionally needs `torch` and the encoder libraries;
see the [reproduction guide](docs/reproduction.md#retriever-embeddings).

### Render the figures and tables (no download)

One command rebuilds the paper's analysis tables and figures:

```bash
python scripts/reproduce.py
```

The runner uses your active Python environment, executes tables before figures
that consume them, and checks that every expected output was refreshed. It
regenerates numerical tables in `results/` and plots in `figs/output/`.
Each step's log is saved under `figs/output/logs/`; a failed step or missing
output stops the run. No embeddings download, GPU, or API key is needed.
The runner does not compare values. The rebuilt tables overwrite the committed
ones, so run `git status --short` afterwards to see which committed files
changed; on the tested stack, none do.

To render figures from the existing results, rebuild only tables, or inspect
all commands and output filenames:

```bash
python scripts/reproduce.py --only figures
python scripts/reproduce.py --only tables
python scripts/reproduce.py --list
```

Some figure steps also write the result CSVs they compute, so `--only tables`
does not rebuild those; `--list` shows every step's outputs. Each plotted
paper figure is written to `figs/output/` under its filename in the paper; the
hand-composed teaser (Fig. 1) and the prompt figures are not produced here.

### Re-run retrieval from the embeddings

Download the [embeddings bundle](docs/data.md#the-embeddings-bundle) (access on
request), then re-run each experiment's retrieval. This
regenerates the committed `results/` (retrieval candidates, per-query metrics,
`bias_metrics.csv`) that the figures read. The committed LLM-judge labels
(`judgments*.jsonl`) cover the committed rankings, and the judge-filtered
analyses count unjudged (query, article) pairs as not relevant. If rankings
change, judge the new pairs before recomputing filtered results: the Qwen judge
skips pairs already in `judgments_qwen.jsonl`, and the gpt-5-mini judge
re-judges every pair. Running the API judge requires an OpenAI key (see the
experiment READMEs).

```bash
bash scripts/fetch_embeddings.sh     # ~10 GB embeddings from Google Drive (rclone; see docs/data.md)
python scripts/verify_embeddings.py  # check sizes and sha256 against scripts/embeddings_manifest.csv

# AllSides — dense retrieve + bias metrics (writes retrieval_candidates/{enc}.csv)
python allsides_synth/scripts/retrieve.py --config allsides_synth/configs/retrieval.yaml

# HCMagic — retrieve from the downloaded embeddings, then aggregate metrics
python hcmagic/scripts/retrieve.py --config hcmagic/configs/retrieval.yaml
python hcmagic/scripts/analyze_retrieval.py --config hcmagic/configs/retrieval.yaml

# Reddit — retrieve against the AllSides corpus (writes retrieval_candidates/{enc}.csv)
python reddit_nat/scripts/retrieve.py --config reddit_nat/configs/retrieval.yaml
```

Retrieval is not bit-exact across hardware and BLAS/NumPy versions: near-tied
similarities can reorder results. The committed scores are the reference
artifacts for reproducing the paper figures. Re-run HCMagic metrics also differ
slightly because `qrels.json` credits byte-identical duplicate answers (see
[hcmagic/README.md](hcmagic/README.md#reproduce)).

Then run `python scripts/reproduce.py` to recompute the tables and figures (not
`--only figures`). Each `<exp>/README.md` covers that experiment's other
stages (LLM judging, lexical residuals, …).

**Reddit post text is deliberately withheld** from this release, per the
paper's Ethical considerations, so the Reddit stages that read it cannot be
re-run from the release alone; every Reddit figure and table rebuilds from the
ID-keyed results (see [reddit_nat/README.md](reddit_nat/README.md#released-data)).
The AllSides BM25 outputs ship without a regeneration script (see
[docs/data.md](docs/data.md#deliberately-absent)).

### Re-encode the corpus

Re-encoding needs a GPU for the four open-weight encoders and an OpenAI key for
`text-embedding-3-large`. Follow the
[reproduction guide](docs/reproduction.md#retriever-embeddings) (model prefixes,
pooling, 512-token cap, L2 normalization); the encode phases of the `run.sh`
drivers implement it. Then re-run retrieval as above. The bundled Reddit query
embeddings cannot be regenerated from the release alone: re-encoding needs the
post text rehydrated by ID (see [reddit_nat/README.md](reddit_nat/README.md)).

### §4 bias-gap analysis: probing and lexical residuals

The §4 probing analysis (per-layer linear probes for lean and dialect, scored
by F1 and V-information) lives in `probing/`: per-layer embedding generation,
probe training and evaluation, and the figure script, with configs per dataset.
`probing/results/consolidated_scores.csv` holds the normal, conditional, and
layer-0 baseline scores for the four open-weight retrievers, and
`plot_paper_probes.py` renders the paper figure from them with the base
dependencies and no model downloads. Regenerating embeddings and probes needs
the extra dependencies listed in `probing/README.md`. The lexical-residual half
of §4 is produced by the `analyze_lexical*.py` scripts under
`allsides_synth/`, `hcmagic/`, and `reddit_nat/` and needs no extra setup.

## Tests

For the CPU regression checks, run `pip install -e '.[test]'` and then
`python -m unittest discover -s tests`. These checks use small fixtures and the
committed Reddit lexical-control results; run them before
`scripts/reproduce.py`, which rewrites those results. They do not download
models or make API calls.

## Citation

```bibtex
@misc{tang2026retrieval,
  title         = {Retrieval Sensitivity to Identity Signals in Queries},
  author        = {Tang, Andrew and Deas, Nicholas and McKeown, Kathleen and Misra, Vishal},
  year          = {2026},
  eprint        = {2609.36534},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.36534}
}
```

## License

The code is released under the MIT License (see [LICENSE](LICENSE)).
Redistributed third-party data stays under its upstream terms; see
[Sources and terms](docs/data.md#sources-and-terms).
