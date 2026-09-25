# Representation probing

Layer-wise logistic-regression probes measure how readily political lean and
dialect can be decoded from retriever representations (§4 of the paper).
Conditional probes measure the predictive gain from a layer beyond the
non-contextual input embeddings.

## Render the paper figure

With the base environment installed (`pip install -e .`), run from the
repository root:

```bash
python probing/scripts/plot_paper_probes.py
```

This reads `results/consolidated_scores.csv` and writes
`figs/output/probe_combined_subset.pdf` and `.png` (relative to the repository
root): HCMagic-synth on the left, AllSides-synth on the right; normal-probe F1
above and conditional V-information below. No model downloads, API calls, or
probe training are needed.

The CSV contains 2,068 score rows for the four open-weight retrievers on the
two synthetic datasets. Its `source` column distinguishes normal, conditional,
and dedicated layer-0 baseline runs. Each conditional curve subtracts the mean
baseline NCE for that model and dataset. Baseline rows have blank `layer`
fields because the baseline is fixed across depths; they are not
interchangeable with the normal run's layer-0 scores.

## Regenerate embeddings and probes

In addition to the base install, install:

```bash
pip install torch transformers datasets h5py tqdm
```

| File | Purpose |
| --- | --- |
| `configs/models.yaml` | The four probed checkpoints and their plot order |
| `configs/embeddings/*.yaml` | Input tables, text columns, and embedding outputs |
| `configs/probes/*.yaml` | Labels, paired variants, split and training settings |
| `scripts/generate_embeddings.py` | Generate per-layer mean-pooled embeddings |
| `scripts/train_probes.py` | Fit and evaluate normal and conditional probes |
| `scripts/plot_regenerated_probes.py` | Plot regenerated scores from `results/probe_scores/` |

For example, regenerate the AllSides-synth BGE probes with:

```bash
python probing/scripts/generate_embeddings.py \
  --config probing/configs/embeddings/allsides_synth.yaml \
  --model BAAI/bge-large-en-v1.5 --device 0
python probing/scripts/train_probes.py \
  --config probing/configs/probes/allsides_synth.yaml --only BGE-large
```

`bash probing/scripts/run.sh` embeds both configured datasets with the four
probed retrievers in `configs/models.yaml`, trains their probes, and plots the
scores, all in the active Python environment. Set `PYTHON` to use a different
interpreter. The driver embeds on GPU 0; for another device, run
`generate_embeddings.py` directly with `--device N` (`-1` for CPU). Generated
embedding files use `data/embeddings/{model_slug}/probes.{dataset}.h5`.
Embeddings and trained probe weights are not committed.

Rerunning extraction skips a complete embedding file only if its recorded
input hash, model, settings, and shapes match the request; otherwise it fails
and leaves the file in place. An interrupted extraction requires
`--restart-incomplete`, which the driver also accepts. To resume the driver
after embeddings are ready, use `bash probing/scripts/run.sh --start-stage 2`
(stage 3 is plotting).

For a sparse extraction, pass e.g. `--layers 0 4 8` to `generate_embeddings.py`.
Conditional probes require layer 0, and `plot_regenerated_probes.py` rejects
sparse layer series.

The regeneration pipeline produces new scores. Exact training splits and
activations for the released consolidated scores are not included, so it is
not a byte-for-byte recreation of those runs. Its conditional plots use the
normal layer-0 probe as baseline; the consolidated scores include dedicated
baseline runs. `plot_regenerated_probes.py` writes
`figs/output/probe_combined_subset_regenerated.pdf` and `.png`, next to but
separate from the paper figure; use `plot_paper_probes.py` to reproduce the
paper figure.

## Evaluation conventions

Probe inputs are the raw text, except that Qwen3 checkpoints get their
web-search query instruction; unlike retrieval, the BGE and Nemotron query
prefixes are not applied. Tokenizers pad on the left, in batches of 8, with no
explicit position IDs, so for BGE-large, whose position embeddings are
absolute, pooled embeddings depend on which rows share a batch. The
HCMagic-synth probe embeds all 5,000 rows of `hcmagic_wme_synth.csv`,
including the one row whose three AAL paraphrases are empty, which the
4,999-query retrieval set drops.

AllSides uses shuffled 5-fold cross-validation over query rows. HCMagic-synth
uses an 80/20 split over concatenated AAL and WME rows, separately for each of
three AAL variants. For each variant, the same seed gives matching partitions
across layers and normal/conditional probes. Scaling is fitted on training
rows only, with a logistic-regression iteration limit of 5,000.

The split unit is a query row. Variants of the same source question or article
can occur in both training and test partitions; the evaluation does not
measure generalization to unseen sources.

Extraction includes hidden states 0 through N−1 and intentionally omits the
final state N. The paper-figure script normalizes depth by `max(layer) + 1`;
`plot_regenerated_probes.py` divides layer indices by the number of layers in
each series.
