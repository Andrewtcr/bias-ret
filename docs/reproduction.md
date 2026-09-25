# Reproducing embeddings and query data

Run shell commands from the repository root unless a command block explicitly
changes directories. File paths in this guide are relative to the repository
root. The Python examples below share the normalization helper defined first.

- [Retriever embeddings](#retriever-embeddings): model configuration and encoding examples.
- [Dialect query data](#dialect-query-data): external tools for HealthCareMagic labeling and synthesis.

## Retriever embeddings

This paper evaluates five dense retrievers (four open-weight Hugging Face
embedding models + OpenAI `text-embedding-3-large`) plus a BM25 sparse
baseline. The snippets below describe the embedding configuration used in the experiments: model-specific query/document instruction
prefixes, pooling, the 512-token cap, and L2 normalization. They are
extracted from `src/embedding.py` (`encode_texts_multi` / `_prepare_hf_texts`),
which is the single source of truth.

### Conventions used everywhere

- **Retrieval score** is cosine similarity. Every embedding is
  L2-normalized, so cosine = dot product.
- **Max sequence length** is capped at **512 tokens** for all HF models
  (`model.max_seq_length = 512`).
- **dtype**: the three 8B models run in `bfloat16`; BGE-large runs in the
  default float32.
- A query is `kind="query"`; a corpus document is `kind="document"`. The
  prefix/prompt differs by kind for some models (below).

```bash
pip install "sentence-transformers>=3.0" torch
pip install openai            # for text-embedding-3-large
pip install "transformers==4.57.*"   # tested version; nvidia/llama-embed-nemotron-8b needs trust_remote_code
```

A shared helper:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)
```

---

### 1. BGE-large-en-v1.5 (`BAAI/bge-large-en-v1.5`)

Query gets the canonical BGE retrieval instruction; documents get no
prefix. No `trust_remote_code`, default float32, batch size 32
(64 in `hcmagic/configs/retrieval.yaml`).

```python
model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
model.max_seq_length = 512

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def embed_bge(texts, kind):
    if kind == "query":
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    embs = model.encode(texts, batch_size=32, convert_to_numpy=True)
    return l2_normalize(embs)
```

### 2. Qwen3-Embedding-8B (`Qwen/Qwen3-Embedding-8B`)

Uses sentence-transformers' built-in Qwen3 prompt templates via
`prompt_name` (`"query"` for queries, `"document"` for documents). Run in
`bfloat16`, batch size 8.

```python
import torch
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-8B",
    device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
)
model.max_seq_length = 512

def embed_qwen3(texts, kind):
    prompt_name = "query" if kind == "query" else "document"
    embs = model.encode(
        texts, batch_size=8, prompt_name=prompt_name, convert_to_numpy=True,
    )
    return l2_normalize(embs)
```

### 3. Llama-Nemotron-8B (`nvidia/llama-embed-nemotron-8b`)

Query gets an `Instruct:`/`Query:` retrieval prompt; documents get no
prefix. Requires `trust_remote_code=True`; `bfloat16`, batch size 8.

```python
import torch
model = SentenceTransformer(
    "nvidia/llama-embed-nemotron-8b",
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16},
)
model.max_seq_length = 512

NEMOTRON_QUERY_PREFIX = (
    "Instruct: Given a search query, retrieve relevant passages.\nQuery: "
)

def embed_nemotron(texts, kind):
    if kind == "query":
        texts = [NEMOTRON_QUERY_PREFIX + t for t in texts]
    embs = model.encode(texts, batch_size=8, convert_to_numpy=True)
    return l2_normalize(embs)
```

### 4. Octen-Embedding-8B (`Octen/Octen-Embedding-8B`)

No query prefix (per the model card). Documents get a `"- "` prefix (the
model card's recommended workaround for an upstream Qwen3 quirk).
`bfloat16`, batch size 8.

```python
import torch
model = SentenceTransformer(
    "Octen/Octen-Embedding-8B",
    device="cuda",
    model_kwargs={"torch_dtype": torch.bfloat16},
)
model.max_seq_length = 512

def embed_octen(texts, kind):
    if kind == "document":
        texts = ["- " + t for t in texts]
    # queries: no prefix
    embs = model.encode(texts, batch_size=8, convert_to_numpy=True)
    return l2_normalize(embs)
```

### 5. OpenAI `text-embedding-3-large`

API embeddings, default 3072 dimensions (no Matryoshka truncation used in
the paper), L2-normalized after. API batch of 100. Requires
`OPENAI_API_KEY` in the environment.

```python
from openai import OpenAI

client = OpenAI()

def embed_openai(texts, kind=None, batch_size=100):
    # OpenAI embeddings take no query/document distinction.
    rows = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        resp = client.embeddings.create(model="text-embedding-3-large", input=chunk)
        rows.extend(d.embedding for d in resp.data)  # returned in input order
    return l2_normalize(np.asarray(rows, dtype=np.float32))
```

---

### Putting it together (retrieval)

For each retriever: embed the corpus once with `kind="document"`, embed
queries with `kind="query"`, and rank documents by cosine similarity
(dot product on the normalized embeddings):

```python
doc_embs = embed_bge(corpus_texts, kind="document")    # (N_docs, d)
q_embs   = embed_bge(query_texts,  kind="query")        # (N_queries, d)
scores   = q_embs @ doc_embs.T                          # cosine sims
ranking  = (-scores).argsort(axis=1)                    # full sorted ranking per query
```

For HealthCareMagic, MRR is computed on the **full** sorted ranking (no
top-`k` cap); see `hcmagic/metrics.py::reciprocal_rank`. AllSides retrieval
excludes each query's source article from its ranking
(`src/retrieval.py::dense_retrieve_with_mask`, called by
`allsides_synth/scripts/retrieve.py`).

### BM25 baseline (not an embedding model)

BM25 via [`bm25s`](https://github.com/xhluca/bm25s), using its default Lucene variant,
is used for the HealthCareMagic baseline. The AllSides and Reddit baselines
use `rank_bm25.BM25Okapi`, as in `reddit_nat/scripts/retrieve_bm25.py`. BM25
ranks documents by score over tokenized text and does not produce embeddings.

### Execution and configuration

- The encoding pipeline shards the corpus across GPUs (one model load per
  device) for speed; the snippets above are the single-GPU equivalent.
  See `src/embedding.py::encode_texts_multi` for the data-parallel path.
- Per-experiment encoder lists, devices, and batch sizes live in the YAML
  configs (e.g. `allsides_synth/configs/retrieval.yaml`).

## Dialect query data

HealthCareMagic dialect data preparation depends on third-party research
tools that are not vendored into this repo. None of them is needed to rebuild
any committed figure or table; they are only needed to *regenerate*
`hcmagic/data/healthq_aal/*.csv` from the raw upstream corpus (see
[Sources and terms](data.md#sources-and-terms)).

### 1. `twitteraae` — dialect-proxy classifier

Used by `src/dialect.py` (imported by `hcmagic/scripts/label_dialect.py`).
Clone its repository:

```bash
git clone https://github.com/slanglab/twitteraae ../twitteraae
```

No install step beyond the clone — `src/dialect.py` imports
`twitteraae/code/predict.py` directly off `sys.path` and points its model
file paths at the clone.

### 2. `multivalue` (Multi-VALUE) — dialect transformation

Used by `hcmagic/scripts/synthesize_aal.py` for
`Dialects.AfricanAmericanVernacular().transform(...)`. Published on PyPI as
`value-nlp`:

```bash
pip install value-nlp
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet cmudict
python -c "import stanza; stanza.download('en')"   # coref pipeline models
```

`synthesize_aal.py` aligns Stanza coreference mentions to spaCy tokens by
character offset and skips mentions that do not align. This handles tokenizer
disagreements without editing the installed Multi-VALUE package.

### 3. `PhonATe` — phonological AAL augmentation

Used by `hcmagic/scripts/synthesize_aal.py` for
`AALPhonate(...).full_phon_aug(...)`, applied after the Multi-VALUE
morphosyntactic transform to produce each AAL paraphrase variant. Not on
PyPI — clone and set up per its own README:

```bash
git clone https://github.com/NickDeas/PhonATe ../PhonATe
cd ../PhonATe
pip install -r requirements.txt
tar -xzf byt5-aal-p2g.tar.gz          # phoneme-to-grapheme model
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('words')"
cd -
```

The synthesis script loads `phonate/default_config.json` and resolves the
phoneme-to-grapheme model path relative to the PhonATe clone.

### Layout assumed by the defaults

```text
<parent dir>/
├── bias-ret/           # this repo
├── twitteraae/
└── PhonATe/
```

Every script needing `twitteraae` or `PhonATe` resolves it in this order: the
CLI flag (`--twitteraae-dir`, `--phonate-dir`), the path in the script's YAML
config, the env var (`TWITTERAAE_HOME`, `PHONATE_HOME`), then the sibling path
shown above.
