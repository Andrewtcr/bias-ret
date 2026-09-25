#!/usr/bin/env bash
# HCMagic full-pipeline driver (encode -> retrieve -> analyze).
# Each encoder is pinned to a disjoint device set in
# hcmagic/configs/retrieval.yaml; encoders run concurrently to use
# the configured GPUs and OpenAI API concurrently.
#
# Usage: bash hcmagic/scripts/run.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CFG="hcmagic/configs/retrieval.yaml"
LOGDIR="hcmagic/results/logs"
mkdir -p "$LOGDIR"

# Read full checkpoint identifiers from the same config passed to each stage.
ENCODERS=$("$PYTHON" - "$CFG" <<'PYAML'
import sys
import yaml

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream)
encoders = config["retrieval"]["encoders"]
if not isinstance(encoders, list) or not encoders:
    raise ValueError("retrieval.encoders must be a nonempty list")
names = [encoder["name"] if isinstance(encoder, dict) else encoder
         for encoder in encoders]
if any(not isinstance(name, str) or not name or any(c.isspace() for c in name)
       for name in names):
    raise ValueError("encoder names must be nonempty checkpoint identifiers")
if len(names) != len(set(names)):
    raise ValueError("retrieval.encoders contains duplicate checkpoint identifiers")
print("\n".join(names))
PYAML
)
ENCS=()
while IFS= read -r enc; do
  ENCS+=("$enc")
done <<< "$ENCODERS"

echo "==== Phase 1: parallel corpus embedding ===="
pids=()
for enc in "${ENCS[@]}"; do
  slug="${enc//\//__}"
  log="$LOGDIR/01_corpus_${slug}.log"
  echo "[run] start corpus embed: $enc -> $log"
  "$PYTHON" hcmagic/scripts/embed_corpus.py \
      --config "$CFG" --only "$enc" >"$log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
if [[ $fail -ne 0 ]]; then
  echo "[run] Phase 1 had failures; see $LOGDIR"
  exit 1
fi

echo "==== Phase 2: parallel query embedding ===="
pids=()
for enc in "${ENCS[@]}"; do
  slug="${enc//\//__}"
  log="$LOGDIR/02_queries_${slug}.log"
  echo "[run] start query embed: $enc -> $log"
  "$PYTHON" hcmagic/scripts/embed_queries.py \
      --config "$CFG" --only "$enc" >"$log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done
if [[ $fail -ne 0 ]]; then
  echo "[run] Phase 2 had failures; see $LOGDIR"
  exit 1
fi

echo "==== Phase 3: retrieve + per-query scores (sequential, fast) ===="
"$PYTHON" hcmagic/scripts/retrieve.py \
    --config "$CFG" 2>&1 | tee "$LOGDIR/03_retrieve.log"

echo "==== Phase 4: paired + unpaired tests ===="
"$PYTHON" hcmagic/scripts/analyze_retrieval.py \
    --config "$CFG" 2>&1 | tee "$LOGDIR/04_analyze.log"

echo "[run] done."
