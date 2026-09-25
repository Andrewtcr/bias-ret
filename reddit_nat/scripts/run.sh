#!/usr/bin/env bash
# Parallel encoder driver for the Reddit real-query arm.
#
# Each encoder is pinned to a disjoint device set in
# configs/retrieval.yaml; run the configured encoders concurrently.
#
# Usage: bash reddit_nat/scripts/run.sh

set -euo pipefail
PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")/../.."
CFG="reddit_nat/configs/retrieval.yaml"
LOGDIR="reddit_nat/results/logs"
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

pids=()
for enc in "${ENCS[@]}"; do
  slug="${enc//\//__}"
  log="$LOGDIR/embed_${slug}.log"
  echo "[run] starting embed for $enc -> $log"
  "$PYTHON" -m reddit_nat.scripts.embed_queries \
      --config "$CFG" --only "$enc" >"$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[run] one or more encoders failed; logs in $LOGDIR"
  exit 1
fi

echo "[run] all encoders done; running retrieval"
"$PYTHON" -m reddit_nat.scripts.retrieve --config "$CFG" \
    2>&1 | tee "$LOGDIR/retrieve.log"
echo "[run] done"
