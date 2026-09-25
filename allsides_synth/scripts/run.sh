#!/usr/bin/env bash
# Drive embed (one process per encoder, in parallel) + retrieve for one
# config. Each encoder is launched as a separate process so they can run on
# their assigned (disjoint) device sets concurrently. Per-encoder stdout goes
# to LOG_DIR/encode_<slug>.log; this script reports completion of each.
# Activate the Python environment first, or set PYTHON explicitly.
#
# Usage:  bash allsides_synth/scripts/run.sh CONFIG_PATH
#
# To run a single encoder serially, invoke embed.py directly with --only.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 CONFIG_PATH" >&2
    exit 2
fi

CONFIG="$1"
PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-allsides_synth/results/logs}"

cd "$(dirname "$0")/../.."   # repo root

echo "[run.sh] config=$CONFIG"

# Extract encoder slugs from the config so we can launch one --only invocation
# per encoder.
ENCODERS=$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as stream:
    cfg = yaml.safe_load(stream)
for encoder in cfg['retrieval']['encoders']:
    name = encoder['name'] if isinstance(encoder, dict) else encoder
    print(name.split('/')[-1])
PY
)

if [ -z "$ENCODERS" ]; then
    echo "[run.sh] no encoders found in $CONFIG" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "[run.sh] === embed (parallel) ==="
echo "[run.sh] encoders:"
echo "$ENCODERS" | sed 's/^/  - /'

PIDS=()
SLUGS=()
LOGS=()
for slug in $ENCODERS; do
    log="$LOG_DIR/encode_${slug}.log"
    echo "[run.sh] launching --only $slug  (log: $log)"
    "$PYTHON" allsides_synth/scripts/embed.py --config "$CONFIG" --only "$slug" \
        > "$log" 2>&1 &
    PIDS+=($!)
    SLUGS+=("$slug")
    LOGS+=("$log")
done

echo "[run.sh] launched ${#PIDS[@]} processes; waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    slug="${SLUGS[$i]}"
    log="${LOGS[$i]}"
    if wait "$pid"; then
        echo "[run.sh] OK  $slug (pid $pid)"
    else
        rc=$?
        echo "[run.sh] FAILED  $slug (pid $pid) exited $rc - tail of $log:" >&2
        tail -30 "$log" >&2
        FAIL=1
    fi
done
[ $FAIL -ne 0 ] && exit 1

echo "[run.sh] === retrieve ==="
"$PYTHON" allsides_synth/scripts/retrieve.py --config "$CONFIG"

echo "[run.sh] done."
