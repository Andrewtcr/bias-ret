#!/usr/bin/env bash
# Probing pipeline driver: embed each dataset with every roster model, train
# the probes, render figures.
#
# Usage: bash probing/scripts/run.sh [--start-stage 1|2|3] [--restart-incomplete]

set -euo pipefail
PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

START_STAGE=1
EMBED_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      if [[ $# -lt 2 || ! "$2" =~ ^[123]$ ]]; then
        echo "error: --start-stage requires 1 (embed), 2 (train), or 3 (figures)" >&2
        exit 2
      fi
      START_STAGE="$2"
      shift 2
      ;;
    --restart-incomplete)
      EMBED_ARGS+=("--restart-incomplete")
      shift
      ;;
    -h|--help)
      echo "usage: $0 [--start-stage 1|2|3] [--restart-incomplete]"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

LOGDIR="probing/results/logs"
mkdir -p "$LOGDIR"

DATASETS=(allsides_synth hcmagic_synth)

if [[ "$START_STAGE" -le 1 ]]; then
  # The embedding, training, and figure stages share this checkpoint roster.
  MODEL_NAMES=$("$PYTHON" - probing/configs/models.yaml <<'PYAML'
import sys
import yaml

with open(sys.argv[1]) as stream:
    config = yaml.safe_load(stream)
models = config["models"]
if not isinstance(models, dict) or not models:
    raise ValueError("models must be a nonempty mapping")
names = list(models.values())
if any(not isinstance(name, str) or not name or any(c.isspace() for c in name)
       for name in names):
    raise ValueError("model values must be nonempty checkpoint identifiers")
# Shared checkpoints need one embedding run even if multiple labels refer to them.
print("\n".join(dict.fromkeys(names)))
PYAML
  )
  MODELS=()
  while IFS= read -r model; do
    MODELS+=("$model")
  done <<< "$MODEL_NAMES"

  echo "==== Phase 1: embed each dataset x model ===="
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      slug="${model//\//__}"
      log="$LOGDIR/01_${ds}_${slug}.log"
      echo "[run] generate_embeddings: $ds / $model -> $log"
      "$PYTHON" probing/scripts/generate_embeddings.py \
          --config "probing/configs/embeddings/${ds}.yaml" --model "$model" \
          ${EMBED_ARGS[@]+"${EMBED_ARGS[@]}"} >"$log" 2>&1
    done
  done
fi

if [[ "$START_STAGE" -le 2 ]]; then
  echo "==== Phase 2: train + evaluate probes ===="
  PROBE_CONFIGS=(allsides_synth hcmagic_synth)
  for cfg in "${PROBE_CONFIGS[@]}"; do
    log="$LOGDIR/02_${cfg}.log"
    echo "[run] train_probes: $cfg -> $log"
    "$PYTHON" probing/scripts/train_probes.py \
        --config "probing/configs/probes/${cfg}.yaml" \
        2>&1 | tee "$log"
  done
fi

echo "==== Phase 3: figures ===="
"$PYTHON" probing/scripts/plot_regenerated_probes.py \
    --config probing/configs/figures.yaml 2>&1 | tee "$LOGDIR/03_plot_regenerated_probes.log"

echo "[run] done."
