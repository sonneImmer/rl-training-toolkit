#!/usr/bin/env bash
set -Eeuo pipefail

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MODEL_HOST:=/mnt/local/transformer_files/zkx/runs/amlsim-sft-20261021/amlsim-sft.ERWFretm/checkpoints/global_step_30/huggingface}"
: "${DATA_HOST:=/mnt/local/transformer_files/zkx/data/amlsim_typology_rules_v1_20261021/rl}"
: "${RUNS_HOST:=/mnt/local/transformer_files/zkx/runs}"
: "${TOTAL_STEPS:=50}"
: "${SAVE_FREQ:=$TOTAL_STEPS}"
: "${RANDOM_SEED:=42}"
: "${NPU_COUNT:=8}"
: "${ALGORITHMS:=grpo ngrpo ppo}"

for algorithm in $ALGORITHMS; do
  container="amlsim-compare-${algorithm}-$(date +%Y%m%d-%H%M%S)"
  echo ">>> Starting ${algorithm}: ${container}"
  container_id=$(
    MODEL_HOST="$MODEL_HOST" DATA_HOST="$DATA_HOST" RUNS_HOST="$RUNS_HOST" \
    TOTAL_STEPS="$TOTAL_STEPS" SAVE_FREQ="$SAVE_FREQ" RANDOM_SEED="$RANDOM_SEED" \
    NPU_COUNT="$NPU_COUNT" ALGORITHM_PROFILE="$algorithm" CONTAINER_NAME="$container" \
    "$TOOLS_DIR/launch_amlsim_rl.sh"
  )
  status=$(docker wait "$container_id")
  if [[ "$status" != "0" ]]; then
    echo ">>> ${algorithm} failed with exit code ${status}" >&2
    docker logs --tail 200 "$container_id" >&2
    exit "$status"
  fi
  echo ">>> ${algorithm} completed"
done
