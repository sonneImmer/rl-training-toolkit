#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:=/mnt/model}"
: "${DATA_DIR:=/mnt/data}"
: "${RUN_ROOT:=/mnt/runs/amlsim-sft-$(date +%Y%m%d)}"
: "${TOTAL_STEPS:=50}"
: "${ASCEND_RT_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${RANDOM_SEED:=42}"

TOOLS_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_ROOT"
RUN_DIR=$(mktemp -d "$RUN_ROOT/amlsim-sft.XXXXXXXX")
export HCCL_CONNECT_TIMEOUT=600
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-vlan699}"
export HCCL_WHITELIST_DISABLE=1
export ASCEND_RT_VISIBLE_DEVICES
IFS=',' read -ra DEVICE_LIST <<< "$ASCEND_RT_VISIBLE_DEVICES"
DEVICE_COUNT=${#DEVICE_LIST[@]}
export MODEL_PATH DATA_DIR TOTAL_STEPS RANDOM_SEED DEVICE_COUNT

python3 "$TOOLS_DIR/preflight_amlsim_sft.py" \
  --data "$DATA_DIR" --model "$MODEL_PATH" --output "$RUN_DIR/preflight" --max-length 2048

python3 - "$RUN_DIR/run_manifest.json" <<'PY_MANIFEST'
import json, os, sys
from pathlib import Path
manifest = {
    "schema_version": "1.0.0",
    "task": {"domain": "amlsim", "type": "aml_risk_typology_evidence_sft"},
    "dataset": {"id": os.environ["DATA_DIR"], "typologies": ["FAN_IN", "FAN_OUT", "CYCLE", "NORMAL"]},
    "prompt": {"contains_cot": False, "target": "Risk/Typology/Evidence three-line format"},
    "model": {"id": os.environ["MODEL_PATH"]},
    "training": {"algorithm": "sft", "seed": int(os.environ["RANDOM_SEED"]), "total_steps": int(os.environ["TOTAL_STEPS"])},
    "resources": {"npu_count": int(os.environ["DEVICE_COUNT"]), "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")},
}
Path(sys.argv[1]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
PY_MANIFEST

torchrun --standalone --nnodes=1 --nproc_per_node="$DEVICE_COUNT" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_DIR/train.parquet" data.val_files="$DATA_DIR/dev.parquet" \
  data.prompt_key=prompt data.response_key=response \
  data.train_batch_size=16 data.micro_batch_size_per_gpu=1 data.max_length=2048 data.truncation=error \
  model.partial_pretrain="$MODEL_PATH" model.trust_remote_code=False \
  model.enable_gradient_checkpointing=True model.fsdp_config.cpu_offload=True \
  model.strategy=fsdp use_remove_padding=False ulysses_sequence_parallel_size=1 \
  optim.lr=1e-5 trainer.project_name=amlsim_sft trainer.experiment_name="$(basename "$RUN_DIR")" \
  trainer.logger=console trainer.total_epochs=2 trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.seed="$RANDOM_SEED" \
  trainer.nnodes=1 trainer.n_gpus_per_node="$DEVICE_COUNT" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.resume_mode=disable \
  trainer.save_freq="$TOTAL_STEPS" trainer.test_freq=-1 \
  trainer.checkpoint.save_contents='[model,optimizer,extra,hf_model]' \
  trainer.checkpoint.load_contents='[model,optimizer,extra]' \
  2>&1 | tee "$RUN_DIR/train.log"
