#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${IMAGE:-openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix}"
CONTAINER_NAME="${CONTAINER_NAME:-amlsim-rl-$(date +%Y%m%d-%H%M%S)}"
MODEL_HOST="${MODEL_HOST:-/mnt/local/pangu_rl/models/qwen3-8b}"
DATA_HOST="${DATA_HOST:-/mnt/local/transformer_files/zkx/data/amlsim_typology_rules_v1_20261021/rl}"
RUNS_HOST="${RUNS_HOST:-/mnt/local/transformer_files/zkx/runs}"
TOOLS_HOST="${TOOLS_HOST:-/mnt/local/rl-training-toolkit/tools/amlsim}"
PATCH_HOST="${PATCH_HOST:-/mnt/local/rl-training-toolkit/patch_tools}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
CODE_REVISION="${CODE_REVISION:-$(git -C /mnt/local/rl-training-toolkit rev-parse HEAD)}"
NPU_COUNT="${NPU_COUNT:-8}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
HCCL_INTERFACE="${HCCL_INTERFACE:-vlan699}"
case "$NPU_COUNT" in
  2) VISIBLE_DEVICES="0,1" ;;
  4) VISIBLE_DEVICES="0,1,2,3" ;;
  8) VISIBLE_DEVICES="0,1,2,3,4,5,6,7" ;;
  *) echo "NPU_COUNT must be 2, 4, or 8" >&2; exit 1 ;;
esac

exec docker run -d \
  --name "$CONTAINER_NAME" --network=host --ipc=host --hostname "$(hostname)" --add-host "$(hostname):$HOST_IP" \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_HOST:/mnt/model:ro" -v "$DATA_HOST:/mnt/data:ro" \
  -v "$RUNS_HOST:/mnt/runs" -v "$TOOLS_HOST:/mnt/tools:ro" -v "$PATCH_HOST:/mnt/patch_tools:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" -e TOTAL_STEPS="${TOTAL_STEPS:-5}" -e SAVE_FREQ="${SAVE_FREQ:-${TOTAL_STEPS:-5}}" \
  -e HCCL_WHITELIST_DISABLE=1 -e HCCL_SOCKET_IFNAME="$HCCL_INTERFACE" -e HCCL_CONNECT_TIMEOUT=600 \
  -e MODEL_PATH=/mnt/model -e DATA_DIR=/mnt/data -e RUN_ROOT=/mnt/runs/amlsim-rl-$(date +%Y%m%d) \
  -e IMAGE_NAME="$IMAGE" -e CODE_REVISION="$CODE_REVISION" -e RANDOM_SEED="${RANDOM_SEED:-42}" -e ADV_ESTIMATOR="$ADV_ESTIMATOR" \
  -e REWARD_FILE="${REWARD_FILE:-amlsim_reward.py}" -e REWARD_RULE_ID="${REWARD_RULE_ID:-amlsim_risk_typology_evidence_v1}" \
  -e REWARD_MIN="${REWARD_MIN:--4.0}" -e REWARD_MAX="${REWARD_MAX:-4.0}" -e NGRPO_MAX_REWARD="${NGRPO_MAX_REWARD:-${REWARD_MAX:-4.0}}" \
  "$IMAGE" bash -lc 'if [[ "$ADV_ESTIMATOR" == "ngrpo" ]]; then python3 /mnt/patch_tools/patch_verl_ngrpo.py apply --verl-root /workspace/verl --backup-dir /tmp/verl_ngrpo_backup && python3 /mnt/patch_tools/patch_verl_ngrpo.py check --verl-root /workspace/verl; fi; exec bash /mnt/tools/run_amlsim_rl.sh'
