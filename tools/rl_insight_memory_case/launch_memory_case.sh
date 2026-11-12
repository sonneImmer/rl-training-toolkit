#!/usr/bin/env bash
set -Eeuo pipefail

IMAGE="${IMAGE:-openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix}"
GRAD_SYNC_MODE="${GRAD_SYNC_MODE:-per_micro_batch}"
CONTAINER_NAME="${CONTAINER_NAME:-rl-insight-memory-${GRAD_SYNC_MODE}-$(date +%Y%m%d-%H%M%S)}"
MODEL_HOST="${MODEL_HOST:-/mnt/local/transformer_files/zkx/runs/amlsim-sft-20261021/amlsim-sft.ERWFretm/checkpoints/global_step_30/huggingface}"
DATA_HOST="${DATA_HOST:-/mnt/local/transformer_files/zkx/data/amlsim_typology_rules_v1_20261021/rl}"
RUNS_HOST="${RUNS_HOST:-/mnt/local/transformer_files/zkx/runs}"
PATCH_HOST="${PATCH_HOST:-/mnt/local/rl-training-toolkit/patch_tools}"
TOOLS_HOST="${TOOLS_HOST:-/mnt/local/rl-training-toolkit/tools/rl_insight_memory_case}"
AMLSIM_TOOLS_HOST="${AMLSIM_TOOLS_HOST:-/mnt/local/rl-training-toolkit/tools/amlsim}"
RL_INSIGHT_HOST="${RL_INSIGHT_HOST:-/mnt/local/rl-insight-official-20261031}"
RL_INSIGHT_OTEL_HOST="${RL_INSIGHT_OTEL_HOST:-/mnt/local/rl-insight-runtime/container-otel-1.39.1-20261103}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"

exec docker run -d \
  --name "$CONTAINER_NAME" --network=host --ipc=host --hostname "$(hostname)" --add-host "$(hostname):$HOST_IP" \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_HOST:/mnt/model:ro" -v "$DATA_HOST:/mnt/data:ro" -v "$RUNS_HOST:/mnt/runs" \
  -v "$PATCH_HOST:/mnt/patch_tools:ro" -v "$TOOLS_HOST:/mnt/tools:ro" -v "$AMLSIM_TOOLS_HOST:/mnt/amlsim_tools:ro" \
  -v "$RL_INSIGHT_HOST:/mnt/rl_insight:ro" -v "$RL_INSIGHT_OTEL_HOST:/mnt/rl_insight_otel:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e GRAD_SYNC_MODE="$GRAD_SYNC_MODE" \
  -e PYTHONPATH=/mnt/rl_insight:/mnt/rl_insight_otel -e RL_INSIGHT_ENABLE=1 \
  -e RL_INSIGHT_SERVER_URL="${RL_INSIGHT_SERVER_URL:-http://$HOST_IP:18080}" \
  -e RL_INSIGHT_SERVER_BACKEND=ray -e RL_INSIGHT_PROJECT=rl_insight_memory_case \
  -e TOTAL_STEPS="${TOTAL_STEPS:-3}" -e SAVE_FREQ="${SAVE_FREQ:-999}" \
  -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}" \
  -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}" -e PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}" \
  -e RANDOM_SEED="${RANDOM_SEED:-42}" -e HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-vlan699}" \
  -e HCCL_WHITELIST_DISABLE=1 -e HCCL_CONNECT_TIMEOUT=600 \
  -e MODEL_PATH=/mnt/model -e DATA_DIR=/mnt/data -e RUN_ROOT=/mnt/runs/rl-insight-memory-$(date +%Y%m%d) \
  "$IMAGE" bash -lc 'python3 /mnt/patch_tools/patch_verl_gradient_sync_case.py apply --verl-root /workspace/verl --backup-dir /tmp/verl_gradient_sync_case_backup && python3 /mnt/patch_tools/patch_verl_gradient_sync_case.py check --verl-root /workspace/verl && python3 /mnt/patch_tools/patch_verl_rl_insight.py apply --verl-root /workspace/verl --backup-dir /tmp/verl_rl_insight_backup && python3 /mnt/patch_tools/patch_verl_rl_insight.py check --verl-root /workspace/verl && python3 /mnt/patch_tools/patch_verl_rl_insight_rollout.py apply --verl-root /workspace/verl --backup-dir /tmp/verl_rl_insight_rollout_backup && python3 /mnt/patch_tools/patch_verl_rl_insight_rollout.py check --verl-root /workspace/verl && exec bash /mnt/tools/run_memory_case.sh'
