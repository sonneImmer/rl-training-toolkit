#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

IMAGE="${IMAGE:-openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix}"
MODEL_HOST="${MODEL_HOST:?MODEL_HOST is required}"
DATA_HOST="${DATA_HOST:-/mnt/local/transformer_files/zkx/data/finance_taiwanese_bankruptcy_v2_20261020/test.parquet}"
OUTPUT_HOST="${OUTPUT_HOST:?OUTPUT_HOST is required}"
TOOLS_HOST="${TOOLS_HOST:-$REPO_ROOT/tools/finance_risk}"
TP_SIZE="${TP_SIZE:-4}"
VISIBLE_DEVICES="${VISIBLE_DEVICES:-0,1,2,3}"
LIMIT="${LIMIT:-}"
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
HCCL_INTERFACE="${HCCL_INTERFACE:-vlan699}"
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
mkdir -p "$OUTPUT_HOST"

exec docker run --rm --network=host --ipc=host --hostname "$(hostname)" --add-host "$(hostname):$HOST_IP" \
  --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_HOST:/mnt/model:ro" -v "$DATA_HOST:/mnt/test.parquet:ro" \
  -v "$OUTPUT_HOST:/mnt/output" -v "$TOOLS_HOST:/mnt/tools:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES" -e VLLM_ASCEND_ENABLE_NZ=0 \
  -e HCCL_WHITELIST_DISABLE=1 -e HCCL_SOCKET_IFNAME="$HCCL_INTERFACE" -e HCCL_CONNECT_TIMEOUT=600 \
  "$IMAGE" python3 /mnt/tools/evaluate_finance_risk.py \
  --model /mnt/model --data /mnt/test.parquet --output /mnt/output \
  --tensor-parallel-size "$TP_SIZE" "${LIMIT_ARGS[@]}"
