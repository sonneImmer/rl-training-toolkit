#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?请设置容器内真实模型目录 MODEL_PATH}"
: "${DATA_DIR:?请设置修正后数据目录 DATA_DIR}"
: "${RUN_ROOT:?请设置新实验根目录 RUN_ROOT}"
: "${ASCEND_RT_VISIBLE_DEVICES:?请明确设备编号，例如 1,2}"
: "${TOTAL_STEPS:=1}"
: "${MAX_RESPONSE_LENGTH:=512}"
: "${SAVE_FREQ:=-1}"
: "${TEST_FREQ:=-1}"
: "${ADV_ESTIMATOR:=grpo}"
: "${NGRPO_MAX_REWARD:=1.0}"
: "${NGRPO_VIRTUAL_COUNT:=1}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$RUN_ROOT"
RUN_DIR=$(mktemp -d "$RUN_ROOT/math-input-check.XXXXXXXX")
export VLLM_ASCEND_ENABLE_NZ=0
export HYDRA_FULL_ERROR=1
python3 "$SCRIPT_DIR/math_input_check.py" preflight --data "$DATA_DIR" --model "$MODEL_PATH" --output "$RUN_DIR/preflight" --max-prompt-length 2048
IFS=',' read -ra DEVICE_LIST <<< "$ASCEND_RT_VISIBLE_DEVICES"
DEVICE_COUNT=${#DEVICE_LIST[@]}
if [[ "$DEVICE_COUNT" -ne 2 && "$DEVICE_COUNT" -ne 4 && "$DEVICE_COUNT" -ne 8 ]]; then
    echo '该短程复验配置仅支持明确分配 2、4 或 8 张卡。' >&2
    exit 1
fi
ALGORITHM_ARGS=("algorithm.adv_estimator=$ADV_ESTIMATOR")
if [[ "$ADV_ESTIMATOR" == "ngrpo" ]]; then
    ALGORITHM_ARGS+=(
        "algorithm.max_reward=$NGRPO_MAX_REWARD"
        "algorithm.add_max_reward_num=$NGRPO_VIRTUAL_COUNT"
    )
fi
python3 -m verl.trainer.main_ppo \
    "${ALGORITHM_ARGS[@]}" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=8 data.val_batch_size=8 \
    data.max_prompt_length=2048 data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$DEVICE_COUNT" actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 trainer.val_before_train=False trainer.logger=console \
    trainer.project_name=math_input_verification trainer.experiment_name="$(basename "$RUN_DIR")" \
    trainer.n_gpus_per_node="$DEVICE_COUNT" trainer.nnodes=1 trainer.save_freq="$SAVE_FREQ" trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs=1 trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.rollout_data_dir="$RUN_DIR/rollouts" 2>&1 | tee "$RUN_DIR/train.log"
python3 "$SCRIPT_DIR/math_input_check.py" audit --data "$DATA_DIR" --run "$RUN_DIR"
