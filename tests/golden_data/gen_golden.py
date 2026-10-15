"""Generate golden dataset for testing REQ-01 and REQ-02.

Creates:
- golden_log.txt: simulated training log with Hydra params, metric lines, ANSI codes
- golden_rollouts/: directory with step JSONL files covering various test cases
"""
import json
import os
import math

OUT_DIR = os.path.join(os.path.dirname(__file__))
ROLLOUT_DIR = os.path.join(OUT_DIR, "golden_rollouts")

# ── Simulated training log ──────────────────────────────────────────
GOLDEN_LOG = r"""
+ export VLLM_ASCEND_ENABLE_NZ=0
+ VLLM_ASCEND_ENABLE_NZ=0
+ python3 -m verl.trainer.main_ppo algorithm.adv_estimator=grpo data.train_files=/data/math/train.parquet data.val_files=/data/math/test.parquet data.train_batch_size=8 data.val_batch_size=8 data.max_prompt_length=1024 data.max_response_length=1024 actor_rollout_ref.model.path=/models/qwen7b actor_rollout_ref.actor.optim.lr=5e-7 actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.tensor_model_parallel_size=2 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 trainer.logger=console trainer.project_name=golden_test trainer.experiment_name=qwen_golden trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq=7 trainer.test_freq=7 trainer.total_epochs=3 trainer.total_training_steps=174
\x1b[36m(pid=12345)\x1b[0m Some warning message
\x1b[36m(TaskRunner pid=12345)\x1b[0m step:1 - actor/entropy:0.733 - actor/pg_loss:0.1 - actor/kl_loss:0.001 - actor/pg_clipfrac:0.05 - actor/grad_norm:1.2 - actor/ppo_kl:0.0 - actor/lr:5e-07 - training/global_step:1 - training/epoch:0 - critic/score/mean:0.4 - critic/score/max:1.0 - critic/score/min:-1.0 - response_length/mean:542.9 - response_length/max:1024.0 - response_length/min:40.0 - response_length/clip_ratio:0.5 - perf/throughput:21.6 - perf/total_num_tokens:8735 - perf/time_per_step:201.8 - timing_s/step:201.8 - timing_s/gen:162.0
\x1b[36m(TaskRunner pid=12345)\x1b[0m step:2 - actor/entropy:0.65 - actor/pg_loss:0.05 - actor/kl_loss:0.002 - actor/pg_clipfrac:0.08 - actor/grad_norm:1.5 - actor/ppo_kl:0.001 - actor/lr:5e-07 - training/global_step:2 - training/epoch:0 - critic/score/mean:0.5 - response_length/mean:500.0 - response_length/max:1024.0 - response_length/min:30.0 - response_length/clip_ratio:0.3 - perf/throughput:22.0 - timing_s/step:195.0
\x1b[36m(TaskRunner pid=12345)\x1b[0m step:3 - actor/entropy:0.3 - actor/pg_loss:0.02 - actor/kl_loss:0.01 - actor/pg_clipfrac:0.25 - actor/grad_norm:5.0 - actor/ppo_kl:0.005 - training/global_step:3 - training/epoch:0 - response_length/clip_ratio:0.6 - perf/throughput:15.0
\x1b[36m(TaskRunner pid=12345)\x1b[0m ("Initial validation metrics: {'val-aux/math/reward/mean@1': -0.5, 'val-aux/math/reward/max@1': 1.0, 'val-aux/math/reward/min@1': -1.0}")
\x1b[36m(TaskRunner pid=12345)\x1b[0m step:4 - actor/entropy:0.1 - actor/pg_loss:0.0 - actor/kl_loss:0.05 - actor/pg_clipfrac:0.3 - actor/grad_norm:10.0 - actor/ppo_kl:0.02 - training/global_step:4 - training/epoch:1 - response_length/clip_ratio:0.8 - perf/throughput:10.0
"""


def write_step_file(step: int, records: list, filename: str = None):
    """Write a step JSONL file. Records can be dicts or raw strings."""
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    fname = filename or f"{step}.jsonl"
    path = os.path.join(ROLLOUT_DIR, fname)
    with open(path, "w") as f:
        for rec in records:
            if isinstance(rec, str):
                # Raw string - write as-is (for corrupted JSON tests)
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def gen_golden():
    # Clean previous data
    import shutil
    if os.path.exists(ROLLOUT_DIR):
        shutil.rmtree(ROLLOUT_DIR)
    os.makedirs(ROLLOUT_DIR, exist_ok=True)

    # Step 1: 2 groups, 5 rollouts each
    # Group A: effective group (rewards differ)
    group_a = []
    for resp, score in [
        ("The answer is 4.", 1.0),
        ("The answer is 5.", -1.0),
        ("The answer is 4.", 1.0),
        ("I don't know.", -1.0),
        ("The answer is 4.", 1.0),
    ]:
        group_a.append({
            "input": "What is 2+2?",
            "output": resp,
            "gts": "4",
            "score": score,
            "step": 1,
            "uid": "group_a_step1",
            "data_source": "math",
        })

    # Group B: constant_score (all same reward)
    group_b = []
    for resp, score in [
        ("Answer: 10.", 1.0),
        ("Answer: 10.", 1.0),
        ("Answer: 10.", 1.0),
        ("Answer: 10.", 1.0),
        ("Answer: 10.", 1.0),
    ]:
        group_b.append({
            "input": "What is 5+5?",
            "output": resp,
            "gts": "10",
            "score": score,
            "step": 1,
            "uid": "group_b_step1",
            "data_source": "math",
        })

    write_step_file(1, group_a + group_b)

    # Step 2: near_constant group + insufficient group + negative rewards
    group_c = []
    for resp, score in [
        ("Result: 0.50", 0.5),
        ("Result: 0.51", 0.51),
        ("Result: 0.49", 0.49),
        ("Result: 0.50", 0.5),
        ("Result: 0.50", 0.5),
    ]:
        group_c.append({
            "input": "Calculate 1/2.",
            "output": resp,
            "gts": "0.5",
            "score": score,
            "step": 2,
            "uid": "group_c_step2",
            "data_source": "math",
        })

    group_d = [
        {"input": "Solve x^2=4.", "output": "x=2", "gts": "2", "score": 1.0, "step": 2, "uid": "group_d_step2"},
        {"input": "Solve x^2=4.", "output": "x=-2", "gts": "2", "score": 1.0, "step": 2, "uid": "group_d_step2"},
        {"input": "Solve x^2=4.", "output": "x=3", "gts": "2", "step": 2, "uid": "group_d_step2"},
        {"input": "Solve x^2=4.", "output": "", "gts": "2", "score": float("nan"), "step": 2, "uid": "group_d_step2"},
    ]

    write_step_file(2, group_c + group_d)

    # Step 3: array-form record (single record with multiple responses)
    array_record = {
        "input": "What is 3+3?",
        "response": ["Answer: 6.", "Answer: 5.", "Answer: 6.", "Answer: 7.", "Answer: 6."],
        "reward": [1.0, -1.0, 1.0, -1.0, 1.0],
        "gts": "6",
        "step": 3,
        "uid": "group_e_step3",
        "data_source": "math",
    }
    write_step_file(3, [array_record])

    # Step 4: JSON corrupted line + schema error (response/reward mismatch)
    step4_records = [
        {"input": "Q1", "output": "A1", "gts": "1", "score": 1.0, "step": 4, "uid": "g4_1"},
        {"input": "Q2", "output": "A2", "gts": "2", "score": -1.0, "step": 4, "uid": "g4_2"},
        # Corrupted JSON (raw string, not a dict - written as-is)
        '{"input": "broken", "output": "brok',
        # Array mismatch
        {"input": "Q3", "response": ["a", "b", "c"], "reward": [1.0, -1.0], "step": 4, "uid": "g4_3"},
    ]
    write_step_file(4, step4_records)

    # Step 5: no prompt, no ID (should still parse reward stats)
    step5_records = [
        {"output": "Answer A", "score": 1.0, "step": 5},
        {"output": "Answer B", "score": -1.0, "step": 5},
    ]
    write_step_file(5, step5_records)

    # Invalid filename (should be ignored with warning)
    write_step_file(0, [{"input": "x", "output": "y", "score": 1.0, "step": 0}], filename="abc.jsonl")

    # Write golden log
    log_path = os.path.join(OUT_DIR, "golden_log.txt")
    with open(log_path, "w") as f:
        f.write(GOLDEN_LOG.replace("\\x1b", "\x1b"))

    print(f"Golden data written to {OUT_DIR}")
    print(f"  Log: {log_path}")
    print(f"  Rollouts: {ROLLOUT_DIR}")
    print(f"  Steps: 1,2,3,4,5 + invalid abc.jsonl")


if __name__ == "__main__":
    gen_golden()
