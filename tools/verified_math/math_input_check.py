import argparse
import hashlib
import html
import importlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def pandas_module():
    try:
        import pandas
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pangu_rl/pylib"))
        import pandas
    return pandas


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(source, output):
    pandas = pandas_module()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "prepared_not_runtime_verified", "splits": {}}
    previews = []
    for split in ("train", "test"):
        source_path = source / f"{split}.parquet"
        source_hash = sha256(source_path)
        frame = pandas.read_parquet(source_path)
        original_questions = frame["prompt"].tolist()
        original_rewards = frame["reward_model"].tolist()
        identities = []
        messages = []
        for index, question in enumerate(original_questions):
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{split}:{index}: expected nonempty original question")
            if not isinstance(original_rewards[index], dict) or "ground_truth" not in original_rewards[index]:
                raise ValueError(f"{split}:{index}: missing ground_truth")
            messages.append([{"role": "user", "content": question}])
            previous = frame.iloc[index].get("extra_info")
            extra = dict(previous) if isinstance(previous, dict) else {}
            extra.update({"index": index, "sample_id": f"historical-math/{split}/{index}",
                          "source_sha256": source_hash})
            identities.append(extra)
            previews.append({"split": split, "source_row": index, "sample_id": extra["sample_id"],
                             "question": question, "ground_truth": original_rewards[index]["ground_truth"]})
        frame["prompt"] = messages
        frame["extra_info"] = identities
        target = output / f"{split}.parquet"
        frame.to_parquet(target, index=False)
        restored = pandas.read_parquet(target)
        for index, question in enumerate(original_questions):
            actual = list(restored.iloc[index]["prompt"])
            if actual != [{"role": "user", "content": question}] or restored.iloc[index]["reward_model"] != original_rewards[index]:
                raise AssertionError(f"round trip mismatch: {split}:{index}")
        manifest["splits"][split] = {"source": str(source_path), "source_sha256": sha256(source_path),
                                    "output": str(target), "output_sha256": sha256(target),
                                    "rows": len(frame), "all_questions_and_answers_preserved": True}
    with (output / "original_questions.jsonl").open("w", encoding="utf-8") as handle:
        for row in previews:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    cards = ''.join(f'<details><summary>{html.escape(row["sample_id"])}</summary><pre>{html.escape(row["question"])}</pre><p>参考答案：{html.escape(str(row["ground_truth"]))}</p></details>' for row in previews)
    (output / "questions.html").write_text('<!doctype html><meta charset="utf-8"><title>重新提取的数学原题</title><style>body{max-width:1000px;margin:30px auto;font:16px/1.7 system-ui}details{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}summary{cursor:pointer}</style><h1>重新提取的 5000 道原题</h1><p>原题与答案逐行保留；训练 4500，测试 500。此页不是模型新生成的回答。旧 96 条无法按参考答案可靠关联回原题。</p>' + cards, encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    print(f"已逐行验证并重新导出 {len(previews)} 道原题：{output}")


def format_prompts(source, output):
    pandas = pandas_module()
    output.mkdir(parents=True, exist_ok=False)
    instruction = (
        "Give a concise solution in at most 200 words. End the final line exactly as: "
        "Answer: <answer>\n/no_think"
    )
    manifest = {"status": "formatted", "instruction": instruction, "splits": {}}
    for split in ("train", "test"):
        source_path = source / f"{split}.parquet"
        frame = pandas.read_parquet(source_path)
        formatted = []
        for index, raw_prompt in enumerate(frame["prompt"].tolist()):
            messages = list(raw_prompt)
            if len(messages) != 1 or messages[0].get("role") != "user":
                raise ValueError(f"{split}:{index}: expected one user message")
            question = messages[0].get("content")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{split}:{index}: expected nonempty question")
            formatted.append([{"role": "user", "content": f"{question.rstrip()}\n\n{instruction}"}])
        frame["prompt"] = formatted
        target = output / f"{split}.parquet"
        frame.to_parquet(target, index=False)
        manifest["splits"][split] = {
            "source": str(source_path),
            "source_sha256": sha256(source_path),
            "output": str(target),
            "output_sha256": sha256(target),
            "rows": len(frame),
        }
    write_json(output / "manifest.json", manifest)
    print(f"已生成短回答格式数据：{output}")


def preflight(data, model, output, limit):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.reward_score import math_dapo
    pandas = pandas_module()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    sources = {}
    for name in ("verl.utils.dataset.rl_dataset", "verl.experimental.agent_loop.single_turn_agent_loop", "verl.trainer.ppo.ray_trainer", "verl.utils.reward_score.math_dapo"):
        module = importlib.import_module(name)
        path = Path(module.__file__)
        sources[name] = {"path": str(path), "sha256": sha256(path)}
    probes = []
    for split in ("train", "test"):
        frame = pandas.read_parquet(data / f"{split}.parquet")
        config = OmegaConf.create({"prompt_key": "prompt", "max_prompt_length": limit,
                                   "filter_overlong_prompts": False, "filter_prompts": False,
                                   "shuffle": False, "cache_dir": str(output / "cache")})
        dataset = RLHFDataset([str(data / f"{split}.parquet")], tokenizer, config)
        if len(dataset) != len(frame):
            raise AssertionError("loader changed row count")
        for index in range(len(frame)):
            raw = dataset[index]["raw_prompt"]
            if isinstance(raw, str):
                raise AssertionError("loader still outputs string raw_prompt")
            messages = list(raw)
            question = list(frame.iloc[index]["prompt"])[0]["content"]
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
            if question not in rendered or question not in decoded or len(token_ids) <= 3:
                raise AssertionError(f"question lost: {split}:{index}")
            probes.append({"split": split, "row": index, "sample_id": frame.iloc[index]["extra_info"]["sample_id"],
                           "question": question, "rendered": rendered, "decoded": decoded,
                           "token_count": len(token_ids), "within_budget": len(token_ids) <= limit})
    checks = []
    for response, reference, expected in [
        ("Answer: 4", "4", True),
        ("Answer: 5", "4", False),
        ("Answer: 90%", "90", True),
    ]:
        result = math_dapo.compute_score(response, reference)
        correct = result.get("acc") if isinstance(result, dict) else result > 0
        if bool(correct) != expected:
            raise AssertionError(f"scoring contract failed: {result}")
        checks.append({"response": response, "reference": reference, "result": result})
    with (output / "tokenizer_probes.jsonl").open("w", encoding="utf-8") as handle:
        for probe in probes:
            handle.write(json.dumps(probe, ensure_ascii=False) + "\n")
    oversized = sum(not probe["within_budget"] for probe in probes)
    write_json(output / "result.json", {"status": "pass" if oversized == 0 else "blocked_overlength",
               "model": model, "loaded_sources": sources, "questions_checked": len(probes),
               "overlength": oversized, "scoring_checks": checks})
    if oversized:
        raise SystemExit(f"{oversized} 道题超过输入预算；未自动丢弃，请先确定处理方案")
    print("实际 loader、tokenizer 与评分检查通过；尚未验证生成与参数更新")


def audit(data, run):
    pandas = pandas_module()
    references = {}
    for split in ("train", "test"):
        frame = pandas.read_parquet(data / f"{split}.parquet")
        for _, row in frame.iterrows():
            question = list(row["prompt"])[0]["content"]
            references.setdefault(question, set()).add(str(row["reward_model"]["ground_truth"]))
    records = []
    failures = []
    groups = defaultdict(list)
    for path in sorted((run / "rollouts").glob("*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            matched = [question for question in references if question in row.get("input", "")]
            if not matched or not any(str(row.get("gts")) in references[question] for question in matched):
                failures.append(f"{path.name}:{number}: 原题或参考答案无法对应")
            if not row.get("uid"):
                failures.append(f"{path.name}:{number}: 缺少分组编号，无法验收同题四个回答")
            else:
                groups[(path.stem, str(row["uid"]))].append(row)
            if not isinstance(row.get("output"), str) or not row["output"].strip():
                failures.append(f"{path.name}:{number}: 回答为空")
            records.append({"file": path.name, "line": number, "record": row,
                            "candidate_question_count": len(matched), "question_matches": matched})
    for identity, members in groups.items():
        if len(members) != 4 or len({(member.get("input"), str(member.get("gts"))) for member in members}) != 1:
            failures.append(f"{identity}: 分组数量不是 4 或同组输入/答案不一致")
    text = (run / "train.log").read_text(errors="replace")
    metric_rows = []
    for line in text.splitlines():
        match = re.search(r"step:(\d+) - ", line)
        if not match or int(match.group(1)) == 0:
            continue
        values = {}
        for key in ("actor/grad_norm", "critic/advantages/min", "critic/advantages/max", "prompt_length/mean"):
            value = re.search(re.escape(key) + r":([-+\d.eE]+)", line)
            if value:
                values[key] = float(value.group(1))
        metric_rows.append({"step": int(match.group(1)), **values})
    if not records:
        failures.append("没有新 rollout")
    if not metric_rows:
        failures.append("没有训练指标")
    nonzero = any(row.get("actor/grad_norm", 0) > 0 and (row.get("critic/advantages/min", 0) != 0 or row.get("critic/advantages/max", 0) != 0) for row in metric_rows)
    checkpoint_files = sorted(
        (run / "checkpoints").glob("global_step_*/actor/model_world_size_*_rank_0.pt"),
        key=lambda path: int(path.parents[1].name.rsplit("_", 1)[-1]),
    )
    checkpoint_fingerprints = [
        {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in checkpoint_files
    ]
    parameter_change_verified = (
        len(checkpoint_fingerprints) >= 2
        and checkpoint_fingerprints[0]["sha256"] != checkpoint_fingerprints[-1]["sha256"]
    )
    if failures:
        status = "failed_input_or_missing_evidence"
    elif parameter_change_verified:
        status = "parameter_update_verified_pending_learning_evaluation"
    elif nonzero:
        status = "signal_observed_pending_parameter_verification"
    else:
        status = "input_preserved_but_no_learning_signal_observed"
    result = {"status": status,
              "errors": failures, "records": records, "training_metrics": metric_rows,
              "nonzero_gradient_and_advantage_observed": nonzero,
              "checkpoint_fingerprints": checkpoint_fingerprints,
              "parameter_change_verified": parameter_change_verified,
              "learning_improvement_verified": False}
    write_json(run / "audit.json", result)
    print(f"原题检查错误：{len(failures)}；观察到非零优势与梯度：{nonzero}。不代表效果已改善。")
    if failures or not nonzero:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    formatting = commands.add_parser("format-prompts")
    formatting.add_argument("--source", type=Path, required=True)
    formatting.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--data", type=Path, required=True)
    check.add_argument("--model", required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--max-prompt-length", type=int, default=1024)
    inspection = commands.add_parser("audit")
    inspection.add_argument("--data", type=Path, required=True)
    inspection.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source.resolve(), args.output.resolve())
    elif args.command == "format-prompts":
        format_prompts(args.source.resolve(), args.output.resolve())
    elif args.command == "preflight":
        preflight(args.data.resolve(), args.model, args.output.resolve(), args.max_prompt_length)
    else:
        audit(args.data.resolve(), args.run.resolve())
