"""Evidence-bound algorithm mechanism diagnostics and candidate experiments."""
from __future__ import annotations

from .algorithm_profiles import resolve_algorithm_profile


def _card(
    rule_id: str,
    title: str,
    status: str,
    observation: str,
    mechanism: str,
    boundary: str,
    actions: list[str],
    revalidation: list[str],
    evidence_steps: list[int],
    confidence: str = "standard",
) -> dict:
    return {
        "rule_id": rule_id,
        "title": title,
        "status": status,
        "confidence": confidence,
        "observation": observation,
        "mechanism": mechanism,
        "boundary": boundary,
        "candidate_actions": actions,
        "revalidation": revalidation,
        "evidence_steps": sorted(set(evidence_steps)),
    }


def build_mechanism_diagnostics(
    algorithm: str | None,
    advantage_estimator: str | None,
    group_results: list,
    step_aggs: list,
) -> dict:
    profile = resolve_algorithm_profile(algorithm, advantage_estimator)
    analyzable = [item for item in group_results if item.signal_type != "group_unresolved"]
    recorded = [item for item in analyzable if item.advantage_recorded_count > 0]
    homogeneous = [
        item for item in recorded
        if item.signal_type in ("constant_score", "near_constant_score")
    ]
    homogeneous_zero = [item for item in homogeneous if item.homogeneous_reward_zero_advantage]
    homogeneous_nonzero = [item for item in homogeneous if item.homogeneous_reward_nonzero_advantage]
    all_wrong_negative = [item for item in recorded if item.all_wrong_negative_advantage]
    cards = []

    coverage = len(recorded) / len(analyzable) if analyzable else 0.0
    cards.append(_card(
        "advantage_evidence_coverage",
        "Advantage证据覆盖",
        "observed" if recorded else "not_recorded",
        f"{len(recorded)}/{len(analyzable)}个可分析Group记录了Advantage（{coverage:.1%}）。",
        profile["mechanism"],
        "记录的序列级Advantage表示策略损失方向，不等于单Group真实梯度贡献。",
        ["优先补齐异常Step的Advantage和Mask证据。"] if not recorded else [],
        ["复验时保持模型、数据、seed和训练预算一致。"],
        [item.step for item in recorded],
        confidence="standard" if recorded else "none",
    ))

    if profile["family"] == "grpo":
        if homogeneous_zero:
            cards.append(_card(
                "group_advantage_collapse",
                "同质Group缺少相对优势",
                "requires_experiment",
                f"{len(homogeneous_zero)}/{len(homogeneous)}个已记录的同质Reward Group，其Advantage全部为零。",
                "GRPO依赖组内Reward差异；同组得分相同时，相对优势可能退化为零。",
                "零Advantage不等于总梯度为零，KL等损失仍可能参与更新。",
                ["检查Reward判别能力和Rollout多样性。", "安排同预算NGRPO单变量对照。"],
                ["比较零Advantage Group、独立评测、KL、耗时和显存。"],
                [item.step for item in homogeneous_zero],
            ))
    elif profile["family"] == "ngrpo":
        if homogeneous_nonzero:
            cards.append(_card(
                "ngrpo_homogeneous_reward_nonzero_advantage",
                "同质Reward仍有NGRPO优势信号",
                "observed",
                f"{len(homogeneous_nonzero)}/{len(homogeneous)}个已记录的同质Reward Group存在非零Advantage。",
                "当前NGRPO补丁使用虚拟最高Reward参考，可使部分同质低分Group获得非零方向。",
                "非零Advantage不证明更新有效，也不保证独立任务指标改善。",
                ["核对max_reward是否等于Reward真实上限。"],
                ["比较固定测试集及类别指标，检查负优势是否伴随局部退化。"],
                [item.step for item in homogeneous_nonzero],
            ))
        if all_wrong_negative:
            cards.append(_card(
                "ngrpo_negative_advantage_concentration",
                "全错Group出现负优势",
                "requires_experiment",
                f"{len(all_wrong_negative)}个全错Group记录到负Advantage。",
                "负优势提供抑制方向，但其强度可能改变类别和输出分布。",
                "该现象不能单独说明NGRPO优于GRPO。",
                ["检查虚拟最大奖励、Reward范围及类别采样。"],
                ["按业务类别复评，并与GRPO保持同seed、预算和数据顺序。"],
                [item.step for item in all_wrong_negative],
            ))
    elif profile["family"] == "ppo":
        cards.append(_card(
            "ppo_gae_signal",
            "PPO价值优势信号",
            "observed" if recorded else "not_recorded",
            f"已记录{len(recorded)}个Group的GAE Advantage；同质Reward Group为{len(homogeneous)}个。",
            "PPO通过Critic价值估计和GAE计算优势，同组Reward是否相同不是其信号是否为零的充分条件。",
            "需要同时检查Critic损失、Value预测和实际Actor更新，当前尚未进行逐Group梯度归因。",
            ["联看Critic指标、Advantage分布和Clip Fraction。"],
            ["与GRPO/NGRPO使用相同数据、seed、Rollout数和总Step独立评测。"],
            [item.step for item in recorded],
        ))

    cards.append(_card(
        "low_signal_compute_waste",
        "低信号采样成本",
        "not_recorded",
        "当前未按Group记录生成Token、重试次数和Wall Time，不能量化低信号样本的计算浪费。",
        "只有把Group信息量与实际生成成本关联，才能评价动态采样是否节省训练资源。",
        "不能用低信息Group占比直接换算NPU浪费比例。",
        ["后续增加过滤前后Group数、生成Token和阶段耗时埋点。"],
        ["单变量比较样本利用率、吞吐和独立评测。"],
        [item.step for item in step_aggs],
        confidence="none",
    ))

    return {
        "algorithm_profile": profile,
        "cards": cards,
        "summary": {
            "analyzable_groups": len(analyzable),
            "advantage_recorded_groups": len(recorded),
            "homogeneous_reward_groups": len(homogeneous),
            "homogeneous_zero_advantage_groups": len(homogeneous_zero),
            "homogeneous_nonzero_advantage_groups": len(homogeneous_nonzero),
            "all_wrong_negative_advantage_groups": len(all_wrong_negative),
        },
    }
