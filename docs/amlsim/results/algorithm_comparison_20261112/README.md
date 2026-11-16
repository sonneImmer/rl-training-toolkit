# AMLSim 2026-11-12 三算法长训练评测

## 实验范围

- 算法：GRPO、NGRPO、PPO。
- 训练预算：相同模型、相同数据、相同随机种子、150 step、8 张 Ascend 910B3。
- 独立评测：固定 `test.parquet`，800 条样本，每类 200 条。
- 数据集 SHA-256：见 `test_dataset.sha256`。

## 结果结论

PPO 在本轮 800 条独立测试中最高：风险准确率和类型准确率均为 100%；NGRPO 为 99.875%；GRPO 为 99.75%。三者差距只有 0--2 条样本，当前只能作为本数据集、单 seed、单 checkpoint 的实验结果，不能据此宣称算法普遍优越。

训练过程中的平均 Reward 已接近饱和，不能替代独立评测。算法比较应优先查看风险召回率、误报/漏报、类型分层准确率和证据 F1。

## 文件说明

- `comparison.md`：可直接用于分析和汇报的汇总报告。
- `comparison.json`：结构化对比结果。
- `*_metrics.json`：三种算法的独立评测指标。
- `*_evaluation_manifest.json`：checkpoint、测试集和数据哈希关联信息。
- `ERR99999_FIX.md`：训练结束阶段 ERR99999 的定位和修复验证。

完整预测、训练日志、checkpoint 和模型权重不进入 Git；需要时根据评测清单中的本地路径和数据哈希重新生成。
