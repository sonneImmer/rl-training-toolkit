const state = {
  experiments: [],
  rolloutReading: false,
  selectedId: null,
  detail: null,
  rolloutEvidence: null,
  timer: null,
};

const statusNames = {
  running: "运行中",
  completed: "已完成",
  completed_with_errors: "完成但有尾部错误",
  failed: "失败",
  stopped: "停止更新",
  unreadable: "不可读取",
};

const el = (id) => document.getElementById(id);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
}[char]));

function percent(value) {
  return value === null || value === undefined ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
}

function number(value, digits = 4) {
  return value === null || value === undefined ? "-" : Number(value).toFixed(digits);
}
function advantageDirection(value) {
  if (value === null || value === undefined) return "未记录";
  const numeric = Number(value);
  if (Math.abs(numeric) < 1e-8) return "零信号";
  return numeric > 0 ? "正向强化" : "负向抑制";
}

function groupRollouts(records) {
  const groups = new Map();
  records.forEach((record, index) => {
    const groupId = String(record.uid ?? record.group_id ?? record.input ?? record.prompt ?? `unknown-${index}`);
    const group = groups.get(groupId) || { records: [] };
    group.records.push(record);
    groups.set(groupId, group);
  });
  return [...groups.values()];
}

function rolloutGroupsHtml(records) {
  return groupRollouts(records).map((group, groupIndex) => {
    const advantages = group.records
      .map((record) => record.advantage)
      .filter((value) => value !== null && value !== undefined)
      .map(Number);
    const positive = advantages.filter((value) => value > 1e-8).length;
    const negative = advantages.filter((value) => value < -1e-8).length;
    const zero = advantages.length - positive - negative;
    const signal = advantages.length ? `正 ${positive} · 负 ${negative} · 零 ${zero}` : "Advantage 未记录";
    return `<details class="rollout-group">
      <summary><strong>Group #${groupIndex + 1}</strong><span>${group.records.length} responses</span><span>${signal}</span></summary>
      <div class="rollout-group-body">${groupComponentHtml(group.records)}${group.records.map((record, index) => `<details class="rollout-item">
        <summary><strong>#${index + 1}</strong><span>reward ${escapeHtml(record.score ?? record.reward ?? "-")}</span><span>adv ${escapeHtml(record.advantage ?? "-")}</span><span>${escapeHtml(advantageDirection(record.advantage))}</span><span>${escapeHtml(selectionText(record))}</span><span>acc ${escapeHtml(record.acc ?? "-")}</span><span>pred ${escapeHtml(record.pred ?? "-")}</span></summary>
        ${responseComponentsHtml(record)}
        <div class="selection-evidence"><span>response mask token ${record.response_mask_tokens ?? "未记录"}</span><span>policy signal token ${record.policy_signal_tokens ?? "未记录"}</span></div>
        <pre>${escapeHtml(`PROMPT\n${record.input ?? record.prompt ?? ""}\n\nOUTPUT\n${record.output ?? record.response ?? ""}`)}</pre>
      </details>`).join("")}</div>
    </details>`;
  }).join("");
}


function selectionText(record) {
  if (record.selected_for_update === null || record.selected_for_update === undefined) return "更新状态未记录";
  return record.selected_for_update ? "已进入 Actor 更新批次" : "未进入 Actor 更新批次";
}

function componentValue(component) {
  if (component && typeof component === "object") {
    const value = component.weighted ?? component.score ?? component.raw;
    return typeof value === "number" ? value : null;
  }
  return typeof component === "number" ? component : null;
}

function groupComponentHtml(records) {
  const values = new Map();
  records.forEach((record) => Object.entries(record.reward_components || {}).forEach(([name, component]) => {
    const value = componentValue(component);
    if (value === null) return;
    const items = values.get(name) || [];
    items.push(value);
    values.set(name, items);
  }));
  if (!values.size) return '<div class="component-note">Reward 分量未记录</div>';
  return `<div class="component-grid">${[...values.entries()].map(([name, items]) => {
    const low = Math.min(...items); const high = Math.max(...items);
    const signal = high - low > 1e-8 ? "有组内区分" : "组内无区分";
    return `<div class="component-chip"><span>${escapeHtml(name)}</span><strong>${number(low, 3)} ~ ${number(high, 3)}</strong><small>${signal}</small></div>`;
  }).join("")}</div>`;
}

function responseComponentsHtml(record) {
  const entries = Object.entries(record.reward_components || {});
  if (!entries.length) return "";
  return `<div class="response-components">${entries.map(([name, component]) => `<span><b>${escapeHtml(name)}</b> ${escapeHtml(number(componentValue(component), 3))}</span>`).join("")}</div>`;
}

function evaluationAlertHtml(alert) {
  return `<div class="evaluation-alert"><strong>总体上升但单类下降</strong><span>step ${alert.from_step} → ${alert.to_step}：${escapeHtml(alert.metric)} ${percent(alert.overall_before)} → ${percent(alert.overall_after)}，${escapeHtml(alert.class_name)} ${percent(alert.class_before)} → ${percent(alert.class_after)}</span></div>`;
}

function evaluationHtml(evaluation) {
  const summary = evaluation.summary || {};
  const percentKeys = new Set(["risk_accuracy", "risk_precision", "risk_recall", "risk_f1", "risk_f2", "typology_accuracy", "format_valid_rate", "evidence_f1", "accuracy", "precision", "recall", "f1", "f2"]);
  const summaryCards = Object.entries(summary).map(([key, value]) => diagnosticMetric(key, percentKeys.has(key) ? percent(value) : value)).join("");
  const perClass = Object.entries(evaluation.per_class || {});
  return `<details class="evaluation-item" open><summary><strong>${escapeHtml(evaluation.evaluation_id)}</strong><span>checkpoint step ${evaluation.checkpoint_step ?? "未记录"}</span><span>${escapeHtml(evaluation.split || "test")}</span></summary>
    <div class="evaluation-body"><div class="diagnostic-grid">${summaryCards || diagnosticMetric("状态", "暂无摘要")}</div>
    ${perClass.length ? `<div class="table-scroll"><table class="facts-table"><thead><tr><th>类别</th><th>样本</th><th>正确</th><th>准确率</th></tr></thead><tbody>${perClass.map(([name, row]) => `<tr><td>${escapeHtml(name)}</td><td>${row.samples ?? "-"}</td><td>${row.correct ?? "-"}</td><td>${percent(row.accuracy)}</td></tr>`).join("")}</tbody></table></div>` : ""}</div></details>`;
}

function compact(value) {
  if (!value) return "-";
  const text = String(value);
  return text.length > 34 ? `...${text.slice(-31)}` : text;
}

function timeText(value) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  }).format(new Date(value));
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function loadExperiments(force = false) {
  try {
    const detailState = captureDetailState();
    if (force) await fetchJson("/api/refresh");
    state.experiments = await fetchJson("/api/experiments");
    el("connectionStatus").textContent = `已连接 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    renderList();
    if (!state.selectedId && state.experiments.length) {
      await selectExperiment(state.experiments[0].experiment_id);
    } else if (state.selectedId && (!state.rolloutReading || force)) {
      if (force) state.rolloutReading = false;
      await selectExperiment(state.selectedId, false, detailState);
    }
  } catch (error) {
    el("connectionStatus").textContent = `连接失败 · ${error.message}`;
  }
}

function filteredExperiments() {
  const query = el("searchInput").value.trim().toLowerCase();
  const status = el("statusFilter").value;
  return state.experiments.filter((item) => {
    if (status !== "all" && item.status !== status) return false;
    const haystack = [item.name, item.config?.algorithm, item.config?.model, item.config?.dataset]
      .filter(Boolean).join(" ").toLowerCase();
    return !query || haystack.includes(query);
  });
}

function renderList() {
  const items = filteredExperiments();
  el("experimentCount").textContent = `${items.length} 个记录`;
  el("experimentList").innerHTML = items.map((item) => {
    const latest = item.latest_metrics || {};
    const progress = item.progress || {};
    const active = item.experiment_id === state.selectedId ? " active" : "";
    return `<button class="experiment-row${active}" data-id="${escapeHtml(item.experiment_id)}">
      <div class="row-primary">
        <span class="status-dot ${escapeHtml(item.status)}"></span>
        <strong title="${escapeHtml(item.name)}">${escapeHtml(item.name || item.experiment_id)}</strong>
        <span>${escapeHtml(statusNames[item.status] || item.status)}</span>
      </div>
      <div class="row-meta">
        <span>${escapeHtml(item.config?.algorithm || "算法未知")}</span>
        <span>step ${progress.current_step ?? "-"}/${progress.total_steps ?? "-"}</span>
        <span>${timeText(item.updated_at)}</span>
      </div>
      <div class="row-metrics">
        <span title="最新训练 Step 内全部 rollout 奖励的算术平均值">最新步平均奖励 ${number(latest.reward_mean, 3)}</span>
        <span>有效 ${percent(latest.valid_ratio)}</span>
        <span>标签 ${item.labels?.length || 0}</span>
      </div>
    </button>`;
  }).join("") || `<div class="empty-state"><p>没有符合条件的实验。</p></div>`;
  document.querySelectorAll(".experiment-row").forEach((button) => {
    button.addEventListener("click", () => selectExperiment(button.dataset.id));
  });
}

function captureDetailState() {
  if (!state.selectedId) return null;
  const content = el("rolloutContent");
  const step = el("rolloutStep");
  if (!content || !step) return state.rolloutEvidence;
  return {
    rolloutStep: step.value,
    rolloutHtml: content.innerHTML,
    rolloutOpen: [...content.querySelectorAll("details")].map((item) => item.open),
  };
}

function restoreDetailState(detailState) {
  if (!detailState || !el("rolloutContent")) return;
  const content = el("rolloutContent");
  const step = el("rolloutStep");
  if (detailState.rolloutHtml && detailState.rolloutHtml !== '<span class="no-labels">选择 step 查看 prompt、输出、reward、pred 和 acc。</span>') {
    content.innerHTML = detailState.rolloutHtml;
    step.value = detailState.rolloutStep || step.value;
    [...content.querySelectorAll("details")].forEach((item, index) => {
      item.open = Boolean(detailState.rolloutOpen?.[index]);
    });
  }
}

async function selectExperiment(experimentId, scroll = true, detailState = null) {
  const changed = state.selectedId !== experimentId;
  state.selectedId = experimentId;
  if (changed) {
    state.rolloutEvidence = null;
    state.rolloutReading = false;
  }
  renderList();
  try {
    state.detail = await fetchJson(`/api/experiments/${encodeURIComponent(experimentId)}`);
    renderDetail();
    restoreDetailState(changed ? null : detailState);
    if (scroll) el("detail").scrollTop = 0;
  } catch (error) {
    el("detail").innerHTML = `<section class="empty-state"><h2>读取失败</h2><p>${escapeHtml(error.message)}</p></section>`;
  }
}

function metricItem(label, value) {
  return `<div class="metric-item"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function labelHtml(label) {
  const threshold = label.threshold === undefined ? "" : ` · 阈值 ${number(label.threshold, 3)}`;
  return `<div class="label-chip ${escapeHtml(label.severity)}">
    <strong>${escapeHtml(label.title)}</strong>
    <span>step ${label.first_step ?? "-"}-${label.last_step ?? "-"} · 当前 ${number(label.observed_value, 3)}${threshold}</span>
  </div>`;
}

function configCell(label, value) {
  return `<div class="config-cell"><span>${escapeHtml(label)}</span><strong title="${escapeHtml(value)}">${escapeHtml(compact(value))}</strong></div>`;
}

function diagnosticMetric(label, value) {
  return `<div class="diagnostic-metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function manifestValue(manifest, section, key) {
  return manifest?.[section]?.[key] ?? "未记录";
}

function readinessHtml(item) {
  const readiness = item.comparison_readiness || {};
  const ready = readiness.status === "ready";
  const missing = readiness.missing_fields || [];
  const missingText = missing.length ? `缺少：${missing.join("、")}` : "关键条件已记录";
  return `<div class="readiness ${ready ? "ready" : "incomplete"}">
    <strong>${ready ? "具备算法对比条件" : "暂不具备严格对比条件"}</strong>
    <span>${escapeHtml(missingText)}</span>
  </div>`;
}

function ruleStatusHtml(item) {
  const names = { triggered: "已触发", not_triggered: "未触发", not_evaluated: "未评估", early_observation: "早期正常", early_warning: "早期预警" };
  const ruleNames = {
    explore_collapse: "探索能力收缩", kl_drift: "KL 漂移", clip_saturation: "裁剪饱和",
    gradient_spike: "梯度尖峰", output_truncation: "输出截断", reward_signal_degradation: "奖励信号退化",
    performance_degradation: "训练性能下降",
  };
  const detail = [
    item.reason ? `原因：${item.reason}` : "",
    item.description ? `说明：${item.description}` : "",
    item.metric_name ? `指标：${item.metric_name}` : "",
    item.trigger_value !== undefined ? `当前窗口：${number(item.trigger_value, 6)}` : "",
    item.baseline_value !== undefined ? `基线窗口：${number(item.baseline_value, 6)}` : "",
    item.threshold !== undefined ? `阈值：${number(item.threshold, 6)}` : "",
    item.window_size !== undefined ? `窗口大小：${item.window_size}` : "",
    item.baseline_window !== undefined ? `基线长度：${item.baseline_window}` : "",
    item.window_steps?.length ? `证据 step：${item.window_steps.join(", ")}` : "",
    item.confidence === "low" ? "置信度：低（短实验，仅供观察）" : "",
  ].filter(Boolean);
  return `<details class="rule-row ${escapeHtml(item.status)}"><summary><strong>${escapeHtml(ruleNames[item.rule_name] || item.rule_name)}</strong><span>${escapeHtml(names[item.status] || item.status)}</span><small>${escapeHtml(item.reason || item.description || "点击展开详情")}</small></summary><div class="rule-detail">${detail.map((line) => `<div>${escapeHtml(line)}</div>`).join("") || "暂无更多信息"}</div></details>`;
}

function experimentOptions(selectedId) {
  return state.experiments.map((item) => `<option value="${escapeHtml(item.experiment_id)}" ${item.experiment_id === selectedId ? "selected" : ""}>${escapeHtml(item.name || item.experiment_id)}</option>`).join("");
}

function renderDetail() {
  const item = state.detail;
  const latest = item.latest_metrics || {};
  const progress = item.progress || {};
  const labels = item.labels || [];
  const rows = item.steps || [];
  const diagnostics = item.diagnostics || {};
  const latestSignal = rows.at(-1) || {};
  const quality = latestSignal.rollout_quality_counts || {};
  const stability = diagnostics.stability || {};
  el("detail").innerHTML = `
    <section class="detail-header">
      <div class="detail-title">
        <h2>${escapeHtml(item.name || item.experiment_id)}</h2>
        <p>${escapeHtml(item.experiment_id)} · 更新于 ${timeText(item.updated_at)}</p>
      </div>
      <span class="status-badge ${escapeHtml(item.status)}">${escapeHtml(statusNames[item.status] || item.status)}</span>
    </section>

    <section class="metric-strip">
      ${metricItem("训练进度", `${progress.current_step ?? "-"}/${progress.total_steps ?? "-"}`)}
      ${metricItem("最新 Step 平均奖励", number(latest.reward_mean, 4))}
      ${metricItem("Reward 范围", `${number(latest.reward_min, 3)} ~ ${number(latest.reward_max, 3)}`)}
      ${metricItem("Reward 波动", number(latest.reward_std, 4))}
      ${metricItem("有效回答", percent(latest.valid_ratio))}
      ${metricItem("正奖励", percent(latest.positive_ratio))}
      ${metricItem("截断率", percent(latest.clip_ratio))}
      ${metricItem("梯度范数", number(latest.grad_norm, 4))}
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>独立评测</h3><span>${(item.evaluations || []).length} 个结果 · 不使用训练 Rollout 代替</span></div>
      <div class="section-content evaluation-list">${(item.evaluation_alerts || []).map(evaluationAlertHtml).join("")}${(item.evaluations || []).map(evaluationHtml).join("") || '<span class="no-labels">尚未关联独立评测结果。</span>'}</div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>实验可信度</h3><span>配置完整性</span></div>
      <div class="section-content">
        ${readinessHtml(item)}
        <p class="muted-note">对比资格要求记录算法、模型、数据集、Reward 定义与范围、随机种子、训练预算、镜像和代码版本。</p>
      </div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>实时事实标签</h3><span>规则版本 ${escapeHtml(item.rule_version)}</span></div>
      <div class="section-content labels">${labels.length ? labels.map(labelHtml).join("") : '<span class="no-labels">当前没有规则被触发</span>'}</div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>训练指标</h3><span>按 step 更新</span></div>
      <div class="section-content chart-grid">
        <div class="chart-block"><h4>每 Step 平均奖励 / 有效回答</h4><canvas id="rewardChart"></canvas></div>
        <div class="chart-block" title="KL loss 衡量当前策略偏离参考模型所产生的惩罚项；Entropy 衡量输出随机性。"><h4>KL Loss / Entropy</h4><canvas id="policyChart"></canvas></div>
        <div class="chart-block" title="本步参数更新前的整体梯度大小。应结合组内学习信号判断，单个尖峰不能直接判定训练崩溃。"><h4>梯度范数</h4><canvas id="gradientChart"></canvas></div>
        <div class="chart-block" title="每个 prompt 的多次 rollout 组成一组；全对或全错表示组内没有正确性差异。"><h4>正确性分组（每个 Prompt）</h4><canvas id="groupChart"></canvas></div>
        <div class="chart-block"><h4>组内学习信号</h4><canvas id="signalChart"></canvas></div>
        <div class="chart-block"><h4>Rollout 质量</h4><canvas id="qualityChart"></canvas></div>
      </div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>Rollout 与组内学习信号</h3><span>标准诊断快照</span></div>
      <div class="section-content diagnostic-grid">
        ${diagnosticMetric("可用 rollout", percent(latestSignal.rollout_usable_ratio))}
        ${diagnosticMetric("有效 group", percent(latestSignal.effective_group_ratio))}
        ${diagnosticMetric("低信息 group", percent(latestSignal.low_information_group_ratio))}
        ${diagnosticMetric("可区分 pair", percent(latestSignal.mean_informative_pair_ratio))}
        ${diagnosticMetric("全同 reward", latestSignal.constant_score_groups ?? "-")}
        ${diagnosticMetric("近全同 reward", latestSignal.near_constant_groups ?? "-")}
        ${Object.entries(quality).map(([key, value]) => diagnosticMetric(key, value)).join("")}
      </div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>稳定性规则</h3><span>${escapeHtml(stability.config_version || "-")}</span></div>
      <div class="section-content rule-list">${(stability.evaluations || []).map(ruleStatusHtml).join("") || '<span class="no-labels">暂无稳定性评估数据。</span>'}</div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>最近步骤</h3><span>${rows.length} 个 step</span></div>
      <div class="table-scroll"><table class="facts-table"><thead><tr>
        <th>step</th><th>rollout</th><th>有效</th><th>正奖励</th><th>全对</th><th>近全对</th><th>混合</th><th>近全错</th><th>全错</th><th>截断</th><th>KL</th><th>grad</th>
      </tr></thead><tbody>${rows.slice(-30).map((row) => `<tr>
        <td>${row.step}</td><td>${row.rollouts}</td><td>${percent(row.valid_ratio)}</td><td>${percent(row.positive_ratio)}</td>
        <td>${row.has_correctness ? row.all_correct_groups : "-"}</td><td>${row.has_correctness ? row.mostly_correct_groups : "-"}</td>
        <td>${row.has_correctness ? row.mixed_groups : "-"}</td><td>${row.has_correctness ? row.mostly_wrong_groups : "-"}</td>
        <td>${row.has_correctness ? row.all_wrong_groups : "-"}</td><td>${percent(row.clip_ratio)}</td><td>${number(row.kl_loss, 4)}</td><td>${number(row.grad_norm, 4)}</td>
      </tr>`).join("")}</tbody></table></div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>运行配置</h3><span>日志与实验清单</span></div>
      <div class="section-content config-grid">
        ${configCell("算法", item.config.algorithm || "未记录")}${configCell("模型", item.config.model || "未记录")}${configCell("数据集", item.config.dataset || "未记录")}
        ${configCell("领域", manifestValue(item.run_manifest, "task", "domain"))}${configCell("任务", manifestValue(item.run_manifest, "task", "type"))}${configCell("输入维度", manifestValue(item.run_manifest, "dataset", "feature_count"))}
        ${configCell("Reward 定义", manifestValue(item.run_manifest, "reward", "definition"))}${configCell("Reward 最小值", manifestValue(item.run_manifest, "reward", "min"))}${configCell("Reward 最大值", manifestValue(item.run_manifest, "reward", "max"))}
        ${configCell("是否包含 CoT", manifestValue(item.run_manifest, "prompt", "contains_cot"))}${configCell("随机种子", manifestValue(item.run_manifest, "training", "seed"))}${configCell("NPU 数量", manifestValue(item.run_manifest, "resources", "npu_count"))}
        ${configCell("镜像", manifestValue(item.run_manifest, "environment", "image"))}${configCell("Patch 版本", manifestValue(item.run_manifest, "environment", "patch_revision"))}${configCell("代码版本", manifestValue(item.run_manifest, "environment", "code_revision"))}
        ${configCell("Rollout N", item.config.rollout_n)}${configCell("Batch Size", item.config.train_batch_size)}${configCell("Learning Rate", item.config.learning_rate)}
      </div>
    </section>

    <section class="section-band">
      <div class="section-heading"><h3>实验参数对比</h3><span>配置差异与最新指标</span></div>
      <div class="section-content comparison-controls">
        <div class="comparison-picker"><select id="compareExperiment">${experimentOptions(state.experiments.find((entry) => entry.experiment_id !== item.experiment_id)?.experiment_id || item.experiment_id)}</select><button id="compareButton" type="button">开始对比</button></div>
        <div id="comparisonContent" class="comparison-content"><span class="no-labels">选择另一个实验后查看参数和指标差异。</span></div>
      </div>
    </section>

    <section class="section-band">
      <div class="section-heading">
        <h3>Rollout 证据</h3>
        <div class="rollout-toolbar"><input id="rolloutStep" type="number" min="1" value="${progress.current_step ?? ""}" placeholder="step"><button id="loadRollouts" type="button">读取</button></div>
      </div>
      <div id="rolloutContent" class="section-content"><span class="no-labels">选择 step 查看 prompt、输出、reward、pred 和 acc。</span></div>
    </section>`;

  el("loadRollouts").addEventListener("click", loadRollouts);
  el("compareButton").addEventListener("click", loadComparison);
  drawCharts(rows);
}

async function loadComparison() {
  const otherId = el("compareExperiment").value;
  const content = el("comparisonContent");
  if (!otherId || otherId === state.selectedId) {
    content.innerHTML = '<span class="no-labels">请选择另一个实验。</span>';
    return;
  }
  content.textContent = "正在读取对比...";
  try {
    const result = await fetchJson(`/api/compare?id=${encodeURIComponent(state.selectedId)}&id=${encodeURIComponent(otherId)}`);
    const names = result.experiments.map((entry) => entry.name || entry.experiment_id);
    const configRows = result.config_rows.filter((row) => row.changed);
    content.innerHTML = `
      <h4>参数差异</h4>
      <div class="comparison-table"><table class="facts-table"><thead><tr><th>参数</th>${names.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr></thead><tbody>
        ${configRows.map((row) => `<tr><td>${escapeHtml(row.key)}</td>${row.values.map((value) => `<td>${escapeHtml(value ?? "-")}</td>`).join("")}</tr>`).join("") || '<tr><td colspan="3">配置没有差异</td></tr>'}
      </tbody></table></div>
      <h4>最新指标</h4>
      <div class="comparison-table"><table class="facts-table"><thead><tr><th>指标</th>${names.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}<th>差值</th></tr></thead><tbody>
        ${result.latest_metric_rows.map((row) => `<tr><td>${escapeHtml(row.key)}</td>${row.values.map((value) => `<td>${number(value, 4)}</td>`).join("")}<td>${number(row.delta, 4)}</td></tr>`).join("") || '<tr><td colspan="4">暂无可比较指标</td></tr>'}
      </tbody></table></div>
      <div class="comparison-chart-heading"><h4>对齐 step 趋势</h4><select id="comparisonMetric">
        <option value="reward_mean">平均奖励</option><option value="effective_group_ratio">有效 group</option>
        <option value="low_information_group_ratio">低信息 group</option><option value="mean_informative_pair_ratio">可区分 pair</option>
        <option value="kl_loss">KL</option><option value="entropy">Entropy</option><option value="grad_norm">Gradient Norm</option>
      </select></div><canvas id="comparisonChart"></canvas>`;
    const redraw = () => drawComparisonChart(result.series, el("comparisonMetric").value);
    el("comparisonMetric").addEventListener("change", redraw);
    redraw();
  } catch (error) {
    content.innerHTML = `<span class="no-labels">对比失败：${escapeHtml(error.message)}</span>`;
  }
}

function setupCanvas(id) {
  const canvas = el(id);
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, Math.floor(rect.width * ratio));
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return { ctx, width: rect.width, height: rect.height };
}

function drawLineChart(id, rows, series) {
  const { ctx, width, height } = setupCanvas(id);
  const padding = { left: 38, right: 14, top: 16, bottom: 26 };
  const values = series.flatMap((item) => rows.map((row) => row[item.key]).filter((value) => value !== null && value !== undefined).map(Number));
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#d9dee5";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(padding.left, padding.top); ctx.lineTo(padding.left, height - padding.bottom); ctx.lineTo(width - padding.right, height - padding.bottom); ctx.stroke();
  if (!rows.length || !values.length) {
    ctx.fillStyle = "#657180"; ctx.fillText("暂无数据", padding.left + 10, padding.top + 20); return;
  }
  let min = Math.min(...values); let max = Math.max(...values);
  if (min === max) { min -= 0.5; max += 0.5; }
  const x = (index) => padding.left + index / Math.max(rows.length - 1, 1) * (width - padding.left - padding.right);
  const y = (value) => padding.top + (max - value) / (max - min) * (height - padding.top - padding.bottom);
  ctx.font = "11px system-ui"; ctx.fillStyle = "#657180";
  ctx.fillText(max.toFixed(3), 2, padding.top + 4); ctx.fillText(min.toFixed(3), 2, height - padding.bottom);
  series.forEach((item, seriesIndex) => {
    ctx.strokeStyle = item.color; ctx.lineWidth = 2; ctx.beginPath(); let started = false;
    rows.forEach((row, index) => {
      const value = row[item.key]; if (value === null || value === undefined) return;
      if (!started) { ctx.moveTo(x(index), y(Number(value))); started = true; } else { ctx.lineTo(x(index), y(Number(value))); }
    });
    ctx.stroke();
    ctx.fillStyle = item.color; ctx.fillRect(padding.left + seriesIndex * 110, 2, 10, 3);
    ctx.fillStyle = "#657180"; ctx.fillText(item.label, padding.left + 15 + seriesIndex * 110, 7);
  });
  ctx.fillStyle = "#657180"; ctx.fillText(`step ${rows[0].step}`, padding.left, height - 7); ctx.fillText(`step ${rows.at(-1).step}`, width - 62, height - 7);
}

function drawCharts(rows) {
  drawLineChart("rewardChart", rows, [
    { key: "reward_mean", label: "平均奖励", color: "#176b5b" },
    { key: "valid_ratio", label: "valid", color: "#285a8e" },
  ]);
  drawLineChart("policyChart", rows, [
    { key: "kl_loss", label: "KL loss", color: "#8c4f15" },
    { key: "entropy", label: "entropy", color: "#5d4a8d" },
  ]);
  drawLineChart("gradientChart", rows, [{ key: "grad_norm", label: "grad norm", color: "#a22b2b" }]);
  drawLineChart("groupChart", rows, [
    { key: "all_correct_groups", label: "全对", color: "#176b5b" },
    { key: "all_wrong_groups", label: "全错", color: "#a22b2b" },
    { key: "mixed_groups", label: "混合", color: "#285a8e" },
  ]);
  drawLineChart("signalChart", rows, [
    { key: "effective_group_ratio", label: "有效", color: "#176b5b" },
    { key: "low_information_group_ratio", label: "低信息", color: "#a22b2b" },
    { key: "mean_informative_pair_ratio", label: "pair", color: "#285a8e" },
  ]);
  drawLineChart("qualityChart", rows, [
    { key: "quality_usable", label: "可用", color: "#176b5b" },
    { key: "quality_invalid_empty", label: "空输出", color: "#8c4f15" },
    { key: "quality_invalid_truncated", label: "截断", color: "#a22b2b" },
    { key: "quality_invalid_reward", label: "reward", color: "#5d4a8d" },
  ]);
}

function drawComparisonChart(series, metric) {
  const canvas = el("comparisonChart");
  if (!canvas) return;
  const rowsByStep = new Map();
  series.forEach((experiment, index) => {
    experiment.steps.forEach((row) => {
      const combined = rowsByStep.get(row.step) || { step: row.step };
      combined[`value_${index}`] = row[metric];
      rowsByStep.set(row.step, combined);
    });
  });
  const rows = [...rowsByStep.values()].sort((a, b) => a.step - b.step);
  drawLineChart("comparisonChart", rows, series.map((experiment, index) => ({
    key: `value_${index}`,
    label: experiment.name || `实验 ${index + 1}`,
    color: index === 0 ? "#176b5b" : "#a22b2b",
  })));
}

async function loadRollouts() {
  state.rolloutReading = true;
  const step = el("rolloutStep").value;
  const content = el("rolloutContent");
  content.textContent = "正在读取...";
  try {
    const records = await fetchJson(`/api/experiments/${encodeURIComponent(state.selectedId)}/rollouts?step=${encodeURIComponent(step)}&limit=100`);
    content.innerHTML = records.length
      ? `<div class="rollout-list">${rolloutGroupsHtml(records)}</div>`
      : '<span class="no-labels">该 step 没有可读取的 rollout。</span>';
    state.rolloutEvidence = captureDetailState();
  } catch (error) {
    content.innerHTML = `<span class="no-labels">读取失败：${escapeHtml(error.message)}</span>`;
  }
}

el("refreshButton").addEventListener("click", () => loadExperiments(true));
el("searchInput").addEventListener("input", renderList);
el("statusFilter").addEventListener("change", renderList);
window.addEventListener("resize", () => { if (state.detail) drawCharts(state.detail.steps || []); });

loadExperiments();
state.timer = setInterval(() => loadExperiments(false), 10000);
