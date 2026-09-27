# 可追溯的正式可视化交付链

这条链只为正式实验生成对比图。当前仓库尚未冻结候选方法，因而没有生成任何 MVTec AD 或 VisA 正式样本清单、失败案例或正式图片。现有测试全部使用 `pytest` 临时目录中的合成图片、掩码和预测数组，不枚举也不读取真实测试集或 GT。

## 两类样本必须分开

`result_independent_preregistered` 是 F0 推理前冻结的样本。可以在配置中显式给出 `sample_id`，也可以按固定盐值哈希分层选择。默认分层键为：

- 正常样本：`category / normal`；
- 异常样本：`category / defect_type / anomaly_scale`。

配置只接受上述两种方式。按模型分数、误差或可视化效果选择固定样本会被 schema 校验拒绝。

`outcome_dependent_machine_selected` 是推理完成后由 F0 已冻结规则选出的失败候选。当前支持两种排序：

- 候选方法单图像素 AP 从低到高；
- `候选像素 AP - 基线像素 AP` 从低到高。

每类数量、升序规则和 `sample_id` 升序 tie-break 都在 F0 固定。解析工具输出所有候选的分数、完整名次和入选标记，不提供手工 ID 参数。它展示的是预定义规则下的失败候选，不可替代整体指标。

## 正式测试清单 schema

输入文件是一个 JSON 对象：

```json
{
  "schema_version": 1,
  "kind": "formal_test_inventory",
  "inventory_id": "optional-stable-id",
  "dataset": "mvtec_ad",
  "split": "test",
  "entries": [
    {
      "dataset": "mvtec_ad",
      "category": "bottle",
      "sample_id": "bottle/crack/001",
      "label": 1,
      "defect_type": "crack",
      "anomaly_area_fraction": 0.0123,
      "anomaly_scale": "small",
      "source_image": "bottle/test/crack/001.png",
      "source_image_sha256": "<64 lowercase hex>",
      "source_mask": "bottle/ground_truth/crack/001_mask.png",
      "source_mask_sha256": "<64 lowercase hex>"
    }
  ]
}
```

`anomaly_scale` 只能是 `normal`、`small`、`medium` 或 `large`。正常样本必须使用 `label=0`、`defect_type=good`、`anomaly_area_fraction=0`、`anomaly_scale=normal`，且两个 mask 字段为 `null`。异常样本必须提供源 mask 和哈希。所有路径必须是数据根目录下的 POSIX 相对路径；冻结时会重新计算源图片和源 mask 的 SHA-256。

## F0 选择配置 schema

确定性分层配置如下：

```json
{
  "schema_version": 1,
  "kind": "formal_visualization_selection_config",
  "dataset": "mvtec_ad",
  "model_seed": 1887537248,
  "selection_salt": "project-and-round-specific-salt",
  "fixed_selection": {
    "mode": "deterministic_hash_stratified",
    "normal_per_category": 1,
    "anomaly_per_defect_scale_stratum": 1
  },
  "failure_case_rule": {
    "baseline_method": "DRAEM",
    "candidate_method": "FrozenCandidate",
    "candidate_filter": "anomalies_with_positive_evaluation_mask",
    "rankings": [
      {
        "name": "candidate_absolute_failures",
        "metric": "candidate_pixel_average_precision",
        "order": "ascending",
        "per_category": 2,
        "tie_break": "sample_id_ascending"
      },
      {
        "name": "candidate_regressions_vs_baseline",
        "metric": "candidate_minus_baseline_pixel_average_precision",
        "order": "ascending",
        "per_category": 2,
        "tie_break": "sample_id_ascending"
      }
    ]
  }
}
```

如需显式样本，把 `fixed_selection` 改成：

```json
{
  "mode": "explicit_sample_ids",
  "sample_ids": ["bottle/good/000", "bottle/crack/001"]
}
```

冻结入口还要求 `configs/preregistered_experiments.json` 中的 `formal_method_registry.status` 已是 `frozen_before_formal_evaluation`，并核对基线、候选和方法列表。输出文件拒绝覆盖，包含输入清单 SHA、配置 SHA、方法注册 SHA、每个样本的源文件 SHA，以及完整的失败候选池。应在任何 F0 推理前把该文件提交或归档为本轮唯一清单。

## 运行顺序

所有命令先加载项目环境：

```powershell
. .\scripts\Set-ProjectEnvironment.ps1 | Out-Null
```

方法和正式测试清单都冻结后，且仍未运行正式推理时：

```powershell
.\.venv\Scripts\python.exe scripts\freeze_visualization_selection.py `
  --test-inventory manifests\formal\mvtec_test_inventory.json `
  --dataset-root data\mvtec_ad `
  --selection-config configs\formal_visualization_mvtec.json `
  --method-registry configs\preregistered_experiments.json `
  --output manifests\formal\mvtec_visualization_selection.json
```

正式基线和候选预测完成后，按冻结规则解析失败候选。每个 `--prediction-dir` 必须是一个类别的一次完整正式运行；多类别时重复传入：

```powershell
.\.venv\Scripts\python.exe scripts\resolve_visualization_failures.py `
  --selection-manifest manifests\formal\mvtec_visualization_selection.json `
  --prediction-dir artifacts\formal\DRAEM\bottle\seed_1887537248 `
  --prediction-dir artifacts\formal\FrozenCandidate\bottle\seed_1887537248 `
  --output manifests\formal\mvtec_visualization_failures.json
```

渲染时可同时包含预注册样本和机器选出的失败候选：

```powershell
.\.venv\Scripts\python.exe scripts\render_visualization_comparisons.py `
  --selection-manifest manifests\formal\mvtec_visualization_selection.json `
  --dataset-root data\mvtec_ad `
  --prediction-dir artifacts\formal\DRAEM\bottle\seed_1887537248 `
  --prediction-dir artifacts\formal\FrozenCandidate\bottle\seed_1887537248 `
  --failure-resolution manifests\formal\mvtec_visualization_failures.json `
  --output-dir artifacts\formal_visualizations\mvtec
```

## 完整性与显示口径

每个预测目录必须满足以下条件：

- `run_state.json` 为完整的 `formal_full`、`full`、`formal_matrix_candidate` 运行；
- `run_state.json` 中的 run config、metrics 和 prediction index SHA 与磁盘文件一致；
- `run_config.json` 明确声明 metrics 从保存后的数组重载计算；
- `metrics.json` 明确指向 `predictions/index.jsonl` 中的 `raw_map` 和 `evaluation_mask`；
- 每个被展示数组的 SHA、dtype、维度和有限性再次核对；
- 各方法的 `sample_id`、标签、缺陷类型、源图路径和 metric mask 完全一致。

PNG 的列名明确标出 `GT (metric mask)` 和每个方法的 `raw_map (metric source)`。同一张图中所有方法共用该样本的原始异常分数最小值和最大值作为 `magma` 色标。这个缩放只发生在 Matplotlib 显示阶段；工具不会重写 `.npy`，也不会把显示归一化送回指标计算。`figure_manifest.json` 保存源文件、预测索引、数组和 PNG 的 SHA，以及色标上下界。

合成契约测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_visualization_delivery.py
```

测试覆盖源文件和预测数组篡改、跨方法样本错配、结果驱动固定选样拒绝、冻结失败规则的自动解析，以及两次渲染逐 PNG SHA 一致。
