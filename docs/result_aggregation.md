# 正式结果聚合与统计

正式结果不能手工挑选或从日志截图抄表。`src/industrial_anomaly_lab/result_aggregation.py` 只接受状态为 `completed`、`run_purpose=formal_full`、`claim_scope=formal_matrix_candidate`、正式 eligibility 为真且协议为 `industrial_anomaly_raw_map_max_256_v2` 的落盘运行。`execution_scope=full` 本身不具备正式含义；作者复现和开发全程即使是 full 也会被拒绝。聚合器会按当前冻结配置重新验证 inference recipe、训练 checkpoint manifest、method/dataset/category/seed、checkpoint 字节与结果文件哈希，然后才读取指标。完整字段见 `docs/formal_provenance.md`。

schema-v2 指标必须声明共同 `image_auroc` 来自 `raw_map_max_score`，方法原生 I-AUROC 来自 `method_native_image_score`。聚合器会重新读取预测索引以及每个哈希绑定的 `.npy` raw map 和 evaluation mask，逐行复核路径、SHA-256、形状、有限值与二值掩码约束；随后从这些落盘数组重算 I-AUROC、P-AUROC、P-AP、AUPRO@0.30，并另外重算方法原生 I-AUROC。它还会从 `formal_provenance` 重新定位并验证正式测试 inventory bundle，要求 `run_config.json`、`metrics.json`、`run_state.json` 的清单证据完全一致，并逐行核对样本 ID、标签、缺陷类型、源图/掩码相对路径与 SHA-256。旧协议、旧 schema、通用 `image_score`、缺失双分数字段、缺失正式测试清单、raw map、evaluation mask、源文件哈希或重算数值不一致都会被拒绝，旧结果不能仅靠改字段名进入新主表。

每个数据集先逐类别计算指标，再在每个种子内做类别宏平均，随后报告多个种子的均值与样本标准差。改进与基线的配对比较先在相同类别、相同种子上求差，再把种子差值按类别平均；同时报告严格改善比例与不回退比例，预注册的“至少 60% 类别不回退”读取 `nonnegative_category_fraction`。95% 置信区间对类别做固定种子 bootstrap，显著性使用类别级双侧精确符号翻转检验。这样不会把像素或多个种子误当成大量独立样本。

提供 `--baseline` 与 `--candidate` 时，聚合器还会把 `configs/preregistered_experiments.json` 中冻结的数值口径机械化为 `claim_adjudication`：两个数据集主指标均为正、等权平均增益至少 1.0 个百分点、每个数据集不回退类别比例至少 60%、至少一个数据集 bootstrap 下界高于零且另一数据集点估计为正、每项次要指标在每个数据集回退不超过 0.5 个百分点。任一项失败时 decision 为 `fail`，完整结果仍然保留。

主比较固定为 AUPRO@0.30；I-AUROC、P-AUROC 与 P-AP 同样输出。统计显著不替代实际效应量：报告必须同时保留平均差、置信区间、改善类别比例和逐类别差值。

冻结正式矩阵后使用 `scripts/aggregate_results.py`。命令必须显式给出方法与种子，而且方法顺序、基线/候选对、三个种子、bootstrap 次数和 bootstrap 种子必须与 `configs/preregistered_experiments.json` 完全一致；当前候选与消融尚未冻结，聚合器会故意拒绝正式运行。脚本读取 `configs/project.json` 中完整的 MVTec AD 15 类和 VisA 12 类，只有注册完成且所有组合精确齐全时才生成 `per_category_seed.csv`、`macro_by_seed.csv`、`method_summary.csv` 和带配对统计的 `aggregation_manifest.json`。输出目录必须为空，防止把不同协议的表混在一起。
