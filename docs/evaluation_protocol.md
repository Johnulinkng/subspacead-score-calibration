# 统一评测协议：`industrial_anomaly_raw_map_max_256_v2`

状态：AUPRO 核心实现已通过官方代码交叉验证；完整数据适配与正式运行协议仍待冻结。

## 口径

- 输入为 256×256 未量化浮点异常图；不得对每张图单独 min-max 后用于指标。
- GT 使用最近邻缩放并保持二值。图像级标签来自数据清单，不从缩放后的 mask 推断。
- 主表 I-AUROC 的图像分数固定为每张**已落盘、未逐图归一化 raw anomaly map 的最大值**。指标计算和可视化读取同一个 `raw_map` 文件；不得让方法专属图像分数进入通用 `metrics.image_auroc`。
- `predictions/index.jsonl` 必须同时保存 `raw_map_max_score` 和 `method_native_image_score`。DRAEM 的 21×21 平均池化后最大值、PatchCore 的作者 patch-score 聚合、RD4AD 的作者平滑图最大值只用于单列 `metrics.image_auroc_method_native`；不同方法的 native 指标不作公平横向主表比较。
- P-AUROC 与 P-AP 使用完整类别全部测试图的所有像素；P-AP 使用 `average_precision_score`，不是 PR 曲线梯形面积。
- AUPRO@0.30 使用 8 连通异常区域、区域等权、完整类别所有背景像素定义 FPR、相同分数共同更新阈值，在 FPR=0.30 线性插值并除以 0.30。
- 每类别单独计算；MVTec AD 对完整 15 类、VisA 对完整 12 类分别等权宏平均。每个种子先独立评测，再汇总种子；不先平均多个种子的异常图。

实现位于 `src/industrial_anomaly_lab/metrics.py`，合成小数组契约测试位于 `tests/test_metrics.py`。公共 `evaluate_category()` 不再接收外部 `image_scores`，从 API 层阻止方法原生分数混入共同 I-AUROC。

指标文件使用 schema v2，并显式记录 `image_score_contract`。正式聚合器重新读取哈希绑定的预测索引和每个 `.npy` raw map，复核文件 SHA-256、逐样本最大值、共同 I-AUROC 与 native I-AUROC。旧 schema、旧协议名、通用 `image_score` 字段或缺少双分数字段的结果全部失败关闭，不能静默迁移。

官方交叉验证覆盖 7 个固定边界案例、500 个确定性随机案例和 5 个 FPR 上限，共 2,535 次比较；最大绝对差为 `3.193105974075294e-08`，在 `1e-6` 容差内全部通过。脚本、来源哈希和机器报告见 `scripts/validate_aupro_against_official.py`、`docs/aupro_official_crosscheck.md` 与 `artifacts/validation/aupro_official_crosscheck.json`。

## 与作者旧口径的边界

DRAEM 作者脚本对 GT 使用默认线性插值，再转为 `uint8`，会丢弃一部分边缘像素；本统一口径改为最近邻。作者发布权重的核对结果与统一结果必须分栏，不能把统一口径结果冒充作者原始 97.5/68.9。

MVTec 官方评测包 v1.0 要求原图分辨率 TIFF，并默认以异常图 raw max 作为图像分数。项目采用其 AUPRO 与 raw-map-max 图像分数定义，但当前 256 像素协议不等于完整执行官方原尺寸协议。

官方评测包的只读核验 SHA-256：`dfcda7d67eee25316ec6ae5042c0b1684a4cabf33b2346be351e2ce36013f220`。
