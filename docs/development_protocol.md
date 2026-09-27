# 训练侧开发协议 v1

状态：已冻结数据划分、随机种子、合成开发配方和试验预算；方法模块本身仍需在第一次候选运行前单独冻结。本协议不包含模型训练结果。

## 数据边界

开发阶段只允许读取 VisA 官方 1-class 布局下各类别的 `train/good` 原图，以及 DTD R1.0.1 图像。构建脚本不会遍历、列举或打开 VisA 的 `test` 与 `ground_truth`。MVTec AD 和 VisA 正式 test 在开发阶段均不能用于选模块、超参数、checkpoint、阈值、种子或可视化样本。

VisA 每类别按原始文件内容组划分：目标开发比例为 20%，剩余为训练集。每个文件记录相对路径、大小和 SHA-256；内容完全相同的文件作为一组进入同一侧。划分顺序由公开盐值和内容哈希决定，不受文件系统枚举顺序影响。增强与裁剪只能在划分之后执行。

DTD 在 47 个纹理类别内分别按同一原则划分 80% 训练源和 20% 开发源。模型训练合成异常只能取 DTD-train；合成开发集只能取 DTD-dev。开发完成并冻结方法后，正式重训是否合并全部 DTD 必须在首次正式评测前登记，并对所有方法一致。

## 固定随机性与合成开发集

正式模型运行种子固定为 `1887537248, 1337854654, 1609762728`。它们不是根据结果选择：对固定字符串 `industrial-anomaly-lab|formal-v1|2026-09-27` 计算 SHA-256，得到 `f08188604fbe0abedff307a881d63b5c32a99bc27c1fdf5a294b9098cbab4055`，取偏移 0、8、16 的三个 8 位十六进制块并分别执行 `& 0x7fffffff`。生成器会核验该推导。数据划分与合成开发配方使用独立固定盐值，因此不同模型种子和不同方法看到完全相同的开发样本。

每张 VisA-dev 正常原图保留一个未增强正常样本，并登记两个合成正例配方。recipe schema v3 对每个正例固定：目标图和 SHA-256、目标是否执行作者式 30% 旋转及角度、DTD-dev 来源和 SHA-256、32 位配方种子、三个增强器及其递归子种子、Perlin 子种子/重试序号/横纵尺度/旋转角/阈值 `0.5`、混合系数及输入尺寸。生成器只计算 Perlin mask 来排除空 mask，不解码或渲染 VisA/DTD 图像。后续评测必须按清单渲染，不能现场重采样。

builder 与 consumer 共用源划分和完整 recipe 推导函数。consumer 先按 `split_salt`、`dev_fraction` 和原文件内容组重推每个 VisA/DTD train/dev 行，再从 recipe salt、目标记录、variant 和有序 DTD-dev 清单重算全部 3,468 条 recipe。比较使用 canonical JSON，因此字段集合、顺序对应关系、值和 JSON 数值类型都必须完全相同；仅把条目改成另一个合法 DTD-dev 文件、交换等量 train/dev 行或把整数改写为等值浮点数，整体重签后仍会失败。根清单中的 `artifact_files.path` 和 `config_path` 都是以项目根为基准的相对路径。

开发集固定为每个原图 1 个正常样本和 2 个合成正例，比例为 1:2。它不同于作者训练 loader 最终约 1:1 的随机正常/合成输出，只用于固定候选排序；P-AP 因而是该固定 prevalence 下的开发指标。正常误报单独以冻结阈值 `0.5` 计算图像级和像素级 FPR。所有结果必须标记 `synthetic-development-only`、`claim_eligible=false`，不能充当真实缺陷或正式数据集效果证据。

## 预注册矩阵与停止规则

完整机器可读矩阵在 `configs/preregistered_experiments.json`。流程为：管线 smoke、三种子全类别基线、最多三个候选族的固定预算筛选、三种子确认、逐项消融，最后才进入 MVTec AD 与 VisA 的正式评测。

候选进入三种子确认需同时满足：合成开发集宏平均 AUPRO@0.30 至少提高 1.0 个百分点；至少 8/12 类不回退；正常开发误报率增加不超过 1.0 个百分点。单种子全类别筛选若宏平均落后基线至少 0.5 个百分点，或超过两个类别回退超过 3.0 个百分点，立即停止该候选。每个候选族最多八组超参数，总候选族最多三个。不得用最好 seed、最好 epoch 或类别子集代替注册矩阵。

正式 test 一旦在 F0 前被读取，本轮注册失效，需要披露并建立新的探索轮次。F0 后发生的模型改动也属于新一轮，不能回写成本轮的未接触测试结果。

## 生成与核验

```powershell
Set-Location E:\3k
. .\scripts\Set-ProjectEnvironment.ps1
.\.venv\Scripts\python.exe .\scripts\build_development_manifests.py
.\.venv\Scripts\python.exe -m pytest .\tests\test_development_protocol.py -q
```

输出固定在 `manifests/development/visa_v1`。`manifest.json` 记录每类计数、清单哈希、允许读取的目录、`formal_test_content_accessed=false` 和 `training_started=false`。JSONL 文件使用项目相对路径，便于迁移和复核。

## 合成开发验证与评测入口

正式候选运行前先执行只读校验和有限渲染审计：

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_draem_synthetic_dev.py `
  --category candle --seed 1887537248 --validate-only `
  --output-dir .\artifacts\validation\synthetic_dev_candle_validate

& .\.venv\Scripts\python.exe .\scripts\evaluate_draem_synthetic_dev.py `
  --category candle --seed 1887537248 --render-audit-count 2 `
  --output-dir .\artifacts\validation\synthetic_dev_render_candle_v3
```

只读校验只打开清单登记的当前类别 VisA-dev 与 DTD-dev 文件，并逐文件复核大小和 SHA-256；不会遍历 `test` 或 `ground_truth`。路径解析前会拒绝符号链接和 Windows reparse point。有限审计把同一 recipe 渲染两次，要求逐字节相同，并保存 `float32` BGR 输入、布尔 mask、索引和 SHA-256，不加载模型也不计算性能。

renderer-v2 固定了 DRAEM 的算子族和顺序、OpenCV BGR/线性缩放、目标旋转概率、imgaug 参数族、vendor Perlin 公式、阈值和 blend 公式。这是算子与分布口径的复现，不声称重放作者训练时未登记的全局 RNG 轨迹。逐字节约束指同一 renderer-v2 recipe 在锁定环境中的内部重放；imgaug 0.4.0 的全局 RNG 区段已串行化，直接多线程调用也必须得到同一哈希。

根清单还冻结 renderer/evaluator 的完整 provenance：环境锁文件、recipe builder、训练与评测 manifest resolver、模型运行/安全加载、推理、指标、原子落盘、两个生产 CLI，以及 DRAEM 的 data loader、Perlin 和网络源码。训练侧与合成开发侧都会按当前文件字节和包版本重算该块；任一登记文件漂移时二者同时失败关闭。validate-only、render audit 与后续模型评测的 `input_evidence` 均保留完整 provenance 及其 canonical SHA-256。

模型 checkpoint 产生后，`--evaluate` 会对完整的正常 dev 加合成正例集合运行 DRAEM，逐样本保存原始 `float32` 异常图和布尔 mask。统一 I-AUROC 使用落盘 raw map 最大值；DRAEM 21×21 池化分数只作为 `method_native_image_score` 和 `image_auroc_method_native` 单列。P-AUROC、P-AP、AUPRO@0.30 和主正常误报指标也只从这些落盘数组重新加载计算，确保指标与后续可视化共用同一份异常图；native 正常误报另存，不能替代共同口径。该入口不包含任何 optimizer 或训练调用。

## DRAEM 运行链约束

清单已接入 `scripts/train_draem.py`。开发运行必须显式传入 `--input-protocol visa_train_dev_dtd_v1` 和一个注册种子；默认的 `author_full_train` 仅用于作者式复现，不能作为开发协议证据。运行链依据 `configs/development_protocol.json` 的 `frozen_manifest.sha256` 锁定根清单，并逐层验证全部子清单、所选文件大小与 SHA-256、dataset/category/split 字段和路径边界，只把 VisA 当前类别与 DTD 的 `split=train` 路径传入 Dataset。

如重新生成 `manifests/development/visa_v1/manifest.json`，即使划分内容相同，生成时间也会改变根哈希。新的清单必须先经过审计，再显式更新 `frozen_manifest.sha256`；运行链不会自动接受新哈希。
