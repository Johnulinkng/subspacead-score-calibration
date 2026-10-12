# 工业异常检测实验项目

项目目录：`E:\3k`。本项目从公开实现和真实公开数据集开始，此前个人研究不进入本项目的方法、代码或结果。

当前阶段（2026-10-12）：按用户新的约一天、低资源整合要求，已完成独立的 PaDiM 风格 ResNet18 快速复现轮次，配置见 `configs/quick_reproduction_20261012.json`。这轮完整读取 MVTec AD 15 类与 VisA 12 类共 12,288 张正常训练图，对 3,887 张真实测试图固定一个种子、四组收缩/平滑组合，得到 108 行逐类结果及 15,548 张原始异常图。仅使用两个 CPU 线程，不进行反向训练。2026-10-09 的 DRAEM 700 轮训练仍按用户要求暂停，暂停记录见 `docs/training_paused_20261009.md`。

本轮固定协方差收缩候选在两个数据集的四项宏平均指标均低于基线；AUPRO 在 MVTec AD 下降 0.407 个百分点，在 VisA 下降 1.416 个百分点。完整实验矩阵及模型/预测完整性审计通过，但改善性能与论文级研究目标尚未达成。最终快速交付包位于 `artifacts/deliveries/quick-padim-20261012/`。

快速轮次已在新配置冻结后读取真实 test/GT，旧 DRAEM 开发注册此后不能继续宣称这些测试数据仍未接触。快速轮次的本地集成、评测与材料由本项目产生，PaDiM 方法、torchvision 骨干与公开权重保留其实际来源；完整运行与效果提升是分别核验的事项。

DRAEM 作者发布的 5.43 GB 权重归档已完成 SHA-256、ZIP CRC 与路径安全检查；解出的 15 类 30 个 checkpoint 均使用 `torch.load(weights_only=True)` 审计，并严格加载进当前作者网络结构。它们只用于作者发布权重核对，不能替代本项目多种子训练结果。

## 原 DRAEM 研究路线（暂停）

- 主基线：作者官方 DRAEM，提交 `2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`，MIT。
- 工程参考：Anomalib。库内实现与作者实现必须逐项核对，暂不替代官方基线。
- 强对照：PatchCore 与 RD4AD。不得因为测试成绩不利而替换基线或对照。
- 正式数据范围：MVTec AD 全 15 类和 VisA 全 12 类；DRAEM 的训练异常源为 DTD 纹理。目标是形成达到 SCI 三区研究完整度的双数据集证据，不能将其表述为录用保证。
- 可检查的论文级实验门槛见 `docs/research_acceptance.md`；主基线采用 DRAEM 作者实现，同时冻结 PatchCore 与 RD4AD 为两类更强参照，避免只与较老基线比较。具体公平协议见 `docs/comparator_protocol.md`。
- 增加第三个数据集前，先说明要解决的审稿风险、预计成本和是否影响现有实验矩阵。
- 开发阶段只使用 VisA `train/good` 与内容隔离的 DTD 划分；正式测试集不参与模块、超参数、checkpoint 或样本选择。固定划分、三组派生种子、候选预算和停止规则见 `docs/development_protocol.md`。

## 已有文件

- `docs/baseline_selection.md`：社区、来源、方法匹配与工程风险。
- `docs/experiment_protocol.md`：开发/正式隔离、统一评测和交付协议。
- `docs/formal_training_inputs.md`：MVTec AD、VisA 与 DTD 的冻结 train-only 输入及重验链。
- `docs/formal_test_inventory.md`：仅在 F0 冻结后创建的完整 test/GT 身份清单。
- `docs/formal_source_provenance.md`：正式运行的项目/vendor Git HEAD、tree 与干净状态门禁。
- `docs/visualization_delivery.md`：固定样本、机械失败案例和异常图可追溯渲染。
- `docs/monitored_experiments.md`：训练前预检、原子锁和运行中资源证据。
- `configs/project.json`：路径、候选配置、阶段状态。
- `manifests/repository_snapshot.json`：GitHub API 社区状态快照；热度不是性能证据。
- `manifests/migration.json`：D 盘到 E 盘的迁移记录。
- `third_party/DRAEM`：未改动的作者代码，保留 Git 历史和许可证。
- `scripts/Set-ProjectEnvironment.ps1`：将本次 shell 的缓存、临时文件和后续运行路径设在项目内。
- `scripts/Test-ProjectReadiness.ps1`：只读检查源代码、数据目录、项目环境和空间；不启动训练。
- `scripts/Invoke-MonitoredExperiment.ps1`：先执行固定资源门禁，再以原子锁启动并监控项目 Python 子进程。
- `scripts/build_formal_test_inventory.py`：正式方法冻结后构建或重验逐文件哈希的完整测试清单。
- `artifacts/validation/draem_manifest_inputs_candle.json`：冻结 train 清单的 candle 只读核验；未启动训练。
- `artifacts/validation/synthetic_dev_candle_validate/validation.json`：VisA-dev/DTD-dev 全部登记文件的大小与 SHA-256 核验；渲染数和模型加载数均为 0。
- `artifacts/validation/synthetic_dev_render_candle_v3`：两个固定 recipe 的重复渲染完整性审计；保存 `float32` 输入与布尔 mask，不计算性能。

```powershell
Set-Location E:\3k
. .\scripts\Set-ProjectEnvironment.ps1
.\scripts\Sync-Environment.ps1
.\scripts\Test-Environment.ps1
.\scripts\Test-ProjectReadiness.ps1
```

## 后续研究

1. 本轮交付用于验证低资源流程与真实效果，不据测试成绩再次挑选收缩系数，也不删除下降类别。
2. 原 DRAEM 训练保持暂停；恢复需要用户新的启动指令。
3. 新的改进研究需要另行冻结开发协议，以独立验证、多种子结果与强对照检验改进；已有真实测试访问记录必须保留。

本轮快速交付包括实验包、完整指标/消融表、54 组可视化对照和复现核验说明。它不代表原论文级研究目标已经完成。
