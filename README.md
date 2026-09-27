# 工业异常检测实验项目

项目目录：`E:\3k`。本项目从公开实现和真实公开数据集开始，与用户此前的 DDAD、MDG-AD 研究完全分离。

当前阶段：环境与数据准备。已取得 DRAEM 官方代码并锁定现代运行环境；DTD R1.0.1 已从官方入口下载、通过归档检查并核对为 47 类 5,640 张图。VisA 官方归档已完成长度、SHA-256、路径安全和完整划分审计，并按官方 1-class 协议准备为 12 类、10,821 张图和 1,200 张异常掩码。MVTec AD 仍待通过官方表单取得，因此尚未运行正式模型实验，也没有性能提升结论。

## 当前路线

- 首个复现候选：作者官方 DRAEM，提交 `2dbf67397ab5c10a1494e5ae70ab59a25d7c35ef`，MIT。
- 工程参考：Anomalib。库内实现与作者实现必须逐项核对，暂不替代官方基线。
- 条件备选：RD4AD；较强参照：PatchCore。不得因为测试成绩更低而更换基线。
- 正式数据范围：MVTec AD 全 15 类和 VisA 全 12 类；DRAEM 的训练异常源为 DTD 纹理。目标是形成达到 SCI 三区研究完整度的双数据集证据，不能将其表述为录用保证。
- 可检查的论文级实验门槛见 `docs/research_acceptance.md`；主基线采用 DRAEM 作者实现，同时保留更强方法作公平参照，避免只与较老基线比较。
- 增加第三个数据集前，先说明要解决的审稿风险、预计成本和是否影响现有实验矩阵。

## 已有文件

- `docs/baseline_selection.md`：社区、来源、方法匹配与工程风险。
- `docs/experiment_protocol.md`：开发/测试隔离、统一评测和交付约定草案。
- `configs/project.json`：路径、候选配置、阶段状态。
- `manifests/repository_snapshot.json`：GitHub API 社区状态快照；热度不是性能证据。
- `manifests/migration.json`：D 盘到 E 盘的迁移记录。
- `third_party/DRAEM`：未改动的作者代码，保留 Git 历史和许可证。
- `scripts/Set-ProjectEnvironment.ps1`：将本次 shell 的缓存、临时文件和后续运行路径设在项目内。
- `scripts/Test-ProjectReadiness.ps1`：只读检查源代码、数据目录、项目环境和空间；不启动训练。

```powershell
Set-Location E:\3k
. .\scripts\Set-ProjectEnvironment.ps1
.\scripts\Sync-Environment.ps1
.\scripts\Test-Environment.ps1
.\scripts\Test-ProjectReadiness.ps1
```

## 下一步

1. 验证真实数据下载入口、归档完整性和许可证；保存文件清单及哈希。
2. 完成项目内专用环境，处理官方旧 Linux 依赖与本机兼容性，记录所有修复。
3. 单类运行链验证、资源测量及官方权重核验；这些结果不得当成完整基线成绩。
4. 冻结完整训练与评测协议后，再开展全类别复现和改进开发。

本轮只完成准备材料。四项最终交付物（实验包、指标/消融表、可视化、复现说明）尚未形成。
