# 资源恢复与低占用运行记录

记录日期：2026-10-07。项目主目录为 `E:\3k`；本文件记录资源恢复路径，不改变正式实验的资源门禁。

## 当前门禁

正式或完整开发运行由 `scripts/Invoke-MonitoredExperiment.ps1` 先执行
`scripts/Test-TrainingPreflight.ps1`。当前固定要求为：可用主机内存至少 8 GiB、项目盘至少 40 GiB、无 GPU 计算进程、无 WSL 运行实例和无项目训练锁。门禁失败时不会创建子进程、不会写入 checkpoint，也不会产生可进入指标表的结果。

2026-10-07 的最新检查为：可用内存 6.759 GiB、项目盘可用空间 214.054 GiB、GPU 显存空闲 8018 MiB、无计算进程、无 WSL、无训练锁。结果保存于
`artifacts/preflight/preflight_20261007T151053011Z.json`。

## 已采用的低占用设置

- DRAEM、RD4AD 和 PatchCore 的项目层 DataLoader 默认 `workers=0`。这避免 Windows 多进程 worker 复制父进程数据集和预取队列。
- DRAEM 保留原有的 CUDA 自动 pinned-memory 行为，同时支持在资源 smoke 中显式传入 `--no-pin-memory`；最终解析值写入 `run_config.json`，避免运行口径不可追溯。
- PatchCore 的 FAISS 查询 worker 在正式 smoke 前不盲目提高；设备分工和 worker 数必须通过受监控 smoke 冻结。
- 正式训练仍保留完整作者轮数、统一输入尺寸和预注册随机种子。降低 batch、轮数或关闭数据增强只能标为 debug/smoke，不能进入正式结果。

## 公开资料依据

PyTorch 官方 DataLoader 文档说明，多 worker 的总内存开销会随 worker 数量和父进程大小增长，并且 Windows 使用 spawn，启动行为不同于 Unix。项目因此默认使用 `num_workers=0`，同时保留显式配置并在 run_config 中记录。参见：
<https://docs.pytorch.org/docs/stable/data.html>

DRAEM 作者仓库存在 CUDA 内存不足的公开 issue；这只说明运行资源需要实测，不能替代本项目的训练或性能证据。参见：
<https://github.com/VitjanZ/DRAEM/issues/2>

## 恢复顺序

1. 释放或关闭占用主机内存的应用，等待可用内存达到 8 GiB；不终止用户进程，不修改门禁阈值。
2. 重新执行项目环境脚本和训练前预检，并保存新的 JSON 证据。
3. 先运行一个 MVTec 类和一个 VisA 类的受监控 smoke，记录峰值主机内存、显存、耗时和输出契约。
4. 只有 smoke 通过后，才冻结 DRAEM、PatchCore 和 RD4AD 的完整运行配方，然后按三种子开发矩阵推进。

任何只为绕过门禁而减少数据、轮数、类别或随机种子的运行都不进入正式矩阵，也不用于支持改进结论。
